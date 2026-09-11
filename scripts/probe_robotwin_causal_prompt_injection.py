#!/usr/bin/env python3
"""Frozen-foundation probe for ZeVA causal-prefix injection.

The script has two deliberately separate entry points:

* ``run_feature_probe`` audits Eq. (8) prompt construction from a saved
  ``torch.save`` feature payload and is runnable without a RoboTwin runtime.
* ``run_frozen_prefix_probe`` accepts a caller-supplied frozen PI0.5 forward
  function and a :class:`PrefixPromptHook`.  It compares Base/zero/correct/
  permuted-retrieval conditions and performs the gradient/RMS audit needed
  before any Stage-2 unfreezing.
* ``run_pi05_foundation_probe`` loads the actual LeRobot PI0.5 checkpoint and
  executes its normal training forward path.  This is the non-toy smoke test;
  it is deliberately short and never runs an optimizer.

The command-line path never trains or updates parameters.  It intentionally
does not import ``RobotWinZevaPolicy``: the new prefix hook can be installed
around the exact LeRobot ``foundation.model`` without changing the active
policy implementation.

Feature payload format (``torch.save``):

    {
        "global": Tensor[B, Dg], "phase": Tensor[B, Dp],
        "brief": Tensor[B, Hb, Ds], "retrieved": Tensor[B, R, Ds],
        "brief_mask": Optional[bool Tensor[B, Hb]],
        "retrieved_mask": Optional[bool Tensor[B, R]],
    }

The aliases ``global_token``, ``phase_token``, ``brief_signals`` and
``retrieved_signals`` are accepted as well.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from dataclasses import asdict
import json
from pathlib import Path
import sys
from types import ModuleType
import typing
from typing import Any

import torch
from torch import Tensor
import typing_extensions

from openpi.zeva.causal_prompt_injection import CausalPromptInjection
from openpi.zeva.causal_prompt_injection import CausalPromptResult
from openpi.zeva.causal_prompt_injection import PrefixPromptHook
from openpi.zeva.causal_prompt_injection import PromptBranches
from openpi.zeva.causal_prompt_injection import PromptGradientAudit
from openpi.zeva.causal_prompt_injection import PromptMode
from openpi.zeva.causal_prompt_injection import audit_prompt_gradients
from openpi.zeva.causal_prompt_injection import tensor_rms


def _pick(payload: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in payload:
            return payload[name]
    return None


def _inputs_from_payload(payload: Mapping[str, Any]) -> dict[str, Tensor | None]:
    values = {
        "global_token": _pick(payload, "global_token", "global", "g"),
        "phase_token": _pick(payload, "phase_token", "phase", "p"),
        "brief_signals": _pick(payload, "brief_signals", "brief", "hbrief"),
        "retrieved_signals": _pick(payload, "retrieved_signals", "retrieved", "pim", "r"),
        "brief_mask": _pick(payload, "brief_mask"),
        "retrieved_mask": _pick(payload, "retrieved_mask"),
    }
    for name, value in values.items():
        if value is not None and not isinstance(value, Tensor):
            values[name] = torch.as_tensor(value)
    return values


def _infer_dimensions(inputs: Mapping[str, Tensor | None]) -> tuple[int, int, int, int]:
    global_token = inputs["global_token"]
    phase_token = inputs["phase_token"]
    brief_signals = inputs["brief_signals"]
    retrieved_signals = inputs["retrieved_signals"]
    if global_token is None or phase_token is None:
        raise ValueError("The feature payload must contain global and phase tensors.")
    if brief_signals is None and retrieved_signals is None:
        raise ValueError("The feature payload must contain brief or retrieved signals.")
    signal = brief_signals if brief_signals is not None else retrieved_signals
    assert signal is not None
    return int(global_token.shape[-1]), int(phase_token.shape[-1]), int(signal.shape[-1]), int(global_token.shape[0])


def _default_probe_inputs(*, batch_size: int = 1) -> dict[str, Tensor | None]:
    """Make deterministic dimensions for a checkpoint-only foundation smoke.

    Real causal features should be supplied with ``--features`` for a causal
    attribution claim.  The fallback exists only to exercise the actual
    frozen PI0.5 prefix/KV path when no encoder artifact is available.
    """

    generator = torch.Generator(device="cpu").manual_seed(17)
    return {
        "global_token": torch.randn(batch_size, 256, generator=generator),
        "phase_token": torch.randn(batch_size, 128, generator=generator),
        "brief_signals": torch.randn(batch_size, 2, 256, generator=generator),
        "retrieved_signals": torch.randn(batch_size, 3, 256, generator=generator),
        "brief_mask": torch.ones(batch_size, 2, dtype=torch.bool),
        "retrieved_mask": torch.ones(batch_size, 3, dtype=torch.bool),
    }


def _condition_result(
    injector: CausalPromptInjection,
    inputs: Mapping[str, Tensor | None],
    branches: PromptBranches,
    *,
    retrieved_override: Tensor | None = None,
) -> CausalPromptResult:
    retrieved = inputs["retrieved_signals"] if retrieved_override is None else retrieved_override
    return injector.build_prompt(
        inputs["global_token"],
        inputs["phase_token"],
        inputs["brief_signals"],
        retrieved,
        brief_mask=inputs["brief_mask"],
        retrieved_mask=inputs["retrieved_mask"],
        branches=branches,
    )


def _permuted_retrieved(inputs: Mapping[str, Tensor | None]) -> Tensor | None:
    retrieved = inputs["retrieved_signals"]
    if retrieved is None or retrieved.shape[0] < 2:
        return None
    return retrieved.roll(1, dims=0)


def _prompt_metrics(
    correct: CausalPromptResult,
    zero: CausalPromptResult,
    permuted: CausalPromptResult | None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "correct_prompt_rms": float(tensor_rms(correct.tokens)),
        "zero_prompt_rms": float(tensor_rms(zero.tokens)),
        "correct_minus_zero_rms": float(tensor_rms(correct.tokens - zero.tokens)),
        "correct_memory_rms": float(tensor_rms(correct.memory_tokens)),
        "zero_memory_rms": float(tensor_rms(zero.memory_tokens)),
        "branch_rms": correct.branch_rms,
        "gate": float(correct.gate.detach().float()),
    }
    if permuted is not None:
        metrics["permuted_prompt_rms"] = float(tensor_rms(permuted.tokens))
        metrics["correct_minus_permuted_rms"] = float(
            tensor_rms(correct.tokens - permuted.tokens)
        )
    return metrics


def run_feature_probe(
    injector: CausalPromptInjection,
    inputs: Mapping[str, Tensor | None],
) -> dict[str, Any]:
    """Run prompt-only attribution without invoking the foundation model."""

    correct = _condition_result(injector, inputs, PromptBranches())
    zero = _condition_result(
        injector,
        inputs,
        PromptBranches(
            global_enabled=False,
            phase_enabled=False,
            brief_enabled=False,
            retrieved_enabled=False,
        ),
    )
    permuted_inputs = _permuted_retrieved(inputs)
    permuted = (
        _condition_result(injector, inputs, PromptBranches(), retrieved_override=permuted_inputs)
        if permuted_inputs is not None
        else None
    )
    branch_metrics: dict[str, float] = {}
    for branch in ("global", "phase", "brief", "retrieved"):
        result = _condition_result(injector, inputs, PromptBranches.only(branch))
        branch_metrics[branch] = float(tensor_rms(result.tokens))
    metrics = _prompt_metrics(correct, zero, permuted)
    metrics["branch_prompt_rms"] = branch_metrics
    metrics["foundation_called"] = False
    metrics["stage2_allowed"] = False
    metrics["stage2_block_reason"] = "Foundation attribution was not run."
    return metrics


def _as_tensor_output(value: Any) -> Tensor:
    if isinstance(value, Tensor):
        return value
    if isinstance(value, (tuple, list)):
        tensors = [item for item in value if isinstance(item, Tensor)]
        if len(tensors) == 1:
            return tensors[0]
    raise TypeError("Frozen foundation probe must return one tensor.")


def _run_with_hook(
    hook: PrefixPromptHook,
    forward: Callable[[], Any],
    prompt: Tensor | None,
    *,
    mode: PromptMode = "residual",
) -> Tensor:
    return _as_tensor_output(hook.run(forward, prompt, mode=mode))


def run_frozen_prefix_probe(
    injector: CausalPromptInjection,
    hook: PrefixPromptHook,
    inputs: Mapping[str, Tensor | None],
    forward: Callable[[], Any],
    *,
    target_output: Tensor | None = None,
    frozen_parameters: Mapping[str, torch.nn.Parameter] | None = None,
    loss_fn: Callable[[Tensor], Tensor] | None = None,
    min_effect_rms: float = 1e-8,
    injection_mode: PromptMode = "residual",
) -> dict[str, Any]:
    """Audit a frozen PI0.5 with Base/zero/correct/permuted conditions.

    ``forward`` must call the frozen policy's normal diffusion/action path.
    The default ``injection_mode="residual"`` adds one learned vector to all
    existing prefix embeddings.  It does not append or prepend a new prompt
    token, and therefore changes neither prefix length nor masks/positions.
    A ``None`` prompt bypasses the hook and is the exact Base condition.  The
    explicit ``injection_mode="append"`` option is retained only for legacy
    attribution experiments and cannot claim exact Base equivalence.  No
    optimizer is invoked.  ``target_output`` is optional: without it the
    function reports attribution but refuses to claim that the correct
    retrieval is causally preferred, because a larger output perturbation is
    not automatically a better action.
    """

    correct_result = _condition_result(injector, inputs, PromptBranches())
    zero_result = _condition_result(
        injector,
        inputs,
        PromptBranches(
            global_enabled=False,
            phase_enabled=False,
            brief_enabled=False,
            retrieved_enabled=False,
        ),
    )
    permuted_inputs = _permuted_retrieved(inputs)
    permuted_result = (
        _condition_result(injector, inputs, PromptBranches(), retrieved_override=permuted_inputs)
        if permuted_inputs is not None
        else None
    )

    # The Base and zero conditions both bypass prefix concatenation.  Keeping
    # these calls separate makes a caller-visible bit-equivalence check easy.
    base_output = _run_with_hook(hook, forward, None)
    zero_output = _run_with_hook(hook, forward, None)
    correct_prompt = (
        injector.prompt_residual_or_none(correct_result)
        if injection_mode == "residual"
        else injector.prompt_tokens_or_none(correct_result)
    )
    permuted_prompt = (
        injector.prompt_residual_or_none(permuted_result)
        if injection_mode == "residual" and permuted_result is not None
        else injector.prompt_tokens_or_none(permuted_result)
        if permuted_result is not None
        else None
    )
    correct_output = _run_with_hook(hook, forward, correct_prompt, mode=injection_mode)
    permuted_output = (
        _run_with_hook(hook, forward, permuted_prompt, mode=injection_mode)
        if permuted_result is not None
        else None
    )

    correct_delta = tensor_rms(correct_output - base_output)
    permuted_delta = (
        tensor_rms(permuted_output - base_output) if permuted_output is not None else None
    )
    output_metrics: dict[str, Any] = {
        "base_zero_max_abs": float((base_output - zero_output).abs().max()),
        "correct_delta_rms": float(correct_delta),
        "permuted_delta_rms": None if permuted_delta is None else float(permuted_delta),
        "correct_output_rms": float(tensor_rms(correct_output)),
        "base_output_rms": float(tensor_rms(base_output)),
    }
    if target_output is not None:
        target_output = target_output.to(device=correct_output.device, dtype=correct_output.dtype)
        correct_error = tensor_rms(correct_output - target_output)
        permuted_error = (
            tensor_rms(permuted_output - target_output)
            if permuted_output is not None
            else None
        )
        output_metrics["correct_target_rms"] = float(correct_error)
        output_metrics["permuted_target_rms"] = (
            None if permuted_error is None else float(permuted_error)
        )
        retrieval_preferred = (
            permuted_error is not None and bool(correct_error < permuted_error)
        )
    else:
        retrieval_preferred = None

    # The gradient audit is run on a fresh correct condition.  It uses the
    # same foundation callable and therefore tests the real prefix-KV path.
    audit_prompt = (
        injector.prompt_residual_or_none(correct_result)
        if injection_mode == "residual"
        else injector.prompt_tokens_or_none(correct_result)
    )
    gradient_audit: PromptGradientAudit = audit_prompt_gradients(
        injector,
        correct_result,
        lambda _prompt: _run_with_hook(
            hook,
            forward,
            audit_prompt,
            mode=injection_mode,
        ),
        loss_fn=loss_fn,
        frozen_parameters=frozen_parameters,
    )
    prompt_gradient = any(value > 0.0 for value in gradient_audit.gradient_norms.values())
    projector_gradient = any(
        value > 0.0
        for name, value in gradient_audit.gradient_norms.items()
        if name.startswith("prompt_projection.")
    )
    frozen_gradient_free = not gradient_audit.frozen_parameter_gradients
    prompt_effect = float(correct_delta) > float(min_effect_rms)
    causal_consistent = (
        target_output is not None
        and retrieval_preferred is True
        and prompt_effect
        and prompt_gradient
        and frozen_gradient_free
    )
    block_reasons: list[str] = []
    if target_output is None:
        block_reasons.append("target_output was not supplied")
    elif retrieval_preferred is not True:
        block_reasons.append("correct retrieval was not preferred over permuted retrieval")
    if not prompt_effect:
        block_reasons.append(
            "zero-init residual is intentionally an exact Base at step zero; "
            "measure post-update effect before claiming causal consistency"
        )
    if not prompt_gradient:
        block_reasons.append("prompt gradient was zero")
    if not frozen_gradient_free:
        block_reasons.append("frozen foundation received gradients")

    metrics = _prompt_metrics(correct_result, zero_result, permuted_result)
    metrics.update(
        {
            "foundation_called": True,
            "output": output_metrics,
            "gradient": asdict(gradient_audit),
            "prompt_gradient_nonzero": prompt_gradient,
            "prompt_projection_gradient_nonzero": projector_gradient,
            "frozen_gradient_free": frozen_gradient_free,
            "exact_base_at_initialization": output_metrics["correct_delta_rms"] == 0.0,
            "stage2_gradient_ready": projector_gradient and frozen_gradient_free,
            "injection_mode": injection_mode,
            "retrieval_preferred": retrieval_preferred,
            "causal_consistent": causal_consistent,
            "stage2_allowed": causal_consistent,
            "stage2_block_reason": None if causal_consistent else "; ".join(block_reasons),
        }
    )
    return metrics


def run_pi05_foundation_probe(
    checkpoint: str | Path,
    *,
    inputs: Mapping[str, Tensor | None] | None = None,
    device: str = "cuda:0",
    runtime_root: str | Path | None = None,
    text: str = "pick up the object",
    min_effect_rms: float = 1e-8,
) -> dict[str, Any]:
    """Run the residual probe through a real frozen LeRobot PI0.5 model.

    The checkpoint is loaded with ``compile_model=False`` and
    ``gradient_checkpointing=False`` for a deterministic one-step audit.  A
    synthetic image/state/action batch is used because this function is a
    mechanism smoke, not an offline policy-quality evaluation.  Pass a real
    ``inputs`` feature payload when testing retrieval attribution.  The
    model's complete PI0.5 training forward is used so the action expert sees
    the same prefix path as training; no final action residual is installed.
    """

    checkpoint = Path(checkpoint).resolve()
    if runtime_root is not None:
        runtime_root = Path(runtime_root).resolve()
        source = runtime_root / "lerobot-main-py311-v1" / "src"
        if source.exists():
            sys.path.insert(0, str(source))

    # Keep this import lazy: feature-only probes must remain runnable without
    # the optional LeRobot/Transformers runtime.
    # The handoff runtime is Python 3.10 while the vendored LeRobot sources
    # use the 3.11 typing names.  Keep the compatibility shim local to the
    # optional real-foundation path.
    for name in ("Self", "Unpack", "NotRequired"):
        if not hasattr(typing, name):
            setattr(typing, name, getattr(typing_extensions, name))

    import lerobot  # noqa: PLC0415

    if "lerobot.policies" not in sys.modules:
        package = ModuleType("lerobot.policies")
        package.__path__ = [str(Path(next(iter(lerobot.__path__))) / "policies")]
        package.__package__ = "lerobot.policies"
        sys.modules["lerobot.policies"] = package

    from lerobot.configs import PreTrainedConfig  # noqa: PLC0415
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy  # noqa: PLC0415

    config = PreTrainedConfig.from_pretrained(str(checkpoint), local_files_only=True)
    config.device = device
    config.compile_model = False
    config.gradient_checkpointing = False
    config.use_visual_memory = False
    config.use_proprioceptive_memory = False
    policy = PI05Policy.from_pretrained(
        str(checkpoint),
        config=config,
        strict=True,
        local_files_only=True,
    ).to(device)
    policy.eval()
    core = policy.model
    for parameter in core.parameters():
        parameter.requires_grad_(requires_grad=False)

    # The checkpoint's tokenizer is local in the handoff.  We only need its
    # IDs/mask; the synthetic image/action values keep the test deterministic.
    from transformers import AutoTokenizer  # noqa: PLC0415

    tokenizer_path = checkpoint / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    encoded = tokenizer(
        [text],
        max_length=int(config.tokenizer_max_length),
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    image_resolution = tuple(config.image_resolution)
    generator = torch.Generator(device="cpu").manual_seed(23)
    batch: dict[str, Tensor] = {
        str(key): torch.rand((1, 3, *image_resolution), generator=generator)
        for key in config.image_features
    }
    # The PI0.5 checkpoint in the handoff has no proprioceptive memory.  Keep
    # state/action tensors at the native padded dimensions used by the model.
    images, image_masks = policy._preprocess_images(batch)  # noqa: SLF001
    tokens = encoded["input_ids"].to(device)
    masks = encoded["attention_mask"].to(device).bool()
    actions = torch.randn(
        1,
        int(config.chunk_size),
        int(config.max_action_dim),
        generator=generator,
    ).to(device)
    noise = torch.randn(actions.shape, generator=generator).to(device)
    time = torch.tensor([0.37], device=device)

    injector_inputs = dict(_default_probe_inputs()) if inputs is None else dict(inputs)
    injector_inputs = {
        name: value.to(device) if isinstance(value, Tensor) else value
        for name, value in injector_inputs.items()
    }
    global_dim, phase_dim, signal_dim, _ = _infer_dimensions(injector_inputs)
    q_proj = core.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj
    injector = CausalPromptInjection(
        global_dim=global_dim,
        phase_dim=phase_dim,
        signal_dim=signal_dim,
        context_dim=256,
        foundation_dim=int(q_proj.in_features),
        prompt_tokens=1,
        num_heads=4,
        gate_init=1.0,
        zero_init_projection=True,
    ).to(device)
    hook = PrefixPromptHook(core).install()

    def forward() -> Tensor:
        return core.forward(
            images,
            image_masks,
            tokens,
            masks,
            actions,
            noise,
            time,
            states=None,
            state_masks=None,
        )

    try:
        report = run_frozen_prefix_probe(
            injector,
            hook,
            injector_inputs,
            forward,
            frozen_parameters=dict(core.named_parameters()),
            min_effect_rms=min_effect_rms,
            injection_mode="residual",
        )
        # The training forward above proves gradient flow.  Also exercise the
        # released inference path once: ``sample_actions`` builds the prefix
        # KV cache, then calls the action expert for one deterministic denoise
        # step.  This remains inference-only and does not update parameters.
        sample_noise = torch.randn(
            1,
            int(config.chunk_size),
            int(config.max_action_dim),
            generator=generator,
        ).to(device)
        sample_result = _condition_result(injector, injector_inputs, PromptBranches())
        sample_prompt = injector.prompt_residual_or_none(sample_result)

        def sample_forward() -> Tensor:
            return core.sample_actions(
                images,
                image_masks,
                tokens,
                masks,
                states=None,
                state_masks=None,
                noise=sample_noise,
                num_steps=1,
            )

        base_sample = _run_with_hook(hook, sample_forward, None, mode="residual")
        prompt_sample = _run_with_hook(
            hook,
            sample_forward,
            sample_prompt,
            mode="residual",
        )
        report["inference"] = {
            "runtime": "lerobot.PI05Pytorch.sample_actions(num_steps=1)",
            "base_prompt_max_abs": float((base_sample - prompt_sample).abs().max()),
            "output_shape": list(prompt_sample.shape),
            "finite": bool(torch.isfinite(prompt_sample).all()),
        }
    finally:
        hook.uninstall()
    report["foundation_runtime"] = "lerobot.PI05Pytorch.forward"
    report["checkpoint"] = str(checkpoint)
    report["device"] = str(device)
    report["synthetic_batch"] = True
    report["adaptation"] = {
        "mechanism": "additive_existing_prefix_residual",
        "new_prefix_tokens": 0,
        "existing_prefix_positions_modified": "all",
        "prefix_length_masks_positions_unchanged": True,
        "is_appended_prompt_token": False,
    }
    report["limitations"] = [
        "The checkpoint smoke uses deterministic synthetic images/actions, not a policy-quality evaluation.",
        "No target_output or labeled correct-vs-permuted retrieval target was supplied; causal consistency and Stage-2 allowance remain false.",
        "At exact zero initialization the final prompt projector receives the first gradient; upstream memory-fuser gradients appear after that projector moves.",
        "The one-step audit verifies the real PI0.5 prefix/action-expert gradient path but does not establish closed-loop task success.",
    ]
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=None, help="torch.save feature payload")
    parser.add_argument(
        "--pi05-checkpoint",
        type=Path,
        default=None,
        help="optional local LeRobot PI0.5 checkpoint for a real frozen-foundation smoke",
    )
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=None,
        help="handoff root containing runtime/lerobot-main-py311-v1 when LeRobot is not installed",
    )
    parser.add_argument("--text", default="pick up the object")
    parser.add_argument("--output", type=Path, default=None, help="optional JSON report path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--foundation-dim", type=int, default=2048)
    parser.add_argument("--context-dim", type=int, default=256)
    parser.add_argument("--prompt-tokens", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument(
        "--zero-init-projection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use zero-init final projection (the exact-Base, gradient-ready default)",
    )
    parser.add_argument("--gate", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.pi05_checkpoint is not None:
        inputs = None
        if args.features is not None:
            payload = torch.load(args.features, map_location=args.device, weights_only=False)
            if not isinstance(payload, Mapping):
                raise TypeError("Feature payload must be a mapping.")
            inputs = _inputs_from_payload(payload)
        report = run_pi05_foundation_probe(
            args.pi05_checkpoint,
            inputs=inputs,
            device=args.device,
            runtime_root=args.runtime_root,
            text=args.text,
        )
    else:
        if args.features is None:
            raise ValueError("Provide --features for a feature probe or --pi05-checkpoint for a real foundation probe.")
        payload = torch.load(args.features, map_location=args.device, weights_only=False)
        if not isinstance(payload, Mapping):
            raise TypeError("Feature payload must be a mapping.")
        inputs = _inputs_from_payload(payload)
        global_dim, phase_dim, signal_dim, _ = _infer_dimensions(inputs)
        injector = CausalPromptInjection(
            global_dim=global_dim,
            phase_dim=phase_dim,
            signal_dim=signal_dim,
            context_dim=args.context_dim,
            foundation_dim=args.foundation_dim,
            prompt_tokens=args.prompt_tokens,
            num_heads=args.num_heads,
            zero_init_projection=args.zero_init_projection,
            gate_init=args.gate,
        ).to(args.device)
        report = run_feature_probe(injector, inputs)
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
