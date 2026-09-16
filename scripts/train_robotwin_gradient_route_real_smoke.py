#!/usr/bin/env python3
"""Read-only real PI0.5/Stage1 gradient-routing equivalence smoke.

This is a bounded mechanism check for the opt-in action-expert routing path in
``train_robotwin_stage2.py``.  It loads one real RoboTwin Stage 2 sample,
preprocesses that sample once, and then compares two backwards passes:

* the ordinary ``foundation_only`` Base flow; and
* a routed ZeVA pair, where the residual-on backward masks action-expert
  gradients and the matched residual-off backward passes them through.

The pair replays the same foundation CPU/CUDA RNG state.  The action-expert
gradient comparison is intentionally limited to a deterministic representative
subset, while the routing hook still covers every trainable action-expert
parameter.  At least one ZeVA gradient must be nonzero before the residual-off
backward.  No optimizer is constructed and no checkpoint is written.

The command expects the same local handoff, Stage 1, retrieval, and dataset
artifacts as Stage 2.  It prints a JSON provenance/metrics report and can
optionally write that report to ``--output`` (which is not a checkpoint).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch.utils.data import DataLoader
import tyro

from openpi.zeva.robotwin_contract import RobotWinHandoff
from openpi.zeva.robotwin_policy import stage1_artifact_schema
from openpi.zeva.robotwin_policy import validate_stage1_v2_artifact_status
from openpi.zeva.stage1_checkpoint import LEGACY_SCHEMAS
from openpi.zeva.stage1_checkpoint import V2_SCHEMA
from openpi.zeva.stage1_checkpoint import stage1_transition_horizon

try:
    from openpi.zeva.robotwin_policy import RobotWinZevaPolicy
    from scripts.train_robotwin_stage2 import RobotWinStage2Dataset
    from scripts.train_robotwin_stage2 import _ActionExpertGradientRouter
    from scripts.train_robotwin_stage2 import _baseline_losses
    from scripts.train_robotwin_stage2 import _diagnostic_rng_state
    from scripts.train_robotwin_stage2 import _gradient_routed_off_forward
    from scripts.train_robotwin_stage2 import _gradient_routed_on_forward
    from scripts.train_robotwin_stage2 import _load_task_subset
    from scripts.train_robotwin_stage2 import _preprocess_with_task_only_goal
    from scripts.train_robotwin_stage2 import _restore_diagnostic_rng_state
    from scripts.train_robotwin_stage2 import _retrieve
except ModuleNotFoundError:  # Direct ``python scripts/...py`` execution.
    from train_robotwin_stage2 import RobotWinStage2Dataset
    from train_robotwin_stage2 import _ActionExpertGradientRouter
    from train_robotwin_stage2 import _baseline_losses
    from train_robotwin_stage2 import _diagnostic_rng_state
    from train_robotwin_stage2 import _gradient_routed_off_forward
    from train_robotwin_stage2 import _gradient_routed_on_forward
    from train_robotwin_stage2 import _load_task_subset
    from train_robotwin_stage2 import _preprocess_with_task_only_goal
    from train_robotwin_stage2 import _restore_diagnostic_rng_state
    from train_robotwin_stage2 import _retrieve

    from openpi.zeva.robotwin_policy import RobotWinZevaPolicy


@dataclasses.dataclass
class Args:
    """CLI paths and bounded smoke controls.

    Defaults mirror the formal Stage 2 launcher, but every artifact path is
    exposed so a host can point this smoke at a relocated handoff.
    """

    handoff_root: str = "/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1"
    dataset_root: str = "/data1/dingxin/robotwin-lerobot-sidney-eef16-v1/data"
    foundation_checkpoint: str | None = None
    initial_stage2_checkpoint: str | None = None
    adapter_checkpoint: str | None = None
    goal_embedding_checkpoint: str | None = None
    zte_checkpoint: str = "/data1/dingxin/zeva-runs/robotwin-v5-h15-tasklang/stage1-zte/zte_best.pth"
    causal_bank: str = "/data1/dingxin/zeva-runs/robotwin-v5-h15-tasklang/stage1-zte/train_causal_bank.pt"
    live_queries: str = "/data1/dingxin/zeva-runs/robotwin-v5-h15-tasklang/stage1-zte/live_queries_h15.pt"
    task_retrieval: str = "/data1/dingxin/zeva-runs/robotwin-v5-h15-tasklang/stage1.5-task-retrieval/task_retrieval.pth"
    task_subset: str | None = None
    subset: str = "train"
    video_backend: str = "torchcodec"
    decoder_threads: int = 1
    # Per-rank batch is deliberately bounded to one by default. Two is useful
    # for checking per-example reduction while remaining modest on an H100.
    batch_size: int = 1
    num_workers: int = 0
    device: str = "cuda:0"
    seed: int = 1000
    # Retrieval is done once before the matched foundation RNG is captured.
    phase_noise_std: float = 0.0
    memory_dropout: float = 0.0
    retrieval_confidence_floor: float = 0.2
    prior_loss_weight: float = 0.01
    gate_regularization_weight: float = 1e-3
    prior_residual_dropout_probability: float = 0.0
    prior_injection_horizon: int = 50
    # Number of action-expert tensors compared. Hooks still cover all action
    # parameters, so this only bounds report/copy overhead.
    representative_parameter_limit: int = 8
    max_gradient_diff: float = 1e-6
    gradient_checkpointing: bool = True
    output: str | None = None


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_payload(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping payload at {path}, got {type(payload)!r}.")
    return payload


def _seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=device.type == "cuda")
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


def _processed_fingerprint(processed: dict[str, Any]) -> str:
    """Hash the exact tensor mapping shared by the Base and routed forwards."""
    digest = hashlib.sha256()
    for key in sorted(processed):
        digest.update(str(key).encode("utf-8"))
        value = processed[key]
        if isinstance(value, torch.Tensor):
            tensor = value.detach().contiguous()
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(repr(tuple(tensor.shape)).encode("ascii"))
            # ``view(uint8)`` handles CUDA and non-byte dtypes without going
            # through a potentially lossy text representation.
            digest.update(tensor.view(torch.uint8).cpu().numpy().tobytes())
        else:
            digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def _state_fingerprint(state: tuple[torch.Tensor, torch.Tensor | None]) -> str:
    digest = hashlib.sha256()
    digest.update(state[0].cpu().numpy().tobytes())
    if state[1] is not None:
        digest.update(state[1].cpu().numpy().tobytes())
    return digest.hexdigest()


def _parameter_fingerprint(parameters: list[tuple[str, torch.nn.Parameter]]) -> str:
    digest = hashlib.sha256()
    for name, parameter in parameters:
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def _action_parameter_groups(
    policy: torch.nn.Module,
) -> list[tuple[str, torch.nn.Parameter, str]]:
    """Return trainable PI action parameters with stable module group labels."""
    core = policy.foundation.model
    modules = (
        ("gemma_expert", core.paligemma_with_expert.gemma_expert.model),
        ("action_in_proj", core.action_in_proj),
        ("action_out_proj", core.action_out_proj),
        ("time_mlp_in", core.time_mlp_in),
        ("time_mlp_out", core.time_mlp_out),
    )
    named_parameters = dict(policy.foundation.named_parameters())
    result: list[tuple[str, torch.nn.Parameter, str]] = []
    seen: set[int] = set()
    for group, module in modules:
        module_ids = {id(parameter) for parameter in module.parameters()}
        for name, parameter in named_parameters.items():
            if id(parameter) in module_ids and parameter.requires_grad and id(parameter) not in seen:
                result.append((name, parameter, group))
                seen.add(id(parameter))
    if not result:
        raise RuntimeError("The configured PI0.5 action-expert parameter set is empty.")
    return result


def _representative_parameters(
    groups: list[tuple[str, torch.nn.Parameter, str]], limit: int
) -> list[tuple[str, torch.nn.Parameter]]:
    if limit <= 0:
        raise ValueError("representative_parameter_limit must be positive.")
    selected: list[tuple[str, torch.nn.Parameter]] = []
    # Include at least one tensor from every action-path module when possible.
    for group in ("gemma_expert", "action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"):
        candidates = [(name, parameter) for name, parameter, label in groups if label == group]
        if candidates:
            selected.append(candidates[0])
    selected_names = {name for name, _ in selected}
    for name, parameter, _ in groups:
        if len(selected) >= limit:
            break
        if name not in selected_names:
            selected.append((name, parameter))
            selected_names.add(name)
    return selected[:limit]


def _gradient_stats(
    named_parameters: list[tuple[str, torch.nn.Parameter]],
    *,
    require_nonzero: bool = False,
) -> dict[str, Any]:
    count = 0
    nonzero_elements = 0
    nonzero_parameters = 0
    sum_squares = 0.0
    max_abs = 0.0
    missing: list[str] = []
    nonfinite: list[str] = []
    for name, parameter in named_parameters:
        gradient = parameter.grad
        if gradient is None:
            missing.append(name)
            continue
        gradient_float = gradient.detach().float()
        if not bool(torch.isfinite(gradient_float).all().item()):
            nonfinite.append(name)
        elements = int(gradient_float.numel())
        count += elements
        parameter_nonzero = int(torch.count_nonzero(gradient_float).item())
        nonzero_elements += parameter_nonzero
        nonzero_parameters += int(parameter_nonzero > 0)
        sum_squares += float(gradient_float.square().sum().item())
        max_abs = max(max_abs, float(gradient_float.abs().max().item()))
    if nonfinite:
        raise FloatingPointError(f"Non-finite gradients in {nonfinite[:8]}.")
    if require_nonzero and nonzero_elements == 0:
        raise RuntimeError("ZeVA residual-on backward produced entirely zero gradients.")
    return {
        "parameter_count": len(named_parameters),
        "missing_parameter_count": len(missing),
        "missing_parameters": missing[:8],
        "gradient_element_count": count,
        "nonzero_parameter_count": nonzero_parameters,
        "nonzero_element_count": nonzero_elements,
        "max_abs": max_abs,
        "rms": (sum_squares / count) ** 0.5 if count else 0.0,
    }


def _compare_gradients(
    baseline: dict[str, torch.Tensor],
    routed: list[tuple[str, torch.nn.Parameter]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    max_abs = 0.0
    total_square = 0.0
    total_count = 0
    baseline_square = 0.0
    routed_square = 0.0
    for name, parameter in routed:
        gradient = parameter.grad
        if gradient is None:
            raise RuntimeError(f"Routed residual-off backward omitted representative gradient {name}.")
        expected = baseline[name]
        actual = gradient.detach().float().cpu()
        if actual.shape != expected.shape:
            raise RuntimeError(
                f"Gradient shape changed for {name}: {tuple(actual.shape)} versus {tuple(expected.shape)}."
            )
        difference = actual - expected
        diff_max = float(difference.abs().max().item()) if difference.numel() else 0.0
        diff_square = float(difference.square().sum().item())
        element_count = int(difference.numel())
        max_abs = max(max_abs, diff_max)
        total_square += diff_square
        total_count += element_count
        baseline_square += float(expected.square().sum().item())
        routed_square += float(actual.square().sum().item())
        rows.append(
            {
                "name": name,
                "shape": list(actual.shape),
                "numel": element_count,
                "max_abs": diff_max,
                "rms": (diff_square / element_count) ** 0.5 if element_count else 0.0,
                "baseline_rms": (float(expected.square().mean().item())) ** 0.5 if element_count else 0.0,
                "routed_off_rms": (float(actual.square().mean().item())) ** 0.5 if element_count else 0.0,
            }
        )
    return {
        "representative_parameter_count": len(rows),
        "representative_parameters": rows,
        "max_abs_gradient_diff": max_abs,
        "rms_gradient_diff": (total_square / total_count) ** 0.5 if total_count else 0.0,
        "baseline_gradient_rms": (baseline_square / total_count) ** 0.5 if total_count else 0.0,
        "routed_off_gradient_rms": (routed_square / total_count) ** 0.5 if total_count else 0.0,
        "gradient_element_count": total_count,
    }


def _validate_inputs(args: Args, device: torch.device) -> None:
    if args.subset not in {"train", "validation"}:
        raise ValueError("subset must be 'train' or 'validation'.")
    if args.batch_size not in {1, 2}:
        raise ValueError("batch_size must be 1 or 2 for the bounded real smoke.")
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative.")
    if args.video_backend not in {"torchcodec", "ffmpeg"}:
        raise ValueError("video_backend must be 'torchcodec' or 'ffmpeg'.")
    if args.decoder_threads <= 0:
        raise ValueError("decoder_threads must be positive.")
    if args.representative_parameter_limit <= 0:
        raise ValueError("representative_parameter_limit must be positive.")
    if not torch.isfinite(torch.tensor(args.max_gradient_diff)) or args.max_gradient_diff < 0:
        raise ValueError("max_gradient_diff must be finite and non-negative.")
    if args.prior_loss_weight < 0:
        raise ValueError("prior_loss_weight must be non-negative.")
    if args.gate_regularization_weight < 0:
        raise ValueError("gate_regularization_weight must be non-negative.")
    if not 0.0 <= args.prior_residual_dropout_probability < 1.0:
        raise ValueError("prior_residual_dropout_probability must be in [0, 1).")
    if not 0 < args.prior_injection_horizon <= 50:
        raise ValueError("prior_injection_horizon must be in [1, 50].")
    if not 0.0 <= args.memory_dropout <= 1.0:
        raise ValueError("memory_dropout must be in [0, 1].")
    if args.phase_noise_std < 0:
        raise ValueError("phase_noise_std must be non-negative.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device}, but CUDA is unavailable.")


def main(args: Args) -> None:
    device = torch.device(args.device)
    _validate_inputs(args, device)
    _seed(args.seed)

    handoff = RobotWinHandoff.from_root(args.handoff_root)
    foundation_checkpoint = (
        Path(args.foundation_checkpoint).resolve()
        if args.foundation_checkpoint is not None
        else handoff.checkpoint.resolve()
    )
    zte_path = Path(args.zte_checkpoint).resolve()
    bank_path = Path(args.causal_bank).resolve()
    live_path = Path(args.live_queries).resolve()
    retrieval_path = Path(args.task_retrieval).resolve()
    adapter_manifest = Path(args.dataset_root).resolve() / "adapter.json"
    for path, label in (
        (zte_path, "Stage1 checkpoint"),
        (bank_path, "causal bank"),
        (live_path, "live-query cache"),
        (retrieval_path, "task retrieval checkpoint"),
        (adapter_manifest, "dataset adapter manifest"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")

    zte_payload = _load_payload(zte_path)
    zte_schema = zte_payload.get("schema")
    if zte_schema not in LEGACY_SCHEMAS | {V2_SCHEMA}:
        raise ValueError(f"Unsupported Stage1 checkpoint schema: {zte_schema!r}.")
    transition_horizon = stage1_transition_horizon(zte_payload)
    if transition_horizon != 15:
        raise ValueError(f"Expected Stage1 executed horizon 15, got {transition_horizon}.")

    zte_sha256 = _sha256(zte_path)
    statistics_sha256 = _sha256(handoff.statistics)
    causal_bank_payload = _load_payload(bank_path)
    bank_payload_schema_value = causal_bank_payload.get("schema")
    if zte_schema == V2_SCHEMA:
        validate_stage1_v2_artifact_status(causal_bank_payload, artifact_name="causal bank")
    retrieval_payload = _load_payload(retrieval_path)
    if retrieval_payload.get("causal_bank_sha256") != _sha256(bank_path):
        raise ValueError("Task retrieval checkpoint was trained against a different causal bank.")
    if float(retrieval_payload.get("validation_accuracy", 0.0)) < 0.95:
        raise ValueError("Task retrieval validation accuracy is below the 95% Stage1.5 gate.")

    policy = RobotWinZevaPolicy.from_handoff(
        args.handoff_root,
        device=str(device),
        foundation_checkpoint=args.foundation_checkpoint,
        goal_embedding_checkpoint=args.goal_embedding_checkpoint,
        stage2_checkpoint=args.initial_stage2_checkpoint,
        adapter_checkpoint=args.adapter_checkpoint,
        zte_checkpoint=str(zte_path),
        retrieval_checkpoint=str(retrieval_path),
        causal_bank=str(bank_path),
    )
    bank = policy.causal_bank
    if bank is None:
        raise RuntimeError("Stage1 causal bank was not loaded into the policy.")
    if bank.manifest.get("stage1_checkpoint_sha256") != zte_sha256:
        raise ValueError("Causal bank was exported from a different Stage1 checkpoint.")
    if bank.manifest.get("statistics_sha256") != statistics_sha256:
        raise ValueError("Causal bank uses different PI0.5 normalization statistics.")
    bank_schema = stage1_artifact_schema(bank.manifest)
    if zte_schema == V2_SCHEMA:
        if bank_schema != V2_SCHEMA:
            raise ValueError("v2 Stage1 requires a causal bank with explicit v2 provenance.")
    elif bank_schema is not None and bank_schema != zte_schema:
        raise ValueError("Causal bank schema differs from the selected Stage1 checkpoint.")
    del causal_bank_payload

    selected_tasks = None
    if args.task_subset is not None:
        # Keep task filtering at the same Stage2 sample layer as the trainer.
        selected_tasks = _load_task_subset(args.task_subset)
    dataset = RobotWinStage2Dataset(
        adapter_manifest,
        str(live_path),
        subset=args.subset,
        config=policy.zeva_config,
        selected_tasks=selected_tasks,
        video_backend=args.video_backend,
        decoder_threads=args.decoder_threads,
    )
    if dataset.task_names != bank.task_names:
        raise ValueError("Stage2 dataset task ordering differs from the Stage1 causal bank.")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    try:
        raw_batch = next(iter(loader))
    except StopIteration as error:
        raise RuntimeError(f"Stage2 dataset split {args.subset!r} is empty.") from error
    raw_batch = _move_to_device(raw_batch, device)
    sample_indices = raw_batch.pop("zeva.sample_index").detach().cpu().reshape(-1).tolist()
    task_ids = raw_batch.pop("zeva.task_id")
    phase_queries = raw_batch.pop("zeva.phase_query")
    live_brief = raw_batch.pop("zeva.live_brief")
    live_brief_mask = raw_batch.pop("zeva.live_brief_mask")
    live_retrieved = raw_batch.pop("zeva.live_retrieved")
    live_retrieved_mask = raw_batch.pop("zeva.live_retrieved_mask")
    raw_tasks = raw_batch.get("task")
    processed = _preprocess_with_task_only_goal(policy, policy.preprocessor, raw_batch)
    processed = _move_to_device(processed, device)
    processed_signature_before = _processed_fingerprint(processed)

    retrieval_args = SimpleNamespace(
        phase_noise_std=args.phase_noise_std,
        memory_dropout=args.memory_dropout,
        retrieval_confidence_floor=args.retrieval_confidence_floor,
    )
    policy.reset(scope="episode")
    bank_batch, confidence, retrieval_accuracy = _retrieve_for_smoke(
        policy,
        processed,
        phase_queries,
        task_ids,
        live_brief,
        live_brief_mask,
        live_retrieved,
        live_retrieved_mask,
        bank,
        retrieval_args,
    )

    trainable = policy.configure_action_expert_finetune_stage2()
    del trainable
    policy.train()
    policy.enforce_action_expert_stage2_mode()
    core = policy.foundation.model
    if hasattr(core, "gradient_checkpointing_disable"):
        core.gradient_checkpointing_disable()
    expert_model = core.paligemma_with_expert.gemma_expert.model
    if args.gradient_checkpointing:
        expert_model.gradient_checkpointing = True
    elif hasattr(expert_model, "gradient_checkpointing"):
        expert_model.gradient_checkpointing = False

    action_groups = _action_parameter_groups(policy)
    action_parameters = [(name, parameter) for name, parameter, _ in action_groups]
    representative = _representative_parameters(action_groups, args.representative_parameter_limit)
    zeva_parameters = [
        (name, parameter)
        for name, parameter in policy.named_parameters()
        if parameter.requires_grad and not name.startswith("foundation.")
    ]
    if not zeva_parameters:
        raise RuntimeError("Configured standard ZeVA path has no trainable ZeVA parameters.")

    parameter_fingerprint_before = _parameter_fingerprint(representative)
    pair_rng_state = _diagnostic_rng_state(device)
    pair_rng_fingerprint = _state_fingerprint(pair_rng_state)

    # Standard Base-only gradient. It is run first with the pair's exact RNG
    # state, then all gradients are cleared before installing routing hooks.
    _restore_diagnostic_rng_state(pair_rng_state, device)
    policy.zero_grad(set_to_none=True)
    baseline_losses = _baseline_losses(policy, processed)
    baseline_losses["total"].backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    baseline_gradients: dict[str, torch.Tensor] = {}
    for name, parameter in representative:
        if parameter.grad is None:
            raise RuntimeError(f"Base-only backward omitted representative gradient {name}.")
        gradient = parameter.grad.detach().float()
        if not bool(torch.isfinite(gradient).all().item()):
            raise FloatingPointError(f"Base-only gradient for {name} is non-finite.")
        baseline_gradients[name] = gradient.cpu().clone()
    baseline_loss = float(baseline_losses["total"].detach().cpu())
    policy.zero_grad(set_to_none=True)

    router = _ActionExpertGradientRouter([parameter for _, parameter in action_parameters])
    try:
        router.begin_first_step_audit()
        router.set_phase("residual_on")
        _restore_diagnostic_rng_state(pair_rng_state, device)
        on_losses, routed_rng_state = _gradient_routed_on_forward(
            policy,
            processed,
            bank_batch,
            confidence,
            None,
            pair_rng_state,
            args.prior_loss_weight,
            0.0,
            args.gate_regularization_weight,
            args.prior_residual_dropout_probability,
            0.0,
            1.0,
            args.prior_injection_horizon,
        )
        on_losses["total"].backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        zeva_on_stats = _gradient_stats(zeva_parameters, require_nonzero=True)
        on_action_max_abs = max(
            (
                float(parameter.grad.detach().float().abs().max().item())
                for _, parameter in action_parameters
                if parameter.grad is not None
            ),
            default=0.0,
        )

        # The off forward is intentionally constructed only after the on
        # backward. This is the same ordering enforced by the DDP trainer.
        router.set_phase("residual_off")
        off_total, off_flow = _gradient_routed_off_forward(
            policy,
            processed,
            routed_rng_state,
            [parameter for _, parameter in zeva_parameters],
        )
        off_total.backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        router.assert_first_step_audit()
        router_audit = router.first_step_audit()
        gradient_comparison = _compare_gradients(baseline_gradients, representative)
        routed_off_loss = float(off_flow.detach().cpu())
    finally:
        router.close()

    processed_signature_after = _processed_fingerprint(processed)
    if processed_signature_before != processed_signature_after:
        raise RuntimeError("The shared processed batch changed between the two gradient flows.")
    if gradient_comparison["max_abs_gradient_diff"] > args.max_gradient_diff:
        raise RuntimeError(
            "Base and routed residual-off action gradients differ beyond the configured "
            f"bound: {gradient_comparison['max_abs_gradient_diff']:.6g} > "
            f"{args.max_gradient_diff:.6g}."
        )
    if not torch.isfinite(torch.tensor([baseline_loss, routed_off_loss])).all():
        raise FloatingPointError("Base or routed residual-off flow loss is non-finite.")
    if abs(baseline_loss - routed_off_loss) > args.max_gradient_diff:
        raise RuntimeError(
            "Base and routed residual-off losses differ beyond the configured bound: "
            f"{abs(baseline_loss - routed_off_loss):.6g}."
        )
    if parameter_fingerprint_before != _parameter_fingerprint(representative):
        raise RuntimeError("A gradient-only smoke unexpectedly changed model parameters.")

    report = {
        "schema": "zeva-robotwin-gradient-route-real-smoke-v1",
        "passed": True,
        "device": str(device),
        "seed": int(args.seed),
        "batch_size": int(args.batch_size),
        "subset": args.subset,
        "sample_indices": sample_indices,
        "task_ids": task_ids.detach().cpu().reshape(-1).tolist(),
        "tasks": list(raw_tasks) if isinstance(raw_tasks, (list, tuple)) else raw_tasks,
        "processed_batch": {
            "same_object_for_base_and_routed_pair": True,
            "fingerprint_before": processed_signature_before,
            "fingerprint_after": processed_signature_after,
            "keys": sorted(processed),
            "action_shape": list(processed["action"].shape),
        },
        "foundation_rng": {
            "captured_once_before_base": True,
            "replayed_for_routed_on": True,
            "replayed_for_routed_off": True,
            "state_fingerprint": pair_rng_fingerprint,
            "post_pair_stream_contract": "one residual-off current-student foundation forward",
        },
        "provenance": {
            "handoff_root": str(Path(args.handoff_root).resolve()),
            "statistics": str(handoff.statistics.resolve()),
            "statistics_sha256": statistics_sha256,
            "dataset_adapter": str(adapter_manifest),
            "dataset_adapter_sha256": _sha256(adapter_manifest),
            "foundation_checkpoint": str(foundation_checkpoint),
            "foundation_model": str(foundation_checkpoint / "model.safetensors"),
            "foundation_model_sha256": _sha256(foundation_checkpoint / "model.safetensors"),
            "initial_stage2_checkpoint": (
                str(Path(args.initial_stage2_checkpoint).resolve())
                if args.initial_stage2_checkpoint is not None
                else None
            ),
            "initial_stage2_model_sha256": (
                _sha256(Path(args.initial_stage2_checkpoint).resolve() / "model.safetensors")
                if args.initial_stage2_checkpoint is not None
                else None
            ),
            "adapter_checkpoint": (
                str(Path(args.adapter_checkpoint).resolve()) if args.adapter_checkpoint is not None else None
            ),
            "adapter_checkpoint_sha256": (
                _sha256(args.adapter_checkpoint) if args.adapter_checkpoint is not None else None
            ),
            "zte_checkpoint": str(zte_path),
            "zte_checkpoint_sha256": zte_sha256,
            "zte_schema": zte_schema,
            "zte_step": int(zte_payload.get("step", -1)),
            "stage1_transition_horizon": int(transition_horizon),
            "causal_bank": str(bank_path),
            "causal_bank_sha256": _sha256(bank_path),
            "causal_bank_schema": bank_payload_schema_value,
            "live_queries": str(live_path),
            "live_queries_sha256": _sha256(live_path),
            "task_retrieval": str(retrieval_path),
            "task_retrieval_sha256": _sha256(retrieval_path),
            "task_retrieval_validation_accuracy": float(retrieval_payload["validation_accuracy"]),
        },
        "routing_contract": {
            "enabled": True,
            "variant": "standard_zeva_only",
            "base_flow": "foundation_only current PI0.5 flow",
            "residual_on": "ZeVA total loss; action-expert hooks return zero",
            "residual_off": "current student foundation-only flow; action-expert hooks pass through",
            "prior_nll_shared_input_gradient": "detached",
            "same_processed_batch": processed_signature_before == processed_signature_after,
            "same_foundation_rng_noise_time": True,
            "optimizer_created": False,
            "checkpoint_written": False,
            "remote_training_launched": False,
        },
        "losses": {
            "base_only": baseline_loss,
            "routed_residual_off": routed_off_loss,
            "absolute_difference": abs(baseline_loss - routed_off_loss),
            "retrieval_accuracy": float(retrieval_accuracy.detach().cpu()),
            "retrieval_confidence_mean": float(confidence.detach().float().mean().cpu()),
        },
        "action_expert_gradients": {
            "all_parameter_count": len(action_parameters),
            "all_parameter_numel": sum(parameter.numel() for _, parameter in action_parameters),
            "representative_subset": gradient_comparison,
            "residual_on_max_abs_after_hook": on_action_max_abs,
            "equivalence_bound": args.max_gradient_diff,
        },
        "zeva_gradients_after_residual_on": zeva_on_stats,
        "router_audit": router_audit,
        "memory": {
            "gradient_checkpointing": bool(args.gradient_checkpointing),
            "max_memory_allocated_bytes": (torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None),
            "max_memory_reserved_bytes": (torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None),
        },
    }
    if args.output is not None:
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        report["report_written_to"] = str(output)
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        output.write_text(encoded + "\n")
    print(encoded)


def _retrieve_for_smoke(
    policy: torch.nn.Module,
    processed: dict[str, torch.Tensor],
    phase_queries: torch.Tensor,
    task_ids: torch.Tensor,
    live_brief: torch.Tensor,
    live_brief_mask: torch.Tensor,
    live_retrieved: torch.Tensor,
    live_retrieved_mask: torch.Tensor,
    bank,
    args: SimpleNamespace,
):
    """Call the shared Stage 2 retrieval helper without constructing a trainer."""
    return _retrieve(
        policy,
        processed,
        phase_queries,
        task_ids,
        live_brief,
        live_brief_mask,
        live_retrieved,
        live_retrieved_mask,
        bank,
        policy.zeva_config,
        args,
        training=False,
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
