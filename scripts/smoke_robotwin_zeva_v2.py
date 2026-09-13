"""Bounded native PI0.5 + Stage1-v2 serving smoke.

This command loads the selected best-v1 foundation and a real v2 ZTE
checkpoint, then checks zero-residual Base equivalence and one H15 recurrent
update. Bank/live paths are validated against their checkpoint lineage; when a
retrieval checkpoint is supplied, the real bank/retrieval path is loaded too.
An optional compiled-anchor check verifies immutable teacher weights after a
temporary student perturbation, without creating an optimizer or checkpoint.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import tyro

from openpi.zeva.robotwin_contract import ROBOTWIN_CAMERA_KEYS
from openpi.zeva.robotwin_contract import prepare_robotwin_pi_image
from openpi.zeva.robotwin_contract import RobotWinHandoff
from openpi.zeva.robotwin_policy import RobotWinZevaPolicy
from openpi.zeva.robotwin_policy import stage1_artifact_schema
from openpi.zeva.robotwin_policy import validate_stage1_v2_artifact_status
from openpi.zeva.stage1_checkpoint import LEGACY_SCHEMAS
from openpi.zeva.stage1_checkpoint import V2_SCHEMA
from openpi.zeva.stage1_checkpoint import stage1_transition_horizon


@dataclasses.dataclass
class Args:
    handoff_root: str = (
        "/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/"
        "robotwin-memory-baseline-v1"
    )
    foundation_checkpoint: str = (
        "/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/"
        "robotwin-memory-baseline-v1/checkpoint/pretrained_model-best-v1"
    )
    zte_checkpoint: str = (
        "/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/"
        "stage1-zte-v2-phase-vector-mse-4096-20260911h/zte_v2_step_000512.pth"
    )
    causal_bank: str | None = None
    live_queries: str | None = None
    initial_stage2_checkpoint: str | None = None
    goal_embedding_checkpoint: str | None = None
    retrieval_checkpoint: str | None = None
    check_compiled_anchor: bool = False
    device: str = "cuda:0"
    task: str = "pick up the object"
    seed: int = 1000
    output: str | None = None


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_artifact(
    path: str | Path,
    *,
    artifact_name: str,
    checkpoint: dict[str, Any],
    checkpoint_sha256: str,
    statistics_sha256: str,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"{artifact_name} must be a mapping payload.")
    expected_schema = checkpoint.get("schema")
    if expected_schema not in LEGACY_SCHEMAS | {V2_SCHEMA}:
        raise ValueError(f"Unsupported Stage1 schema in checkpoint: {expected_schema!r}.")
    artifact_schema = stage1_artifact_schema(payload)
    if expected_schema == V2_SCHEMA:
        if artifact_schema != V2_SCHEMA:
            raise ValueError(
                f"v2 {artifact_name} must explicitly declare "
                "stage1_checkpoint_schema=v2."
            )
        validate_stage1_v2_artifact_status(payload, artifact_name=artifact_name)
    elif artifact_schema is not None and artifact_schema != expected_schema:
        raise ValueError(f"{artifact_name} was exported from a different Stage1 schema.")

    manifest = payload.get("manifest")
    if not isinstance(manifest, dict):
        raise ValueError(f"{artifact_name} is missing its manifest.")
    if manifest.get("stage1_checkpoint_sha256") != checkpoint_sha256:
        top_hash = payload.get("zte_checkpoint_sha256")
        if artifact_name == "live-query cache" and top_hash == checkpoint_sha256:
            pass
        else:
            raise ValueError(f"{artifact_name} was exported from a different Stage1 checkpoint.")
    if manifest.get("statistics_sha256") != statistics_sha256 and payload.get(
        "statistics_sha256"
    ) != statistics_sha256:
        raise ValueError(f"{artifact_name} uses different PI0.5 statistics.")
    expected_horizon = stage1_transition_horizon(checkpoint)
    artifact_horizon = payload.get("transition_horizon", manifest.get("causal_transition_horizon"))
    if artifact_horizon is None or int(artifact_horizon) != expected_horizon:
        raise ValueError(f"{artifact_name} uses a different Stage1 transition horizon.")
    return {
        "path": str(Path(path).resolve()),
        "schema": payload.get("schema"),
        "stage1_checkpoint_schema": artifact_schema,
        "incomplete": payload.get("incomplete"),
        "usable_for_training": payload.get("usable_for_training"),
        "transition_horizon": int(artifact_horizon),
        "checkpoint_sha256": checkpoint_sha256,
        "validated": True,
    }


def _seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(args: Args) -> None:
    handoff = RobotWinHandoff.from_root(args.handoff_root)
    checkpoint = torch.load(args.zte_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != V2_SCHEMA:
        raise ValueError("This real v2 smoke requires a v2 Stage1 checkpoint.")
    checkpoint_sha256 = _sha256(args.zte_checkpoint)
    statistics_sha256 = _sha256(handoff.statistics)
    artifact_validation: dict[str, Any] = {}
    if args.causal_bank is not None:
        artifact_validation["causal_bank"] = _validate_artifact(
            args.causal_bank,
            artifact_name="causal bank",
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            statistics_sha256=statistics_sha256,
        )
    else:
        artifact_validation["causal_bank"] = {"supplied": False}
    if args.live_queries is not None:
        artifact_validation["live_queries"] = _validate_artifact(
            args.live_queries,
            artifact_name="live-query cache",
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            statistics_sha256=statistics_sha256,
        )
    else:
        artifact_validation["live_queries"] = {"supplied": False}

    policy = RobotWinZevaPolicy.from_handoff(
        args.handoff_root,
        device=args.device,
        foundation_checkpoint=args.foundation_checkpoint,
        zte_checkpoint=args.zte_checkpoint,
        stage2_checkpoint=args.initial_stage2_checkpoint,
        goal_embedding_checkpoint=args.goal_embedding_checkpoint,
        retrieval_checkpoint=args.retrieval_checkpoint,
        causal_bank=args.causal_bank if args.retrieval_checkpoint is not None else None,
    )
    raw_observation = {
        "observation.state": torch.zeros(14, dtype=torch.float32),
        "task": args.task,
        **{
            key: prepare_robotwin_pi_image(
                torch.zeros(3, 480, 640, dtype=torch.uint8), name=key
            )
            for key in ROBOTWIN_CAMERA_KEYS
        },
    }
    processed = policy.preprocessor(raw_observation)
    processed["zeva.goal_embedding"] = policy._task_only_goal_embedding(  # noqa: SLF001
        args.task, processed["observation.language.tokens"].device
    )

    _seed(args.seed)
    with torch.no_grad():
        foundation_baseline = policy.foundation.predict_action_chunk(processed)

    policy.reset(scope="episode")
    policy._direct_context_injection_enabled = False  # noqa: SLF001
    _seed(args.seed)
    with torch.no_grad():
        residual_off = policy.predict_action_chunk(processed)

    policy.reset(scope="episode")
    policy._direct_context_injection_enabled = True  # noqa: SLF001
    _seed(args.seed)
    with torch.no_grad():
        residual_on = policy.predict_action_chunk(processed)

    if not torch.equal(foundation_baseline, residual_off):
        raise AssertionError("Residual-off policy changed the best-v1 foundation output.")
    if not torch.equal(foundation_baseline, residual_on):
        difference = float((foundation_baseline - residual_on).abs().max())
        raise AssertionError(f"Zero-init residual changed best-v1 output (max_abs={difference}).")

    raw_actions = policy.postprocessor(residual_on)
    with torch.no_grad():
        second_chunk = policy.predict_action_chunk(
            processed,
            executed_actions=raw_actions[:, :15],
        )
    state = policy._stage1_state  # noqa: SLF001
    if state is None or state.transition_count != 1:
        raise AssertionError("v2 policy did not record exactly one H15 recurrent transition.")
    if second_chunk.shape != (1, 50, 16):
        raise AssertionError(f"Online v2 step returned {tuple(second_chunk.shape)}.")
    finite = bool(
        torch.isfinite(foundation_baseline).all()
        and torch.isfinite(residual_off).all()
        and torch.isfinite(residual_on).all()
        and torch.isfinite(second_chunk).all()
        and torch.isfinite(raw_actions).all()
    )
    if not finite:
        raise AssertionError("Best-v1/v2 smoke emitted non-finite EEF16 actions.")

    anchor_check = {"requested": args.check_compiled_anchor}
    if args.check_compiled_anchor:
        if args.initial_stage2_checkpoint is None:
            raise ValueError("Compiled anchor smoke requires an explicit initial Stage2 checkpoint.")
        policy.configure_action_expert_finetune_stage2()
        policy.load_foundation_anchor(args.initial_stage2_checkpoint)
        policy.eval()
        policy._clear_active_residuals()  # noqa: SLF001
        core = policy.foundation.model
        core.forward = torch.compile(core.forward, mode="default")
        flow_batch = dict(processed)
        # Synthetic normalized actions: this checks weight swapping, not accuracy.
        flow_batch["action"] = torch.zeros(1, 50, 16, device=args.device)

        def flow(*, anchor: bool) -> torch.Tensor:
            _seed(args.seed)
            with torch.no_grad():
                result = (
                    policy.foundation_anchor_forward(flow_batch, reduction="none")
                    if anchor else policy.foundation(flow_batch, reduction="none")
                )
            loss = result[0] if isinstance(result, tuple) else result
            if not isinstance(loss, torch.Tensor) or not torch.isfinite(loss).all():
                raise AssertionError("Anchor smoke expected finite per-example flow tensors.")
            return loss.detach().clone()

        reference = flow(anchor=False)
        teacher_before = flow(anchor=True)
        if not torch.equal(reference, teacher_before):
            raise AssertionError("Fresh compiled student does not equal its fixed teacher.")
        bias = core.action_out_proj.bias
        original_bias = bias.detach().clone()
        try:
            with torch.no_grad():
                bias.add_(0.125)
            changed_student = flow(anchor=False)
            teacher_after = flow(anchor=True)
            changed_student_after_teacher = flow(anchor=False)
            if torch.equal(changed_student, reference):
                raise AssertionError("Perturbation did not change compiled student flow.")
            if not torch.equal(teacher_before, teacher_after):
                raise AssertionError("Compiled fixed teacher drifted with the student.")
            if not torch.equal(changed_student, changed_student_after_teacher):
                raise AssertionError("Anchor forward failed to restore the student weights.")
        finally:
            with torch.no_grad():
                bias.copy_(original_bias)
        anchor_check.update({
            "passed": True,
            "teacher_unchanged_after_student_perturbation": True,
            "student_restored_after_teacher": True,
            "teacher_matches_initial_student": True,
            "torch_compile": True,
            "synthetic_normalized_actions": True,
            "optimizer_created": False,
        })

    report = {
        "schema": "zeva-robotwin-pi05-bestv1-zte-v2-smoke-v1",
        "passed": True,
        "foundation_runtime": "RobotWinZevaPolicy.from_handoff/native LeRobot PI0.5",
        "foundation_checkpoint": str(Path(args.foundation_checkpoint).resolve()),
        "initial_stage2_checkpoint": args.initial_stage2_checkpoint,
        "initial_stage2_model_sha256": (
            _sha256(Path(args.initial_stage2_checkpoint) / "model.safetensors")
            if args.initial_stage2_checkpoint is not None else None
        ),
        "compiled_anchor_check": anchor_check,
        "zte_checkpoint": str(Path(args.zte_checkpoint).resolve()),
        "zte_checkpoint_sha256": checkpoint_sha256,
        "zte_schema": checkpoint["schema"],
        "zte_step": int(checkpoint["step"]),
        "stage1_transition_horizon": stage1_transition_horizon(checkpoint),
        "foundation_output_shape": list(foundation_baseline.shape),
        "residual_off_shape": list(residual_off.shape),
        "residual_on_shape": list(residual_on.shape),
        "second_chunk_shape": list(second_chunk.shape),
        "zero_residual_off_bit_exact": True,
        "zero_residual_on_bit_exact": True,
        "zero_residual_on_max_abs": 0.0,
        "finite_eef16": finite,
        "pending_normalized_shape": list(policy._pending_normalized_actions.shape),  # noqa: SLF001
        "state_transition_count": int(state.transition_count),
        "state_has_visual_cache": state.visual_cache is not None,
        "state_has_action_cache": state.action_cache is not None,
        "state_has_effect_cache": state.effect_cache is not None,
        "artifact_validation": artifact_validation,
        "synthetic_zero_image": True,
        "closed_loop_evaluation": False,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main(tyro.cli(Args))
