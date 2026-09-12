#!/usr/bin/env python3
"""Run a bounded real-weight RoboTwin v2 serving smoke.

This is an engineering load/inference check, not an evaluator.  It stages the
explicit Base/ZeVA inputs through :mod:`prepare_robotwin_eval_ztev2`, loads one
condition at a time, and runs two synthetic-observation H50 replans.  The
first H15 is committed before the second replan so the ZeVA v2 transition
encoder must exercise its recurrent state.  No simulator, rollout, success
rate, or checkpoint selection is started.

The script is intentionally usable through ``python -`` on the handoff host:
``--robotwin-eval-dir`` points at the serving helper directory and the
preflight module can be placed on ``PYTHONPATH`` without modifying the active
checkout.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any


TASK_DEFAULT = "scan_object"
MODEL_RNG_SEED_DEFAULT = 20260907


class SmokeError(RuntimeError):
    """The serving smoke could not establish its bounded contract."""


def _numpy():
    """Import NumPy only when a fixture or real serving run is requested."""
    import numpy as np  # noqa: PLC0415

    return np


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_metadata(path: str | Path, label: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise SmokeError(f"{label} is missing or empty: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _module_paths(robotwin_eval_dir: Path | None) -> None:
    """Make the standalone serving/preflight modules importable."""
    candidates: list[Path] = []
    if robotwin_eval_dir is not None:
        candidates.append(robotwin_eval_dir.expanduser().resolve())
    script_path = globals().get("__file__")
    if script_path and not str(script_path).startswith("<"):
        candidates.append(Path(script_path).resolve().parent)
    for candidate in candidates:
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


def _synthetic_observation(task: str, image_value: int, joint_value: float) -> dict[str, Any]:
    """Build a deterministic image/state fixture, never a simulator observation."""
    np = _numpy()
    image = np.full((480, 640, 3), image_value, dtype=np.uint8)
    return {
        "observation": {
            "head_camera": {"rgb": image.copy()},
            "left_camera": {"rgb": np.full_like(image, min(255, image_value + 32))},
            "right_camera": {"rgb": np.full_like(image, min(255, image_value + 64))},
        },
        "joint_action": {"vector": np.full((14,), joint_value, dtype=np.float32)},
        # This is explicit task text supplied by the smoke caller.  It is not
        # read from an episode, simulator, or dataset row.
        "task": task,
    }


class _SyntheticRobotWinEnv:
    """Minimal pose provider for the pure action-coordinate conversion."""

    def get_arm_pose(self, arm: str) -> np.ndarray:
        np = _numpy()
        if arm not in {"left", "right"}:
            raise ValueError(f"unexpected arm request: {arm!r}")
        # RoboTwin uses world-frame [xyz, wxyz].  The smoke does not execute it.
        return np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _finite_action_summary(actions: Any, controller_actions: Any) -> dict[str, Any]:
    np = _numpy()
    model_actions = np.asarray(actions)
    converted = np.asarray(controller_actions)
    return {
        "action_shape": list(model_actions.shape),
        "action_dtype": str(model_actions.dtype),
        "finite": bool(np.isfinite(model_actions).all()),
        "controller_shape": list(converted.shape),
        "controller_dtype": str(converted.dtype),
        "controller_finite": bool(np.isfinite(converted).all()),
        "controller_executed_h15_shape": list(converted[:15].shape),
        "controller_executed_h15_finite": bool(np.isfinite(converted[:15]).all()),
        "executed_h15_shape": list(model_actions[:15].shape),
        "executed_h15_finite": bool(np.isfinite(model_actions[:15]).all()),
    }


def _cache_summary(policy: Any) -> dict[str, Any]:
    state = getattr(policy, "_stage1_state", None)
    if state is None:
        return {"state_present": False, "transition_count": None, "caches": {}}
    caches: dict[str, Any] = {}
    for name in ("visual_cache", "action_cache", "effect_cache"):
        value = getattr(state, name, None)
        caches[name] = {
            "present": value is not None,
            "seqlen_offset": (
                None if value is None or getattr(value, "seqlen_offset", None) is None
                else int(value.seqlen_offset)
            ),
        }
    return {
        "state_present": True,
        "transition_count": int(getattr(state, "transition_count", -1)),
        "caches": caches,
        "task_sum_present": getattr(state, "task_sum", None) is not None,
        "goal_embedding_present": getattr(state, "goal_embedding", None) is not None,
    }


def _adapter_flags(policy: Any, saved: dict[str, Any]) -> dict[str, Any]:
    names = (
        "direct_context_injection_enabled",
        "output_action_correction_enabled",
        "output_residual_correction_enabled",
        "prior_injection_horizon",
        "output_correction_horizon",
        "output_residual_horizon",
    )
    loaded = {
        name: getattr(policy, f"_{name}", None)
        for name in names
    }
    saved_subset = {name: saved.get(name) for name in names}
    return {
        "saved": saved_subset,
        "loaded": loaded,
        "saved_loaded_match": all(
            saved.get(name) is None or loaded.get(name) == saved.get(name)
            for name in names
        ),
    }


def _run_condition(
    config: dict[str, Any],
    *,
    condition: str,
    task: str,
    model_seed: int,
    adapter_metadata: dict[str, Any] | None,
    zeva_stage1: dict[str, Any] | None,
) -> dict[str, Any]:
    """Load one actual condition, run exactly two synthetic replans, release it."""
    # Importing this module imports the production PI0.5 runtime only on the
    # handoff host.  Keeping it here means a preflight failure does not reserve
    # GPU memory and local unit tests need no torch installation.
    serving = importlib.import_module("zeva_policy")
    ZevaModel = serving.ZevaModel
    relative_to_robotwin = serving._relative_chunk_to_robotwin
    env = _SyntheticRobotWinEnv()
    np = _numpy()
    model = None
    try:
        model = ZevaModel(config)
        model.reset_model({"seed": model_seed})
        policy = model.policy
        if condition == "zeva" and getattr(policy, "_stage1_schema", None) != (
            zeva_stage1 or {}
        ).get("schema"):
            raise SmokeError(
                "loaded ZeVA policy Stage1 schema does not match preflight metadata: "
                f"{getattr(policy, '_stage1_schema', None)!r}"
            )
        replans: list[dict[str, Any]] = []
        for index, (image_value, joint_value) in enumerate(((32, 0.0), (48, 0.01))):
            response = model.predict(_synthetic_observation(task, image_value, joint_value))
            actions = np.asarray(response.get("actions"))
            converted = relative_to_robotwin(env, actions)
            retrieval = response.get("retrieval", [])
            if not isinstance(retrieval, list):
                raise SmokeError(f"{condition} retrieval diagnostics are not a list")
            summary = _finite_action_summary(actions, converted)
            if summary["action_shape"] != [50, 16] or not summary["finite"]:
                raise SmokeError(f"{condition} replan {index} returned invalid model actions: {summary}")
            if summary["controller_shape"] != [50, 16] or not summary["controller_finite"]:
                raise SmokeError(
                    f"{condition} replan {index} returned invalid controller actions: {summary}"
                )
            summary.update(
                {
                    "replan_index": index,
                    "task_text": task,
                    "retrieval": retrieval,
                    "cache": _cache_summary(policy),
                }
            )
            replans.append(summary)
            if index == 0:
                # Only the executed H15 is fed into the next transition.  The
                # remaining H35 is deliberately not committed or simulated.
                model.commit_executed_actions(actions[:15])
        final_cache = _cache_summary(policy)
        if condition == "zeva":
            counts = [item["cache"]["transition_count"] for item in replans]
            if counts != [0, 1]:
                raise SmokeError(f"v2 ZeVA H15 state did not advance exactly once: {counts}")
            if not final_cache["state_present"]:
                raise SmokeError("v2 ZeVA serving smoke ended without a Stage1 state")
        return {
            "condition": condition,
            "loaded": True,
            "stage2_checkpoint": config.get("stage2_checkpoint"),
            "baseline_only": bool(config.get("baseline_only", False)),
            "replans": replans,
            "final_cache": final_cache,
            "retrieval_source": getattr(policy, "retrieval_source", None),
            "retrieval_task_names": list(getattr(policy, "retrieval_task_names", ())),
            "adapter": (
                _adapter_flags(policy, adapter_metadata or {}) if condition == "zeva" else None
            ),
            "stage1_schema": getattr(policy, "_stage1_schema", None),
            "stage1_transition_horizon": getattr(policy, "_stage1_transition_horizon", None),
            "episode_oracle_used": False,
            "synthetic_observation_fixture": True,
            "rollout_started": False,
            "sr_evaluation_started": False,
        }
    finally:
        # Do not leave a serving model/process on a formal-evaluation GPU.
        if model is not None:
            del model
        gc.collect()
        try:
            import torch  # noqa: PLC0415

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n").encode()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o644)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    required = (
        ("--handoff-root", "handoff root"),
        ("--foundation-checkpoint", "frozen best-v1 foundation checkpoint"),
        ("--goal-embedding-checkpoint", "frozen language/goal checkpoint"),
        ("--base-stage2-checkpoint", "completed normal Base Stage2 checkpoint"),
        ("--zeva-stage2-checkpoint", "completed ZeVA Stage2 checkpoint plus adapter"),
        ("--zte-checkpoint", "corrected v2 Stage1 checkpoint"),
        ("--causal-bank", "corrected complete train95 bank"),
        ("--retrieval-checkpoint", "corrected task-language retrieval checkpoint"),
        ("--output", "new JSON report path"),
        ("--prep-dir", "new preflight staging directory"),
    )
    for option, help_text in required:
        parser.add_argument(option, required=True, help=help_text)
    parser.add_argument("--live-queries", type=Path, default=None, help="optional lineage artifact to hash")
    parser.add_argument("--task-manifest", type=Path, default=None)
    parser.add_argument("--robotwin-eval-dir", type=Path, default=None)
    parser.add_argument("--task", default=TASK_DEFAULT, help="explicit synthetic task text")
    parser.add_argument("--model-rng-seed", type=int, default=MODEL_RNG_SEED_DEFAULT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episodes-per-task", type=int, default=20)
    parser.add_argument("--absolute-start-seed", type=int, default=1000)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.task.strip():
        raise SmokeError("--task must be non-empty explicit task text")
    output_path = Path(args.output).expanduser().resolve()
    prep_dir = Path(args.prep_dir).expanduser().resolve()
    if output_path.exists():
        raise SmokeError(f"refusing to overwrite existing report: {output_path}")
    _module_paths(args.robotwin_eval_dir)
    staging = importlib.import_module("prepare_robotwin_eval_ztev2")
    inputs = staging.Inputs(
        handoff_root=Path(args.handoff_root).expanduser().resolve(),
        foundation_checkpoint=Path(args.foundation_checkpoint).expanduser().resolve(),
        anchor_foundation_checkpoint=Path(args.foundation_checkpoint).expanduser().resolve(),
        goal_embedding_checkpoint=Path(args.goal_embedding_checkpoint).expanduser().resolve(),
        base_stage2_checkpoint=Path(args.base_stage2_checkpoint).expanduser().resolve(),
        zeva_stage2_checkpoint=Path(args.zeva_stage2_checkpoint).expanduser().resolve(),
        zte_checkpoint=Path(args.zte_checkpoint).expanduser().resolve(),
        causal_bank=Path(args.causal_bank).expanduser().resolve(),
        retrieval_checkpoint=Path(args.retrieval_checkpoint).expanduser().resolve(),
        task_manifest=(
            None if args.task_manifest is None else Path(args.task_manifest).expanduser().resolve()
        ),
        model_rng_seed=int(args.model_rng_seed),
        episodes_per_task=int(args.episodes_per_task),
        absolute_start_seed=int(args.absolute_start_seed),
    )
    staging_manifest = staging.stage(inputs, prep_dir)
    configs = {
        name: json.loads(Path(path).read_text(encoding="utf-8"))
        for name, path in staging_manifest["config_files"].items()
    }
    base_config = configs["robotwin_eval_ztev2_baseline.yml"]
    zeva_config = configs["robotwin_eval_ztev2_zeva.yml"]
    # The helper's config deliberately uses the physical device string.  The
    # caller controls the actual visible GPU (CUDA_VISIBLE_DEVICES=2 here).
    base_config["device"] = args.device
    zeva_config["device"] = args.device
    adapter_metadata = staging_manifest["inputs"]["zeva_stage2"]["adapter"]
    live_metadata = None if args.live_queries is None else _file_metadata(args.live_queries, "live queries")

    runtime: dict[str, Any] = {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    try:
        import torch  # noqa: PLC0415

        runtime.update({
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
            "cuda_device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        })
        try:
            import transformers  # noqa: PLC0415

            runtime["transformers"] = transformers.__version__
        except ImportError:
            runtime["transformers"] = None
    except ImportError:
        runtime["torch"] = None

    base_result = _run_condition(
        base_config,
        condition="baseline",
        task=args.task.strip(),
        model_seed=args.model_rng_seed,
        adapter_metadata=None,
        zeva_stage1=None,
    )
    zeva_result = _run_condition(
        zeva_config,
        condition="zeva",
        task=args.task.strip(),
        model_seed=args.model_rng_seed,
        adapter_metadata=adapter_metadata,
        zeva_stage1=staging_manifest["inputs"]["stage1"],
    )
    return {
        "schema": "zeva-robotwin-ztev2-serving-smoke-v1",
        "status": "passed",
        "passed": True,
        "engineering_only": True,
        "formal_eval_started": False,
        "rollout_started": False,
        "sr_evaluation_started": False,
        "checkpoint_selection_started": False,
        "synthetic_observation_fixture": True,
        "task_text": args.task.strip(),
        "task_source": "explicit_cli_task_text",
        "episode_oracle_used": False,
        "preflight": {
            "status": staging_manifest["status"],
            "prep_dir": str(prep_dir),
            "staging_manifest": str(prep_dir / "robotwin_eval_ztev2_staging_manifest.json"),
            "stage1": staging_manifest["inputs"]["stage1"],
            "causal_bank": staging_manifest["inputs"]["causal_bank"],
            "retrieval": staging_manifest["inputs"]["retrieval"],
            "live_queries": live_metadata,
            "zeva_adapter": adapter_metadata,
        },
        "conditions": {"baseline": base_result, "zeva": zeva_result},
        "comparison": {
            "trained_zeva_may_differ_from_base": True,
            "base_vs_zeva_equal_assertion": "not_applicable_trained_adapter",
            "comparison_scope": "both conditions only required finite H50/H15 serving and protocol-compatible conversion",
        },
        "runtime": runtime,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_path = Path(args.output).expanduser().resolve()
    try:
        report = run(args)
    except Exception as error:  # Persist bounded failure evidence for handoff.
        report = {
            "schema": "zeva-robotwin-ztev2-serving-smoke-v1",
            "status": "failed",
            "passed": False,
            "engineering_only": True,
            "formal_eval_started": False,
            "rollout_started": False,
            "sr_evaluation_started": False,
            "checkpoint_selection_started": False,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
        _write_exclusive(output_path, report)
        print(json.dumps(report, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    _write_exclusive(output_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
