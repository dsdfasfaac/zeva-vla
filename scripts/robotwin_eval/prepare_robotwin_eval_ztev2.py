#!/usr/bin/env python3
"""Stage explicit RoboTwin v2 paired-evaluation configs.

This is deliberately a preparation step, not an evaluator.  It validates the
serving inputs and writes three configs for
``launch_paired_formal_eval.sh``:

* a normal trained Base (the explicit Base Stage2 checkpoint),
* the ZeVA Stage2 checkpoint plus the corrected v2 Stage1/bank/retrieval, and
* an untouched best-v1 foundation anchor.

The script never chooses a checkpoint from a run directory.  Every model or
artifact path is required on the command line and the output directory must be
new.  Configs are written as YAML-compatible JSON so this preflight has no
PyYAML dependency on a workstation; ``yaml.safe_load`` accepts the files used
by the existing serving launcher.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


STAGE1_V2_SCHEMA = "zeva-robotwin-zte-stage1-v2-checkpoint"
CAUSAL_BANK_SCHEMA = "zeva-robotwin-train-causal-bank-v2"
RETRIEVAL_SCHEMA = "zeva-robotwin-task-language-retrieval-v1"
ADAPTER_SCHEMAS = frozenset(
    {
        "zeva-robotwin-pi05-adapter-v3",
        "zeva-robotwin-pi05-adapter-v4",
        "zeva-robotwin-stage2-adapter-v4",
        "zeva-robotwin-stage2-adapter-v5",
    }
)
TASK_COUNT = 50
TRAIN_EPISODES = 26150
VALIDATION_EPISODES = 1350
EXECUTED_HORIZON = 15
POLICY_HORIZON = 50
ACTION_DIM = 16
ABSOLUTE_START_SEED = 1000
EPISODES_PER_TASK = 20
MODEL_RNG_SEED = 20260907


class PreflightError(RuntimeError):
    """An input is missing, ambiguous, or violates the serving contract."""


@dataclass(frozen=True)
class Inputs:
    handoff_root: Path
    foundation_checkpoint: Path
    anchor_foundation_checkpoint: Path
    goal_embedding_checkpoint: Path
    base_stage2_checkpoint: Path
    zeva_stage2_checkpoint: Path
    zte_checkpoint: Path
    causal_bank: Path
    retrieval_checkpoint: Path
    task_manifest: Path | None
    model_rng_seed: int
    episodes_per_task: int
    absolute_start_seed: int


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_required(value: str | Path, label: str) -> Path:
    raw = str(value).strip()
    if not raw or raw.lower() in {"none", "null"} or raw.startswith("<"):
        raise PreflightError(f"{label} must be an explicit checkpoint/artifact path.")
    if "__REQUIRED" in raw or raw.startswith("${"):
        raise PreflightError(f"{label} contains a placeholder instead of a selected path.")
    path = Path(raw).expanduser()
    if not path.exists():
        raise PreflightError(f"{label} does not exist: {path}")
    return path.resolve()


def _require_file(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise PreflightError(f"{label} is missing or empty: {path}")


def _require_checkpoint_dir(
    path: Path,
    label: str,
    *,
    adapter: bool = False,
    require_config: bool = False,
) -> dict[str, Any]:
    if not path.is_dir():
        raise PreflightError(f"{label} must be a checkpoint directory: {path}")
    _require_file(path / "model.safetensors", f"{label} model weights")
    if require_config:
        _require_file(path / "config.json", f"{label} config")
    if adapter:
        _require_file(path / "zeva_adapter.pth", f"{label} ZeVA adapter")
    metadata = {
        "path": str(path),
        "model_safetensors_sha256": sha256_file(path / "model.safetensors"),
    }
    if (path / "config.json").is_file():
        metadata["config_sha256"] = sha256_file(path / "config.json")
    if adapter:
        metadata["zeva_adapter_sha256"] = sha256_file(path / "zeva_adapter.pth")
    return metadata


def _require_goal_checkpoint(path: Path) -> dict[str, Any]:
    metadata = _require_checkpoint_dir(path, "goal embedding checkpoint", require_config=True)
    _require_file(path / "tokenizer" / "tokenizer.json", "goal embedding tokenizer")
    metadata["tokenizer_sha256"] = sha256_file(path / "tokenizer" / "tokenizer.json")
    return metadata


def _validate_pi_config(path: Path, label: str) -> dict[str, Any]:
    try:
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PreflightError(f"cannot read {label} config: {error}") from error
    expected = {
        "type": "pi05",
        "chunk_size": POLICY_HORIZON,
        "n_action_steps": POLICY_HORIZON,
        "normalization_mapping": {"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"},
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    state_shape = config.get("input_features", {}).get("observation.state", {}).get("shape")
    action_shape = config.get("output_features", {}).get("action", {}).get("shape")
    if state_shape != [14]:
        mismatches["observation.state.shape"] = (state_shape, [14])
    if action_shape != [ACTION_DIM]:
        mismatches["action.shape"] = (action_shape, [ACTION_DIM])
    if mismatches:
        raise PreflightError(f"{label} is outside the RoboTwin PI0.5 contract: {mismatches!r}")
    for relative in (
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
    ):
        _require_file(path / relative, f"{label} {relative}")
    return {
        "type": config["type"],
        "chunk_size": config["chunk_size"],
        "n_action_steps": config["n_action_steps"],
        "state_shape": state_shape,
        "action_shape": action_shape,
    }


def _validate_adapter(path: Path) -> dict[str, Any]:
    """Check the saved adapter's deployment contract before serving it."""
    payload = _torch_load(path, "ZeVA Stage2 adapter")
    if payload.get("schema") not in ADAPTER_SCHEMAS:
        raise PreflightError(
            "ZeVA Stage2 adapter has an unsupported schema: "
            f"{payload.get('schema')!r}"
        )
    if payload.get("physical_contract") != "joint14-relative-eef16-h50-meanstd":
        raise PreflightError("ZeVA Stage2 adapter is not the RoboTwin Joint14/EEF16/H50 contract.")
    return {
        "schema": payload["schema"],
        "physical_contract": payload["physical_contract"],
        "direct_context_injection_enabled": payload.get("direct_context_injection_enabled"),
        "output_action_correction_enabled": payload.get("output_action_correction_enabled"),
        "output_residual_correction_enabled": payload.get("output_residual_correction_enabled"),
        "sha256": sha256_file(path),
    }


def _torch_load(path: Path, label: str) -> dict[str, Any]:
    try:
        import torch  # noqa: PLC0415
    except ImportError as error:  # pragma: no cover - only workstation preflight.
        raise PreflightError(
            f"validating {label} requires the training runtime's torch installation"
        ) from error
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch versions before the weights_only keyword.
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise PreflightError(f"{label} must contain a mapping payload, got {type(payload)!r}.")
    return payload


def _validate_stage1(path: Path) -> tuple[dict[str, Any], str]:
    payload = _torch_load(path, "v2 Stage1 checkpoint")
    if payload.get("schema") != STAGE1_V2_SCHEMA:
        raise PreflightError(
            f"Stage1 checkpoint schema must be {STAGE1_V2_SCHEMA!r}, got {payload.get('schema')!r}"
        )
    config = payload.get("zte_config")
    if not isinstance(config, dict):
        raise PreflightError("v2 Stage1 checkpoint is missing zte_config.")
    expected = {
        "action_dim": ACTION_DIM,
        "action_horizon": POLICY_HORIZON,
        "executed_action_steps": EXECUTED_HORIZON,
    }
    mismatches = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
    if mismatches:
        raise PreflightError(f"v2 Stage1 physical contract mismatch: {mismatches!r}")
    manifest = payload.get("manifest")
    if not isinstance(manifest, dict):
        raise PreflightError("v2 Stage1 checkpoint is missing manifest.")
    if manifest.get("goal_embeddings_sha256") is None:
        raise PreflightError("v2 Stage1 checkpoint is missing goal_embeddings_sha256 provenance.")
    checkpoint_sha = sha256_file(path)
    return (
        {
            "schema": payload["schema"],
            "step": int(payload.get("step", -1)),
            "sha256": checkpoint_sha,
            "goal_embeddings_sha256": str(manifest["goal_embeddings_sha256"]),
            "statistics_sha256": manifest.get("statistics_sha256"),
            "executed_horizon": EXECUTED_HORIZON,
            "policy_horizon": POLICY_HORIZON,
        },
        checkpoint_sha,
    )


def _validate_status(payload: dict[str, Any], label: str) -> None:
    if payload.get("incomplete") is not False:
        raise PreflightError(f"{label} must explicitly declare incomplete=false.")
    if payload.get("usable_for_training") is not True:
        raise PreflightError(f"{label} must explicitly declare usable_for_training=true.")


def _as_task_names(payload: dict[str, Any], label: str) -> tuple[str, ...]:
    names = payload.get("task_names")
    if not isinstance(names, (list, tuple)) or len(names) != TASK_COUNT or len(set(names)) != TASK_COUNT:
        raise PreflightError(f"{label} must contain exactly {TASK_COUNT} unique task names.")
    return tuple(str(name) for name in names)


def _validate_bank(path: Path, stage1: dict[str, Any]) -> tuple[dict[str, Any], str, tuple[str, ...]]:
    payload = _torch_load(path, "v2 causal bank")
    if payload.get("schema") != CAUSAL_BANK_SCHEMA:
        raise PreflightError(f"causal bank schema must be {CAUSAL_BANK_SCHEMA!r}.")
    if payload.get("split") != "train95":
        raise PreflightError("causal bank must be train95; validation entries cannot enter deployment bank.")
    _validate_status(payload, "causal bank")
    names = _as_task_names(payload, "causal bank")
    manifest = payload.get("manifest")
    if not isinstance(manifest, dict):
        raise PreflightError("causal bank is missing manifest.")
    expected_manifest = {
        "stage1_checkpoint_schema": STAGE1_V2_SCHEMA,
        "stage1_checkpoint_sha256": stage1["sha256"],
        "causal_transition_horizon": EXECUTED_HORIZON,
        "episodes": TRAIN_EPISODES,
        "source_episodes": TRAIN_EPISODES,
    }
    mismatches = {
        key: (manifest.get(key), expected)
        for key, expected in expected_manifest.items()
        if manifest.get(key) != expected
    }
    if mismatches:
        raise PreflightError(f"causal bank provenance mismatch: {mismatches!r}")
    if manifest.get("goal_embeddings_sha256") != stage1["goal_embeddings_sha256"]:
        raise PreflightError("causal bank and v2 Stage1 use different goal embedding tables.")
    bank_sha = sha256_file(path)
    return (
        {
            "schema": payload["schema"],
            "split": payload["split"],
            "task_count": len(names),
            "episodes": TRAIN_EPISODES,
            "sha256": bank_sha,
            "manifest": {
                "stage1_checkpoint_sha256": manifest["stage1_checkpoint_sha256"],
                "goal_embeddings_sha256": manifest.get("goal_embeddings_sha256"),
                "dataset_adapter_sha256": manifest.get("dataset_adapter_sha256"),
            },
        },
        bank_sha,
        names,
    )


def _validate_retrieval(
    path: Path,
    *,
    bank_sha: str,
    stage1: dict[str, Any],
    task_names: tuple[str, ...],
) -> dict[str, Any]:
    payload = _torch_load(path, "task-language retrieval checkpoint")
    if payload.get("schema") != RETRIEVAL_SCHEMA:
        raise PreflightError(f"retrieval schema must be {RETRIEVAL_SCHEMA!r}.")
    if tuple(str(name) for name in payload.get("task_names", ())) != task_names:
        raise PreflightError("retrieval task table differs from the causal bank task table.")
    if payload.get("causal_bank_sha256") != bank_sha:
        raise PreflightError("retrieval checkpoint was trained from a different causal bank.")
    if payload.get("goal_embeddings_sha256") != stage1["goal_embeddings_sha256"]:
        raise PreflightError("retrieval checkpoint and v2 Stage1 use different goal embedding tables.")
    return {
        "schema": payload["schema"],
        "sha256": sha256_file(path),
        "causal_bank_sha256": payload["causal_bank_sha256"],
        "goal_embeddings_sha256": payload["goal_embeddings_sha256"],
        "task_count": len(task_names),
    }


def _validate_task_manifest(path: Path | None, task_names: tuple[str, ...]) -> dict[str, Any]:
    if path is None:
        return {
            "path": None,
            "task_count": TASK_COUNT,
            "selection": "all tasks from RoboTwin runtime _eval_step_limit.yml",
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PreflightError(f"cannot read task manifest {path}: {error}") from error
    names = payload.get("task_names")
    if not isinstance(names, list) or not names or len(names) != len(set(names)):
        raise PreflightError("task manifest must contain a non-empty unique task_names list.")
    unknown = sorted(set(names).difference(task_names))
    if unknown:
        raise PreflightError(f"task manifest contains unknown tasks: {unknown}")
    return {"path": str(path), "task_count": len(names), "task_names_sha256": hashlib.sha256(
        json.dumps(names, separators=(",", ":")).encode("utf-8")
    ).hexdigest()}


def validate_inputs(inputs: Inputs) -> dict[str, Any]:
    """Validate all selected inputs and return immutable provenance metadata."""
    if inputs.episodes_per_task <= 0:
        raise PreflightError("episodes_per_task must be positive.")
    if inputs.absolute_start_seed < 0:
        raise PreflightError("absolute_start_seed must be non-negative.")
    if inputs.foundation_checkpoint != inputs.anchor_foundation_checkpoint:
        raise PreflightError(
            "Base/ZeVA foundation and untouched-anchor foundation differ; refusing a drifted paired eval."
        )
    if inputs.base_stage2_checkpoint == inputs.zeva_stage2_checkpoint:
        raise PreflightError("Base and ZeVA must be distinct explicit Stage2 checkpoints.")
    handoff = inputs.handoff_root
    for relative in (
        "runtime/baseline/contract.json",
        "reference/mean-std-eef16-h50-stage1grip-train95-v2.json",
        "checkpoint/pretrained_model/config.json",
    ):
        _require_file(handoff / relative, f"handoff {relative}")
    foundation = _require_checkpoint_dir(
        inputs.foundation_checkpoint, "foundation checkpoint", require_config=True
    )
    anchor = _require_checkpoint_dir(
        inputs.anchor_foundation_checkpoint, "anchor foundation checkpoint", require_config=True
    )
    foundation["physical_contract"] = _validate_pi_config(
        inputs.foundation_checkpoint, "foundation checkpoint"
    )
    anchor["physical_contract"] = _validate_pi_config(
        inputs.anchor_foundation_checkpoint, "anchor foundation checkpoint"
    )
    goal = _require_goal_checkpoint(inputs.goal_embedding_checkpoint)
    goal["physical_contract"] = _validate_pi_config(
        inputs.goal_embedding_checkpoint, "goal embedding checkpoint"
    )
    base = _require_checkpoint_dir(inputs.base_stage2_checkpoint, "trained Base Stage2 checkpoint")
    zeva = _require_checkpoint_dir(inputs.zeva_stage2_checkpoint, "ZeVA Stage2 checkpoint", adapter=True)
    zeva["adapter"] = _validate_adapter(inputs.zeva_stage2_checkpoint / "zeva_adapter.pth")
    stage1, _ = _validate_stage1(inputs.zte_checkpoint)
    bank, bank_sha, task_names = _validate_bank(inputs.causal_bank, stage1)
    retrieval = _validate_retrieval(
        inputs.retrieval_checkpoint,
        bank_sha=bank_sha,
        stage1=stage1,
        task_names=task_names,
    )
    task_manifest = _validate_task_manifest(inputs.task_manifest, task_names)
    return {
        "handoff_root": str(handoff),
        "foundation": foundation,
        "anchor_foundation": anchor,
        "goal_embedding": goal,
        "base_stage2": base,
        "zeva_stage2": zeva,
        "stage1": stage1,
        "causal_bank": bank,
        "retrieval": retrieval,
        "task_manifest": task_manifest,
        "task_names": list(task_names),
    }


def build_configs(inputs: Inputs) -> dict[str, dict[str, Any]]:
    """Build loader-only configs; no evaluation process is started here."""
    common = {
        "policy_name": "zeva_policy",
        "handoff_root": str(inputs.handoff_root),
        "foundation_checkpoint": str(inputs.foundation_checkpoint),
        "device": "cuda",
        "model_rng_seed": inputs.model_rng_seed,
    }
    return {
        "robotwin_eval_ztev2_baseline.yml": {
            **common,
            "stage2_checkpoint": str(inputs.base_stage2_checkpoint),
            "baseline_only": True,
        },
        "robotwin_eval_ztev2_zeva.yml": {
            **common,
            "goal_embedding_checkpoint": str(inputs.goal_embedding_checkpoint),
            "stage2_checkpoint": str(inputs.zeva_stage2_checkpoint),
            "zte_checkpoint": str(inputs.zte_checkpoint),
            "causal_bank": str(inputs.causal_bank),
            "retrieval_checkpoint": str(inputs.retrieval_checkpoint),
            "baseline_only": False,
        },
        "robotwin_eval_ztev2_anchor.yml": {
            "policy_name": "zeva_policy",
            "handoff_root": str(inputs.handoff_root),
            "foundation_checkpoint": str(inputs.anchor_foundation_checkpoint),
            "baseline_only": True,
            "device": "cuda",
            "model_rng_seed": inputs.model_rng_seed,
        },
    }


def _yaml_compatible_json(payload: dict[str, Any]) -> str:
    # JSON is a strict subset of YAML 1.2 and keeps quoting/hash behavior
    # deterministic without importing PyYAML in the local staging environment.
    return json.dumps(payload, indent=2, sort_keys=False) + "\n"


def _write_new(path: Path, content: str) -> None:
    if path.exists():
        raise PreflightError(f"refusing to overwrite existing staging file: {path}")
    path.write_text(content, encoding="utf-8")


def stage(inputs: Inputs, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise PreflightError(f"staging output must be new or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = validate_inputs(inputs)
    configs = build_configs(inputs)
    for filename, config in configs.items():
        _write_new(output_dir / filename, _yaml_compatible_json(config))
    manifest = {
        "schema": "zeva-robotwin-eval-ztev2-staging-v1",
        "status": "staged_not_evaluated",
        "formal_eval_started": False,
        "checkpoint_selection": "all checkpoint paths were explicit CLI arguments; no run-directory selection was performed",
        "conditions": {
            "baseline": "normal trained Base Stage2 model; baseline_only=true; explicit stage2_checkpoint",
            "zeva": "full Stage2 model + ZeVA adapter + corrected v2 Stage1/bank/retrieval",
            "anchor": "untouched best-v1 foundation; baseline_only=true; no Stage2 adapter",
        },
        "protocol": {
            "task_config": "zeva_randomized",
            "instruction_type": "seen",
            "camera": "Large_D435_640x480",
            "state": "absolute Joint14",
            "action": "chunk-start-relative-eef16-h50",
            "execute_horizon": EXECUTED_HORIZON,
            "action_dim": ACTION_DIM,
            "episodes_per_task": inputs.episodes_per_task,
            "absolute_start_seed": inputs.absolute_start_seed,
            "seed_selection": "baseline selects first expert-valid seeds; Anchor and ZeVA replay exact frozen manifest",
            "model_seed_policy": "continuous",
            "model_rng_seed": inputs.model_rng_seed,
            "videos": True,
        },
        "frozen_language_and_cache": {
            "goal_embedding_checkpoint": str(inputs.goal_embedding_checkpoint),
            "goal_embedding_config_sha256": provenance["goal_embedding"]["config_sha256"],
            "goal_embedding_tokenizer_sha256": provenance["goal_embedding"]["tokenizer_sha256"],
            "stage1_goal_embeddings_sha256": provenance["stage1"]["goal_embeddings_sha256"],
            "stage1_schema": STAGE1_V2_SCHEMA,
            "causal_bank_split": "train95",
            "causal_bank_episodes": TRAIN_EPISODES,
            "validation_episodes": VALIDATION_EPISODES,
            "validation_bank_entries": 0,
            "cache_condition": "same explicit foundation, goal encoder, Stage1 artifact, and retrieval bank across paired configs",
        },
        "inputs": provenance,
        "config_files": {name: str((output_dir / name).resolve()) for name in configs},
        "note": "Preparation only; run launch_paired_formal_eval.sh separately after parent checkpoint selection.",
    }
    _write_new(output_dir / "robotwin_eval_ztev2_staging_manifest.json", json.dumps(
        manifest, indent=2, sort_keys=True, default=str
    ) + "\n")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    required = (
        ("--handoff-root", "handoff root"),
        ("--foundation-checkpoint", "Base/ZeVA foundation checkpoint"),
        ("--anchor-foundation-checkpoint", "untouched anchor foundation checkpoint"),
        ("--goal-embedding-checkpoint", "frozen language/goal encoder checkpoint"),
        ("--base-stage2-checkpoint", "normal trained Base Stage2 checkpoint directory"),
        ("--zeva-stage2-checkpoint", "ZeVA Stage2 checkpoint directory"),
        ("--zte-checkpoint", "corrected v2 Stage1 checkpoint"),
        ("--causal-bank", "complete train95 causal bank"),
        ("--retrieval-checkpoint", "task-language retrieval checkpoint"),
    )
    for option, help_text in required:
        parser.add_argument(option, required=True, help=help_text)
    parser.add_argument("--task-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-rng-seed", type=int, default=MODEL_RNG_SEED)
    parser.add_argument("--episodes-per-task", type=int, default=EPISODES_PER_TASK)
    parser.add_argument("--absolute-start-seed", type=int, default=ABSOLUTE_START_SEED)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    inputs = Inputs(
        handoff_root=_resolve_required(args.handoff_root, "handoff root"),
        foundation_checkpoint=_resolve_required(args.foundation_checkpoint, "foundation checkpoint"),
        anchor_foundation_checkpoint=_resolve_required(
            args.anchor_foundation_checkpoint, "anchor foundation checkpoint"
        ),
        goal_embedding_checkpoint=_resolve_required(args.goal_embedding_checkpoint, "goal embedding checkpoint"),
        base_stage2_checkpoint=_resolve_required(args.base_stage2_checkpoint, "Base Stage2 checkpoint"),
        zeva_stage2_checkpoint=_resolve_required(args.zeva_stage2_checkpoint, "ZeVA Stage2 checkpoint"),
        zte_checkpoint=_resolve_required(args.zte_checkpoint, "v2 Stage1 checkpoint"),
        causal_bank=_resolve_required(args.causal_bank, "causal bank"),
        retrieval_checkpoint=_resolve_required(args.retrieval_checkpoint, "retrieval checkpoint"),
        task_manifest=(
            None if args.task_manifest is None else _resolve_required(args.task_manifest, "task manifest")
        ),
        model_rng_seed=args.model_rng_seed,
        episodes_per_task=args.episodes_per_task,
        absolute_start_seed=args.absolute_start_seed,
    )
    manifest = stage(inputs, args.output_dir.resolve())
    print(json.dumps({
        "status": manifest["status"],
        "output_dir": str(args.output_dir.resolve()),
        "stage1_sha256": manifest["inputs"]["stage1"]["sha256"],
        "causal_bank_sha256": manifest["inputs"]["causal_bank"]["sha256"],
        "retrieval_sha256": manifest["inputs"]["retrieval"]["sha256"],
        "formal_eval_started": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI.
    raise SystemExit(main())
