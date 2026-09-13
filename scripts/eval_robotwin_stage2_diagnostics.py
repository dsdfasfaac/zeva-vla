"""Run the opt-in Stage 2 validation diagnostics without training state.

This entry point deliberately reuses the Stage 2 trainer's policy, dataset, and
``evaluate`` implementation.  It only loads an existing checkpoint and writes
one new JSON report; it never creates an optimizer or writes model/checkpoint
artifacts.
"""

from __future__ import annotations

from contextlib import contextmanager
import dataclasses
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from accelerate import Accelerator
from safetensors import safe_open
from safetensors.torch import load_model
import torch
from torch.utils.data import DataLoader
import tyro
import tqdm

try:
    from openpi.zeva.causal_bank import RobotWinCausalBank
    from openpi.zeva.robotwin_contract import RobotWinHandoff
    from openpi.zeva.robotwin_policy import RobotWinZevaPolicy
    from openpi.zeva.robotwin_policy import stage1_artifact_schema
    from openpi.zeva.robotwin_policy import validate_stage1_v2_artifact_status
    from openpi.zeva.stage1_checkpoint import LEGACY_SCHEMAS
    from openpi.zeva.stage1_checkpoint import V2_SCHEMA
    from openpi.zeva.stage1_checkpoint import stage1_transition_horizon
    from scripts.train_robotwin_stage2 import Args as Stage2Args
    from scripts.train_robotwin_stage2 import RobotWinStage2Dataset
    from scripts.train_robotwin_stage2 import _load_task_subset
    from scripts.train_robotwin_stage2 import _manifest as make_training_manifest
    from scripts.train_robotwin_stage2 import _sha256
    from scripts.train_robotwin_stage2 import _validated_runtime_versions
    from scripts.train_robotwin_stage2 import evaluate
except ModuleNotFoundError:  # Direct ``python scripts/...py`` execution.
    from train_robotwin_stage2 import Args as Stage2Args
    from train_robotwin_stage2 import RobotWinStage2Dataset
    from train_robotwin_stage2 import _load_task_subset
    from train_robotwin_stage2 import _manifest as make_training_manifest
    from train_robotwin_stage2 import _sha256
    from train_robotwin_stage2 import _validated_runtime_versions
    from train_robotwin_stage2 import evaluate

    from openpi.zeva.causal_bank import RobotWinCausalBank
    from openpi.zeva.robotwin_contract import RobotWinHandoff
    from openpi.zeva.robotwin_policy import RobotWinZevaPolicy
    from openpi.zeva.robotwin_policy import stage1_artifact_schema
    from openpi.zeva.robotwin_policy import validate_stage1_v2_artifact_status
    from openpi.zeva.stage1_checkpoint import LEGACY_SCHEMAS
    from openpi.zeva.stage1_checkpoint import V2_SCHEMA
    from openpi.zeva.stage1_checkpoint import stage1_transition_horizon


SUPPORTED_VARIANTS = frozenset({"zeva", "adapter", "prior_adapter", "prior_zeva"})
ROBOTWIN_ACTION_HORIZON = 50
ROBOTWIN_EXECUTED_HORIZON = 15


@dataclasses.dataclass
class EvalArgs:
    """Explicit inputs for one immutable-checkpoint diagnostic evaluation."""

    checkpoint: str
    manifest: str
    output: str
    handoff_root: str | None = None
    dataset_root: str | None = None
    foundation_checkpoint: str | None = None
    goal_embedding_checkpoint: str | None = None
    zte_checkpoint: str | None = None
    causal_bank: str | None = None
    live_queries: str | None = None
    task_retrieval: str | None = None
    fixed_teacher_checkpoint: str | None = None
    batch_size: int | None = None
    eval_batches: int = 0
    video_backend: str | None = None
    decoder_threads: int | None = None
    seed: int | None = None
    context_gate_scale: float = 1.0
    prior_gate_scale: float = 1.0


@contextmanager
def _validation_gate_intervention(policy, context_scale: float, prior_scale: float):
    """Scale activated gates for normal forward and diagnostic replay alike."""
    for value in (context_scale, prior_scale):
        if not math.isfinite(value) or not 0 <= value <= 100:
            raise ValueError("Validation gate scales must be finite and in [0,100].")
    if context_scale == 1.0 and prior_scale == 1.0:
        yield
        return
    method_name = "_activate_residual_gates"
    had_override = method_name in vars(policy)
    previous_override = vars(policy).get(method_name)
    original = getattr(policy, method_name)

    def activate(*args, **kwargs):
        result = original(*args, **kwargs)
        policy._active_context_gate = policy._active_context_gate * context_scale
        policy._active_prior_gate = policy._active_prior_gate * prior_scale
        return result

    setattr(policy, method_name, activate)
    try:
        yield
    finally:
        if had_override:
            setattr(policy, method_name, previous_override)
        else:
            delattr(policy, method_name)


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object at {path}, got {type(value).__name__}.")
    return value


def _require_file(path: str | Path, label: str) -> Path:
    value = Path(path).expanduser().resolve()
    if not value.is_file() or value.stat().st_size <= 0:
        raise FileNotFoundError(f"Missing {label}: {value}")
    return value


def _require_checkpoint(path: str | Path, label: str) -> Path:
    value = Path(path).expanduser().resolve()
    if not value.is_dir():
        raise FileNotFoundError(f"Missing {label} directory: {value}")
    _require_file(value / "model.safetensors", f"{label} model.safetensors")
    return value


def _checkpoint_model_identity(path: str | Path, label: str) -> dict[str, Any]:
    checkpoint = _require_checkpoint(path, label)
    model = checkpoint / "model.safetensors"
    return {
        "path": str(checkpoint),
        "model": str(model),
        "model_sha256": _sha256(model),
        "model_bytes": model.stat().st_size,
    }


def _checkpoint_adapter_identity(path: str | Path) -> dict[str, Any]:
    adapter = _require_file(path, "ZeVA adapter")
    return {
        "path": str(adapter),
        "sha256": _sha256(adapter),
        "bytes": adapter.stat().st_size,
    }


def _manifest_train_args(payload: dict[str, Any]) -> dict[str, Any]:
    recorded = payload.get("train_args")
    if not isinstance(recorded, dict):
        raise ValueError("Stage 2 manifest must contain a train_args object.")
    return recorded


def _manifest_setting(
    payload: dict[str, Any],
    recorded: dict[str, Any],
    field: str,
    *,
    override: str | int | None = None,
    top_level_key: str | None = None,
    required: bool = True,
) -> Any:
    """Resolve a setting while preserving the recorded Stage 2 contract."""
    if override is not None:
        return override
    value = recorded.get(field)
    if value is None and top_level_key is not None:
        value = payload.get(top_level_key)
    if value is None and required:
        raise ValueError(
            f"Stage 2 manifest does not record required setting {field!r}; "
            f"pass it explicitly to the diagnostic CLI."
        )
    return value


def _path_setting(
    payload: dict[str, Any],
    recorded: dict[str, Any],
    field: str,
    *,
    override: str | None = None,
    top_level_key: str | None = None,
    required: bool = True,
) -> Path | None:
    value = _manifest_setting(
        payload,
        recorded,
        field,
        override=override,
        top_level_key=top_level_key,
        required=required,
    )
    if value is None:
        return None
    return Path(value).expanduser().resolve()


def _goal_embedding_setting(
    payload: dict[str, Any], recorded: dict[str, Any], override: str | None
) -> Path | None:
    value = _manifest_setting(
        payload,
        recorded,
        "goal_embedding_checkpoint",
        override=override,
        required=False,
    )
    if value is None:
        identity = payload.get("goal_embedding_identity")
        if isinstance(identity, dict):
            value = identity.get("path")
    return Path(value).expanduser().resolve() if value is not None else None


def _stage2_args_from_manifest(payload: dict[str, Any], cli: EvalArgs) -> Stage2Args:
    recorded = _manifest_train_args(payload)
    fields = {field.name for field in dataclasses.fields(Stage2Args)}
    values = {name: value for name, value in recorded.items() if name in fields}
    args = Stage2Args(**values)

    variant = payload.get("training_variant", recorded.get("training_variant"))
    if variant is None:
        raise ValueError("Stage 2 manifest does not record training_variant.")
    if recorded.get("training_variant") not in (None, variant):
        raise ValueError("Stage 2 manifest training_variant disagrees with train_args.")
    args.training_variant = str(variant)

    args.handoff_root = str(
        _path_setting(
            payload,
            recorded,
            "handoff_root",
            override=cli.handoff_root,
            top_level_key="handoff_root",
        )
    )
    args.dataset_root = str(
        _path_setting(
            payload,
            recorded,
            "dataset_root",
            override=cli.dataset_root,
            top_level_key="dataset_root",
            required=True,
        )
    )
    foundation = _path_setting(
        payload,
        recorded,
        "foundation_checkpoint",
        override=cli.foundation_checkpoint,
        top_level_key="foundation_checkpoint",
        required=False,
    )
    if foundation is not None:
        args.foundation_checkpoint = str(foundation)
    else:
        args.foundation_checkpoint = None
    goal = _goal_embedding_setting(payload, recorded, cli.goal_embedding_checkpoint)
    args.goal_embedding_checkpoint = str(goal) if goal is not None else None
    if args.task_subset is not None:
        args.task_subset = str(Path(args.task_subset).expanduser().resolve())
    if args.initial_stage2_checkpoint is not None:
        args.initial_stage2_checkpoint = str(
            Path(args.initial_stage2_checkpoint).expanduser().resolve()
        )

    for field, override, top_level_key in (
        ("zte_checkpoint", cli.zte_checkpoint, "zte_checkpoint"),
        ("causal_bank", cli.causal_bank, "causal_bank"),
        ("live_queries", cli.live_queries, "live_queries"),
        ("task_retrieval", cli.task_retrieval, "task_retrieval"),
    ):
        path = _path_setting(
            payload,
            recorded,
            field,
            override=override,
            top_level_key=top_level_key,
            required=True,
        )
        setattr(args, field, str(path))

    if cli.video_backend is not None:
        args.video_backend = cli.video_backend
    if cli.decoder_threads is not None:
        args.decoder_threads = cli.decoder_threads
    if cli.batch_size is not None:
        args.batch_size = cli.batch_size
    if cli.seed is not None:
        args.seed = cli.seed

    if args.batch_size <= 0:
        raise ValueError(f"Diagnostic batch_size must be positive, got {args.batch_size}.")
    if args.decoder_threads <= 0:
        raise ValueError(
            f"Diagnostic decoder_threads must be positive, got {args.decoder_threads}."
        )
    args.eval_batches = 1  # replaced after the validation loader is constructed
    args.validation_diagnostics = True
    args.resume_checkpoint = None
    args.anchor_stage2_checkpoint = None
    args.anchor_foundation_checkpoint = None
    args.save_checkpoints = False
    args.compile_model = False
    args.num_workers = 0
    return args


def _selected_tasks(
    payload: dict[str, Any], args: Stage2Args
) -> tuple[str, ...] | None:
    task_scope = payload.get("task_scope")
    if not isinstance(task_scope, dict) or task_scope.get("mode") in (None, "all_tasks"):
        return _load_task_subset(args.task_subset)
    if task_scope.get("mode") != "specialization_subset":
        raise ValueError(f"Unsupported Stage 2 task_scope mode: {task_scope.get('mode')!r}.")
    if args.task_subset is not None:
        selected = _load_task_subset(args.task_subset)
        declared = task_scope.get("task_names")
        if isinstance(declared, list) and tuple(declared) != selected:
            raise ValueError("Task subset file disagrees with the Stage 2 manifest task_scope.")
        return selected
    declared = task_scope.get("task_names")
    if not isinstance(declared, list) or not declared:
        raise ValueError(
            "Specialization manifest lacks both task_subset and task_scope.task_names."
        )
    if any(not isinstance(name, str) or not name for name in declared):
        raise ValueError("Stage 2 task_scope.task_names must contain non-empty strings.")
    if len(declared) != len(set(declared)):
        raise ValueError("Stage 2 task_scope.task_names contains duplicates.")
    return tuple(declared)


def _validate_lineage(
    args: Stage2Args,
    handoff: RobotWinHandoff,
    bank: RobotWinCausalBank,
    runtime_versions: dict[str, str],
) -> dict[str, Any]:
    """Apply the same Stage 1/bank/live-query checks used by Stage 2 training."""
    training_manifest = make_training_manifest(args, handoff, bank, runtime_versions)

    live_cache = torch.load(args.live_queries, map_location="cpu", weights_only=False)
    zte_payload = torch.load(args.zte_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(live_cache, dict) or not isinstance(zte_payload, dict):
        raise ValueError("Stage 1 live-query and ZTE artifacts must be mapping payloads.")
    expected_schema = zte_payload.get("schema")
    if expected_schema not in LEGACY_SCHEMAS | {V2_SCHEMA}:
        raise ValueError("Live queries reference an unsupported Stage 1 checkpoint schema.")
    live_schema = stage1_artifact_schema(live_cache)
    if expected_schema == V2_SCHEMA:
        validate_stage1_v2_artifact_status(live_cache, artifact_name="live-query cache")
        if live_schema != V2_SCHEMA:
            raise ValueError("Stage 2 v2 requires live queries with explicit schema=v2.")
    if live_schema is not None and live_schema != expected_schema:
        raise ValueError("Live queries were exported from a different Stage 1 encoder schema.")
    if live_cache.get("zte_checkpoint_sha256") != _sha256(args.zte_checkpoint):
        raise ValueError("Live queries were not exported from the selected ZTE checkpoint.")
    if live_cache.get("statistics_sha256") != _sha256(handoff.statistics):
        raise ValueError("Live queries use different PI0.5 normalization statistics.")
    expected_horizon = stage1_transition_horizon(zte_payload)
    if expected_horizon != ROBOTWIN_EXECUTED_HORIZON:
        raise ValueError("RoboTwin Stage 2 diagnostics require the executed H15 contract.")
    live_horizon = live_cache.get("transition_horizon")
    if expected_schema == V2_SCHEMA and live_horizon is None:
        raise ValueError("Stage 2 v2 requires live queries with an explicit transition horizon.")
    if live_horizon is not None and int(live_horizon) != expected_horizon:
        raise ValueError("Live queries use a different Stage 1 transition horizon.")

    retrieval = torch.load(args.task_retrieval, map_location="cpu", weights_only=False)
    if not isinstance(retrieval, dict):
        raise ValueError("Task retrieval checkpoint must be a mapping payload.")
    if retrieval.get("causal_bank_sha256") != _sha256(args.causal_bank):
        raise ValueError("Task retrieval was not trained against the selected causal bank.")
    validation_accuracy = float(retrieval.get("validation_accuracy", 0.0))
    if validation_accuracy < 0.95:
        raise ValueError("Task-language retrieval validation accuracy is below the Stage 1.5 gate.")

    return {
        "trainer_manifest_schema": training_manifest.get("schema"),
        "stage1_schema": expected_schema,
        "stage1_step": int(zte_payload["step"]),
        "transition_horizon": expected_horizon,
        "zte_checkpoint": str(Path(args.zte_checkpoint).resolve()),
        "zte_checkpoint_sha256": _sha256(args.zte_checkpoint),
        "causal_bank": str(Path(args.causal_bank).resolve()),
        "causal_bank_sha256": _sha256(args.causal_bank),
        "live_queries": str(Path(args.live_queries).resolve()),
        "live_queries_sha256": _sha256(args.live_queries),
        "task_retrieval": str(Path(args.task_retrieval).resolve()),
        "task_retrieval_sha256": _sha256(args.task_retrieval),
        "task_retrieval_validation_accuracy": validation_accuracy,
        "statistics_sha256": _sha256(handoff.statistics),
    }


def _configure_policy(policy: RobotWinZevaPolicy, variant: str, prior_horizon: int) -> None:
    if variant == "zeva":
        policy.configure_action_expert_finetune_stage2()
    elif variant == "adapter":
        policy.configure_adapter_stage2()
    elif variant == "prior_adapter":
        policy.configure_prior_only_adapter_stage2(prior_gate_probability=0.5)
    elif variant == "prior_zeva":
        policy.configure_action_expert_prior_finetune_stage2(
            prior_gate_probability=0.5,
            prior_injection_horizon=prior_horizon,
        )
    else:
        raise ValueError(
            f"Unsupported diagnostic training_variant={variant!r}; "
            f"supported variants are {sorted(SUPPORTED_VARIANTS)}."
        )


def _verify_shared_frozen_weights(policy, student: Path, teacher: Path) -> None:
    """An action-only teacher swap is valid only with identical frozen weights."""
    trainable = {name for name, parameter in policy.foundation.named_parameters() if parameter.requires_grad}
    with safe_open(student / "model.safetensors", framework="pt", device="cpu") as left:
        with safe_open(teacher / "model.safetensors", framework="pt", device="cpu") as right:
            if set(left.keys()) != set(right.keys()):
                raise ValueError("Student and fixed teacher have different stored tensor keys.")
            for name in left.keys():
                if name not in trainable and not torch.equal(left.get_tensor(name), right.get_tensor(name)):
                    raise ValueError(f"Action-only fixed teacher has different frozen tensor: {name}")


def _ensure_new_output(output: str | Path, checkpoint: str | Path, manifest: str | Path) -> Path:
    value = Path(output).expanduser().resolve()
    if value.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic output: {value}")
    protected = {
        Path(checkpoint).expanduser().resolve(),
        Path(manifest).expanduser().resolve(),
    }
    if value in protected:
        raise ValueError("Diagnostic output must not be the checkpoint directory or source manifest.")
    return value


def _evaluation_window(requested: int, available: int) -> tuple[int, int, bool]:
    """Return ``(loop_limit, evaluated, complete)`` for an explicit cap."""
    if requested < 0:
        raise ValueError(f"eval_batches must be >= 0, got {requested}.")
    if available <= 0:
        raise ValueError(f"available validation batches must be positive, got {available}.")
    loop_limit = available if requested == 0 else requested
    return loop_limit, min(loop_limit, available), requested == 0


def _legacy_unmasked_equivalence(
    result: dict[str, Any], *, batch_size: int
) -> dict[str, Any]:
    """Compare ordinary H50 flow with the diagnostic H50 mean when padding-free.

    ``evaluate`` retains the historical unmasked PI0.5 scalar as ``flow``.
    The opt-in diagnostics intentionally compute a valid-step-aware H50 value.
    When every H50 slot is valid, their means are the same-noise
    ``raw.mean((1, 2))`` replay check; with padding, only the two values are
    reported and no equivalence is claimed.
    """
    legacy = result.get("flow")
    diagnostics = result.get("validation_diagnostics")
    flow = diagnostics.get("flow") if isinstance(diagnostics, dict) else None
    summary = flow.get("zeva_residual_on_h50") if isinstance(flow, dict) else None
    if not isinstance(legacy, (float, int)) or not isinstance(summary, dict):
        return {
            "available": False,
            "reason": "ordinary flow or H50 diagnostic summary was unavailable",
        }
    diagnostic_mean = summary.get("mean")
    valid_examples = summary.get("valid_examples")
    valid_steps = summary.get("valid_steps")
    if not isinstance(diagnostic_mean, (float, int)):
        return {
            "available": False,
            "legacy_unmasked_flow": legacy,
            "reason": "H50 diagnostic summary has no finite mean",
        }
    padding_free = (
        isinstance(valid_examples, int)
        and isinstance(valid_steps, int)
        and valid_steps == valid_examples * ROBOTWIN_ACTION_HORIZON
    )
    equal_batch_weighting = (
        isinstance(valid_examples, int)
        and batch_size > 0
        and valid_examples % batch_size == 0
    )
    report: dict[str, Any] = {
        "available": padding_free and equal_batch_weighting,
        "legacy_unmasked_flow": float(legacy),
        "diagnostic_h50_mean": float(diagnostic_mean),
        "valid_examples": valid_examples,
        "valid_steps": valid_steps,
        "padding_free": padding_free,
        "equal_batch_weighting": equal_batch_weighting,
        "raw_replay": "same-noise raw.mean((1,2)) versus existing evaluate(flow)",
    }
    if padding_free and equal_batch_weighting:
        delta = abs(float(legacy) - float(diagnostic_mean))
        report["absolute_delta"] = delta
        report["equivalent"] = math.isclose(
            float(legacy), float(diagnostic_mean), rel_tol=1e-5, abs_tol=1e-7
        )
    else:
        report["reason"] = (
            "explicit action padding was present; masked H50 is intentionally different"
            if not padding_free
            else "ordinary evaluate(flow) averages batch means; the final batch was partial"
        )
    return report


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_new_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents a concurrent invocation from replacing a
    # report after the preflight existence check.
    with path.open("x", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def main(cli: EvalArgs) -> None:
    checkpoint_dir = _require_checkpoint(cli.checkpoint, "Stage 2 checkpoint")
    manifest_path = _require_file(cli.manifest, "Stage 2 manifest")
    output_path = _ensure_new_output(cli.output, checkpoint_dir, manifest_path)

    source_manifest = _read_json(manifest_path)
    args = _stage2_args_from_manifest(source_manifest, cli)
    if args.training_variant not in SUPPORTED_VARIANTS:
        raise ValueError(
            f"This diagnostic entry point supports {sorted(SUPPORTED_VARIANTS)}, "
            f"got {args.training_variant!r}."
        )
    if args.training_variant in {"prior_zeva", "prior_adapter"} and not 0 < args.prior_injection_horizon <= 50:
        raise ValueError("prior_injection_horizon must be in [1, 50].")
    if args.training_variant == "prior_zeva" and args.prior_injection_horizon != 15:
        raise ValueError("prior_zeva diagnostics require the RoboTwin H15 prior contract.")

    adapter_identity = None
    adapter_path = checkpoint_dir / "zeva_adapter.pth"
    if args.training_variant != "baseline":
        adapter_identity = _checkpoint_adapter_identity(adapter_path)

    fixed_teacher_identity: dict[str, Any]
    fixed_teacher_dir = None
    if cli.fixed_teacher_checkpoint is None:
        fixed_teacher_identity = {
            "available": False,
            "reason": "not supplied; no fixed teacher was loaded",
        }
    else:
        fixed_teacher_dir = _require_checkpoint(cli.fixed_teacher_checkpoint, "fixed teacher")
        if args.training_variant in {"adapter", "prior_adapter"}:
            raise ValueError(
                "A fixed action-path teacher is only supported for trainable action-expert variants."
            )
        fixed_teacher_identity = {
            "available": True,
            **_checkpoint_model_identity(fixed_teacher_dir, "fixed teacher"),
        }

    runtime_versions = _validated_runtime_versions()
    accelerator = Accelerator()
    if accelerator.num_processes != 1:
        raise RuntimeError(
            "Stage 2 validation diagnostics require a single process; "
            f"got world_size={accelerator.num_processes}."
        )
    seed = int(args.seed if cli.seed is None else cli.seed)
    torch.manual_seed(seed + accelerator.process_index)

    handoff = RobotWinHandoff.from_root(args.handoff_root)
    if args.foundation_checkpoint is not None:
        handoff = dataclasses.replace(
            handoff,
            checkpoint=Path(args.foundation_checkpoint).resolve(),
        )
        handoff.validate()
    if args.training_variant in SUPPORTED_VARIANTS:
        default_checkpoint = Path(args.handoff_root).resolve() / "checkpoint" / "pretrained_model"
        if (
            args.foundation_checkpoint is not None
            and Path(args.foundation_checkpoint).resolve() != default_checkpoint
            and args.goal_embedding_checkpoint is None
        ):
            raise ValueError(
                "A changed PI foundation requires an explicit goal_embedding_checkpoint."
            )

    bank = RobotWinCausalBank.load(args.causal_bank, device=accelerator.device)
    lineage = _validate_lineage(args, handoff, bank, runtime_versions)
    selected_tasks = _selected_tasks(source_manifest, args)

    policy = RobotWinZevaPolicy.from_handoff(
        args.handoff_root,
        device=str(accelerator.device),
        foundation_checkpoint=args.foundation_checkpoint,
        goal_embedding_checkpoint=args.goal_embedding_checkpoint,
        zte_checkpoint=args.zte_checkpoint,
        retrieval_checkpoint=args.task_retrieval,
        causal_bank=args.causal_bank,
        stage2_checkpoint=args.initial_stage2_checkpoint,
    )
    _configure_policy(policy, args.training_variant, args.prior_injection_horizon)
    # Load the immutable teacher before replacing the student action path with
    # the selected final checkpoint, matching the trainer's resume order.
    if fixed_teacher_dir is not None:
        _verify_shared_frozen_weights(policy, checkpoint_dir, fixed_teacher_dir)
        load_model(policy.foundation, fixed_teacher_dir / "model.safetensors", strict=True)
        policy.load_foundation_anchor(fixed_teacher_dir)
        fixed_teacher_identity["shared_frozen_weights_verified"] = True
    load_model(policy.foundation, checkpoint_dir / "model.safetensors", strict=True)
    if adapter_identity is not None:
        policy.load_adapter(adapter_path)
    policy.eval()

    adapter_manifest = Path(args.dataset_root) / "adapter.json"
    validation_dataset = RobotWinStage2Dataset(
        adapter_manifest,
        args.live_queries,
        subset="validation",
        config=policy.zeva_config,
        selected_tasks=selected_tasks,
        video_backend=args.video_backend,
        decoder_threads=args.decoder_threads,
    )
    if validation_dataset.task_names != bank.task_names:
        raise ValueError("Dataset task ordering differs from the Stage 1 causal bank.")
    if selected_tasks is not None and validation_dataset.selected_task_names != selected_tasks:
        raise ValueError("The requested Stage 2 task subset was not applied exactly.")
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )
    available_batches = len(validation_loader)
    if available_batches <= 0:
        raise ValueError("Stage 2 validation dataset is empty.")
    args.eval_batches, evaluated_batches, complete = _evaluation_window(
        cli.eval_batches, available_batches
    )
    if evaluated_batches <= 0:
        raise ValueError("eval_batches must be positive when supplied.")

    policy, validation_loader = accelerator.prepare(policy, validation_loader)
    policy.eval()
    with _validation_gate_intervention(policy, cli.context_gate_scale, cli.prior_gate_scale):
        result = evaluate(
            policy,
            tqdm.tqdm(validation_loader, total=min(args.eval_batches, len(validation_loader)), desc="Read-only validation"),
            bank,
            policy.preprocessor,
            policy.zeva_config,
            args,
            accelerator,
        )
    accelerator.wait_for_everyone()

    report = {
        "schema": "zeva-robotwin-stage2-validation-diagnostics-v1",
        "checkpoint": {
            **_checkpoint_model_identity(checkpoint_dir, "Stage 2 checkpoint"),
            "adapter": adapter_identity,
        },
        "source_manifest": {
            "path": str(manifest_path),
            "sha256": _sha256(manifest_path),
        },
        "fixed_teacher": fixed_teacher_identity,
        "lineage": lineage,
        "protocol": {
            "validation_only_intervention": {
                "context_gate_scale": cli.context_gate_scale,
                "prior_gate_scale": cli.prior_gate_scale,
                "deployment_or_checkpoint_modified": False,
            },
            "seed": seed,
            "validation_decision_samples": len(validation_dataset),
            "ordered_validation_samples_sha256": hashlib.sha256(
                json.dumps(validation_dataset._samples, separators=(",", ":")).encode()
            ).hexdigest(),
            "dataset_adapter_sha256": _sha256(adapter_manifest),
            "training_variant": args.training_variant,
            "dataset_root": str(Path(args.dataset_root).resolve()),
            "dataset_adapter": str(adapter_manifest.resolve()),
            "validation_split": "validation5",
            "policy_horizon": ROBOTWIN_ACTION_HORIZON,
            "executed_horizon": ROBOTWIN_EXECUTED_HORIZON,
            "batch_size": args.batch_size,
            "video_backend": args.video_backend,
            "decoder_threads": args.decoder_threads,
            "requested_eval_batches": cli.eval_batches,
            "available_validation_batches": available_batches,
            "evaluated_batches": evaluated_batches,
            "complete": complete,
            "validation_diagnostics": True,
            "optimizer_created": False,
            "checkpoint_written": False,
            "torch_compile": False,
            "single_process": True,
        },
        "diagnostic_source_sha256": {
            "entrypoint": _sha256(Path(__file__).resolve()),
            "trainer": _sha256(Path(__file__).resolve().with_name("train_robotwin_stage2.py")),
        },
        "result": result,
        "legacy_unmasked_flow_equivalence": _legacy_unmasked_equivalence(
            result, batch_size=args.batch_size
        ),
    }
    _write_new_json(output_path, report)


if __name__ == "__main__":
    main(tyro.cli(EvalArgs))
