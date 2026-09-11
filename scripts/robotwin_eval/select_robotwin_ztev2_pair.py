#!/usr/bin/env python3
"""Select matched RoboTwin v2 Stage 2 checkpoints after both runs finish.

The selector is intentionally an offline preflight.  It does not inspect
closed-loop success, a paired residual-off proxy, or the ZeVA total loss (the
latter includes a Gaussian prior NLL).  Each run is selected independently by
the finite held-out ``validation.flow`` value saved in its own training state;
an exact tie is resolved in favour of the earlier optimizer step.

The training state contains optimizer tensors and can be several gigabytes.
States are therefore loaded with ``torch.load(..., map_location="cpu",
weights_only=False, mmap=True)``.  The selector never loads model weights into
Torch; it only hashes the selected immutable files after the choice is made.

Normal selection requires explicit PID arguments.  A PID of zero means the
caller has explicitly established that no process remains; a live PID causes
selection to fail.  ``--metadata-only`` is available for bounded audits of an
incomplete run and never writes a selection JSON.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Callable, Mapping


FINAL_STEP = 5_000
SAVE_FREQ = 500
EXPECTED_CANDIDATE_STEPS = tuple(range(SAVE_FREQ, FINAL_STEP + SAVE_FREQ, SAVE_FREQ))
EXPECTED_VARIANTS = {"baseline": "baseline", "zeva": "zeva"}
EXPECTED_STATE_SCHEMAS = {
    "baseline": "robotwin-pi05-action-expert-baseline-training-state-v1",
    "zeva": "zeva-robotwin-stage2-action-expert-training-state-v8",
}
RESUME_PROVENANCE_FILENAME = "resume_provenance.json"
RESUME_SOURCE_MANIFEST_GLOB = "manifest.source_before_resume_*.json"
RESUME_PROVENANCE_SCHEMA = "zeva-robotwin-stage2-resume-provenance-v1"
# The Stage 2 trainer records the resume source in ``train_args``.  This is
# the only run-manifest field that may differ after an in-place resume.  The
# checkpoint lineage, topology, objective, and all scientific settings remain
# part of the immutable source manifest contract.
RESUME_ALLOWED_MANIFEST_DIFFS = frozenset({"train_args.resume_checkpoint"})

# These fields are deliberately the pair contract, not the complete manifest.
# Stage 1/bank/live/retrieval lineage is allowed to differ because Base loads
# those artifacts for validation only while ZeVA uses its corrected artifacts.
PAIR_MANIFEST_FIELDS = (
    "handoff_root",
    "foundation_checkpoint",
    "foundation_identity",
    "goal_embedding_identity",
    "foundation_forward",
    "runtime_versions",
    "pi_image_contract",
    "dataset_adapter",
    "train_split",
    "validation_split",
    "task_scope",
    "effective_global_batch_size",
    "world_size",
    "video_decode",
)
PAIR_TRAIN_ARGS = (
    "steps",
    "warmup_steps",
    "batch_size",
    "gradient_accumulation_steps",
    "num_workers",
    "video_backend",
    "decoder_threads",
    "action_expert_learning_rate",
    "learning_rate",
    "weight_decay",
    "adam_beta2",
    "prior_loss_weight",
    "preserve_loss_weight",
    "paired_improvement_margin",
    "baseline_preserve_interval",
    "gate_regularization_weight",
    "phase_noise_std",
    "memory_dropout",
    "prior_residual_dropout_probability",
    "initial_residual_gate_probability",
    "save_freq",
    "save_checkpoints",
    "eval_batches",
    "log_freq",
    "compile_model",
    "compile_mode",
    "seed",
    "dataset_root",
    "task_subset",
)
ALLOWED_LINEAGE_FIELDS = (
    "zte_checkpoint",
    "zte_checkpoint_sha256",
    "zte_checkpoint_schema",
    "zte_step",
    "causal_bank",
    "causal_bank_sha256",
    "causal_bank_manifest",
    "live_queries",
    "live_queries_sha256",
    "task_retrieval",
    "task_retrieval_sha256",
    "stage1_usage",
)
_STEP_RE = re.compile(r"^(\d{6})$")


class SelectionError(RuntimeError):
    """Raised when a run or pair violates the immutable selection contract."""


@dataclasses.dataclass(frozen=True)
class Candidate:
    step: int
    checkpoint: Path
    training_state: Path
    model: Path
    adapter: Path | None
    validation_flow: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "checkpoint": str(self.checkpoint),
            "training_state": str(self.training_state),
            "model": str(self.model),
            "adapter": str(self.adapter) if self.adapter is not None else None,
            "validation_flow": self.validation_flow,
        }


@dataclasses.dataclass(frozen=True)
class RunAudit:
    role: str
    root: Path
    manifest_path: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    candidates: tuple[Candidate, ...]
    final_step: int | None
    resume_provenance: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "run_dir": str(self.root),
            "manifest": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "training_variant": self.manifest.get("training_variant"),
            "training_mode": self.manifest.get("training_mode"),
            "final_step": self.final_step,
            "candidates": [candidate.as_dict() for candidate in self.candidates],
            "resume_provenance": self.resume_provenance,
        }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SelectionError(f"cannot read {label}: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise SelectionError(f"{label} must be a JSON object: {path}")
    return payload


def _require_file(path: Path, label: str) -> None:
    try:
        stat = path.stat()
    except OSError as error:
        raise SelectionError(f"missing {label}: {path}") from error
    if not path.is_file() or stat.st_size <= 0:
        raise SelectionError(f"missing or empty {label}: {path}")


def _torch_load_training_state(path: Path) -> dict[str, Any]:
    """Load only checkpoint metadata while preserving mmap as a hard invariant."""
    try:
        import torch  # noqa: PLC0415
    except ImportError as error:  # pragma: no cover - workstation without runtime torch.
        raise SelectionError("selection requires the training runtime's torch") from error
    try:
        state = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except TypeError as error:
        raise SelectionError(
            "training runtime must support torch.load(..., mmap=True); "
            f"refusing a materializing fallback for {path}"
        ) from error
    except Exception as error:
        raise SelectionError(f"cannot mmap-load training state {path}: {error}") from error
    if not isinstance(state, dict):
        raise SelectionError(f"training state must be a mapping: {path}")
    return state


def _candidate_steps(root: Path) -> list[int]:
    steps: list[int] = []
    for child in root.iterdir():
        match = _STEP_RE.fullmatch(child.name)
        if match:
            step = int(match.group(1))
            if step > 0 and step % SAVE_FREQ == 0:
                steps.append(step)
    return sorted(set(steps))


def _check_manifest_variant(manifest: Mapping[str, Any], role: str) -> None:
    expected = EXPECTED_VARIANTS[role]
    if manifest.get("training_variant") != expected:
        raise SelectionError(
            f"{role} manifest training_variant must be {expected!r}, "
            f"got {manifest.get('training_variant')!r}"
        )
    train_args = manifest.get("train_args")
    if not isinstance(train_args, dict) or train_args.get("training_variant") != expected:
        raise SelectionError(f"{role} manifest train_args do not declare {expected!r}")
    if manifest.get("causal_transition_horizon") != 15:
        raise SelectionError(f"{role} does not declare the fixed H15 transition horizon")
    residual = manifest.get("causal_context_residual")
    if not isinstance(residual, dict) or residual.get("broadcast_horizon") != 50:
        raise SelectionError(f"{role} does not declare the fixed H50 broadcast horizon")
    if manifest.get("effective_global_batch_size") != 256:
        raise SelectionError(f"{role} effective_global_batch_size is not 256")
    optimizer = manifest.get("optimizer_groups")
    if not isinstance(optimizer, dict):
        raise SelectionError(f"{role} manifest is missing optimizer_groups")
    action_group = optimizer.get("pi05_action_expert")
    try:
        action_lr = float(action_group.get("learning_rate", -1)) if isinstance(action_group, dict) else -1.0
    except (TypeError, ValueError):
        action_lr = -1.0
    if action_lr != 5e-6:
        raise SelectionError(f"{role} action-expert learning rate is not 5e-6")
    for key, expected_value in {
        "steps": FINAL_STEP,
        "save_freq": SAVE_FREQ,
        "warmup_steps": 500,
    }.items():
        if train_args.get(key) != expected_value:
            raise SelectionError(
                f"{role} train_args[{key!r}] must be {expected_value!r}, "
                f"got {train_args.get(key)!r}"
            )


def _manifest_difference_paths(
    source: Any,
    current: Any,
    path: str = "",
) -> list[str]:
    """Return leaf paths whose values differ between two manifests.

    Manifests contain nested dictionaries and a few lists.  Keeping the
    comparison structural gives resume validation a precise allow-list while
    avoiding a brittle full-JSON string comparison (formatting/key order are
    not scientific settings).
    """

    if isinstance(source, Mapping) and isinstance(current, Mapping):
        differences: list[str] = []
        for key in sorted(set(source) | set(current)):
            child = f"{path}.{key}" if path else str(key)
            if key not in source or key not in current:
                differences.append(child)
            else:
                differences.extend(_manifest_difference_paths(source[key], current[key], child))
        return differences
    if source != current:
        return [path or "<root>"]
    return []


def _resolve_run_child(root: Path, value: Any, label: str) -> Path:
    """Resolve a provenance path and require it to stay inside the run."""

    if not isinstance(value, str) or not value:
        raise SelectionError(f"resume provenance is missing {label}")
    path = Path(value).expanduser().resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise SelectionError(
            f"resume provenance {label} must be inside the run directory: {path}"
        ) from error
    if len(relative.parts) < 1:
        raise SelectionError(f"resume provenance {label} must name a file: {path}")
    return path


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise SelectionError(f"resume provenance {label} must be a lowercase SHA256")
    return value


def _load_resume_context(
    root: Path,
    role: str,
    current_manifest: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Validate an in-place resume and return the allowed old-state context.

    A resumed trainer rewrites ``manifest.json`` before saving step 5000,
    while steps 500..4500 retain the original manifest embedded in their
    training states.  The selector may bridge that one metadata transition
    only when an operator left a signed-by-hash provenance record and an
    immutable source-manifest copy beside the run.  No model/objective/data
    field is relaxed here.
    """

    train_args = current_manifest.get("train_args")
    resume_value = train_args.get("resume_checkpoint") if isinstance(train_args, Mapping) else None
    provenance_path = root / RESUME_PROVENANCE_FILENAME
    source_backups = {path.resolve() for path in root.glob(RESUME_SOURCE_MANIFEST_GLOB)}
    if not provenance_path.exists() and not source_backups and resume_value in (None, ""):
        return None
    if not provenance_path.is_file():
        raise SelectionError(
            f"{role} has resume metadata but no {RESUME_PROVENANCE_FILENAME}; "
            "refusing to relax state-manifest validation"
        )
    provenance = _read_json(provenance_path, f"{role} resume provenance")
    if provenance.get("schema") != RESUME_PROVENANCE_SCHEMA:
        raise SelectionError(
            f"{role} resume provenance schema must be {RESUME_PROVENANCE_SCHEMA!r}"
        )
    if provenance.get("role") != role:
        raise SelectionError(
            f"{role} resume provenance role must be {role!r}, got {provenance.get('role')!r}"
        )
    try:
        source_step = int(provenance.get("source_step"))
    except (TypeError, ValueError) as error:
        raise SelectionError(f"{role} resume provenance source_step is invalid") from error
    if source_step != FINAL_STEP - SAVE_FREQ:
        raise SelectionError(
            f"{role} resume provenance must identify source step {FINAL_STEP - SAVE_FREQ}, "
            f"got {source_step}"
        )

    source_checkpoint = _resolve_run_child(root, provenance.get("source_checkpoint"), "source_checkpoint")
    expected_checkpoint = root / f"{source_step:06d}"
    if source_checkpoint != expected_checkpoint:
        raise SelectionError(
            f"{role} resume source checkpoint must be {expected_checkpoint}, got {source_checkpoint}"
        )
    source_manifest_path = _resolve_run_child(root, provenance.get("source_manifest"), "source_manifest")
    expected_manifest_name = f"manifest.source_before_resume_{source_step}.json"
    if source_manifest_path.name != expected_manifest_name:
        raise SelectionError(
            f"{role} resume source manifest must be named {expected_manifest_name!r}"
        )
    if not source_manifest_path.is_file():
        raise SelectionError(f"{role} resume source manifest is missing: {source_manifest_path}")
    if source_manifest_path not in source_backups:
        raise SelectionError(
            f"{role} resume source manifest is not the saved run-local backup: {source_manifest_path}"
        )
    source_manifest_sha256 = _require_sha256(
        provenance.get("source_manifest_sha256"), "source_manifest_sha256"
    )
    actual_source_manifest_sha256 = sha256_file(source_manifest_path)
    if actual_source_manifest_sha256 != source_manifest_sha256:
        raise SelectionError(
            f"{role} resume source manifest SHA256 does not match provenance"
        )
    source_manifest = _read_json(source_manifest_path, f"{role} resume source manifest")
    _check_manifest_variant(source_manifest, role)

    differences = _manifest_difference_paths(source_manifest, current_manifest)
    unexpected = [path for path in differences if path not in RESUME_ALLOWED_MANIFEST_DIFFS]
    if unexpected:
        raise SelectionError(
            f"{role} resume manifest differs outside allowed metadata: {', '.join(unexpected)}"
        )
    source_args = source_manifest.get("train_args")
    current_args = current_manifest.get("train_args")
    if not isinstance(source_args, Mapping) or not isinstance(current_args, Mapping):
        raise SelectionError(f"{role} resume manifests must contain train_args mappings")
    if source_args.get("resume_checkpoint") not in (None, ""):
        raise SelectionError(
            f"{role} saved source manifest already contains resume_checkpoint; refusing chained resume"
        )
    current_resume = current_args.get("resume_checkpoint")
    if not isinstance(current_resume, str) or Path(current_resume).expanduser().resolve() != source_checkpoint:
        raise SelectionError(
            f"{role} current manifest resume_checkpoint does not identify source checkpoint"
        )
    for field in ("world_size", "effective_global_batch_size"):
        if source_manifest.get(field) != current_manifest.get(field):
            raise SelectionError(
                f"{role} resume changed topology field {field!r}; only same-world resume is allowed"
            )
    for field in ("batch_size", "gradient_accumulation_steps"):
        if source_args.get(field) != current_args.get(field):
            raise SelectionError(
                f"{role} resume changed train_args.{field}; only same-topology resume is allowed"
            )

    source_model = _resolve_run_child(root, provenance.get("source_model"), "source_model")
    source_optimizer = _resolve_run_child(
        root, provenance.get("source_optimizer_state"), "source_optimizer_state"
    )
    expected_model = source_checkpoint / "model.safetensors"
    expected_optimizer = source_checkpoint / "training_state.pt"
    if source_model != expected_model:
        raise SelectionError(f"{role} resume source_model does not identify 4500 model.safetensors")
    if source_optimizer != expected_optimizer:
        raise SelectionError(f"{role} resume source_optimizer_state does not identify 4500 training_state.pt")
    _require_file(source_model, f"{role} resume source model")
    _require_file(source_optimizer, f"{role} resume source optimizer state")
    model_sha256 = _require_sha256(provenance.get("source_model_sha256"), "source_model_sha256")
    optimizer_sha256 = _require_sha256(
        provenance.get("source_optimizer_state_sha256"), "source_optimizer_state_sha256"
    )
    if sha256_file(source_model) != model_sha256:
        raise SelectionError(f"{role} source model SHA256 does not match resume provenance")
    if sha256_file(source_optimizer) != optimizer_sha256:
        raise SelectionError(f"{role} source optimizer-state SHA256 does not match resume provenance")

    source_adapter = provenance.get("source_adapter")
    source_adapter_sha256 = provenance.get("source_adapter_sha256")
    if role == "zeva":
        source_adapter_path = _resolve_run_child(root, source_adapter, "source_adapter")
        expected_adapter = source_checkpoint / "zeva_adapter.pth"
        if source_adapter_path != expected_adapter:
            raise SelectionError(f"{role} resume source_adapter does not identify 4500 zeva_adapter.pth")
        _require_file(source_adapter_path, f"{role} resume source adapter")
        source_adapter_sha256 = _require_sha256(source_adapter_sha256, "source_adapter_sha256")
        if sha256_file(source_adapter_path) != source_adapter_sha256:
            raise SelectionError(f"{role} source adapter SHA256 does not match resume provenance")
    elif source_adapter not in (None, "") or source_adapter_sha256 not in (None, ""):
        raise SelectionError("baseline resume provenance must not declare an adapter source")

    return {
        "schema": provenance["schema"],
        "path": str(provenance_path),
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": source_manifest_sha256,
        "source_checkpoint": str(source_checkpoint),
        "source_step": source_step,
        "source_model_sha256": model_sha256,
        "source_optimizer_state_sha256": optimizer_sha256,
        "source_adapter_sha256": source_adapter_sha256,
        "allowed_manifest_differences": sorted(differences),
        "same_topology": True,
        "rng_state_saved": False,
        "continuation": "non_bit_exact_resume_without_saved_rng_state",
        "source_manifest_payload": source_manifest,
    }


def _inspect_candidate(
    root: Path,
    role: str,
    manifest: dict[str, Any],
    step: int,
    *,
    resume_context: Mapping[str, Any] | None = None,
    state_loader: Callable[[Path], dict[str, Any]] = _torch_load_training_state,
) -> Candidate:
    checkpoint = root / f"{step:06d}"
    if not checkpoint.is_dir():
        raise SelectionError(f"{role} missing candidate directory for step {step}: {checkpoint}")
    model = checkpoint / "model.safetensors"
    training_state = checkpoint / "training_state.pt"
    _require_file(model, f"{role} step {step} model.safetensors")
    _require_file(training_state, f"{role} step {step} training_state.pt")
    adapter: Path | None = None
    if role == "zeva":
        adapter = checkpoint / "zeva_adapter.pth"
        _require_file(adapter, f"{role} step {step} zeva_adapter.pth")

    state = state_loader(training_state)
    expected_schema = EXPECTED_STATE_SCHEMAS[role]
    if state.get("schema") != expected_schema:
        raise SelectionError(
            f"{role} step {step} training-state schema must be {expected_schema!r}, "
            f"got {state.get('schema')!r}"
        )
    if state.get("step") != step:
        raise SelectionError(
            f"{role} step {step} training state declares step {state.get('step')!r}"
        )
    state_manifest = state.get("manifest")
    state_manifest_matches = state_manifest == manifest
    if not state_manifest_matches and resume_context is not None:
        source_step = int(resume_context["source_step"])
        state_manifest_matches = (
            step <= source_step and state_manifest == resume_context["source_manifest_payload"]
        )
    if not state_manifest_matches:
        raise SelectionError(
            f"{role} step {step} training-state manifest differs from manifest.json"
        )
    validation = state.get("validation")
    if not isinstance(validation, dict):
        raise SelectionError(f"{role} step {step} training state has no validation mapping")
    flow = validation.get("flow")
    if isinstance(flow, bool) or not isinstance(flow, (int, float)) or not math.isfinite(float(flow)):
        raise SelectionError(
            f"{role} step {step} validation.flow must be finite; got {flow!r}"
        )
    return Candidate(
        step=step,
        checkpoint=checkpoint,
        training_state=training_state,
        model=model,
        adapter=adapter,
        validation_flow=float(flow),
    )


def inspect_run(
    run_dir: str | Path,
    role: str,
    *,
    require_final: bool = True,
    state_loader: Callable[[Path], dict[str, Any]] = _torch_load_training_state,
) -> RunAudit:
    """Audit a run; ``require_final=False`` supports incomplete metadata smoke."""
    if role not in EXPECTED_VARIANTS:
        raise SelectionError(f"unsupported role: {role!r}")
    root = Path(run_dir).expanduser().resolve()
    if not root.is_dir():
        raise SelectionError(f"{role} run directory does not exist: {root}")
    manifest_path = root / "manifest.json"
    _require_file(manifest_path, f"{role} manifest.json")
    manifest = _read_json(manifest_path, f"{role} manifest")
    _check_manifest_variant(manifest, role)
    resume_context = _load_resume_context(root, role, manifest)

    available = _candidate_steps(root)
    if not available:
        raise SelectionError(f"{role} has no checkpoint candidate directories: {root}")
    if require_final:
        missing = [step for step in EXPECTED_CANDIDATE_STEPS if step not in available]
        if missing:
            raise SelectionError(
                f"{role} is not complete through step {FINAL_STEP}; missing candidate steps {missing}"
            )
        selected_steps = list(EXPECTED_CANDIDATE_STEPS)
    else:
        selected_steps = [step for step in available if step <= FINAL_STEP]

    candidates = tuple(
        _inspect_candidate(
            root,
            role,
            manifest,
            step,
            resume_context=resume_context,
            state_loader=state_loader,
        )
        for step in selected_steps
    )
    final_step = FINAL_STEP if FINAL_STEP in selected_steps else (selected_steps[-1] if selected_steps else None)
    return RunAudit(
        role=role,
        root=root,
        manifest_path=manifest_path,
        manifest=manifest,
        manifest_sha256=sha256_file(manifest_path),
        candidates=candidates,
        final_step=final_step,
        resume_provenance=(
            {
                key: value
                for key, value in resume_context.items()
                if key != "source_manifest_payload"
            }
            if resume_context is not None
            else None
        ),
    )


def _train_args_signature(manifest: Mapping[str, Any]) -> dict[str, Any]:
    train_args = manifest.get("train_args")
    if not isinstance(train_args, Mapping):
        return {field: None for field in PAIR_TRAIN_ARGS}
    return {field: train_args.get(field) for field in PAIR_TRAIN_ARGS}


def _pair_checks(base: RunAudit, zeva: RunAudit) -> dict[str, bool]:
    base_manifest = base.manifest
    zeva_manifest = zeva.manifest
    checks: dict[str, bool] = {}
    for field in PAIR_MANIFEST_FIELDS:
        checks[f"matching_manifest.{field}"] = base_manifest.get(field) == zeva_manifest.get(field)
    base_args = _train_args_signature(base_manifest)
    zeva_args = _train_args_signature(zeva_manifest)
    for field in PAIR_TRAIN_ARGS:
        checks[f"matching_train_args.{field}"] = base_args[field] == zeva_args[field]
    checks["base_variant_exact"] = base_manifest.get("training_variant") == "baseline"
    checks["zeva_variant_exact"] = zeva_manifest.get("training_variant") == "zeva"
    checks["base_h15_transition"] = base_manifest.get("causal_transition_horizon") == 15
    checks["zeva_h15_transition"] = zeva_manifest.get("causal_transition_horizon") == 15
    checks["base_h50_broadcast"] = (
        (base_manifest.get("causal_context_residual") or {}).get("broadcast_horizon") == 50
    )
    checks["zeva_h50_broadcast"] = (
        (zeva_manifest.get("causal_context_residual") or {}).get("broadcast_horizon") == 50
    )
    checks["base_action_expert_lr"] = (
        (base_manifest.get("optimizer_groups") or {}).get("pi05_action_expert", {}).get(
            "learning_rate"
        )
        == 5e-6
    )
    checks["zeva_action_expert_lr"] = (
        (zeva_manifest.get("optimizer_groups") or {}).get("pi05_action_expert", {}).get(
            "learning_rate"
        )
        == 5e-6
    )
    checks["base_final_candidate"] = base.final_step == FINAL_STEP
    checks["zeva_final_candidate"] = zeva.final_step == FINAL_STEP
    return checks


def _raise_failed_checks(checks: Mapping[str, bool]) -> None:
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SelectionError("matched pair contract failed: " + ", ".join(failed))


def _select_candidate(audit: RunAudit) -> tuple[Candidate, list[dict[str, Any]]]:
    if not audit.candidates:
        raise SelectionError(f"{audit.role} has no candidates to select")
    selected = min(audit.candidates, key=lambda candidate: (candidate.validation_flow, candidate.step))
    records: list[dict[str, Any]] = []
    for candidate in audit.candidates:
        record = candidate.as_dict()
        if candidate is selected:
            record["selection_reason"] = (
                "selected: minimum finite held-out validation.flow; exact ties choose earlier step"
            )
        elif candidate.validation_flow == selected.validation_flow:
            record["selection_reason"] = "rejected: exact validation.flow tie lost to earlier step"
        else:
            record["selection_reason"] = "rejected: higher held-out validation.flow than selected candidate"
        records.append(record)
    return selected, records


def _pid_status(pid: int, label: str) -> dict[str, Any]:
    if pid < 0:
        raise SelectionError(f"{label} PID must be non-negative, got {pid}")
    if pid == 0:
        return {"pid": 0, "running": False, "check": "explicit_zero_no_process"}
    proc_stat = Path(f"/proc/{pid}/stat")
    if proc_stat.is_file():
        try:
            state = proc_stat.read_text(encoding="utf-8").split()[2]
        except (OSError, IndexError) as error:
            raise SelectionError(f"cannot inspect {label} PID {pid}: {error}") from error
        if state != "Z":
            raise SelectionError(f"{label} PID {pid} is still running (state {state!r})")
        return {"pid": pid, "running": False, "check": "zombie_process"}
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return {"pid": pid, "running": False, "check": "kill_zero_not_found"}
    except PermissionError as error:
        raise SelectionError(f"cannot prove {label} PID {pid} stopped: {error}") from error
    except OSError as error:
        raise SelectionError(f"cannot inspect {label} PID {pid}: {error}") from error
    raise SelectionError(f"{label} PID {pid} is still running")


def _selected_record(candidate: Candidate) -> dict[str, Any]:
    record = {
        "step": candidate.step,
        "checkpoint": str(candidate.checkpoint),
        "model": str(candidate.model),
        "model_sha256": sha256_file(candidate.model),
        "validation_flow": candidate.validation_flow,
    }
    if candidate.adapter is not None:
        record["adapter"] = str(candidate.adapter)
        record["adapter_sha256"] = sha256_file(candidate.adapter)
    else:
        record["adapter"] = None
        record["adapter_sha256"] = None
    return record


def select_pair(
    base_dir: str | Path,
    zeva_dir: str | Path,
    *,
    base_pid: int | None,
    zeva_pid: int | None,
    state_loader: Callable[[Path], dict[str, Any]] = _torch_load_training_state,
) -> dict[str, Any]:
    """Audit and select a completed pair without writing output."""
    if base_pid is None or zeva_pid is None:
        raise SelectionError("normal selection requires explicit --base-pid and --zeva-pid")
    pid_checks = {
        "base": _pid_status(base_pid, "Base"),
        "zeva": _pid_status(zeva_pid, "ZeVA"),
    }
    base = inspect_run(base_dir, "baseline", state_loader=state_loader)
    zeva = inspect_run(zeva_dir, "zeva", state_loader=state_loader)
    checks = _pair_checks(base, zeva)
    _raise_failed_checks(checks)
    selected_base, base_candidates = _select_candidate(base)
    selected_zeva, zeva_candidates = _select_candidate(zeva)
    result = {
        "schema": "zeva-robotwin-ztev2-pair-selection-v1",
        "selection_policy": {
            "metric": "held-out validation.flow",
            "independent_per_run": True,
            "tie_break": "earlier optimizer step",
            "excluded_metrics": [
                "validation.total for ZeVA (includes Gaussian NLL)",
                "residual-off paired proxy",
                "closed-loop success",
            ],
        },
        "final_step_required": FINAL_STEP,
        "pid_checks": pid_checks,
        "pair_checks": checks,
        "allowed_lineage_differences": {
            field: {
                "baseline": base.manifest.get(field),
                "zeva": zeva.manifest.get(field),
            }
            for field in ALLOWED_LINEAGE_FIELDS
            if base.manifest.get(field) != zeva.manifest.get(field)
        },
        "runs": {
            "baseline": {
                **base.as_dict(),
                "candidates": base_candidates,
                "selected": _selected_record(selected_base),
            },
            "zeva": {
                **zeva.as_dict(),
                "candidates": zeva_candidates,
                "selected": _selected_record(selected_zeva),
            },
        },
        "selected_immutable_paths": {
            "baseline_model": str(selected_base.model),
            "zeva_model": str(selected_zeva.model),
            "zeva_adapter": str(selected_zeva.adapter),
        },
    }
    return result


def _write_new(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as error:
        raise SelectionError(f"refusing to overwrite existing output: {path}") from error


def _metadata_report(run_dir: str | Path, role: str) -> dict[str, Any]:
    audit = inspect_run(run_dir, role, require_final=False)
    return {
        "schema": "zeva-robotwin-ztev2-run-metadata-audit-v1",
        **audit.as_dict(),
        "final_selection_not_performed": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--zeva-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--base-pid",
        type=int,
        help="completed Base launcher PID; pass 0 only after establishing no process remains",
    )
    parser.add_argument(
        "--zeva-pid",
        type=int,
        help="completed ZeVA launcher PID; pass 0 only after establishing no process remains",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="audit available candidate metadata without requiring final step or writing output",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.metadata_only:
            result = {
                "schema": "zeva-robotwin-ztev2-metadata-smoke-v1",
                "runs": {
                    "baseline": _metadata_report(args.base_dir, "baseline"),
                    "zeva": _metadata_report(args.zeva_dir, "zeva"),
                },
                "selection_not_performed": True,
            }
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.output is None:
            raise SelectionError("normal selection requires --output; output is never implicit")
        result = select_pair(
            args.base_dir,
            args.zeva_dir,
            base_pid=args.base_pid,
            zeva_pid=args.zeva_pid,
        )
        _write_new(args.output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except SelectionError as error:
        print(f"selection failed: {error}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
