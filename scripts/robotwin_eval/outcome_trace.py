"""Fail-closed RoboTwin closed-loop outcome-trace contract.

The formal evaluator normally does not retain policy internals.  This module
defines the optional, versioned payload used when an evaluator is explicitly
asked to collect outcome data.  It is intentionally dependency-light because
the same validation code is imported by the model server and by the render
host's evaluation client.

Trace collection is opt-in.  Nothing in this module is imported by the normal
evaluation path unless tracing has been enabled.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np


# v2 adds mandatory split/protocol provenance and an explicit episode-only
# outcome-label scope; it must not be read as the older v1 contract.
TRACE_SCHEMA = "zeva-robotwin-outcome-trace-v2"
ACTION_DIM = 16
EXECUTED_HORIZON = 15
VLM_FEATURE_DIM = 2048
# A trace contains per-replan policy diagnostics, but the only target label is
# the completed episode outcome.  Keeping this explicit prevents downstream
# code from treating a failed episode as evidence that every decision in it
# was causal for the failure.
OUTCOME_LABEL_SCOPE = "episode_outcome_only"
SUPERVISION_SPLITS = frozenset({"development", "validation"})


class OutcomeTraceError(ValueError):
    """Raised when an outcome trace violates the closed-loop contract."""


def _array(value: Any, *, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise OutcomeTraceError(f"{name} must be a finite numeric array") from error
    if shape is not None and tuple(array.shape) != shape:
        raise OutcomeTraceError(f"{name} must have shape {shape}, got {tuple(array.shape)}")
    if not np.isfinite(array).all():
        raise OutcomeTraceError(f"{name} contains non-finite values")
    return array


def _integer(value: Any, *, name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        raise OutcomeTraceError(f"{name} must be an integer, got boolean")
    try:
        integer = int(value)
    except (TypeError, ValueError) as error:
        raise OutcomeTraceError(f"{name} must be an integer") from error
    if isinstance(value, float) and not value.is_integer():
        raise OutcomeTraceError(f"{name} must be an integer, got {value!r}")
    if minimum is not None and integer < minimum:
        raise OutcomeTraceError(f"{name} must be >= {minimum}, got {integer}")
    return integer


def _finite_scalar(value: Any, *, name: str) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError) as error:
        raise OutcomeTraceError(f"{name} must be a finite scalar") from error
    if not math.isfinite(scalar):
        raise OutcomeTraceError(f"{name} must be finite")
    return scalar


def validate_decision_trace(record: Any) -> dict[str, Any]:
    """Validate one server-produced replan decision and return a copy.

    Candidate actions are the post-processor action domain used by RoboTwin
    execution.  Pairwise distances are computed in the selector's candidate
    domain but are recorded as finite scalar diagnostics.  Keeping both makes
    the outcome data sufficient for auditing without rerunning the policy.
    This record intentionally contains no success label; success belongs only
    to the completed episode envelope.
    """

    if not isinstance(record, dict):
        raise OutcomeTraceError("each replan trace must be an object")
    required = {
        "replan_index",
        "retrieved_task",
        "candidate_h15_actions",
        "pairwise_distances",
        "selected_candidate",
        "pi_vlm_eos_feature",
        "previous_executed_h15",
    }
    missing = sorted(required.difference(record))
    if missing:
        raise OutcomeTraceError(f"replan trace is missing fields: {missing}")

    replan_index = _integer(record["replan_index"], name="replan_index", minimum=0)
    candidates = np.asarray(record["candidate_h15_actions"], dtype=np.float32)
    if candidates.ndim != 3 or tuple(candidates.shape[1:]) != (EXECUTED_HORIZON, ACTION_DIM):
        raise OutcomeTraceError(
            "candidate_h15_actions must have shape [K,15,16], "
            f"got {tuple(candidates.shape)}"
        )
    if candidates.shape[0] < 1:
        raise OutcomeTraceError("candidate_h15_actions must contain at least one candidate")
    if not np.isfinite(candidates).all():
        raise OutcomeTraceError("candidate_h15_actions contains non-finite values")

    pairwise = np.asarray(record["pairwise_distances"], dtype=np.float32)
    expected_pairwise = (candidates.shape[0], candidates.shape[0])
    if tuple(pairwise.shape) != expected_pairwise:
        raise OutcomeTraceError(
            f"pairwise_distances must have shape {expected_pairwise}, got {tuple(pairwise.shape)}"
        )
    if not np.isfinite(pairwise).all():
        raise OutcomeTraceError("pairwise_distances contains non-finite values")

    selected_candidate = _integer(
        record["selected_candidate"], name="selected_candidate", minimum=0
    )
    if selected_candidate >= candidates.shape[0]:
        raise OutcomeTraceError(
            f"selected_candidate={selected_candidate} is outside K={candidates.shape[0]}"
        )

    retrieved_task = record["retrieved_task"]
    if retrieved_task is not None and (
        not isinstance(retrieved_task, str) or not retrieved_task.strip()
    ):
        raise OutcomeTraceError("retrieved_task must be a non-empty string or null")

    eos = record["pi_vlm_eos_feature"]
    if eos is not None:
        eos = _array(eos, name="pi_vlm_eos_feature", shape=(VLM_FEATURE_DIM,))

    previous = record["previous_executed_h15"]
    if previous is not None:
        previous = _array(
            previous,
            name="previous_executed_h15",
            shape=(EXECUTED_HORIZON, ACTION_DIM),
        )

    normalized: dict[str, Any] = {
        "replan_index": replan_index,
        "retrieved_task": retrieved_task,
        "candidate_h15_actions": candidates,
        "pairwise_distances": pairwise,
        "selected_candidate": selected_candidate,
        "pi_vlm_eos_feature": eos,
        "previous_executed_h15": previous,
    }
    # Optional diagnostics are validated when present, but are not required
    # for a baseline provider that has no ZeVA phase score.
    for key in ("phase_confidence", "retrieval_score"):
        if key in record and record[key] is not None:
            normalized[key] = _finite_scalar(record[key], name=key)
    return normalized


def _string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OutcomeTraceError(f"{name} must be a non-empty string")
    return value


def _label(value: Any, *, name: str) -> str:
    """Validate and canonicalize a provenance label without changing text IDs."""

    return _string(value, name=name).strip()


def bind_episode_trace(
    *,
    task: str,
    seed: int,
    episode_index: int,
    instruction: str,
    condition: str,
    policy_name: str,
    split: str,
    protocol: str,
    success: bool,
    steps: int,
    step_limit: int,
    replans: list[dict[str, Any]],
) -> dict[str, Any]:
    """Bind server decisions to one completed RoboTwin episode.

    The client, which is the only component that knows the simulator seed and
    success result, calls this only after the episode has ended.  Every record
    receives the same identity fields, preventing accidental cross-episode
    joins when files are copied between Base and ZeVA condition directories.
    The success field is an episode-level outcome only; it must not be
    interpreted as per-decision failure causality.
    """

    task = _string(task, name="task")
    instruction = _string(instruction, name="instruction")
    condition = _string(condition, name="condition")
    policy_name = _string(policy_name, name="policy_name")
    split = _label(split, name="split")
    protocol = _label(protocol, name="protocol")
    seed = _integer(seed, name="seed", minimum=1000)
    episode_index = _integer(episode_index, name="episode_index", minimum=0)
    steps = _integer(steps, name="steps", minimum=0)
    step_limit = _integer(step_limit, name="step_limit", minimum=1)
    if steps > step_limit:
        raise OutcomeTraceError(f"steps={steps} exceeds step_limit={step_limit}")
    if not isinstance(success, bool):
        raise OutcomeTraceError("success must be boolean")
    if not isinstance(replans, list) or not replans:
        raise OutcomeTraceError("completed episode must contain at least one replan trace")

    bound_replans: list[dict[str, Any]] = []
    for expected_index, raw_record in enumerate(replans):
        record = validate_decision_trace(raw_record)
        if record["replan_index"] != expected_index:
            raise OutcomeTraceError(
                "replan_index values must be contiguous starting at zero; "
                f"expected {expected_index}, got {record['replan_index']}"
            )
        # Convert server arrays to JSON-friendly lists only after validation.
        bound_replans.append(
            {
                "task": task,
                "seed": seed,
                "episode_index": episode_index,
                "split": split,
                "protocol": protocol,
                **_jsonable(record),
            }
        )

    return {
        "schema": TRACE_SCHEMA,
        "task": task,
        "seed": seed,
        "episode_index": episode_index,
        "instruction": instruction,
        "condition": condition,
        "policy_name": policy_name,
        "split": split,
        "protocol": protocol,
        "outcome_label_scope": OUTCOME_LABEL_SCOPE,
        "success": success,
        "steps": steps,
        "step_limit": step_limit,
        "replans": bound_replans,
    }


def validate_episode_trace(payload: Any) -> dict[str, Any]:
    """Validate a complete atomically-written episode trace."""

    if not isinstance(payload, dict):
        raise OutcomeTraceError("episode trace must be a JSON object")
    if payload.get("schema") != TRACE_SCHEMA:
        raise OutcomeTraceError(
            f"unsupported episode trace schema: {payload.get('schema')!r}"
        )
    required = {
        "task",
        "seed",
        "episode_index",
        "instruction",
        "condition",
        "policy_name",
        "split",
        "protocol",
        "outcome_label_scope",
        "success",
        "steps",
        "step_limit",
        "replans",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise OutcomeTraceError(f"episode trace is missing fields: {missing}")
    # A trace copied from another episode must not be made valid merely by
    # rebinding its outer metadata.  If nested identity fields are present,
    # they must agree exactly with the episode envelope.
    for index, record in enumerate(payload["replans"] if isinstance(payload["replans"], list) else []):
        if not isinstance(record, dict):
            continue
        for key in ("task", "seed", "episode_index", "split", "protocol"):
            if key in record and record[key] != payload[key]:
                raise OutcomeTraceError(
                    f"replans[{index}].{key} does not match the episode envelope"
                )
    if payload["outcome_label_scope"] != OUTCOME_LABEL_SCOPE:
        raise OutcomeTraceError(
            "outcome_label_scope must be "
            f"{OUTCOME_LABEL_SCOPE!r}; per-decision success labels are not supported"
        )
    expected = bind_episode_trace(
        task=payload["task"],
        seed=payload["seed"],
        episode_index=payload["episode_index"],
        instruction=payload["instruction"],
        condition=payload["condition"],
        policy_name=payload["policy_name"],
        split=payload["split"],
        protocol=payload["protocol"],
        success=payload["success"],
        steps=payload["steps"],
        step_limit=payload["step_limit"],
        replans=payload["replans"],
    )
    # The rebinding above also ensures every nested identity is present and
    # equal.  Keep the original scalar spelling only where it is equivalent.
    return expected


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


def trace_filename(task: str, episode_index: int, seed: int) -> str:
    """Return a path-safe deterministic filename for one episode."""

    safe_task = _SAFE_NAME.sub("_", str(task)).strip("._") or "task"
    return f"{safe_task}__episode{int(episode_index):04d}__seed{int(seed)}.json"


def atomic_write_episode_trace(path: str | Path, payload: dict[str, Any]) -> Path:
    """Validate and atomically write one completed trace."""

    validated = validate_episode_trace(payload)
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(validated, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_episode_trace(path: str | Path) -> dict[str, Any]:
    """Load and validate one trace file."""

    source = Path(path).expanduser()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise OutcomeTraceError(f"cannot read trace {source}: {error}") from error
    try:
        return validate_episode_trace(payload)
    except OutcomeTraceError as error:
        raise OutcomeTraceError(f"{source}: {error}") from error
