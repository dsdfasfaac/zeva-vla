"""Comparable frozen phase probes for the old ZTE and v2 encoders.

This is deliberately a diagnostic, policy-independent probe.  It evaluates
the *exported* 128-dimensional phase token with one fixed linear/ridge
implementation for the old live-query cache and both v2 checkpoints.  The
native action/progress heads are never used as evidence.  The main target is
the next executed H15 chunk (240 normalized values); initial phase -> action0
is reported separately because it is a different time boundary.

The default split is deterministic (seed 1000): four train episodes and two
validation episodes per task.  Selection, standardisation, ridge alpha and
all source hashes are recorded in the JSON report.  Validation episodes never
enter fitting or standardisation.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch


SCHEMA = "zeva-robotwin-old-v2-next-h15-probe-v1"
DEFAULT_HANDOFF_ROOT = (
    "/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1"
)
DEFAULT_DATASET_ROOT = "/data1/huangbingjia/robotwin-lerobot-sidney-eef16-v1/data"
DEFAULT_GOALS = "/data1/dingxin/zeva-runs/robotwin-v5-h15-tasklang/pi05_task_embeddings.pt"
DEFAULT_OLD_CHECKPOINT = (
    "/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/"
    "stage1-artifacts-v1/stage1-zte/zte_best.pth"
)
DEFAULT_OLD_LIVE_QUERIES = (
    "/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/"
    "stage1-artifacts-v1/stage1-zte/live_queries_h15.pt"
)
DEFAULT_V2_PRE = (
    "/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/"
    "stage1-zte-v2-pilot-taskpaired-20260911e/zte_v2_step_000256.pth"
)
DEFAULT_V2_PHASE = (
    "/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/"
    "stage1-zte-v2-pilot-phaseaction-20260911f/zte_v2_step_000256.pth"
)


@dataclasses.dataclass
class Args:
    handoff_root: str = DEFAULT_HANDOFF_ROOT
    dataset_root: str = DEFAULT_DATASET_ROOT
    goal_embeddings: str = DEFAULT_GOALS
    old_checkpoint: str = DEFAULT_OLD_CHECKPOINT
    old_live_queries: str = DEFAULT_OLD_LIVE_QUERIES
    v2_pre_checkpoint: str = DEFAULT_V2_PRE
    v2_phase_checkpoint: str = DEFAULT_V2_PHASE
    output: str = (
        "/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/"
        "stage1-zte-old-v2-probe-20260911/old_v2_next_h15.json"
    )
    train_episodes_per_task: int = 4
    validation_episodes_per_task: int = 2
    transition_stride: int = 15
    effect_steps: int = 15
    executed_action_steps: int = 15
    ridge_alpha: float = 1.0
    bootstrap_replicates: int = 1000
    seed: int = 1000
    bootstrap_seed: int = 11000
    device: str = "cuda"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def episode_identity(record: dict[str, Any]) -> str:
    key = record["key"]
    return f"{key[0]}:{key[1]}:{int(record['episode_index'])}"


def expected_decision_frames(length: int, effect_steps: int = 15, stride: int = 15) -> tuple[int, ...]:
    if length <= effect_steps:
        return ()
    before = list(range(0, length - effect_steps, stride))
    return tuple([0, *[frame + effect_steps for frame in before]])


def validate_cache_record(
    cached: dict[str, Any],
    *,
    record: dict[str, Any],
    record_index: int,
    task_id: int,
    effect_steps: int = 15,
    stride: int = 15,
) -> None:
    """Reject a live-query row unless identity and frame indexing are exact."""

    if int(cached.get("record_index", -1)) != int(record_index):
        raise ValueError(
            f"Live-query record index mismatch: cache={cached.get('record_index')}, "
            f"dataset={record_index}."
        )
    if int(cached.get("task_id", -1)) != int(task_id):
        raise ValueError(f"Live-query task id mismatch for record {record_index}.")
    frames = tuple(int(item) for item in torch.as_tensor(cached["decision_frames"]).tolist())
    expected = expected_decision_frames(int(record["length"]), effect_steps, stride)
    if frames != expected:
        raise ValueError(f"Live-query frame mismatch for record {record_index}: {frames} != {expected}.")
    phase = torch.as_tensor(cached["phase_queries"])
    if phase.ndim != 2 or phase.shape[0] != len(expected) or phase.shape[1] != 128:
        raise ValueError(
            f"Live-query phase shape for record {record_index} is {tuple(phase.shape)}, "
            f"expected ({len(expected)}, 128)."
        )


def validate_cache_payload(
    payload: dict[str, Any],
    *,
    checkpoint_sha256: str,
    statistics_sha256: str,
    records_by_split: dict[str, Sequence[dict[str, Any]]] | None = None,
    task_names: Sequence[str] | None = None,
    effect_steps: int = 15,
    stride: int = 15,
) -> None:
    """Validate cache provenance and, when supplied, adapter record alignment."""

    if payload.get("schema") != "zeva-robotwin-live-queries-h15-v1":
        raise ValueError("Unsupported old live-query schema.")
    if int(payload.get("transition_horizon", -1)) != effect_steps:
        raise ValueError("Old live-query horizon does not match H15.")
    if payload.get("zte_checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("Old live-query checkpoint SHA does not match the passed checkpoint.")
    if payload.get("statistics_sha256") != statistics_sha256:
        raise ValueError("Old live-query statistics SHA does not match the handoff statistics.")
    cache_tasks = tuple(payload.get("task_names", ()))
    if task_names is not None and cache_tasks != tuple(task_names):
        raise ValueError("Old live-query task ordering differs from the adapter.")
    if records_by_split is None:
        return
    for split, records in records_by_split.items():
        cached_records = payload.get("splits", {}).get(split)
        if not isinstance(cached_records, list) or len(cached_records) != len(records):
            raise ValueError(f"Old live-query {split} record count differs from adapter.")
        names = {name: index for index, name in enumerate(cache_tasks)}
        for index, record in enumerate(records):
            task = record["key"][1]
            if task not in names:
                raise ValueError(f"Adapter task {task!r} is absent from old cache.")
            validate_cache_record(
                cached_records[index],
                record=record,
                record_index=index,
                task_id=names[task],
                effect_steps=effect_steps,
                stride=stride,
            )


def select_episode_records(
    records: Sequence[dict[str, Any]],
    *,
    task_names: Sequence[str],
    count_per_task: int,
    seed: int,
    split_offset: int,
    effect_steps: int = 15,
    stride: int = 15,
) -> list[dict[str, Any]]:
    """Select sorted, deterministic record indices independently per task."""

    if count_per_task < 1:
        raise ValueError("count_per_task must be positive.")
    task_to_id = {name: index for index, name in enumerate(task_names)}
    selected: list[dict[str, Any]] = []
    for task_id, task_name in enumerate(task_names):
        candidates = [
            index
            for index, record in enumerate(records)
            if record["key"][1] == task_name
            and expected_decision_frames(int(record["length"]), effect_steps, stride)
        ]
        if len(candidates) < count_per_task:
            raise ValueError(
                f"Task {task_name!r} has {len(candidates)} valid episodes, "
                f"cannot select {count_per_task}."
            )
        generator = np.random.default_rng(seed + split_offset + 104729 * task_id)
        chosen = sorted(int(item) for item in generator.choice(candidates, count_per_task, replace=False))
        for index in chosen:
            record = records[index]
            selected.append(
                {
                    "record_index": index,
                    "task_id": task_to_id[task_name],
                    "task_name": task_name,
                    "identity": episode_identity(record),
                    "length": int(record["length"]),
                    "decision_frames": list(expected_decision_frames(int(record["length"]), effect_steps, stride)),
                }
            )
    return selected


def assert_disjoint_selection(train: Sequence[dict[str, Any]], validation: Sequence[dict[str, Any]]) -> None:
    train_ids = {row["identity"] for row in train}
    val_ids = {row["identity"] for row in validation}
    overlap = sorted(train_ids & val_ids)
    if overlap:
        raise ValueError(f"Train/validation episode overlap: {overlap[:5]}.")


def phase_rows(
    phase_queries: torch.Tensor | np.ndarray,
    actions: torch.Tensor | np.ndarray,
    decision_frames: Sequence[int],
    episode_id: str,
    task_id: int,
    length: int,
) -> dict[str, np.ndarray]:
    """Create next-action rows and a separate initial-phase row block."""

    phase = torch.as_tensor(phase_queries, dtype=torch.float32)
    action = torch.as_tensor(actions, dtype=torch.float32)
    frames = torch.as_tensor(decision_frames, dtype=torch.long)
    if phase.ndim != 2 or phase.shape[1] != 128:
        raise ValueError(f"Expected phase [T+1,128], got {tuple(phase.shape)}.")
    if action.ndim != 3 or action.shape[0] != phase.shape[0] - 1:
        raise ValueError(f"Expected actions [T,15,16], got {tuple(action.shape)} for phase {tuple(phase.shape)}.")
    if len(decision_frames) != phase.shape[0]:
        raise ValueError("decision_frames, phase and actions have inconsistent lengths.")
    target_next = action[1:].reshape(max(0, len(action) - 1), -1)
    after_phase = phase[1:-1]
    after_frames = frames[1:-1]
    denominator = max(1, int(length) - 1)
    result = {
        "phase": after_phase.numpy(),
        "target": target_next.numpy(),
        "progress": (after_frames.float() / denominator).numpy()[:, None],
        "episode": np.asarray([episode_id] * len(after_phase), dtype=object),
        "task": np.asarray([task_id] * len(after_phase), dtype=np.int64),
        "initial_phase": phase[:1].numpy(),
        "initial_target": action[:1].reshape(1, -1).numpy(),
        "initial_episode": np.asarray([episode_id], dtype=object),
        "initial_task": np.asarray([task_id], dtype=np.int64),
    }
    return result


def _as_2d(value: np.ndarray | torch.Tensor) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"Expected a matrix, got shape {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError("Probe input contains NaN or infinite values.")
    return array


@dataclasses.dataclass
class FixedRidge:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    weight: np.ndarray
    intercept: np.ndarray
    alpha: float

    def predict(self, features: np.ndarray | torch.Tensor) -> np.ndarray:
        matrix = _as_2d(features)
        return (matrix - self.feature_mean) / self.feature_scale @ self.weight + self.intercept


def fit_fixed_ridge(features: np.ndarray, targets: np.ndarray, alpha: float = 1.0) -> FixedRidge:
    """Fit a fixed-alpha multi-output ridge with train-only X standardisation."""

    if alpha < 0 or not math.isfinite(alpha):
        raise ValueError("alpha must be finite and non-negative.")
    x = _as_2d(features)
    y = _as_2d(targets)
    if len(x) != len(y) or len(x) == 0:
        raise ValueError("Ridge features/targets must have the same non-zero row count.")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    xz = (x - mean) / scale
    y_mean = y.mean(axis=0)
    gram = xz.T @ xz
    rhs = xz.T @ (y - y_mean)
    weight = np.linalg.solve(gram + float(alpha) * np.eye(xz.shape[1]), rhs)
    return FixedRidge(mean, scale, weight, y_mean, float(alpha))


def task_mean_predictions(
    train_targets: np.ndarray,
    train_tasks: np.ndarray,
    eval_tasks: np.ndarray,
) -> np.ndarray:
    y = _as_2d(train_targets)
    train_tasks = np.asarray(train_tasks, dtype=np.int64)
    eval_tasks = np.asarray(eval_tasks, dtype=np.int64)
    means = {task: y[train_tasks == task].mean(axis=0) for task in np.unique(train_tasks)}
    missing = sorted(set(eval_tasks.tolist()) - set(means))
    if missing:
        raise ValueError(f"No train task mean for validation tasks {missing}.")
    return np.stack([means[int(task)] for task in eval_tasks])


def _episode_error_map(
    errors: np.ndarray,
    episodes: Sequence[str],
    tasks: Sequence[int],
) -> tuple[dict[str, float], dict[str, float]]:
    values: dict[str, list[float]] = {}
    task_values: dict[int, list[float]] = {}
    for error, episode, task in zip(errors, episodes, tasks, strict=True):
        values.setdefault(str(episode), []).append(float(error))
        task_values.setdefault(int(task), []).append(float(error))
    episode_mse = {key: float(np.mean(value)) for key, value in values.items()}
    task_mse = {str(key): float(np.mean(value)) for key, value in sorted(task_values.items())}
    return episode_mse, task_mse


def episode_block_bootstrap_ci(
    errors: np.ndarray,
    episodes: Sequence[str],
    *,
    replicates: int = 1000,
    seed: int = 11000,
) -> dict[str, Any]:
    """Bootstrap whole episode means, never individual transition rows."""

    if replicates < 1:
        raise ValueError("bootstrap replicates must be positive.")
    values: dict[str, list[float]] = {}
    for error, episode in zip(np.asarray(errors, dtype=np.float64), episodes, strict=True):
        values.setdefault(str(episode), []).append(float(error))
    if not values:
        raise ValueError("Cannot bootstrap an empty error set.")
    episode_means = np.asarray([np.mean(item) for item in values.values()], dtype=np.float64)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(episode_means), size=(replicates, len(episode_means)))
    samples = episode_means[indices].mean(axis=1)
    return {
        "episodes": int(len(episode_means)),
        "replicates": int(replicates),
        "seed": int(seed),
        "mean": float(episode_means.mean()),
        "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
    }


def _spearman(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred).reshape(-1)
    target = np.asarray(target).reshape(-1)
    if len(pred) < 2 or np.all(pred == pred[0]) or np.all(target == target[0]):
        return float("nan")
    pred_rank = np.argsort(np.argsort(pred, kind="stable"), kind="stable").astype(np.float64)
    target_rank = np.argsort(np.argsort(target, kind="stable"), kind="stable").astype(np.float64)
    return float(np.corrcoef(pred_rank, target_rank)[0, 1])


def summarize_regression(
    predictions: np.ndarray,
    targets: np.ndarray,
    episodes: Sequence[str],
    tasks: Sequence[int],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    task_names: Sequence[str],
    progress: bool = False,
) -> dict[str, Any]:
    pred = _as_2d(predictions)
    target = _as_2d(targets)
    if pred.shape != target.shape:
        raise ValueError(f"Prediction/target shape mismatch: {pred.shape} != {target.shape}.")
    row_mse = np.mean((pred - target) ** 2, axis=1)
    episode_mse, task_mse = _episode_error_map(row_mse, episodes, tasks)
    task_named = {task_names[int(key)]: value for key, value in task_mse.items()}
    result: dict[str, Any] = {
        "row_count": int(len(row_mse)),
        "mse": float(row_mse.mean()) if len(row_mse) else float("nan"),
        "episode_mse": episode_mse,
        "task_mse": task_named,
        "episode_block_bootstrap": episode_block_bootstrap_ci(
            row_mse,
            episodes,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        ),
    }
    if progress:
        result["mae"] = float(np.mean(np.abs(pred - target)))
        result["spearman"] = _spearman(pred, target)
    return result


def _concat_rows(rows: Sequence[dict[str, np.ndarray]], key: str, width: int | None = None) -> np.ndarray:
    values = [row[key] for row in rows]
    if not values:
        return np.empty((0, width or 0), dtype=np.float64)
    return np.concatenate(values, axis=0)


def fit_and_evaluate(
    train: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    *,
    alpha: float,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    task_names: Sequence[str],
) -> dict[str, Any]:
    """Fit all three requested probes, always using train rows only."""

    phase_model = fit_fixed_ridge(train["phase"], train["target"], alpha)
    progress_model = fit_fixed_ridge(train["phase"], train["progress"], alpha)
    initial_model = fit_fixed_ridge(train["initial_phase"], train["initial_target"], alpha)

    phase_pred = phase_model.predict(validation["phase"])
    progress_pred = progress_model.predict(validation["phase"])
    initial_pred = initial_model.predict(validation["initial_phase"])
    baseline_pred = task_mean_predictions(train["target"], train["task"], validation["task"])
    result = {
        "alpha": float(alpha),
        "standardization": "feature mean/std fit on train rows only; constant features use scale=1",
        "next_h15": summarize_regression(
            phase_pred,
            validation["target"],
            validation["episode"],
            validation["task"],
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed,
            task_names=task_names,
        ),
        "task_mean_baseline_next_h15": summarize_regression(
            baseline_pred,
            validation["target"],
            validation["episode"],
            validation["task"],
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed + 1,
            task_names=task_names,
        ),
        "progress": summarize_regression(
            progress_pred,
            validation["progress"],
            validation["episode"],
            validation["task"],
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed + 2,
            task_names=task_names,
            progress=True,
        ),
        "initial_phase_to_action0": summarize_regression(
            initial_pred,
            validation["initial_target"],
            validation["initial_episode"],
            validation["initial_task"],
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed + 3,
            task_names=task_names,
        ),
    }
    return result


def _load_records(adapter_manifest: Path, subset: str):
    from scripts.train_robotwin_zte import TorchCodecRoboTwinDataset

    source = TorchCodecRoboTwinDataset(adapter_manifest, subset)
    return source


def _build_selection(source_by_split: dict[str, Any], args: Args) -> dict[str, list[dict[str, Any]]]:
    task_names = tuple(sorted({record["key"][1] for source in source_by_split.values() for record in source.dataset._records}))
    selections = {
        "train": select_episode_records(
            source_by_split["train"].dataset._records,
            task_names=task_names,
            count_per_task=args.train_episodes_per_task,
            seed=args.seed,
            split_offset=0,
            effect_steps=args.effect_steps,
            stride=args.transition_stride,
        ),
        "validation": select_episode_records(
            source_by_split["validation"].dataset._records,
            task_names=task_names,
            count_per_task=args.validation_episodes_per_task,
            seed=args.seed,
            split_offset=1_000_003,
            effect_steps=args.effect_steps,
            stride=args.transition_stride,
        ),
    }
    assert_disjoint_selection(selections["train"], selections["validation"])
    return {"task_names": list(task_names), **selections}


def _episode_inputs(source, record_index: int, *, args: Args, normalizer):
    record = source.dataset._records[record_index]
    length = int(record["length"])
    before_frames = list(range(0, length - args.effect_steps, args.transition_stride))
    after_frames = [frame + args.effect_steps for frame in before_frames]
    images = source.read_images(record, before_frames + after_frames)
    sequence_length = len(before_frames)
    from openpi.zeva.robotwin_policy import robotwin_multiview_image

    before = robotwin_multiview_image({key: value[:sequence_length] for key, value in images.items()})
    after = robotwin_multiview_image({key: value[sequence_length:] for key, value in images.items()})
    start = int(source.dataset._cumulative[record_index])
    actions = torch.stack(
        [source.dataset[start + frame]["action"][: args.executed_action_steps] for frame in before_frames]
    )
    return before, after, actions, tuple([0, *after_frames]), record


def _load_v2_model(checkpoint_path: Path, device: torch.device):
    from openpi.zeva.transition_encoder_v2 import CausalTransitionEncoderV2, TransitionEncoderV2Config

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw_config = dict(payload["zte_config"])
    config_values = {key: value for key, value in raw_config.items() if key in TransitionEncoderV2Config.__dataclass_fields__}
    config = TransitionEncoderV2Config(**config_values)
    model = CausalTransitionEncoderV2(config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval().requires_grad_(False).to(device)
    return model, payload, config


def _empty_rows() -> dict[str, list[np.ndarray]]:
    return {key: [] for key in ("phase", "target", "progress", "episode", "task", "initial_phase", "initial_target", "initial_episode", "initial_task")}


def _append_episode(rows: dict[str, list[np.ndarray]], values: dict[str, np.ndarray]) -> None:
    for key in rows:
        rows[key].append(values[key])


def _finalize_rows(rows: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for key, values in rows.items():
        if not values:
            width = 128 if key in ("phase", "initial_phase") else 240 if key in ("target", "initial_target") else 1
            dtype = object if key.endswith("episode") else np.float64
            result[key] = np.empty((0,) if dtype is object else (0, width), dtype=dtype)
        else:
            result[key] = np.concatenate(values, axis=0)
    return result


def _old_phase_for_record(cache_payload: dict[str, Any], split: str, selected: dict[str, Any]) -> torch.Tensor:
    cached = cache_payload["splits"][split][int(selected["record_index"])]
    return torch.as_tensor(cached["phase_queries"], dtype=torch.float32)


def _collect_source_rows(
    *,
    source_by_split: dict[str, Any],
    selection: dict[str, list[dict[str, Any]]],
    cache_payload: dict[str, Any],
    model=None,
    split: str,
    args: Args,
    normalizer,
    device: torch.device,
) -> dict[str, np.ndarray]:
    rows = _empty_rows()
    source = source_by_split[split]
    for selected in selection[split]:
        before, after, actions, frames, record = _episode_inputs(source, selected["record_index"], args=args, normalizer=normalizer)
        phase = _old_phase_for_record(cache_payload, split, selected) if model is None else None
        if model is not None:
            # The exact PI0.5 goal table is attached by the caller.  It is
            # passed through ``selected['_goal']`` to keep this reader pure.
            goal = selected["_goal"]
            with torch.inference_mode():
                outputs = model(
                    before[None].to(device),
                    normalizer.normalize(actions[None].to(device)),
                    after[None].to(device),
                    goal[None].to(device),
                )
            phase = torch.cat([outputs.initial_phase_token[:, None], outputs.phase_token], dim=1)[0].cpu()
        normalized_actions = normalizer.normalize(actions)
        values = phase_rows(phase, normalized_actions, frames, selected["identity"], selected["task_id"], int(record["length"]))
        _append_episode(rows, values)
    return _finalize_rows(rows)


def _attach_goals(source_by_split: dict[str, Any], selection: dict[str, list[dict[str, Any]]], goal_path: Path) -> None:
    payload = torch.load(goal_path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "zeva-robotwin-pi05-goal-embeddings-v2" or payload.get("embedding_dim") != 2048:
        raise ValueError("Unsupported PI0.5 goal table.")
    for split, source in source_by_split.items():
        records = source.dataset._records
        expected = tuple(episode_identity(record) for record in records)
        table = payload["splits"][split]
        if tuple(table["record_ids"]) != expected:
            raise ValueError(f"Goal table order differs from {split} adapter records.")
        embeddings = torch.as_tensor(table["embeddings"], dtype=torch.float32)
        for selected in selection[split]:
            selected["_goal"] = embeddings[int(selected["record_index"])]


def _merge_sources(train_rows, val_rows, *, alpha, bootstrap_replicates, bootstrap_seed, task_names):
    return fit_and_evaluate(
        train_rows,
        val_rows,
        alpha=alpha,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
        task_names=task_names,
    )


def run(args: Args) -> dict[str, Any]:
    if args.train_episodes_per_task != 4 or args.validation_episodes_per_task != 2:
        raise ValueError("This comparable probe requires exactly 4 train and 2 validation episodes per task.")
    if args.transition_stride != 15 or args.effect_steps != 15 or args.executed_action_steps != 15:
        raise ValueError("This probe is fixed to the H15 replanning contract.")
    if not math.isclose(args.ridge_alpha, 1.0):
        raise ValueError("This report requires fixed ridge alpha=1.0.")
    dataset_root = Path(args.dataset_root)
    adapter_manifest = dataset_root / "adapter.json"
    source_by_split = {split: _load_records(adapter_manifest, split) for split in ("train", "validation")}
    selection = _build_selection(source_by_split, args)
    task_names = selection.pop("task_names")
    _attach_goals(source_by_split, selection, Path(args.goal_embeddings))

    old_checkpoint = Path(args.old_checkpoint)
    old_cache_path = Path(args.old_live_queries)
    old_sha = sha256_file(old_checkpoint)
    handoff_stats = Path(args.handoff_root) / "reference" / "mean-std-eef16-h50-stage1grip-train95-v2.json"
    stats_sha = sha256_file(handoff_stats)
    old_cache = torch.load(old_cache_path, map_location="cpu", weights_only=False)
    records_by_split = {split: source.dataset._records for split, source in source_by_split.items()}
    validate_cache_payload(
        old_cache,
        checkpoint_sha256=old_sha,
        statistics_sha256=stats_sha,
        records_by_split=records_by_split,
        task_names=task_names,
        effect_steps=args.effect_steps,
        stride=args.transition_stride,
    )

    from openpi.zeva.robotwin_contract import MeanStdActionNormalizer

    normalizer = MeanStdActionNormalizer.from_stats_file(handoff_stats)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    common_meta = {
        "train_episode_count": len(selection["train"]),
        "validation_episode_count": len(selection["validation"]),
        "train_episodes_per_task": args.train_episodes_per_task,
        "validation_episodes_per_task": args.validation_episodes_per_task,
        "seed": args.seed,
        "ridge_alpha": args.ridge_alpha,
        "task_names": task_names,
        "selection": selection,
        "source_hashes": {
            "adapter_manifest": sha256_file(adapter_manifest),
            "goal_embeddings": sha256_file(args.goal_embeddings),
            "statistics": stats_sha,
            "old_checkpoint": old_sha,
            "old_live_queries": sha256_file(old_cache_path),
        },
    }
    groups: dict[str, Any] = {}
    old_train = _collect_source_rows(
        source_by_split=source_by_split,
        selection=selection,
        cache_payload=old_cache,
        split="train",
        args=args,
        normalizer=normalizer,
        device=device,
    )
    old_val = _collect_source_rows(
        source_by_split=source_by_split,
        selection=selection,
        cache_payload=old_cache,
        split="validation",
        args=args,
        normalizer=normalizer,
        device=device,
    )
    groups["old_zte"] = {
        "feature_source": "old live_queries phase_queries (initial + exported post-transition phases)",
        "checkpoint_sha256": old_sha,
        "live_queries_sha256": common_meta["source_hashes"]["old_live_queries"],
        "metrics": _merge_sources(
            old_train,
            old_val,
            alpha=args.ridge_alpha,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
            task_names=task_names,
        ),
    }

    checkpoint_specs = (
        ("v2_e_pre", Path(args.v2_pre_checkpoint)),
        ("v2_f_phase", Path(args.v2_phase_checkpoint)),
    )
    for name, checkpoint_path in checkpoint_specs:
        model, payload, config = _load_v2_model(checkpoint_path, device)
        train_rows = _collect_source_rows(
            source_by_split=source_by_split,
            selection=selection,
            cache_payload=old_cache,
            model=model,
            split="train",
            args=args,
            normalizer=normalizer,
            device=device,
        )
        val_rows = _collect_source_rows(
            source_by_split=source_by_split,
            selection=selection,
            cache_payload=old_cache,
            model=model,
            split="validation",
            args=args,
            normalizer=normalizer,
            device=device,
        )
        groups[name] = {
            "feature_source": "actual outputs.phase_token and outputs.initial_phase_token; native heads ignored",
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "step": payload.get("step"),
            "action_prediction_context": getattr(config, "action_prediction_context", "pre"),
            "metrics": _merge_sources(
                train_rows,
                val_rows,
                alpha=args.ridge_alpha,
                bootstrap_replicates=args.bootstrap_replicates,
                bootstrap_seed=args.bootstrap_seed,
                task_names=task_names,
            ),
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report = {
        "schema": SCHEMA,
        "diagnostic_only": True,
        "gate_status": "not_a_full_stage1_gate",
        "missing_gates": [
            "old-ZTE-comparable frozen probe is only one representation diagnostic",
            "no PI rollout robustness or closed-loop control evidence",
            "no old/new task-and-phase intervention equivalence proof",
            "no full Stage1 objective/gate audit",
        ],
        "contract": {
            "target": "next executed H15 action at the next replanning boundary",
            "phase_alignment": "phase_queries[1:-1] at after frame t -> actions[t+1]",
            "initial_alignment": "phase_queries[0] -> actions[0], reported separately",
            "action_shape": [15, 16],
            "flatten_order": "row-major H15 then EEF16",
            "phase_dim": 128,
            "progress": "independent ridge on exported phase token; native progress heads unused",
            "standardization": "train-only feature mean/std",
            "validation_used_for_fit": False,
            "ridge_alpha": args.ridge_alpha,
        },
        "common": common_meta,
        "groups": groups,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    import tyro

    args = tyro.cli(Args)
    report = run(args)
    print(json.dumps({"output": args.output, "groups": list(report["groups"]), "gate_status": report["gate_status"]}))


if __name__ == "__main__":
    main()
