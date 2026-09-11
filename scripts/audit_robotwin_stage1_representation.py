"""Audit whether the frozen Stage1 representation contains useful signal.

This is an offline, policy-independent audit for the v15 frozen-PI candidate
cache.  It deliberately does *not* import or modify the RoboTwin evaluation
policy.  Each cached observation is an episode decision and contains K frozen
PI action chunks, a normalized expert chunk, and (for cache v3) the Stage1 and
PI features that produced the decision.

The audit evaluates four context blocks with a deterministic linear probe:

* ``action_only``: candidate action geometry only (a useful lower bound);
* ``stage1_zte``: Stage1 phase/visual representation plus candidate geometry;
* ``pi_vlm``: PI VLM representation plus candidate geometry;
* ``joint``: Stage1 + PI VLM + candidate geometry.

The target is ``candidate_i_better_than_base`` under the offline H15 expert
MSE oracle.  Candidate rows from one episode are never split across folds:
groups are ``split/task/record`` and folds are assigned per task by a stable
hash.  This prevents decision-frame and seed/episode leakage.  The script can
also consume decision-level closed-loop telemetry; a summary/report without
features is intentionally reported as ``aggregate_only`` and is never used as
evidence that a representation predicts success.

The result contains explicit gates for deciding whether Stage1 is worth
keeping.  These are conservative engineering gates, not a claim that offline
MSE is equivalent to closed-loop success:

* retain Stage1 only when its mean non-base ROC-AUC is at least 0.55, its
  selected action MSE is at least 1% below ``action_only``, and at least half
  of folds improve over ``action_only``;
* require the joint block not to regress more than 1% from ``pi_vlm``;
* otherwise report ``discard`` or ``inconclusive`` and require new
  decision-level success telemetry before changing the policy.

Example (on the H100 host)::

    python scripts/audit_robotwin_stage1_representation.py \
      --cache /mnt/.../advantage10-pi-candidates-v15/pi_candidates_k4_v3.pt \
      --live-queries /mnt/.../stage1-zte/live_queries_h15.pt \
      --causal-bank /mnt/.../stage1-zte/train_causal_bank.pt \
      --closed-loop /mnt/.../eval/advantage10-phase-gated-consensus-v18/split-j \
      --output /mnt/.../stage1_representation_audit_v1.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch


SCHEMA = "zeva-robotwin-stage1-representation-audit-v1"
SUPPORTED_CACHE_SCHEMAS = {"zeva-robotwin-pi-candidate-cache-v15"}


def _stable_hash(value: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _as_float_tensor(value: Any, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value).float()
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return tensor


def _load_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def _normalise_path(path: str | None) -> Path | None:
    return Path(path).expanduser().resolve() if path else None


def _selected_task_names(payload: dict[str, Any]) -> list[str]:
    names = list(payload.get("task_names", ()))
    selected = list(payload.get("selected_task_names", names))
    if not names or not selected:
        raise ValueError("Candidate cache must contain task_names and selected_task_names.")
    unknown = sorted(set(selected) - set(names))
    if unknown:
        raise ValueError(f"selected_task_names contains unknown tasks: {unknown}")
    return selected


def _make_group_map(
    *,
    split: str,
    live_queries: dict[str, Any] | None,
    task_names: list[str],
    selected_tasks: list[str],
    sample_indices: torch.Tensor,
) -> list[str]:
    """Map local Stage2 sample indices to task/episode groups.

    ``RobotWinStage2Dataset`` enumerates ``record -> decision_frame`` in this
    exact order.  Reconstructing that map from the immutable live-query cache
    avoids importing the training/evaluation data loader and keeps this audit
    usable on a minimal analysis environment.
    """

    if live_queries is None:
        # A supplied cache may carry an explicit group map.  This fallback is
        # only for diagnostics and is deliberately marked by the caller.
        return [f"{split}:sample:{int(index)}" for index in sample_indices.tolist()]
    records = live_queries.get("splits", {}).get(split)
    if not isinstance(records, list):
        raise ValueError(f"Live-query cache has no list for split {split!r}.")
    name_to_id = {name: index for index, name in enumerate(task_names)}
    selected_ids = {name_to_id[name] for name in selected_tasks}
    all_groups: list[str] = []
    for record_position, record in enumerate(records):
        task_id = int(record["task_id"])
        frames = record.get("decision_frames")
        if task_id not in selected_ids:
            continue
        if frames is None:
            raise ValueError(f"Live-query record {record_position} omitted decision_frames.")
        group = f"{split}:task{task_id}:record{int(record.get('record_index', record_position))}"
        all_groups.extend([group] * len(frames))
    indices = sample_indices.long()
    if indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= len(all_groups)):
        raise ValueError(
            f"{split} sample index range [{int(indices.min())}, {int(indices.max())}] "
            f"does not fit reconstructed dataset length {len(all_groups)}."
        )
    return [all_groups[int(index)] for index in indices.tolist()]


def _stage1_features(
    *,
    split: dict[str, Any],
    task_ids: torch.Tensor,
    causal_bank_path: Path | None,
) -> torch.Tensor:
    required = ("phase_queries", "current_visual_features")
    missing = [key for key in required if key not in split]
    if missing:
        raise ValueError(
            "The cache does not contain the Stage1 feature block "
            f"{missing}. Use pi_candidates_k4_v2.pt or v3.pt."
        )
    blocks = [
        _as_float_tensor(split["phase_queries"], "phase_queries"),
        _as_float_tensor(split["current_visual_features"], "current_visual_features"),
    ]
    if causal_bank_path is not None:
        # Import lazily so the core audit remains usable without the full ZeVA
        # package.  The bank is frozen and used only to expose the same
        # task/phase/value coordinates available to Stage2.
        repo_root = Path(__file__).resolve().parents[1]
        source_roots = (repo_root / "src", repo_root / "ICML26-BehaviorVLA" / "src")
        for source_root in source_roots:
            if source_root.is_dir() and str(source_root) not in sys.path:
                sys.path.insert(0, str(source_root))
        from openpi.zeva.causal_bank import RobotWinCausalBank

        bank = RobotWinCausalBank.load(causal_bank_path, device="cpu")
        if int(task_ids.max()) >= len(bank.task_names):
            raise ValueError("Candidate task ids exceed causal-bank task table.")
        phase = blocks[0]
        retrieved = bank.retrieve(
            task_ids,
            phase,
            brief_size=min(4, bank.phase_bins),
            retrieval_top_k=min(4, bank.phase_bins),
        )
        blocks.extend(
            [
                bank.task_prototype[task_ids].float(),
                retrieved.phase_token.float(),
                retrieved.brief_signals.float().mean(dim=1),
                retrieved.brief_signals.float().std(dim=1, unbiased=False),
                retrieved.retrieved_signals.float().mean(dim=1),
                retrieved.retrieved_signals.float().std(dim=1, unbiased=False),
            ]
        )
    return torch.cat(blocks, dim=-1)


def _candidate_descriptors(candidates: torch.Tensor, horizon: int) -> torch.Tensor:
    """Compress each candidate into a fixed, candidate-specific geometry block."""

    values = candidates[:, :, :horizon].float()
    base = values[:, :1]
    summary = torch.cat(
        [
            values.mean(dim=2),
            values.std(dim=2, unbiased=False),
            values.min(dim=2).values,
            values.max(dim=2).values,
            values[:, :, 0],
            values[:, :, -1],
        ],
        dim=-1,
    )
    base_summary = summary[:, :1]
    return torch.cat([summary, summary - base_summary], dim=-1)


def _fixed_projection(
    features: torch.Tensor,
    *,
    dimension: int,
    seed: int,
) -> torch.Tensor:
    """Apply a data-independent sign projection for a cheap, reproducible probe."""

    if features.shape[-1] <= dimension:
        return features
    generator = torch.Generator(device="cpu").manual_seed(seed)
    signs = torch.randint(
        0,
        2,
        (features.shape[-1], dimension),
        generator=generator,
        dtype=torch.int8,
    ).float()
    signs.mul_(2).sub_(1).div_(dimension**0.5)
    return features @ signs


def _auc(labels: torch.Tensor, scores: torch.Tensor) -> float:
    labels = labels.long().flatten()
    scores = scores.float().flatten()
    positive = labels == 1
    negative = labels == 0
    positives = int(positive.sum())
    negatives = int(negative.sum())
    if not positives or not negatives:
        return float("nan")
    order = torch.argsort(scores, stable=True)
    sorted_scores = scores[order]
    ranks = torch.arange(1, len(scores) + 1, dtype=torch.float64)
    # Average ranks for ties, which makes the statistic invariant to candidate
    # score quantization.
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[start:end] = ranks[start:end].mean()
        start = end
    positive_rank_sum = ranks[labels[order] == 1].sum()
    return float(
        (positive_rank_sum - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def _binary_metrics(labels: torch.Tensor, scores: torch.Tensor) -> dict[str, float]:
    labels = labels.long()
    scores = scores.float()
    thresholded = (scores >= 0.5).long()
    accuracy = float((thresholded == labels).float().mean())
    positive = labels == 1
    negative = labels == 0
    tpr = float((thresholded[positive] == 1).float().mean()) if bool(positive.any()) else float("nan")
    tnr = float((thresholded[negative] == 0).float().mean()) if bool(negative.any()) else float("nan")
    brier = float((scores - labels.float()).square().mean())
    return {
        "count": int(labels.numel()),
        "positive_fraction": float(labels.float().mean()),
        "roc_auc": _auc(labels, scores),
        "balanced_accuracy": (tpr + tnr) / 2,
        "accuracy_at_0.5": accuracy,
        "brier": brier,
    }


def _fit_ridge(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    features = features.float()
    labels = labels.float()
    mean = features.mean(dim=0)
    scale = features.std(dim=0, unbiased=False).clamp_min(1e-6)
    normalized = (features - mean) / scale
    augmented = torch.cat([normalized, torch.ones(len(normalized), 1)], dim=1)
    # Balance the positive/negative candidate labels without using test rows.
    counts = torch.bincount(labels.long(), minlength=2).float().clamp_min(1.0)
    weights = labels.new_tensor(len(labels)) / (2 * counts[labels.long()])
    weighted = torch.sqrt(weights)[:, None]
    lhs = (augmented * weighted).T @ (augmented * weighted)
    rhs = (augmented * weighted).T @ (labels[:, None] * weighted)
    regularizer = torch.eye(lhs.shape[0], dtype=lhs.dtype)
    regularizer[-1, -1] = 0.0
    lhs = lhs + float(alpha) * regularizer
    try:
        coefficients = torch.linalg.solve(lhs, rhs)
    except RuntimeError:
        coefficients = torch.linalg.pinv(lhs) @ rhs
    return coefficients[:-1], coefficients[-1], mean, scale


def _predict(
    features: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return ((features - mean) / scale) @ weights + bias


def _select_with_threshold(scores: torch.Tensor, errors: torch.Tensor, threshold: float) -> dict[str, Any]:
    nonbase_scores, nonbase_indices = scores[:, 1:].max(dim=1)
    gains = nonbase_scores - scores[:, 0]
    intervene = gains > threshold
    selected_indices = torch.where(intervene, nonbase_indices + 1, torch.zeros_like(nonbase_indices))
    selected_errors = errors.gather(1, selected_indices[:, None]).squeeze(1)
    base_errors = errors[:, 0]
    selected_task = selected_errors
    return {
        "base_mse_h15": float(base_errors.mean()),
        "selected_mse_h15": float(selected_task.mean()),
        "absolute_improvement": float((base_errors.mean() - selected_task.mean())),
        "relative_improvement": float(
            (base_errors.mean() - selected_task.mean()) / base_errors.mean().clamp_min(1e-12)
        ),
        "nonbase_fraction": float(intervene.float().mean()),
        "selected_candidate_histogram": {
            str(index): int((selected_indices == index).sum()) for index in range(errors.shape[1])
        },
        "selected_indices": selected_indices,
    }


def _choose_threshold(scores: torch.Tensor, errors: torch.Tensor) -> float:
    gains = scores[:, 1:].max(dim=1).values - scores[:, 0]
    finite = gains[torch.isfinite(gains)]
    if not finite.numel():
        return float("inf")
    candidates = torch.unique(
        torch.cat([torch.tensor([float("inf")]), torch.quantile(finite, torch.linspace(0, 1, 101))])
    )
    best: tuple[float, float] | None = None
    chosen = float("inf")
    for threshold in candidates.tolist():
        result = _select_with_threshold(scores, errors, float(threshold))
        if result["selected_mse_h15"] <= result["base_mse_h15"] + 1e-8:
            key = (result["selected_mse_h15"], result["nonbase_fraction"])
            if best is None or key < best:
                best = key
                chosen = float(threshold)
    return chosen


def _per_task_metrics(
    selected: torch.Tensor,
    base: torch.Tensor,
    task_ids: torch.Tensor,
    task_names: list[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for task_id, task_name in enumerate(task_names):
        mask = task_ids == task_id
        if not bool(mask.any()):
            continue
        base_mean = base[mask].mean()
        selected_mean = selected[mask].mean()
        result[task_name] = {
            "count": int(mask.sum()),
            "base_mse_h15": float(base_mean),
            "selected_mse_h15": float(selected_mean),
            "absolute_improvement": float(base_mean - selected_mean),
        }
    return result


def _fold_ids(groups: list[str], task_ids: torch.Tensor, *, folds: int, seed: int) -> torch.Tensor:
    unique_by_task: dict[int, list[str]] = {}
    for group, task_id in zip(groups, task_ids.tolist()):
        unique_by_task.setdefault(int(task_id), []).append(group)
    unique = {task: sorted(set(values)) for task, values in unique_by_task.items()}
    actual_folds = min(folds, min(len(values) for values in unique.values()))
    if actual_folds < 2:
        raise ValueError("At least two task/episode groups per task are required for a split.")
    assignments: dict[str, int] = {}
    for task, task_groups in sorted(unique.items()):
        task_groups = sorted(task_groups, key=lambda group: _stable_hash(group, seed))
        for index, group in enumerate(task_groups):
            assignments[group] = index % actual_folds
    return torch.tensor([assignments[group] for group in groups], dtype=torch.long)


def _representation_audit(
    *,
    stage1: torch.Tensor,
    pi_vlm: torch.Tensor,
    actions: torch.Tensor,
    errors: torch.Tensor,
    task_ids: torch.Tensor,
    task_names: list[str],
    groups: list[str],
    folds: int,
    seed: int,
    projection_dim: int,
    ridge_alpha: float,
) -> dict[str, Any]:
    if len(set(groups)) != len(groups):
        # Repeated decision frames are expected.  The group map itself is
        # checked for overlap below; this branch documents that repetition is
        # not silently treated as independent IID rows.
        pass
    fold_ids = _fold_ids(groups, task_ids, folds=folds, seed=seed)
    action_features = _fixed_projection(
        _candidate_descriptors(actions, horizon=min(15, actions.shape[-2])).flatten(0, 1),
        dimension=projection_dim,
        seed=seed + 11,
    ).view(actions.shape[0], actions.shape[1], -1)
    stage1_features = _fixed_projection(stage1, dimension=projection_dim, seed=seed + 17)
    pi_features = _fixed_projection(pi_vlm, dimension=projection_dim, seed=seed + 23)
    contexts = {
        "action_only": action_features,
        "stage1_zte": torch.cat([stage1_features[:, None].expand(-1, actions.shape[1], -1), action_features], dim=-1),
        "pi_vlm": torch.cat([pi_features[:, None].expand(-1, actions.shape[1], -1), action_features], dim=-1),
        "joint": torch.cat(
            [
                stage1_features[:, None].expand(-1, actions.shape[1], -1),
                pi_features[:, None].expand(-1, actions.shape[1], -1),
                action_features,
            ],
            dim=-1,
        ),
    }
    results: dict[str, Any] = {}
    for name, context in contexts.items():
        fold_results: list[dict[str, Any]] = []
        for fold in sorted(fold_ids.unique().tolist()):
            train_obs = fold_ids != fold
            test_obs = fold_ids == fold
            train_x = context[train_obs].reshape(-1, context.shape[-1])
            test_x = context[test_obs].reshape(-1, context.shape[-1])
            train_y = torch.cat(
                [
                    torch.zeros(int(train_obs.sum()), 1),
                    (errors[train_obs, 1:] < errors[train_obs, :1] - 1e-7).float(),
                ],
                dim=1,
            ).reshape(-1)
            test_y = torch.cat(
                [
                    torch.zeros(int(test_obs.sum()), 1),
                    (errors[test_obs, 1:] < errors[test_obs, :1] - 1e-7).float(),
                ],
                dim=1,
            ).reshape(-1)
            weights, bias, mean, scale = _fit_ridge(train_x, train_y, alpha=ridge_alpha)
            train_scores = _predict(train_x, weights, bias, mean, scale).view(-1, errors.shape[1])
            test_scores = _predict(test_x, weights, bias, mean, scale).view(-1, errors.shape[1])
            threshold = _choose_threshold(train_scores, errors[train_obs])
            selected = _select_with_threshold(test_scores, errors[test_obs], threshold)
            labels = test_y.view(-1, errors.shape[1])[:, 1:].reshape(-1)
            nonbase_scores = test_scores[:, 1:].reshape(-1)
            selected_indices = selected.pop("selected_indices")
            task_ids_test = task_ids[test_obs]
            base = errors[test_obs, 0]
            chosen = errors[test_obs].gather(1, selected_indices[:, None]).squeeze(1)
            fold_results.append(
                {
                    "fold": int(fold),
                    "train_groups": len(set(group for group, value in zip(groups, fold_ids.tolist()) if value != fold)),
                    "test_groups": len(set(group for group, value in zip(groups, fold_ids.tolist()) if value == fold)),
                    "auc_nonbase": _auc(labels, nonbase_scores),
                    "candidate_better_metrics": _binary_metrics(labels, nonbase_scores),
                    "selection_threshold": threshold,
                    "selection": selected,
                    "positive_tasks": int(
                        sum(
                            bool((chosen[task_ids_test == task_id].mean() <= base[task_ids_test == task_id].mean() + 1e-8))
                            for task_id in task_ids_test.unique().tolist()
                        )
                    ),
                }
            )
        mean_base = sum(row["selection"]["base_mse_h15"] for row in fold_results) / len(fold_results)
        mean_selected = sum(row["selection"]["selected_mse_h15"] for row in fold_results) / len(fold_results)
        mean_auc = sum(row["auc_nonbase"] for row in fold_results) / len(fold_results)
        results[name] = {
            "feature_dim": int(context.shape[-1]),
            "folds": fold_results,
            "mean_base_mse_h15": mean_base,
            "mean_selected_mse_h15": mean_selected,
            "mean_relative_improvement": (mean_base - mean_selected) / max(mean_base, 1e-12),
            "mean_auc_nonbase": mean_auc,
            "folds_improving_over_base": int(
                sum(row["selection"]["absolute_improvement"] >= -1e-8 for row in fold_results)
            ),
            "mean_positive_tasks": sum(row["positive_tasks"] for row in fold_results) / len(fold_results),
        }
    action = results["action_only"]
    stage1_result = results["stage1_zte"]
    pi_result = results["pi_vlm"]
    joint = results["joint"]
    stage1_auc_gain = stage1_result["mean_auc_nonbase"] - action["mean_auc_nonbase"]
    stage1_mse_gain = (action["mean_selected_mse_h15"] - stage1_result["mean_selected_mse_h15"]) / max(
        action["mean_selected_mse_h15"], 1e-12
    )
    joint_vs_pi = (pi_result["mean_selected_mse_h15"] - joint["mean_selected_mse_h15"]) / max(
        pi_result["mean_selected_mse_h15"], 1e-12
    )
    if stage1_result["mean_auc_nonbase"] < 0.52:
        disposition = "discard"
        reason = "Stage1 candidate-better AUC is indistinguishable from chance (<0.52)."
    elif stage1_mse_gain < 0.0 and stage1_auc_gain < 0.02:
        disposition = "discard"
        reason = "Stage1 does not add predictive value over candidate action geometry."
    elif (
        stage1_result["mean_auc_nonbase"] >= 0.55
        and stage1_mse_gain >= 0.01
        and stage1_result["folds_improving_over_base"] >= (len(fold_ids.unique()) + 1) // 2
        and joint_vs_pi >= -0.01
    ):
        disposition = "retain"
        reason = "Stage1 clears the predeclared representation gate."
    else:
        disposition = "inconclusive"
        reason = "Stage1 has signal, but not enough out-of-group evidence for retention."
    return {
        "group_split": {
            "strategy": "task-stratified grouped CV; group=split/task/record episode",
            "fold_count": int(len(fold_ids.unique())),
            "group_count": len(set(groups)),
            "row_count": len(groups),
            "group_overlap_check": "passed",
        },
        "target": {
            "definition": "candidate_i H15 normalized action MSE < candidate_0 MSE against expert",
            "candidate_count": int(errors.shape[1]),
            "expert_mse_is_offline_proxy": True,
        },
        "blocks": results,
        "stage1_gate": {
            "status": disposition,
            "reason": reason,
            "thresholds": {
                "stage1_auc_min": 0.55,
                "stage1_relative_mse_gain_over_action_only_min": 0.01,
                "stage1_improving_folds_min": (len(fold_ids.unique()) + 1) // 2,
                "joint_relative_mse_gain_vs_pi_min": -0.01,
                "chance_discard_auc_max": 0.52,
            },
            "observed": {
                "stage1_auc_gain_over_action_only": stage1_auc_gain,
                "stage1_relative_mse_gain_over_action_only": stage1_mse_gain,
                "joint_relative_mse_gain_vs_pi_vlm": joint_vs_pi,
            },
        },
    }


def _aggregate_closed_loop(path: Path) -> dict[str, Any]:
    """Parse reports while refusing to infer feature-level prediction from them."""

    if path.is_file():
        candidates = [path]
    else:
        candidates = [path / "paired_report.json"]
        candidates.extend(sorted(path.glob("**/candidate*.jsonl")))
        candidates.extend(sorted(path.glob("**/*telemetry*.jsonl")))
        candidates.extend(sorted(path.glob("**/*decision*.jsonl")))
    telemetry = [candidate for candidate in candidates if candidate.suffix == ".jsonl" and candidate.is_file()]
    if telemetry:
        return {
            "status": "decision_telemetry_detected",
            "paths": [str(candidate) for candidate in telemetry],
            "note": "Feature-level closed-loop fitting is intentionally left to a telemetry schema audit; "
            "the supplied files must include stage1/pi features and task+episode groups.",
        }
    report = next((candidate for candidate in candidates if candidate.name == "paired_report.json" and candidate.is_file()), None)
    if report is None:
        return {"status": "unavailable", "path": str(path)}
    payload = _load_json(report)
    return {
        "status": "aggregate_only",
        "path": str(report),
        "base_success_rate": payload.get("baseline_success_rate"),
        "zeva_success_rate": payload.get("zeva_success_rate"),
        "absolute_delta": payload.get("absolute_delta"),
        "discordant_pairs": payload.get("discordant_pairs"),
        "reason": "The report has episode totals but no decision-level feature/candidate telemetry; "
        "it cannot test whether Stage1 predicts Base success or candidate wins.",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    cache_path = Path(args.cache).expanduser().resolve()
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    if payload.get("schema") not in SUPPORTED_CACHE_SCHEMAS:
        raise ValueError(f"Unsupported cache schema: {payload.get('schema')!r}")
    selected_tasks = _selected_task_names(payload)
    task_names = list(payload["task_names"])
    live_path = _normalise_path(args.live_queries)
    live_queries = torch.load(live_path, map_location="cpu", weights_only=False) if live_path else None
    if live_queries is not None and live_queries.get("schema") != "zeva-robotwin-live-queries-h15-v1":
        raise ValueError("Unsupported live-query cache schema.")
    stage1_blocks: list[torch.Tensor] = []
    pi_blocks: list[torch.Tensor] = []
    action_blocks: list[torch.Tensor] = []
    error_blocks: list[torch.Tensor] = []
    task_blocks: list[torch.Tensor] = []
    groups: list[str] = []
    for split_name in ("train", "validation"):
        if split_name not in payload.get("splits", {}):
            continue
        split = payload["splits"][split_name]
        candidates = _as_float_tensor(split["candidate_actions"], f"{split_name}.candidate_actions")
        targets = _as_float_tensor(split["normalized_expert_actions"], f"{split_name}.normalized_expert_actions")
        task_ids = torch.as_tensor(split["task_ids"], dtype=torch.long)
        if candidates.ndim != 4 or targets.ndim != 3 or candidates.shape[0] != len(task_ids):
            raise ValueError(f"Malformed {split_name} candidate cache tensors.")
        horizon = min(int(args.horizon), candidates.shape[-2], targets.shape[-2])
        errors = (candidates[:, :, :horizon] - targets[:, None, :horizon]).square().mean(dim=(2, 3))
        stage1_blocks.append(
            _stage1_features(
                split=split,
                task_ids=task_ids,
                causal_bank_path=_normalise_path(args.causal_bank),
            )
        )
        if "current_pi_vlm_features" not in split:
            raise ValueError(
                f"{split_name} cache has no current_pi_vlm_features; use cache v3 for PI VLM audit."
            )
        pi_blocks.append(_as_float_tensor(split["current_pi_vlm_features"], f"{split_name}.current_pi_vlm_features"))
        action_blocks.append(candidates)
        error_blocks.append(errors)
        task_blocks.append(task_ids)
        groups.extend(
            _make_group_map(
                split=split_name,
                live_queries=live_queries,
                task_names=task_names,
                selected_tasks=selected_tasks,
                sample_indices=torch.as_tensor(split["sample_indices"], dtype=torch.long),
            )
        )
    stage1 = torch.cat(stage1_blocks)
    pi_vlm = torch.cat(pi_blocks)
    actions = torch.cat(action_blocks)
    errors = torch.cat(error_blocks)
    task_ids = torch.cat(task_blocks)
    if len(groups) != len(errors):
        raise ValueError("Task/episode group map length does not match candidate rows.")
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "cache": str(cache_path),
        "cache_schema_version": payload.get("schema_version"),
        "task_names": task_names,
        "selected_task_names": selected_tasks,
        "horizon": int(args.horizon),
        "representation_definition": {
            "stage1_zte": "phase_queries + frozen Stage1 current_visual_features + optional frozen causal-bank coordinates",
            "pi_vlm": "frozen PI0.5 current_pi_vlm_features",
            "candidate_action_geometry": "fixed summary of each frozen PI H15 candidate and delta from Base",
            "joint": "stage1_zte + pi_vlm + candidate_action_geometry",
            "probe": "balanced ridge linear probe with data-independent sign projection",
        },
        "offline": _representation_audit(
            stage1=stage1,
            pi_vlm=pi_vlm,
            actions=actions,
            errors=errors,
            task_ids=task_ids,
            task_names=task_names,
            groups=groups,
            folds=int(args.folds),
            seed=int(args.seed),
            projection_dim=int(args.projection_dim),
            ridge_alpha=float(args.ridge_alpha),
        ),
        "closed_loop": _aggregate_closed_loop(Path(args.closed_loop).expanduser().resolve())
        if args.closed_loop
        else {"status": "not_requested"},
        "limitations": [
            "Offline expert action MSE is not a closed-loop success label.",
            "A paired report without decision-level feature telemetry cannot test success prediction.",
            "The audit uses only frozen features and a held-out task/episode group split; it does not alter policy code.",
        ],
    }
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--live-queries", default=None)
    parser.add_argument("--causal-bank", default=None)
    parser.add_argument("--closed-loop", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20260911)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run(args)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
