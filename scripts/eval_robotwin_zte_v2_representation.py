"""Bounded, diagnostic-only representation report for a RoboTwin ZTE v2 checkpoint.

This report deliberately does not claim a Stage-1 gate.  It measures the
representation checks that can be run offline from the v2 episode contract,
while naming the two missing deployment-level checks (the old-ZTE comparable
probe and PI rollout robustness) explicitly in the output.

The default selection is deterministic: two train and two validation episodes
per task, with no train/validation episode identity overlap.  Validation
prototypes are never built; language-masked retrieval always uses pooled train
features as the prototype source.
"""

from __future__ import annotations

from collections import defaultdict
import dataclasses
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import tyro

from openpi.zeva.robotwin_contract import ROBOTWIN_ACTION_DIM
from openpi.zeva.robotwin_contract import ROBOTWIN_ACTION_HORIZON
from openpi.zeva.robotwin_contract import MeanStdActionNormalizer
from openpi.zeva.robotwin_contract import RobotWinHandoff
from openpi.zeva.transition_encoder_v2 import CausalTransitionEncoderV2
from openpi.zeva.transition_encoder_v2 import TransitionEncoderV2Config
from scripts.train_robotwin_zte import RobotWinZTEEpisodeDataset
from scripts.train_robotwin_zte_v2 import collate_robotwin_episodes
from scripts.train_robotwin_zte_v2 import use_torchcodec_source


@dataclasses.dataclass
class Args:
    handoff_root: str = (
        "/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1"
    )
    dataset_root: str = "/data1/huangbingjia/robotwin-lerobot-sidney-eef16-v1/data"
    goal_embeddings: str = "/data1/dingxin/zeva-runs/robotwin-v5-h15-tasklang/pi05_task_embeddings.pt"
    zte_checkpoint: str = (
        "/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/"
        "stage1-zte-v2-resume-control-20260911c/zte_v2_step_000004.pth"
    )
    output_path: str = (
        "/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/"
        "stage1-zte-v2-resume-control-20260911c/representation_diagnostic.json"
    )
    episodes_per_task: int = 2
    batch_size: int = 8
    num_workers: int = 2
    device: str = "cuda"
    seed: int = 1000
    expected_task_count: int = 50


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record_identity(record: dict[str, Any]) -> str:
    key = record["key"]
    return f"{key[0]}::{key[1]}::{int(record['episode_index'])}"


def _record_info(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "identity": _record_identity(record),
        "key": [str(value) for value in record["key"]],
        "task_name": str(record["key"][1]),
        "episode_index": int(record["episode_index"]),
        "length": int(record["length"]),
    }


def _select_per_task(
    dataset: RobotWinZTEEpisodeDataset,
    *,
    task_names: tuple[str, ...],
    count: int,
    seed: int,
    split_offset: int,
) -> tuple[list[int], dict[str, list[dict[str, Any]]], list[str]]:
    if count <= 0:
        raise ValueError("episodes_per_task must be positive.")
    selected: list[int] = []
    details: dict[str, list[dict[str, Any]]] = {}
    shortfalls: list[str] = []
    for task_id, task_name in enumerate(task_names):
        candidates = np.asarray(dataset.indices_by_task[task_id], dtype=np.int64)
        if len(candidates) == 0:
            details[task_name] = []
            shortfalls.append(task_name)
            continue
        generator = np.random.default_rng(seed + split_offset + 104729 * task_id)
        take = min(count, len(candidates))
        picked = np.sort(generator.choice(candidates, size=take, replace=False)).tolist()
        selected.extend(int(index) for index in picked)
        details[task_name] = [
            _record_info(dataset.dataset._records[dataset._record_indices[index]])  # noqa: SLF001
            for index in picked
        ]
        if take < count:
            shortfalls.append(task_name)
    return selected, details, shortfalls


class _SelectedEpisodeDataset(Dataset):
    """Map selected v2 episode indices to one shared task-id namespace."""

    def __init__(
        self,
        dataset: RobotWinZTEEpisodeDataset,
        indices: list[int],
        task_to_id: dict[str, int],
    ):
        self.dataset = dataset
        self.indices = list(indices)
        self.task_to_id = task_to_id

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        source_index = self.indices[index]
        item = dict(self.dataset[source_index])
        record_index = self.dataset._record_indices[source_index]  # noqa: SLF001
        task_name = self.dataset.dataset._records[record_index]["key"][1]  # noqa: SLF001
        item["task_id"] = torch.tensor(self.task_to_id[task_name], dtype=torch.long)
        item["episode_valid"] = torch.ones((), dtype=torch.bool)
        return item


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _encoder_mask(valid_mask: torch.Tensor) -> torch.Tensor:
    mask = valid_mask.clone()
    # All selected rows are real episodes, but keep the same API invariant as
    # the trainer in case a future bounded selection adds a padding slot.
    mask[:, 0] = True
    return mask


def _forward(
    model: CausalTransitionEncoderV2,
    batch: dict[str, torch.Tensor],
    normalizer: MeanStdActionNormalizer,
    device: torch.device,
    *,
    after_images: torch.Tensor | None = None,
    actions: torch.Tensor | None = None,
    language_masked: bool = False,
):
    valid_mask = batch["valid_mask"]
    normalized_actions = normalizer.normalize(batch["actions"] if actions is None else actions)
    return model(
        batch["images_before"],
        normalized_actions,
        batch["images_after"] if after_images is None else after_images,
        None if language_masked else batch["goal_embedding"],
        valid_mask=_encoder_mask(valid_mask),
    )


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> float:
    expanded = (
        mask[(...,) + (None,) * (value.ndim - mask.ndim)].expand_as(value)
        if value.ndim > mask.ndim
        else mask
    )
    count = int(expanded.sum())
    return float(value[expanded].mean()) if count else math.nan


def _spearman(prediction: torch.Tensor, target: torch.Tensor) -> float:
    if prediction.numel() < 2:
        return math.nan
    prediction_rank = prediction.argsort().argsort().to(torch.float32)
    target_rank = target.argsort().argsort().to(torch.float32)
    prediction_rank -= prediction_rank.mean()
    target_rank -= target_rank.mean()
    denominator = prediction_rank.norm() * target_rank.norm()
    return float((prediction_rank @ target_rank / denominator).cpu()) if denominator > 0 else math.nan


def _masked_abs_delta(first: torch.Tensor, second: torch.Tensor, mask: torch.Tensor) -> float:
    return _masked_mean((first - second).abs(), mask)


def _task_metric_dict(
    values: dict[int, list[float]],
    task_names: tuple[str, ...],
) -> dict[str, float]:
    return {
        task_names[task_id]: float(np.mean(task_values))
        for task_id, task_values in sorted(values.items())
        if task_values
    }


def _load_model(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[CausalTransitionEncoderV2, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "zeva-robotwin-zte-stage1-v2-checkpoint":
        raise ValueError("Expected a Stage 1 v2 checkpoint.")
    declared = checkpoint.get("zte_config", {})
    config_fields = {field.name for field in dataclasses.fields(TransitionEncoderV2Config)}
    config = TransitionEncoderV2Config(
        **{key: value for key, value in declared.items() if key in config_fields}
    )
    model = CausalTransitionEncoderV2(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, checkpoint


def _build_action_means(
    action_sum: dict[int, torch.Tensor],
    action_count: dict[int, int],
    action_dim: int,
) -> dict[int, torch.Tensor]:
    return {
        task_id: action_sum[task_id] / max(1, action_count[task_id])
        for task_id in sorted(action_sum)
        if action_count[task_id] > 0
        and action_sum[task_id].shape == (ROBOTWIN_ACTION_HORIZON - 35, action_dim)
    }


@torch.inference_mode()
def main(args: Args) -> None:
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers must be non-negative.")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    handoff = RobotWinHandoff.from_root(args.handoff_root)
    normalizer = MeanStdActionNormalizer.from_stats_file(handoff.statistics)
    adapter_manifest = Path(args.dataset_root) / "adapter.json"

    train_dataset = RobotWinZTEEpisodeDataset(
        adapter_manifest,
        subset="train",
        transition_stride=15,
        effect_steps=15,
        executed_action_steps=15,
        goal_embeddings=args.goal_embeddings,
    )
    validation_dataset = RobotWinZTEEpisodeDataset(
        adapter_manifest,
        subset="validation",
        transition_stride=15,
        effect_steps=15,
        executed_action_steps=15,
        goal_embeddings=args.goal_embeddings,
    )
    train_dataset = use_torchcodec_source(train_dataset, adapter_manifest, "train")
    validation_dataset = use_torchcodec_source(validation_dataset, adapter_manifest, "validation")

    task_names = tuple(sorted(set(train_dataset.task_names) | set(validation_dataset.task_names)))
    task_to_id = {name: index for index, name in enumerate(task_names)}
    if tuple(train_dataset.task_names) != task_names or tuple(validation_dataset.task_names) != task_names:
        raise ValueError("Train and validation task namespaces differ; refusing ambiguous prototype labels.")

    train_indices, train_selection, train_shortfalls = _select_per_task(
        train_dataset,
        task_names=task_names,
        count=args.episodes_per_task,
        seed=args.seed,
        split_offset=0,
    )
    validation_indices, validation_selection, validation_shortfalls = _select_per_task(
        validation_dataset,
        task_names=task_names,
        count=args.episodes_per_task,
        seed=args.seed,
        split_offset=1,
    )
    train_ids = {
        item["identity"]
        for values in train_selection.values()
        for item in values
    }
    validation_ids = {
        item["identity"]
        for values in validation_selection.values()
        for item in values
    }
    overlap = sorted(train_ids & validation_ids)
    train_selected = _SelectedEpisodeDataset(train_dataset, train_indices, task_to_id)
    validation_selected = _SelectedEpisodeDataset(validation_dataset, validation_indices, task_to_id)
    train_loader = DataLoader(
        train_selected,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_robotwin_episodes,
    )
    validation_loader = DataLoader(
        validation_selected,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_robotwin_episodes,
    )

    model, checkpoint = _load_model(args.zte_checkpoint, device)

    train_feature_rows: dict[int, list[torch.Tensor]] = defaultdict(list)
    action_sum: dict[int, torch.Tensor] = {}
    action_count: dict[int, int] = defaultdict(int)
    for raw_batch in train_loader:
        batch = _move_batch(raw_batch, device)
        masked = _forward(model, batch, normalizer, device, language_masked=True)
        features = F.normalize(masked.global_prompt, dim=-1)
        labels = batch["task_id"].reshape(-1)
        for task_id in labels.unique().tolist():
            rows = labels == task_id
            train_feature_rows[int(task_id)].extend(features[rows].detach().cpu())

        actions = normalizer.normalize(batch["actions"])
        next_mask = batch["valid_mask"][:, :-1] & batch["valid_mask"][:, 1:]
        next_targets = actions[:, 1:]
        for task_id in labels.unique().tolist():
            row_mask = next_mask & labels.eq(task_id)[:, None]
            values = next_targets[row_mask]
            if not len(values):
                continue
            if int(task_id) not in action_sum:
                action_sum[int(task_id)] = values.sum(dim=0).detach().cpu()
            else:
                action_sum[int(task_id)] += values.sum(dim=0).detach().cpu()
            action_count[int(task_id)] += len(values)

    train_prototypes = {
        task_id: F.normalize(torch.stack(rows).mean(dim=0), dim=-1)
        for task_id, rows in train_feature_rows.items()
        if rows
    }
    train_action_means = _build_action_means(action_sum, action_count, ROBOTWIN_ACTION_DIM)

    validation_feature_masked: list[torch.Tensor] = []
    validation_labels: list[torch.Tensor] = []
    phase_predictions: list[torch.Tensor] = []
    phase_targets: list[torch.Tensor] = []
    phase_predictions_by_task: dict[int, list[torch.Tensor]] = defaultdict(list)
    phase_targets_by_task: dict[int, list[torch.Tensor]] = defaultdict(list)
    phase_pair_correct = 0
    phase_pair_total = 0
    next_model_squared = 0.0
    next_baseline_squared = 0.0
    next_elements = 0
    next_model_by_task: dict[int, list[float]] = defaultdict(list)
    next_baseline_by_task: dict[int, list[float]] = defaultdict(list)
    effect_model_squared = 0.0
    effect_zero_squared = 0.0
    effect_elements = 0
    effect_model_by_task: dict[int, list[float]] = defaultdict(list)
    effect_zero_by_task: dict[int, list[float]] = defaultdict(list)
    intervention_sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    intervention_counts: dict[str, int] = defaultdict(int)

    representation_names = (
        "predicted_effect",
        "predicted_action",
        "pre_context",
        "post_context",
        "phase_token",
        "causal_signal",
        "task_embedding",
        "global_prompt",
        "phase_progress",
    )

    for raw_batch in validation_loader:
        batch = _move_batch(raw_batch, device)
        valid_mask = batch["valid_mask"]
        episode_mask = valid_mask.any(dim=1)
        correct = _forward(model, batch, normalizer, device)
        masked = _forward(model, batch, normalizer, device, language_masked=True)
        validation_feature_masked.append(F.normalize(masked.global_prompt, dim=-1).detach().cpu())
        validation_labels.append(batch["task_id"].detach().cpu())

        phase_predictions.append(correct.phase_progress[valid_mask].detach().cpu())
        phase_targets.append(batch["progress"][valid_mask].detach().cpu())
        pair_mask = valid_mask[:, :-1] & valid_mask[:, 1:]
        if pair_mask.any():
            phase_pair_correct += int(
                (
                    correct.phase_progress[:, 1:] > correct.phase_progress[:, :-1]
                )[pair_mask].sum()
            )
            phase_pair_total += int(pair_mask.sum())
        for task_id in batch["task_id"].unique().tolist():
            task_rows = batch["task_id"].eq(task_id)[:, None] & valid_mask
            phase_predictions_by_task[int(task_id)].append(correct.phase_progress[task_rows].detach().cpu())
            phase_targets_by_task[int(task_id)].append(batch["progress"][task_rows].detach().cpu())

        actions = normalizer.normalize(batch["actions"])
        next_mask = valid_mask[:, :-1] & valid_mask[:, 1:]
        next_prediction = correct.predicted_action[:, :-1]
        next_target = correct.target_action[:, 1:]
        for task_id in batch["task_id"].unique().tolist():
            row_mask = next_mask & batch["task_id"].eq(task_id)[:, None]
            values_prediction = next_prediction[row_mask]
            values_target = next_target[row_mask]
            if not len(values_prediction) or int(task_id) not in train_action_means:
                continue
            task_mean = train_action_means[int(task_id)].to(device)
            model_error = (values_prediction - values_target).square().mean(dim=(1, 2))
            baseline_error = (values_target - task_mean).square().mean(dim=(1, 2))
            next_model_squared += float(model_error.sum())
            next_baseline_squared += float(baseline_error.sum())
            next_elements += len(model_error)
            next_model_by_task[int(task_id)].extend(model_error.detach().cpu().tolist())
            next_baseline_by_task[int(task_id)].extend(baseline_error.detach().cpu().tolist())

        effect_values = correct.target_effect[valid_mask]
        effect_prediction = correct.predicted_effect[valid_mask]
        if len(effect_values):
            model_error = (effect_prediction - effect_values).square().mean(dim=-1)
            zero_error = effect_values.square().mean(dim=-1)
            effect_model_squared += float(model_error.sum())
            effect_zero_squared += float(zero_error.sum())
            effect_elements += len(model_error)
            for task_id in batch["task_id"].unique().tolist():
                row_mask = batch["task_id"].eq(task_id)[:, None] & valid_mask
                task_effect = correct.target_effect[row_mask]
                task_prediction = correct.predicted_effect[row_mask]
                effect_model_by_task[int(task_id)].extend(
                    (task_prediction - task_effect).square().mean(dim=-1).detach().cpu().tolist()
                )
                effect_zero_by_task[int(task_id)].extend(
                    task_effect.square().mean(dim=-1).detach().cpu().tolist()
                )

        interventions = {
            "zero_effect": (batch["images_before"], None),
            "shuffled_effect": (torch.roll(batch["images_after"], shifts=1, dims=0), None),
            "zero_action": (None, torch.zeros_like(batch["actions"])),
            "shuffled_action": (None, torch.roll(batch["actions"], shifts=1, dims=0)),
        }
        for name, (after_override, action_override) in interventions.items():
            changed = _forward(
                model,
                batch,
                normalizer,
                device,
                after_images=after_override,
                actions=action_override,
            )
            intervention_counts[name] += int(episode_mask.sum())
            for representation_name in representation_names:
                baseline_value = getattr(correct, representation_name)
                changed_value = getattr(changed, representation_name)
                value_mask = (
                    valid_mask
                    if baseline_value.ndim >= 2 and baseline_value.shape[:2] == valid_mask.shape
                    else episode_mask
                )
                delta = _masked_abs_delta(
                    baseline_value,
                    changed_value,
                    value_mask,
                )
                if math.isfinite(delta):
                    intervention_sums[name][representation_name] += delta * int(episode_mask.sum())

    train_feature_source = "train_subset_language_masked_global_prompt"
    validation_masked = torch.cat(validation_feature_masked)
    validation_labels_tensor = torch.cat(validation_labels)
    retrieval_correct = 0
    retrieval_total = 0
    retrieval_by_task: dict[int, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    prototype_matrix = torch.stack([train_prototypes[index] for index in range(len(task_names)) if index in train_prototypes])
    prototype_ids = [index for index in range(len(task_names)) if index in train_prototypes]
    if len(prototype_ids) == 0:
        raise RuntimeError("No train pooled features were produced for language-masked retrieval.")
    retrieval_logits = validation_masked @ prototype_matrix.T
    retrieval_predictions = torch.tensor(
        [prototype_ids[index] for index in retrieval_logits.argmax(dim=-1).tolist()], dtype=torch.long
    )
    for task_id, prediction in zip(validation_labels_tensor.tolist(), retrieval_predictions.tolist(), strict=True):
        retrieval_total += 1
        retrieval_by_task[task_id]["total"] += 1
        if task_id == prediction:
            retrieval_correct += 1
            retrieval_by_task[task_id]["correct"] += 1

    phase_prediction_tensor = torch.cat(phase_predictions)
    phase_target_tensor = torch.cat(phase_targets)

    intervention_report = {
        name: {
            "episode_count": intervention_counts[name],
            "mean_abs_delta": {
                representation: intervention_sums[name][representation] / max(1, intervention_counts[name])
                for representation in representation_names
            },
        }
        for name in interventions
    }

    next_model_mse = next_model_squared / max(1, next_elements)
    next_baseline_mse = next_baseline_squared / max(1, next_elements)
    effect_model_mse = effect_model_squared / max(1, effect_elements)
    effect_zero_mse = effect_zero_squared / max(1, effect_elements)
    phase_task_spearman = {
        task_names[task_id]: _spearman(torch.cat(phase_predictions_by_task[task_id]), torch.cat(phase_targets_by_task[task_id]))
        for task_id in phase_predictions_by_task
    }
    report: dict[str, Any] = {
        "schema": "zeva-robotwin-zte-v2-representation-diagnostic-v1",
        "status": "diagnostic_only",
        "stage1_gate_passed": False,
        "seed": args.seed,
        "checkpoint": {
            "path": str(Path(args.zte_checkpoint).resolve()),
            "sha256": _sha256(args.zte_checkpoint),
            "step": int(checkpoint.get("step", -1)),
            "schema": checkpoint.get("schema"),
        },
        "selection": {
            "episodes_per_task": args.episodes_per_task,
            "task_count": len(task_names),
            "expected_task_count": args.expected_task_count,
            "train_selected_episodes": len(train_indices),
            "validation_selected_episodes": len(validation_indices),
            "train": train_selection,
            "validation": validation_selection,
            "shortfall_train_tasks": train_shortfalls,
            "shortfall_validation_tasks": validation_shortfalls,
            "train_validation_identity_overlap": overlap,
        },
        "checks": {
            "episode_selection": {
                "passed": (
                    len(task_names) == args.expected_task_count
                    and not train_shortfalls
                    and not validation_shortfalls
                    and not overlap
                ),
                "train_task_coverage": len(train_selection) - len(train_shortfalls),
                "validation_task_coverage": len(validation_selection) - len(validation_shortfalls),
                "identity_overlap_count": len(overlap),
            },
            "next_h15": {
                "passed": next_elements > 0 and next_model_mse < next_baseline_mse,
                "valid_next_transition_count": next_elements,
                "model_normalized_mse": next_model_mse,
                "train_task_mean_baseline_mse": next_baseline_mse,
                "relative_improvement": 1.0 - next_model_mse / max(next_baseline_mse, 1e-12),
                "per_task_model_mse": _task_metric_dict(next_model_by_task, task_names),
                "per_task_train_mean_baseline_mse": _task_metric_dict(next_baseline_by_task, task_names),
            },
            "forward_effect": {
                "passed": effect_elements > 0 and effect_model_mse < effect_zero_mse,
                "valid_transition_count": effect_elements,
                "model_mse": effect_model_mse,
                "zero_effect_baseline_mse": effect_zero_mse,
                "relative_improvement": 1.0 - effect_model_mse / max(effect_zero_mse, 1e-12),
                "per_task_model_mse": _task_metric_dict(effect_model_by_task, task_names),
                "per_task_zero_effect_baseline_mse": _task_metric_dict(effect_zero_by_task, task_names),
            },
            "language_masked_prototype_retrieval": {
                "passed": retrieval_total > 0 and retrieval_correct / retrieval_total >= 0.5,
                "prototype_source": train_feature_source,
                "validation_bank_used": False,
                "accuracy": retrieval_correct / max(1, retrieval_total),
                "correct": retrieval_correct,
                "total": retrieval_total,
                "per_task": {
                    task_names[task_id]: {
                        "correct": values["correct"],
                        "total": values["total"],
                        "accuracy": values["correct"] / max(1, values["total"]),
                    }
                    for task_id, values in sorted(retrieval_by_task.items())
                },
            },
            "phase_ordering": {
                "readout_source": "progress_head(post_context), not an exported-phase frozen probe",
                "passed": None,
                "above_chance": phase_pair_total > 0 and phase_pair_correct / phase_pair_total > 0.5,
                "required_comparison": "old-ZTE phase probe and episode-group confidence intervals",
                "valid_transition_count": len(phase_prediction_tensor),
                "pair_count": phase_pair_total,
                "pair_order_accuracy": phase_pair_correct / max(1, phase_pair_total),
                "spearman": _spearman(phase_prediction_tensor, phase_target_tensor),
                "per_task_spearman": phase_task_spearman,
                "mae": float((phase_prediction_tensor - phase_target_tensor).abs().mean()),
            },
            "input_interventions": {
                "passed": None,
                "executed": all(intervention_counts[name] > 0 for name in intervention_report),
                "effect_prediction_no_leakage": all(
                    intervention_report[name]["mean_abs_delta"][field] < 1e-7
                    for name in ("zero_effect", "shuffled_effect")
                    for field in ("pre_context", "predicted_effect")
                ),
                "action_prediction_context": model.config.action_prediction_context,
                "next_action_current_after_dependency_allowed": model.config.action_prediction_context == "phase",
                "required_comparison": "causal utility, not merely sensitivity to changed inputs",
                "required": [
                    "zero_effect",
                    "shuffled_effect",
                    "zero_action",
                    "shuffled_action",
                ],
                "results": intervention_report,
            },
        },
        "mandatory_missing_gates": [
            "exported_phase_frozen_linear_action_and_progress_probes_not_run",
            "old_zte_comparable_probe_not_run_by_this_report",
            "action_only_forward_effect_predictor_comparison_not_run",
            "cross_task_effect_retrieval_and_ablation_comparisons_not_run",
            "language_permutation_retrieval_consistency_not_run",
            "episode_group_confidence_intervals_not_computed",
            "pi_rollout_robustness_not_run_by_this_report",
            "full_stage1_gate_requires_external_comparable_and_rollout_evidence",
        ],
        "notes": [
            "All metrics exclude right-padding through the trainer valid_mask contract.",
            "The action check compares predicted_action[t] with executed target_action[t+1] over H15.",
            "This diagnostic must not be interpreted as a full Stage-1 acceptance result.",
        ],
    }
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "report_path": str(output),
        "checkpoint_step": report["checkpoint"]["step"],
        "stage1_gate_passed": False,
        "status": "diagnostic_only",
        "checks": {
            name: {key: value for key, value in check.items()
                   if not key.startswith("per_task") and key != "results"}
            for name, check in report["checks"].items()
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(tyro.cli(Args))
