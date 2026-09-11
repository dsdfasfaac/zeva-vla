#!/usr/bin/env python3
"""Audit Stage-1 v2 objective gradients on real RoboTwin episodes.

This is a read-only diagnostic.  It loads one frozen Stage-1 checkpoint,
selects the first fixed-seed batches from the real paired sampler, evaluates
the trainer's three views and masks in ``eval()`` mode, and uses
``torch.autograd.grad`` without an optimizer.  The output distinguishes a
disconnected (``None``) gradient from a connected parameter whose gradient is
numerically zero.

The report is an estimate for the selected episodes, not a training-
distribution or convergence claim.  The default checkpoint is the completed
pilot-e step-256 run; use ``CUDA_VISIBLE_DEVICES=2`` with ``--device cuda:0``
when running on the reserved physical GPU2.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch


DEFAULT_CHECKPOINT = (
    "/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/"
    "stage1-zte-v2-pilot-taskpaired-20260911e/zte_v2_step_000256.pth"
)
DEFAULT_HANDOFF_ROOT = (
    "/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/"
    "robotwin-memory-baseline-v1"
)
DEFAULT_DATASET_ROOT = "/data1/huangbingjia/robotwin-lerobot-sidney-eef16-v1/data"
DEFAULT_GOAL_EMBEDDINGS = (
    "/data1/dingxin/zeva-runs/robotwin-v5-h15-tasklang/pi05_task_embeddings.pt"
)
DEFAULT_DEVICE = "cuda:0"
DEFAULT_BATCH_SIZE = 8
DEFAULT_BATCHES = 2
DEFAULT_SEED = 1000


# These are exactly the terms returned by train_robotwin_zte_v2.compute_v2_losses.
# Categories are reporting groupings only; gradients are measured from the
# weighted category sum and are never inferred from loss magnitudes.
LOSS_WEIGHT_FIELDS = {
    "effect": "effect_loss_weight",
    "action": "action_loss_weight",
    "task": "task_loss_weight",
    "phase_contrastive": "phase_contrastive_weight",
    "phase_order": "phase_order_weight",
    "language_consistency": "language_consistency_weight",
    "global_contrastive": "global_contrastive_weight",
    "causal_contrastive": "causal_contrastive_weight",
    "variance_covariance": "variance_covariance_weight",
    "causal_alignment": "causal_alignment_weight",
}
CATEGORY_COMPONENTS = {
    "action": ("action",),
    "effect": ("effect",),
    "global": ("task", "global_contrastive"),
    "local": ("phase_contrastive", "phase_order", "variance_covariance"),
    "causal": ("causal_alignment", "causal_contrastive", "language_consistency"),
}
GROUP_PREFIXES = {
    "pre_fusion": ("pre_fusion.",),
    "visual_mamba": ("visual_stream.",),
    "action_mamba": ("action_stream.",),
    "phase_head": ("phase_head.",),
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    return str(value)


def _finite_float(value: torch.Tensor | float) -> float | None:
    scalar = float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)
    return scalar if math.isfinite(scalar) else None


def _parameter_groups(model: torch.nn.Module) -> tuple[dict[str, list[torch.nn.Parameter]], dict[str, list[str]]]:
    named = list(model.named_parameters())
    groups: dict[str, list[torch.nn.Parameter]] = {}
    names: dict[str, list[str]] = {}
    for group_name, prefixes in GROUP_PREFIXES.items():
        selected = [(name, parameter) for name, parameter in named if name.startswith(prefixes)]
        if not selected:
            raise RuntimeError(f"No parameters matched gradient group {group_name!r}.")
        groups[group_name] = [parameter for _, parameter in selected]
        names[group_name] = [name for name, _ in selected]
    return groups, names


def _gradient_snapshot(
    loss: torch.Tensor,
    groups: dict[str, list[torch.nn.Parameter]],
) -> tuple[dict[str, dict[str, Any]], dict[str, torch.Tensor | None]]:
    """Compute one weighted loss gradient and classify None versus numeric zero."""

    ordered_groups = list(groups)
    parameters = [parameter for group in ordered_groups for parameter in groups[group]]
    if loss.requires_grad:
        gradients = torch.autograd.grad(
            loss,
            parameters,
            allow_unused=True,
            retain_graph=True,
        )
    else:
        gradients = (None,) * len(parameters)

    reports: dict[str, dict[str, Any]] = {}
    vectors: dict[str, torch.Tensor | None] = {}
    offset = 0
    for group_name in ordered_groups:
        group_parameters = groups[group_name]
        group_gradients = gradients[offset : offset + len(group_parameters)]
        offset += len(group_parameters)
        non_none = [gradient for gradient in group_gradients if gradient is not None]
        none_count = len(group_gradients) - len(non_none)
        if not non_none:
            vectors[group_name] = None
            reports[group_name] = {
                "status": "none",
                "norm": None,
                "parameter_count": len(group_parameters),
                "gradient_parameter_count": 0,
                "none_parameter_count": none_count,
            }
            continue

        # Keep a stable coordinate system for cosine comparisons.  A None
        # parameter gradient is a zero coordinate, but its count remains
        # explicit in the report so it cannot be confused with a connected
        # numeric zero gradient.
        flat_parts = []
        for parameter, gradient in zip(group_parameters, group_gradients, strict=True):
            flat_parts.append(
                torch.zeros_like(parameter).reshape(-1)
                if gradient is None
                else gradient.detach().reshape(-1)
            )
        vector = torch.cat(flat_parts)
        norm = float(vector.norm().detach().cpu())
        vectors[group_name] = vector
        if none_count:
            status = "partial_none"
        elif norm == 0.0:
            status = "zero"
        else:
            status = "nonzero"
        reports[group_name] = {
            "status": status,
            "norm": norm,
            "parameter_count": len(group_parameters),
            "gradient_parameter_count": len(non_none),
            "none_parameter_count": none_count,
        }
    return reports, vectors


def _pairwise_cosines(
    vectors_by_objective: dict[str, dict[str, torch.Tensor | None]],
    objective_names: list[str],
    group_names: list[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for left_index, left_name in enumerate(objective_names):
        for right_name in objective_names[left_index + 1 :]:
            pair_name = f"{left_name}__vs__{right_name}"
            result[pair_name] = {}
            for group_name in group_names:
                left = vectors_by_objective[left_name][group_name]
                right = vectors_by_objective[right_name][group_name]
                if left is None or right is None:
                    result[pair_name][group_name] = {
                        "cosine": None,
                        "status": "undefined_none_gradient",
                    }
                    continue
                left_norm = left.norm()
                right_norm = right.norm()
                if float(left_norm) == 0.0 or float(right_norm) == 0.0:
                    result[pair_name][group_name] = {
                        "cosine": None,
                        "status": "undefined_zero_norm",
                    }
                    continue
                result[pair_name][group_name] = {
                    "cosine": float(torch.dot(left, right) / (left_norm * right_norm)),
                    "status": "defined",
                }
    return result


def _record_loss_gradients(
    losses: dict[str, torch.Tensor],
    args: Any,
    groups: dict[str, list[torch.nn.Parameter]],
) -> dict[str, Any]:
    weighted_terms = {
        name: getattr(args, field) * losses[name]
        for name, field in LOSS_WEIGHT_FIELDS.items()
    }
    component_reports: dict[str, Any] = {}
    component_vectors: dict[str, dict[str, torch.Tensor | None]] = {}
    for name, weighted in weighted_terms.items():
        gradients, vectors = _gradient_snapshot(weighted, groups)
        component_reports[name] = {
            "raw_value": _finite_float(losses[name]),
            "weight": float(getattr(args, LOSS_WEIGHT_FIELDS[name])),
            "weighted_value": _finite_float(weighted),
            "gradients": gradients,
        }
        component_vectors[name] = vectors

    category_reports: dict[str, Any] = {}
    category_vectors: dict[str, dict[str, torch.Tensor | None]] = {}
    for category, components in CATEGORY_COMPONENTS.items():
        weighted = sum((weighted_terms[name] for name in components), losses[components[0]].new_zeros(()))
        gradients, vectors = _gradient_snapshot(weighted, groups)
        category_reports[category] = {
            "components": list(components),
            "weighted_value": _finite_float(weighted),
            "gradients": gradients,
        }
        category_vectors[category] = vectors

    category_names = list(CATEGORY_COMPONENTS)
    return {
        "raw_losses": {name: _finite_float(losses[name]) for name in LOSS_WEIGHT_FIELDS},
        "weighted_losses": {
            name: _finite_float(weighted_terms[name]) for name in weighted_terms
        },
        "total_loss": _finite_float(losses["total"]),
        "weighted_component_sum": _finite_float(sum(weighted_terms.values())),
        "components": component_reports,
        "categories": category_reports,
        "category_pairwise_cosine": _pairwise_cosines(
            category_vectors, category_names, list(groups)
        ),
        # Component-level cosines are useful for identifying which terms drive
        # a category conflict, while the category table remains the primary
        # action/effect/global/local/causal comparison requested by the audit.
        "component_pairwise_cosine": _pairwise_cosines(
            component_vectors, list(LOSS_WEIGHT_FIELDS), list(groups)
        ),
    }


def _args_from_manifest(trainer: Any, checkpoint: dict[str, Any]) -> Any:
    fields = {field.name for field in dataclasses.fields(trainer.Args)}
    values = {
        key: value
        for key, value in checkpoint["manifest"]["train_args"].items()
        if key in fields
    }
    return trainer.Args(**values)


def _sample_identity(dataset: Any, dataset_index: int) -> dict[str, Any]:
    record_index = int(dataset._record_indices[dataset_index])  # noqa: SLF001
    record = dataset.dataset._records[record_index]  # noqa: SLF001
    key = record["key"]
    return {
        "dataset_index": int(dataset_index),
        "record_index": record_index,
        "key": [str(value) for value in key],
        "task_name": str(key[1]),
        "episode_index": int(record["episode_index"]),
        "length": int(record["length"]),
    }


def _mean_defined(values: list[float | None]) -> float | None:
    defined = [value for value in values if value is not None and math.isfinite(value)]
    return sum(defined) / len(defined) if defined else None


def _aggregate_batch_reports(batch_reports: list[dict[str, Any]]) -> dict[str, Any]:
    categories = list(CATEGORY_COMPONENTS)
    groups = list(GROUP_PREFIXES)
    summary: dict[str, Any] = {"batches": len(batch_reports), "categories": {}}
    for category in categories:
        summary["categories"][category] = {
            "weighted_value_mean": _mean_defined(
                [report["categories"][category]["weighted_value"] for report in batch_reports]
            ),
            "gradient_norm_mean": {},
            "gradient_status_counts": {},
        }
        for group in groups:
            entries = [report["categories"][category]["gradients"][group] for report in batch_reports]
            summary["categories"][category]["gradient_norm_mean"][group] = _mean_defined(
                [entry["norm"] for entry in entries]
            )
            counts: dict[str, int] = {}
            for entry in entries:
                counts[entry["status"]] = counts.get(entry["status"], 0) + 1
            summary["categories"][category]["gradient_status_counts"][group] = counts

    for name in ("category_pairwise_cosine", "component_pairwise_cosine"):
        pairs = sorted({pair for report in batch_reports for pair in report[name]})
        summary[name] = {}
        for pair in pairs:
            summary[name][pair] = {}
            for group in groups:
                entries = [report[name].get(pair, {}).get(group, {}) for report in batch_reports]
                values = [entry.get("cosine") for entry in entries]
                summary[name][pair][group] = {
                    "cosine_mean_defined": _mean_defined(values),
                    "defined_count": sum(value is not None for value in values),
                    "undefined_count": sum(value is None for value in values),
                }
    return summary


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_audit(
    *,
    checkpoint: str | Path,
    handoff_root: str | Path,
    dataset_root: str | Path,
    goal_embeddings: str | Path,
    output: str | Path,
    device: str = DEFAULT_DEVICE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    batches: int = DEFAULT_BATCHES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Run the read-only gradient audit and atomically persist its report."""

    if batch_size < 2 or batch_size % 2 or batches <= 0:
        raise ValueError("batch_size must be an even value >=2 and batches must be positive")
    checkpoint = Path(checkpoint).expanduser().resolve()
    handoff_root = Path(handoff_root).expanduser().resolve()
    dataset_root = Path(dataset_root).expanduser().resolve()
    goal_embeddings = Path(goal_embeddings).expanduser().resolve()
    output = Path(output).expanduser()
    for path in (checkpoint, goal_embeddings):
        if not path.is_file():
            raise FileNotFoundError(path)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False

    # Keep heavyweight trainer/data imports lazy so the pure helper tests can
    # run without the RoboTwin/LeRobot environment.
    from scripts import train_robotwin_zte_v2 as trainer  # noqa: PLC0415
    from openpi.zeva.robotwin_contract import MeanStdActionNormalizer  # noqa: PLC0415
    from openpi.zeva.transition_encoder_v2 import (  # noqa: PLC0415
        CausalTransitionEncoderV2,
        TransitionEncoderV2Config,
    )

    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if checkpoint_payload.get("schema") != "zeva-robotwin-zte-stage1-v2-checkpoint":
        raise ValueError(f"Unsupported checkpoint schema: {checkpoint_payload.get('schema')!r}")
    if int(checkpoint_payload.get("step", -1)) != 256:
        raise ValueError(f"Expected pilot step-256 checkpoint, got {checkpoint_payload.get('step')!r}")
    args = _args_from_manifest(trainer, checkpoint_payload)
    if int(args.seed) != seed:
        raise ValueError(f"Requested seed {seed} differs from checkpoint train seed {args.seed}")
    if batch_size > int(args.batch_size):
        raise ValueError(
            f"Requested batch_size {batch_size} exceeds checkpoint train batch_size {args.batch_size}"
        )

    # Loading the checkpoint overwrites every initialized tensor, so avoid a
    # network lookup for torchvision's initialization weights.  This does not
    # alter the architecture or any loaded parameter value.
    config_values = dict(checkpoint_payload["zte_config"])
    config_values["vision_pretrained"] = False
    config = TransitionEncoderV2Config(**config_values)
    model = CausalTransitionEncoderV2(config)
    model.load_state_dict(checkpoint_payload["model_state_dict"], strict=True)
    target_device = torch.device(device)
    model.to(target_device)
    model.eval()
    if model.training or any(module.training for module in model.modules() if isinstance(module, torch.nn.Dropout)):
        raise RuntimeError("Objective audit requires eval mode with dropout disabled")

    groups, group_parameter_names = _parameter_groups(model)
    normalizer = MeanStdActionNormalizer.from_stats_file(
        checkpoint_payload["manifest"]["statistics"]
    )
    adapter_manifest = dataset_root / "adapter.json"
    train_dataset = trainer.RobotWinZTEEpisodeDataset(
        adapter_manifest,
        subset="train",
        transition_stride=args.transition_stride,
        effect_steps=args.effect_steps,
        executed_action_steps=args.executed_action_steps,
        goal_embeddings=goal_embeddings,
    )
    train_dataset = trainer.use_torchcodec_source(train_dataset, adapter_manifest, "train")
    sampler = trainer.PairedTaskSampler(
        train_dataset,
        batch_size=batch_size,
        seed=seed,
        replicas=1,
        rank=0,
        shuffle=True,
    )
    sampler.set_epoch(0)
    sampler_indices = list(sampler)
    selected_batches = [sampler_indices[index : index + batch_size] for index in range(0, batch_size * batches, batch_size)]
    if any(len(batch) != batch_size or any(index < 0 for index in batch) for batch in selected_batches):
        raise RuntimeError("The requested initial paired sampler batches contain padding or are incomplete")

    batch_reports: list[dict[str, Any]] = []
    sample_batches: list[list[dict[str, Any]]] = []
    for batch_index, indices in enumerate(selected_batches):
        samples = [train_dataset[index] for index in indices]
        raw_batch = trainer.collate_robotwin_episodes(samples)
        sample_ids = [_sample_identity(train_dataset, index) for index in indices]
        sample_batches.append(sample_ids)
        batch = {
            key: value.to(target_device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in raw_batch.items()
        }
        valid_mask = batch["valid_mask"]
        episode_mask = batch["episode_mask"]
        encoder_mask = valid_mask.clone()
        encoder_mask[:, 0] = True
        actions = normalizer.normalize(batch["actions"])

        # The trainer's three-view information flow is preserved verbatim:
        # goal-conditioned, augmented language-masked, and original
        # language-masked.  The only change is eval mode/dropout-off for a
        # repeatable diagnostic gradient, as required by this audit.
        outputs = model(
            batch["images_before"],
            actions,
            batch["images_after"],
            batch["goal_embedding"],
            valid_mask=encoder_mask,
        )
        devices = [target_device] if target_device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed + 100003 * batch_index)
            if target_device.type == "cuda":
                torch.cuda.manual_seed_all(seed + 100003 * batch_index)
            augmented_outputs = model(
                trainer.augment_robotwin_images(
                    batch["images_before"],
                    noise_std=args.augmentation_noise,
                    brightness=args.augmentation_brightness,
                ),
                actions,
                trainer.augment_robotwin_images(
                    batch["images_after"],
                    noise_std=args.augmentation_noise,
                    brightness=args.augmentation_brightness,
                ),
                None,
                valid_mask=encoder_mask,
            )
        language_masked_outputs = model(
            batch["images_before"],
            actions,
            batch["images_after"],
            None,
            valid_mask=encoder_mask,
        )
        losses = trainer.compute_v2_losses(
            outputs,
            augmented_outputs,
            language_masked_outputs,
            batch["progress"],
            batch["task_id"],
            args,
            valid_mask=valid_mask,
            episode_mask=episode_mask,
        )
        if not torch.isfinite(losses["total"]):
            raise FloatingPointError(f"Non-finite total loss in audit batch {batch_index}")
        loss_report = _record_loss_gradients(losses, args, groups)
        loss_report.update(
            {
                "batch_index": batch_index,
                "sampler_indices": [int(index) for index in indices],
                "sample_ids": sample_ids,
                "valid_transition_counts": [int(value) for value in valid_mask.sum(dim=1).cpu()],
                "episode_valid_count": int(episode_mask.sum().cpu()),
                "task_ids": [int(value) for value in batch["task_id"].cpu()],
            }
        )
        batch_reports.append(loss_report)
        del outputs, augmented_outputs, language_masked_outputs, losses, loss_report
        del batch, raw_batch, samples
        if target_device.type == "cuda":
            torch.cuda.empty_cache()

    all_weight_fields = {
        field.name for field in dataclasses.fields(args) if field.name.endswith("_weight")
    }
    used_weight_fields = set(LOSS_WEIGHT_FIELDS.values())
    report = {
        "schema": "zeva-robotwin-zte-v2-objective-gradient-audit-v1",
        "audit": {
            "checkpoint_step": int(checkpoint_payload["step"]),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "checkpoint_schema": checkpoint_payload["schema"],
            "device": str(target_device),
            "physical_gpu_visible_index": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_name": (
                torch.cuda.get_device_name(target_device)
                if target_device.type == "cuda" and torch.cuda.is_available()
                else None
            ),
            "model_eval": not model.training,
            "dropout_disabled": not any(
                module.training for module in model.modules() if isinstance(module, torch.nn.Dropout)
            ),
            "optimizer_used": False,
            "training_step_performed": False,
            "estimate_scope": "selected_real_train95_episodes_only",
            "limitations": [
                "Two fixed first-epoch paired-sampler batches, not a training-distribution estimate.",
                "Eval/dropout-off gradients are intentionally not the stochastic train-mode gradient.",
                "No optimizer, scheduler, weight update, or checkpoint write was performed.",
                "Loss magnitudes alone do not establish objective dominance; compare weighted gradients and cosines.",
            ],
        },
        "data": {
            "subset": "train",
            "dataset_root": str(dataset_root),
            "adapter_manifest": str(adapter_manifest),
            "handoff_root": str(handoff_root),
            "normalization_stats": normalizer.metadata(),
            "normalization_stats_sha256": sha256_file(checkpoint_payload["manifest"]["statistics"]),
            "goal_embeddings": str(goal_embeddings),
            "goal_embeddings_sha256": sha256_file(goal_embeddings),
            "sampler": "PairedTaskSampler",
            "sampler_seed": seed,
            "sampler_epoch": 0,
            "batch_size": batch_size,
            "batches": batches,
            "sample_ids_by_batch": sample_batches,
        },
        "model": {
            "checkpoint_config": checkpoint_payload["zte_config"],
            "instantiation_overrides": {"vision_pretrained": False},
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "gradient_groups": {
                group: {
                    "parameter_names": group_parameter_names[group],
                    "parameter_count": sum(parameter.numel() for parameter in groups[group]),
                }
                for group in groups
            },
        },
        "objective": {
            "loss_weight_fields": LOSS_WEIGHT_FIELDS,
            "loss_weights": {
                field: float(getattr(args, field)) for field in sorted(used_weight_fields)
            },
            "unconsumed_declared_weight_fields": sorted(all_weight_fields - used_weight_fields),
            "categories": {name: list(components) for name, components in CATEGORY_COMPONENTS.items()},
            "gradient_status_definitions": {
                "none": "all parameters in the group are disconnected from this weighted loss",
                "zero": "all group parameters are connected and the flattened gradient norm is exactly zero",
                "partial_none": "some group parameters are disconnected; norm uses zero-filled missing coordinates",
                "nonzero": "all group parameters are connected and flattened norm is nonzero",
            },
        },
        "batches": batch_reports,
        "aggregate": _aggregate_batch_reports(batch_reports),
    }
    _atomic_json(output, report)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path(DEFAULT_CHECKPOINT))
    parser.add_argument("--handoff-root", type=Path, default=Path(DEFAULT_HANDOFF_ROOT))
    parser.add_argument("--dataset-root", type=Path, default=Path(DEFAULT_DATASET_ROOT))
    parser.add_argument("--goal-embeddings", type=Path, default=Path(DEFAULT_GOAL_EMBEDDINGS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--batches", type=int, default=DEFAULT_BATCHES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = run_audit(
        checkpoint=args.checkpoint,
        handoff_root=args.handoff_root,
        dataset_root=args.dataset_root,
        goal_embeddings=args.goal_embeddings,
        output=args.output,
        device=args.device,
        batch_size=args.batch_size,
        batches=args.batches,
        seed=args.seed,
    )
    print(json.dumps({
        "output": str(args.output),
        "schema": report["schema"],
        "checkpoint_sha256": report["audit"]["checkpoint_sha256"],
        "batches": report["aggregate"]["batches"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
