"""Stage 1 v2 training for the causal ZeVA transition encoder.

The trainer is intentionally separate from ``train_robotwin_zte.py``.  It
uses the same frozen RoboTwin/PI0.5 dataset adapter and H15/H50 contract, but
implements measurable representation gates:

* task supervision is a task-name prototype probe, never an episode-index
  lookup;
* JEPA/action prediction is computed from the pre-transition path only;
* a post-transition effect alignment loss trains the causal prompt;
* two colour/noise augmentations provide same-time phase positives, while an
  order loss checks temporal direction;
* language-masked consistency is logged so a high task probe cannot be
  explained by a language shortcut.

The default values are suitable for a real run, but this file is also useful
as a reproducible probe runner with ``--steps 1 --eval_batches 1``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import math
from pathlib import Path
from typing import Any

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Sampler
import tqdm
import tyro

from openpi.zeva.robotwin_contract import ROBOTWIN_ACTION_DIM
from openpi.zeva.robotwin_contract import ROBOTWIN_ACTION_HORIZON
from openpi.zeva.robotwin_contract import ROBOTWIN_CAMERA_KEYS
from openpi.zeva.robotwin_contract import ROBOTWIN_IMAGE_SHAPE
from openpi.zeva.robotwin_contract import MeanStdActionNormalizer
from openpi.zeva.robotwin_contract import RobotWinHandoff
from openpi.zeva.transition_encoder_v2 import CausalTransitionEncoderV2
from openpi.zeva.transition_encoder_v2 import TransitionEncoderV2Config
from openpi.zeva.representation_objectives import global_supervised_contrastive
from openpi.zeva.representation_objectives import causal_effect_contrastive
from openpi.zeva.representation_objectives import variance_covariance_loss
from scripts.train_robotwin_zte import RobotWinZTEEpisodeDataset
from scripts.train_robotwin_zte import TorchCodecRoboTwinDataset


@dataclasses.dataclass
class Args:
    handoff_root: str = (
        "/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1"
    )
    dataset_root: str = "/data1/huangbingjia/robotwin-lerobot-sidney-eef16-v1/data"
    goal_embeddings: str = "/data1/dingxin/zeva-runs/robotwin-v5-h15-tasklang/pi05_task_embeddings.pt"
    save_dir: str = "/data1/dingxin/zeva-runs/robotwin-v5-h15-tasklang/stage1-zte-v2"
    resume_checkpoint: str | None = None
    steps: int = 30_000
    # Complete episodes are padded in ``collate_robotwin_episodes``.  Eight
    # episodes per device gives the task probe useful negatives while the
    # transition mask keeps short episodes from changing the objective.
    batch_size: int = 8
    task_paired_batches: bool = True
    num_workers: int = 4
    transition_stride: int = 15
    effect_steps: int = 15
    executed_action_steps: int = 15
    # Explicit experiment axis; old checkpoints/configurations remain pre.
    action_prediction_context: str = "pre"
    prediction_loss_reduction: str = "mean_coordinate_huber"
    learning_rate: float = 1e-4
    vision_learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    warmup_steps: int = 1_000
    effect_loss_weight: float = 0.2
    action_loss_weight: float = 0.2
    task_loss_weight: float = 0.1
    global_contrastive_weight: float = 2.0
    causal_contrastive_weight: float = 1.0
    variance_covariance_weight: float = 0.1
    causal_alignment_weight: float = 1.0
    phase_contrastive_weight: float = 1.0
    phase_order_weight: float = 1.0
    language_consistency_weight: float = 0.5
    monotonic_loss_weight: float = 0.2
    contrastive_temperature: float = 0.07
    monotonic_margin: float = 0.01
    augmentation_noise: float = 0.02
    augmentation_brightness: float = 0.12
    save_freq: int = 1_000
    eval_batches: int = 0
    log_freq: int = 10
    seed: int = 1000
    min_task_probe_accuracy: float = 0.50
    min_phase_order_accuracy: float = 0.80
    min_effect_cosine: float = 0.05
    min_language_consistency: float = 0.80


class GroupedTaskSampler(Sampler[int]):
    """Epoch sampler that visits every episode exactly once.

    The earlier v2 sampler drew a fresh episode from a task group whenever it
    needed another item.  That made an ``epoch`` a random replacement sample
    and could leave some episodes unseen indefinitely.  This sampler first
    builds a permutation of the complete episode set, then gives each rank a
    strided slice.  ``-1`` is used only as a *padding slot* when the episode
    count is not divisible by the number of ranks; it is never an episode
    replacement and is handled by :class:`PaddedEpisodeDataset`.

    ``shuffle=False`` is used for validation, making its global order stable
    across runs and epochs.
    """

    def __init__(
        self,
        dataset,
        *,
        seed: int,
        replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
    ):
        if replicas <= 0 or not 0 <= rank < replicas:
            raise ValueError("Invalid distributed sampler topology.")
        self.dataset = dataset
        self.seed = int(seed)
        self.replicas = int(replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.epoch = 0
        if len(dataset) <= 0:
            raise ValueError("The sampler dataset must contain at least one episode.")
        self.episodes_per_rank = math.ceil(len(dataset) / replicas)
        self.padded_size = self.episodes_per_rank * replicas
        self._groups = [list(group) for group in dataset.indices_by_task if group]
        if not self._groups:
            raise ValueError("The grouped dataset has no task groups.")
        flattened = sorted(index for group in self._groups for index in group)
        if flattened != list(range(len(dataset))):
            raise ValueError("Task groups must partition dataset indices exactly once.")

    def __len__(self) -> int:
        return self.episodes_per_rank

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _global_order(self) -> list[int]:
        if not self.shuffle:
            return list(range(len(self.dataset)))
        generator = torch.Generator().manual_seed(self.seed + 1009 * self.epoch)
        # Shuffle within each task and interleave task queues.  The interleave
        # retains task diversity for the probe without ever sampling with
        # replacement.
        queues = [
            [group[index] for index in torch.randperm(len(group), generator=generator).tolist()]
            for group in self._groups
        ]
        order: list[int] = []
        while any(queues):
            group_order = torch.randperm(len(queues), generator=generator).tolist()
            order.extend(
                queues[group_index].pop()
                for group_index in group_order
                if queues[group_index]
            )
        return order

    def __iter__(self):
        schedule = self._global_order()
        schedule.extend([-1] * (self.padded_size - len(schedule)))
        return iter(schedule[self.rank :: self.replicas])


class PairedTaskSampler(GroupedTaskSampler):
    """Keep same-task pairs in each local batch without repeating episodes.

    Global interleaving followed by rank striding usually gives every episode
    a different task label. Supervised contrastive learning then degenerates
    to matching only an episode's own augmentation. Pair blocks avoid that
    failure while preserving complete epoch coverage.
    """

    def __init__(self, dataset, *, batch_size: int, **kwargs):
        super().__init__(dataset, **kwargs)
        if batch_size < 2 or batch_size % 2:
            raise ValueError("Task-paired batches require an even per-device batch size >=2.")
        self.batch_size = batch_size
        global_batch = batch_size * self.replicas
        self.padded_size = math.ceil(len(dataset) / global_batch) * global_batch
        self.episodes_per_rank = self.padded_size // self.replicas

    def _global_order(self):
        generator = torch.Generator().manual_seed(self.seed + 1009 * self.epoch)
        pairs, leftovers = [], []
        for group in self._groups:
            ordered = [group[i] for i in torch.randperm(len(group), generator=generator).tolist()]
            even_length = len(ordered) - len(ordered) % 2
            pairs.extend(ordered[i:i + 2] for i in range(0, even_length, 2))
            leftovers.extend(ordered[even_length:])
        shuffled = [pairs[i] for i in torch.randperm(len(pairs), generator=generator).tolist()]
        return [index for pair in shuffled for index in pair] + leftovers

    def __iter__(self):
        order = self._global_order()
        order.extend([-1] * (self.padded_size - len(order)))
        global_batch = self.batch_size * self.replicas
        rank_order = []
        for start in range(0, len(order), global_batch):
            offset = start + self.rank * self.batch_size
            rank_order.extend(order[offset:offset + self.batch_size])
        return iter(rank_order)


class PaddedEpisodeDataset(Dataset):
    """Expose sampler padding slots without decoding a fake video episode."""

    def __init__(self, dataset: RobotWinZTEEpisodeDataset):
        self.dataset = dataset
        height, width, _ = ROBOTWIN_IMAGE_SHAPE
        self._padding_image = torch.zeros(
            (1, 3, height, width * len(ROBOTWIN_CAMERA_KEYS)), dtype=torch.uint8
        )
        self._padding_actions = torch.zeros(
            (1, dataset.executed_action_steps, ROBOTWIN_ACTION_DIM), dtype=torch.float32
        )
        self._padding_goal = torch.zeros(dataset.goals.embeddings.shape[-1], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index >= 0:
            item = dict(self.dataset[index])
            item["episode_valid"] = torch.ones((), dtype=torch.bool)
            return item
        return {
            "images_before": self._padding_image.clone(),
            "images_after": self._padding_image.clone(),
            "actions": self._padding_actions.clone(),
            "progress": torch.zeros(1, dtype=torch.float32),
            "initial_progress": torch.zeros((), dtype=torch.float32),
            "goal_embedding": self._padding_goal.clone(),
            "task_id": torch.zeros((), dtype=torch.long),
            "episode_valid": torch.zeros((), dtype=torch.bool),
        }

    def __getattr__(self, name: str):
        # Keep task metadata visible to callers that need the unpadded dataset.
        return getattr(self.dataset, name)


def collate_robotwin_episodes(
    samples: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Pad complete episodes to one batch-local length and return valid masks.

    ``images_before``/``images_after`` are the dataset's lossless canvas with
    shape ``[T,3,480,1920]``; actions are ``[T,15,16]``.  Padding is always on
    the right, which is the information-flow convention supported by the
    encoder's causal streams.
    """

    if not samples:
        raise ValueError("Cannot collate an empty episode batch.")
    episode_valid = torch.stack(
        [
            torch.as_tensor(
                item.get("episode_valid", True),
                dtype=torch.bool,
            )
            for item in samples
        ]
    ).to(dtype=torch.bool)
    lengths = [
        int(item["actions"].shape[0]) if bool(is_valid) else 0
        for item, is_valid in zip(samples, episode_valid, strict=True)
    ]
    max_length = max(1, max(lengths, default=0))

    def pad_sequence(key: str, *, pad_value: float = 0.0) -> torch.Tensor:
        reference = samples[0][key]
        shape = (len(samples), max_length, *reference.shape[1:])
        result = reference.new_full(shape, pad_value)
        for batch_index, (item, length) in enumerate(zip(samples, lengths, strict=True)):
            if length:
                value = item[key]
                if value.shape[0] != length:
                    raise ValueError(f"{key} length disagrees with actions for sample {batch_index}.")
                result[batch_index, :length] = value
        return result

    progress = pad_sequence("progress")
    actions = pad_sequence("actions")
    valid_mask = torch.arange(max_length).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)
    return {
        "images_before": pad_sequence("images_before"),
        "images_after": pad_sequence("images_after"),
        "actions": actions,
        "progress": progress,
        "valid_mask": valid_mask,
        "episode_mask": episode_valid,
        "initial_progress": torch.stack([item["initial_progress"] for item in samples]),
        "goal_embedding": torch.stack([item["goal_embedding"] for item in samples]),
        "task_id": torch.stack([item["task_id"] for item in samples]),
    }


def use_torchcodec_source(
    dataset: RobotWinZTEEpisodeDataset,
    adapter_manifest: Path,
    subset: str,
) -> RobotWinZTEEpisodeDataset:
    """Swap the v1 dataset's AV1 decoder for the cached TorchCodec backend.

    ``RobotWinZTEEpisodeDataset`` owns the episode/action contract; replacing
    only its image source keeps those records and goal-embedding alignment
    intact while avoiding one FFmpeg process per camera for every sample.
    """

    source = TorchCodecRoboTwinDataset(adapter_manifest, subset)
    old_records = dataset.dataset._records  # noqa: SLF001
    new_records = source.dataset._records  # noqa: SLF001
    old_signature = [
        (record["key"], int(record["episode_index"]), int(record["length"]))
        for record in old_records
    ]
    new_signature = [
        (record["key"], int(record["episode_index"]), int(record["length"]))
        for record in new_records
    ]
    if old_signature != new_signature:
        raise ValueError(f"TorchCodec {subset} records do not match the episode/goal table order.")
    dataset.source = source
    dataset.dataset = source.dataset
    return dataset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_checkpoint(payload: dict, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def _to_float_images(images: torch.Tensor) -> torch.Tensor:
    images = images.to(torch.float32)
    if images.detach().amax() > 2.0:
        images = images / 255.0
    if images.detach().amin() < 0.0:
        images = (images + 1.0) / 2.0
    return images.clamp(0.0, 1.0)


def augment_robotwin_images(
    images: torch.Tensor,
    *,
    noise_std: float,
    brightness: float,
) -> torch.Tensor:
    """Camera-safe colour/noise view used for local phase positives."""

    value = _to_float_images(images)
    batch_shape = value.shape[:2] if value.ndim >= 5 else value.shape[:1]
    scale = 1.0 + (torch.rand(*batch_shape, 1, 1, 1, device=value.device) * 2.0 - 1.0) * brightness
    while scale.ndim < value.ndim:
        scale = scale.unsqueeze(-1)
    noise = torch.randn_like(value) * noise_std
    return (value * scale + noise).clamp(0.0, 1.0)


def _zero_loss(value: torch.Tensor) -> torch.Tensor:
    return value.new_zeros(())


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean only over right-padded-valid elements, preserving gradients."""

    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    expanded = mask.to(dtype=value.dtype).expand_as(value)
    return (value * expanded).sum() / expanded.sum().clamp_min(1.0)


def prediction_loss(predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, reduction: str) -> torch.Tensor:
    """Average valid times, optionally retaining squared-vector error units.

    For H15 actions, vector_mse sums EEF coordinates but averages the 15
    substeps; changing H must not multiply the prediction objective.
    """
    if reduction == "mean_coordinate_huber":
        error = F.smooth_l1_loss(predicted, target, reduction="none")
    elif reduction == "vector_mse":
        error = F.mse_loss(predicted, target, reduction="none").sum(dim=-1)
    else:
        raise ValueError(f"Unknown prediction loss reduction: {reduction}")
    return _masked_mean(error, mask)


def temporal_phase_contrastive_loss(
    first: torch.Tensor,
    second: torch.Tensor,
    temperature: float,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """InfoNCE over time indices; same-time augmentations are positives."""

    if valid_mask is None:
        valid_mask = torch.ones(first.shape[:2], dtype=torch.bool, device=first.device)
    losses = []
    for first_row, second_row, row_mask in zip(first, second, valid_mask, strict=True):
        left = F.normalize(first_row[row_mask], dim=-1)
        right = F.normalize(second_row[row_mask], dim=-1)
        if left.shape[0] <= 1:
            continue
        logits = left @ right.T / temperature
        labels = torch.arange(left.shape[0], device=left.device)
        losses.extend((F.cross_entropy(logits, labels), F.cross_entropy(logits.T, labels)))
    return torch.stack(losses).mean() if losses else first.new_zeros(())


def temporal_order_loss(
    phase_progress: torch.Tensor,
    target_progress: torch.Tensor,
    margin: float,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Calibrate progress and enforce strict local temporal order."""

    if valid_mask is None:
        valid_mask = torch.ones_like(phase_progress, dtype=torch.bool)
    calibration = _masked_mean(
        F.smooth_l1_loss(phase_progress, target_progress, reduction="none"), valid_mask
    )
    pair_mask = valid_mask[:, 1:] & valid_mask[:, :-1]
    if phase_progress.shape[1] <= 1 or not pair_mask.any():
        return calibration
    predicted_difference = phase_progress[:, 1:] - phase_progress[:, :-1]
    order = _masked_mean(F.relu(margin - predicted_difference), pair_mask)
    return calibration + order


def _masked_task_embedding(
    task_embedding: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    weights = valid_mask.to(task_embedding.dtype).unsqueeze(-1)
    return (task_embedding * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _task_probe_loss(
    outputs,
    task_ids: torch.Tensor,
    temperature: float,
    valid_mask: torch.Tensor,
    episode_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if outputs.task_prototypes is None:
        return _zero_loss(outputs.global_prompt)
    if episode_mask is None:
        episode_mask = valid_mask.any(dim=1)
    if not episode_mask.any():
        return _zero_loss(outputs.global_prompt)
    # The encoder's public global prompt is the representation consumed by
    # downstream PI0.5.  Older checkpoints expose only the per-transition task
    # stream, so retain the masked fallback for compatibility.
    if outputs.global_prompt.ndim == 2:
        pooled = F.normalize(outputs.global_prompt, dim=-1)
    else:
        pooled = F.normalize(_masked_task_embedding(outputs.task_embedding, valid_mask), dim=-1)
    prototypes = F.normalize(outputs.task_prototypes, dim=-1)
    return F.cross_entropy(
        (pooled[episode_mask] @ prototypes.T) / temperature,
        task_ids.reshape(-1)[episode_mask],
    )


def compute_v2_losses(
    outputs,
    augmented_outputs,
    language_masked_outputs,
    progress: torch.Tensor,
    task_ids: torch.Tensor,
    args: Args,
    *,
    valid_mask: torch.Tensor | None = None,
    episode_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """All Stage-1 v2 objectives with explicit information-flow separation."""

    if valid_mask is None:
        valid_mask = torch.ones_like(progress, dtype=torch.bool)
    if episode_mask is None:
        episode_mask = valid_mask.any(dim=1)
    target_action = outputs.target_action
    predicted_action = outputs.predicted_action[..., : target_action.shape[-2], :]
    effect = prediction_loss(
        outputs.predicted_effect, outputs.target_effect, valid_mask, args.prediction_loss_reduction,
    )
    # ``predicted_action[t]`` is explicitly a next-action auxiliary target:
    # it must be compared with the executed H15 chunk at t+1.  The final
    # transition has no successor and is excluded, as are right-padding slots.
    if predicted_action.shape[1] <= 1:
        action = _zero_loss(predicted_action)
    else:
        next_mask = valid_mask[:, :-1] & valid_mask[:, 1:]
        action = prediction_loss(
            predicted_action[:, :-1],
            target_action[:, 1:, : predicted_action.shape[-2], :],
            next_mask, args.prediction_loss_reduction,
        )
    # Task supervision is intentionally taken from the language-masked pass;
    # otherwise a strong PI0.5 goal embedding can solve the probe without a
    # visual transition representation.
    task = _task_probe_loss(
        language_masked_outputs,
        task_ids,
        args.contrastive_temperature,
        valid_mask,
        episode_mask,
    )
    causal_alignment = _masked_mean(
        1.0
        - F.cosine_similarity(
            outputs.causal_signal, outputs.causal_target_signal.detach(), dim=-1
        ),
        valid_mask,
    )
    phase_contrastive = temporal_phase_contrastive_loss(
        outputs.phase_token,
        augmented_outputs.phase_token,
        args.contrastive_temperature,
        valid_mask,
    )
    phase_order = temporal_order_loss(
        outputs.phase_progress,
        progress,
        args.monotonic_margin,
        valid_mask,
    )
    language_consistency = 0.5 * (
        _masked_mean(
            1.0
            - F.cosine_similarity(
                outputs.phase_token,
                language_masked_outputs.phase_token.detach(),
                dim=-1,
            ),
            valid_mask,
        )
        + _masked_mean(
            1.0
            - F.cosine_similarity(
                outputs.causal_signal,
                language_masked_outputs.causal_signal.detach(),
                dim=-1,
            ),
            valid_mask,
        )
    )
    # Both global views are language-masked. Episode-padding rows contribute
    # neither positives nor negatives to the contrastive objective.
    global_contrastive = global_supervised_contrastive(
        language_masked_outputs.global_prompt[episode_mask],
        augmented_outputs.global_prompt[episode_mask],
        task_ids[episode_mask],
        args.contrastive_temperature,
    ) if episode_mask.any() else outputs.global_prompt.sum() * 0
    causal_contrastive = causal_effect_contrastive(
        language_masked_outputs.causal_signal[valid_mask],
        language_masked_outputs.causal_target_signal[valid_mask],
        args.contrastive_temperature,
    )
    variance_covariance = 0.5 * (
        variance_covariance_loss(language_masked_outputs.phase_token[valid_mask])
        + variance_covariance_loss(language_masked_outputs.causal_signal[valid_mask])
    )
    total = (
        args.effect_loss_weight * effect
        + args.action_loss_weight * action
        + args.task_loss_weight * task
        + args.causal_alignment_weight * causal_alignment
        + args.phase_contrastive_weight * phase_contrastive
        + args.phase_order_weight * phase_order
        + args.language_consistency_weight * language_consistency
        + args.global_contrastive_weight * global_contrastive
        + args.causal_contrastive_weight * causal_contrastive
        + args.variance_covariance_weight * variance_covariance
    )
    return {
        "total": total,
        "effect": effect,
        "action": action,
        "task": task,
        "causal_alignment": causal_alignment,
        "phase_contrastive": phase_contrastive,
        "phase_order": phase_order,
        "language_consistency": language_consistency,
        "global_contrastive": global_contrastive,
        "causal_contrastive": causal_contrastive,
        "variance_covariance": variance_covariance,
    }


@torch.no_grad()
def evaluate_v2(
    model: nn.Module,
    loader: DataLoader,
    normalizer: MeanStdActionNormalizer,
    args: Args,
    accelerator: Accelerator,
) -> dict[str, float | bool | int | str]:
    """Evaluate each validation episode once and reduce metrics globally."""

    def model_mask(valid_mask: torch.Tensor) -> torch.Tensor:
        # The encoder requires one valid token per row.  Sampler padding rows
        # are synthetic and remain excluded from every metric/loss below.
        result = valid_mask.clone()
        result[:, 0] = True
        return result

    def global_sum(value: torch.Tensor) -> torch.Tensor:
        gathered = accelerator.gather_for_metrics(value.detach().reshape(1))
        return gathered.sum()

    model.eval()
    sums = {
        key: torch.zeros((), device=accelerator.device, dtype=torch.float64)
        for key in ("validation_loss", "task_correct", "phase_order_correct", "effect_cosine", "language_consistency")
    }
    counts = {
        key: torch.zeros((), device=accelerator.device, dtype=torch.float64)
        for key in ("episodes", "task", "phase_order", "transitions")
    }
    for batch_index, raw_batch in enumerate(loader):
        if args.eval_batches > 0 and batch_index >= args.eval_batches:
            break
        batch = {
            key: value.to(accelerator.device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in raw_batch.items()
        }
        valid_mask = batch["valid_mask"]
        episode_mask = batch["episode_mask"]
        encoder_mask = model_mask(valid_mask)
        actions = normalizer.normalize(batch["actions"])
        outputs = model(
            batch["images_before"],
            actions,
            batch["images_after"],
            batch["goal_embedding"],
            valid_mask=encoder_mask,
        )
        masked = model(
            batch["images_before"],
            actions,
            batch["images_after"],
            None,
            valid_mask=encoder_mask,
        )
        # Use a reproducible but genuinely augmented masked-language view;
        # comparing the representation to itself would trivialize InfoNCE.
        devices = [accelerator.device] if accelerator.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(args.seed + 100003 * accelerator.process_index + batch_index)
            augmented = model(
                augment_robotwin_images(batch["images_before"], noise_std=args.augmentation_noise,
                                        brightness=args.augmentation_brightness),
                actions,
                augment_robotwin_images(batch["images_after"], noise_std=args.augmentation_noise,
                                        brightness=args.augmentation_brightness),
                None,
                valid_mask=encoder_mask,
            )
        losses = compute_v2_losses(
            outputs,
            augmented,
            masked,
            batch["progress"],
            batch["task_id"],
            args,
            valid_mask=valid_mask,
            episode_mask=episode_mask,
        )
        episode_count = episode_mask.sum().to(torch.float64)
        sums["validation_loss"] += losses["total"].detach().to(torch.float64) * episode_count
        counts["episodes"] += episode_count

        if outputs.task_prototypes is not None:
            if masked.global_prompt.ndim == 2:
                pooled = F.normalize(masked.global_prompt, dim=-1)
            else:
                pooled = F.normalize(_masked_task_embedding(masked.task_embedding, valid_mask), dim=-1)
            logits = pooled @ F.normalize(masked.task_prototypes, dim=-1).T
            correct = logits.argmax(dim=-1).eq(batch["task_id"].reshape(-1))
            sums["task_correct"] += correct[episode_mask].sum().to(torch.float64)
            counts["task"] += episode_count

        if outputs.phase_progress.shape[1] > 1:
            pair_mask = valid_mask[:, 1:] & valid_mask[:, :-1]
            predicted_order = outputs.phase_progress[:, 1:] > outputs.phase_progress[:, :-1]
            target_order = batch["progress"][:, 1:] > batch["progress"][:, :-1]
            sums["phase_order_correct"] += (predicted_order == target_order)[pair_mask].sum().to(torch.float64)
            counts["phase_order"] += pair_mask.sum().to(torch.float64)

        transition_mask = valid_mask
        effect_cosine = F.cosine_similarity(
            outputs.predicted_effect, outputs.target_effect, dim=-1
        )
        language_consistency = 0.5 * (
            F.cosine_similarity(outputs.phase_token, masked.phase_token, dim=-1)
            + F.cosine_similarity(outputs.causal_signal, masked.causal_signal, dim=-1)
        )
        sums["effect_cosine"] += effect_cosine[transition_mask].sum().to(torch.float64)
        sums["language_consistency"] += language_consistency[transition_mask].sum().to(torch.float64)
        counts["transitions"] += transition_mask.sum().to(torch.float64)
    model.train()

    reduced_sums = {key: global_sum(value) for key, value in sums.items()}
    reduced_counts = {key: global_sum(value) for key, value in counts.items()}

    def mean(sum_key: str, count_key: str) -> float:
        denominator = float(reduced_counts[count_key].cpu())
        return float(reduced_sums[sum_key].cpu()) / denominator if denominator else math.nan

    episode_count = round(float(reduced_counts["episodes"].cpu()))
    expected_episodes = len(loader.dataset)
    result = {
        "validation_loss": mean("validation_loss", "episodes"),
        "task_probe_accuracy": mean("task_correct", "task"),
        "phase_order_accuracy": mean("phase_order_correct", "phase_order"),
        "effect_cosine": mean("effect_cosine", "transitions"),
        "language_consistency": mean("language_consistency", "transitions"),
        "validation_episode_count": episode_count,
        "validation_complete": bool(args.eval_batches <= 0 and episode_count == expected_episodes),
        # These probes are useful diagnostics, but are not Stage-1 acceptance
        # gates until the full representation audit is run separately.
        "probe_gate_status": "diagnostic_only",
    }
    result["probe_gate_passed"] = False
    return result


def _manifest(args: Args, handoff: RobotWinHandoff, config: TransitionEncoderV2Config, dataset) -> dict[str, Any]:
    return {
        "schema": "zeva-robotwin-zte-stage1-v2",
        "architecture": "three-causal-mamba-streams-plus-per-transition-cross-attention",
        "information_flow": {
            "b0": "task-only-pi05-language-plus-first-three-view-visual-state",
            "forward_effect_prediction": "pre-transition-visual-plus-ordered-executed-H15-no-current-after-image",
            "next_action_prediction": (
                "normalized-exported-post-transition-phase-current-after-visible-future-hidden"
                if config.action_prediction_context == "phase"
                else "pre-transition-context-no-current-after-image"
            ),
            "causal_signal": "post-transition-EMA-effect-stream",
            "effect_target": "loss-only-target-for-pre-transition-JEPA",
            "language_shortcut_control": "task-probe-from-visual-effect-stream-plus-goal-masked-consistency",
            "episode_index_oracle": False,
        },
        "contract": {
            "camera_order": list(ROBOTWIN_CAMERA_KEYS),
            "image_shape": list(ROBOTWIN_IMAGE_SHAPE),
            "state": "absolute-joint14",
            "action": "chunk-start-relative-eef16",
            "executed_horizon": args.executed_action_steps,
            "policy_horizon": ROBOTWIN_ACTION_HORIZON,
        },
        "grouping": {
            "train_subset": "train",
            "validation_subset": "validation",
            "group_key": "task_name",
            "task_names": list(dataset.task_names),
        },
        "batching": {
            "batch_size_per_device": args.batch_size,
            "padding": "right",
            "mask_key": "valid_mask",
            "sampler": "same-task-pairs-with-full-coverage" if args.task_paired_batches else "interleaved-task-queues-with-rank-striding",
            "distributed_loader_sharding": "sampler-only",
            "image_decoder": "torchcodec",
        },
        "handoff_root": str(handoff.root),
        "statistics": str(handoff.statistics),
        "statistics_sha256": _sha256(handoff.statistics),
        "goal_embeddings": str(Path(args.goal_embeddings).resolve()),
        "goal_embeddings_sha256": _sha256(Path(args.goal_embeddings).resolve()),
        "zte_config": dataclasses.asdict(config),
        "train_args": dataclasses.asdict(args),
        "source_files": {
            "trainer": _sha256(Path(__file__).resolve()),
            "encoder": _sha256(Path(inspect.getfile(CausalTransitionEncoderV2)).resolve()),
        },
    }


def _resume_metadata(checkpoint: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Canonicalize only the documented pre-head default in old checkpoints.

    Missing settings must not silently inherit arbitrary current defaults.
    The phase option is new; all checkpoints predating it used the pre head.
    """
    previous_config = dict(checkpoint["zte_config"])
    previous_args = dict(checkpoint["manifest"]["train_args"])
    previous_config.setdefault("action_prediction_context", "pre")
    previous_args.setdefault("action_prediction_context", "pre")
    previous_args.setdefault("prediction_loss_reduction", "mean_coordinate_huber")
    return previous_config, previous_args


def main(args: Args) -> None:
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if args.executed_action_steps != 15 or args.transition_stride != 15 or args.effect_steps != 15:
        raise ValueError("RoboTwin v2 is fixed to the H15 replanning contract.")
    if (Path(args.save_dir) / "manifest.json").exists() and not args.resume_checkpoint:
        raise FileExistsError("Run already exists; provide its resume checkpoint or a new save_dir.")
    # Several auxiliary heads are intentionally diagnostic or inference-only;
    # allow DDP to mark those parameters unused instead of hanging at the
    # first all-reduce.
    accelerator = Accelerator(
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
    )
    torch.manual_seed(args.seed + accelerator.process_index)
    handoff = RobotWinHandoff.from_root(args.handoff_root)
    adapter_manifest = Path(args.dataset_root) / "adapter.json"
    train_dataset = RobotWinZTEEpisodeDataset(
        adapter_manifest,
        subset="train",
        transition_stride=args.transition_stride,
        effect_steps=args.effect_steps,
        executed_action_steps=args.executed_action_steps,
        goal_embeddings=args.goal_embeddings,
    )
    validation_dataset = RobotWinZTEEpisodeDataset(
        adapter_manifest,
        subset="validation",
        transition_stride=args.transition_stride,
        effect_steps=args.effect_steps,
        executed_action_steps=args.executed_action_steps,
        goal_embeddings=args.goal_embeddings,
    )
    train_dataset = use_torchcodec_source(train_dataset, adapter_manifest, "train")
    validation_dataset = use_torchcodec_source(validation_dataset, adapter_manifest, "validation")
    config = TransitionEncoderV2Config(
        action_dim=ROBOTWIN_ACTION_DIM,
        action_horizon=ROBOTWIN_ACTION_HORIZON,
        executed_action_steps=args.executed_action_steps,
        num_views=len(ROBOTWIN_CAMERA_KEYS),
        action_prediction_context=args.action_prediction_context,
        task_count=train_dataset.task_count,
        vision_pretrained=True,
    )
    model = CausalTransitionEncoderV2(config)
    vision_parameters = list(model.vision_encoder.parameters())
    vision_ids = {id(parameter) for parameter in vision_parameters}
    other_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in vision_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": vision_parameters, "lr": args.vision_learning_rate},
            {"params": other_parameters, "lr": args.learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, (step + 1) / max(1, args.warmup_steps))
        * 0.5
        * (1.0 + math.cos(math.pi * min(step, args.steps) / max(1, args.steps))),
    )
    start_step = 0
    best_validation = math.inf
    resume_rng = None
    if args.resume_checkpoint:
        checkpoint = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        if checkpoint.get("schema") != "zeva-robotwin-zte-stage1-v2-checkpoint":
            raise ValueError("The resume checkpoint is not a Stage 1 v2 checkpoint.")
        previous_config, previous_args = _resume_metadata(checkpoint)
        if previous_config != dataclasses.asdict(config):
            raise ValueError("Resume encoder configuration differs from checkpoint.")
        if checkpoint.get("world_size") != accelerator.num_processes or "rng_by_rank" not in checkpoint:
            raise ValueError("Exact resume requires saved RNG state and the same distributed world size.")
        # Output location and diagnostic verbosity can change; the training
        # schedule, data order, and objective cannot silently change on resume.
        mutable = {"resume_checkpoint", "save_dir", "num_workers", "log_freq"}
        for key, value in dataclasses.asdict(args).items():
            if key not in mutable and previous_args.get(key) != value:
                raise ValueError(f"Resume changes training setting {key}; start a new experiment instead.")
        if checkpoint["manifest"]["statistics_sha256"] != _sha256(handoff.statistics):
            raise ValueError("Resume normalization statistics changed.")
        if checkpoint["manifest"]["goal_embeddings_sha256"] != _sha256(Path(args.goal_embeddings)):
            raise ValueError("Resume goal embedding coordinates changed.")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_step = int(checkpoint["step"])
        best_validation = float(checkpoint.get("best_validation", math.inf))
        resume_rng = checkpoint["rng_by_rank"][accelerator.process_index]

    train_sampler_type = PairedTaskSampler if args.task_paired_batches else GroupedTaskSampler
    sampler_kwargs = {"batch_size": args.batch_size} if args.task_paired_batches else {}
    train_sampler = train_sampler_type(
        train_dataset,
        seed=args.seed,
        replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=True,
        **sampler_kwargs,
    )
    validation_sampler = GroupedTaskSampler(
        validation_dataset,
        seed=args.seed + 1,
        replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=False,
    )
    padded_train_dataset = PaddedEpisodeDataset(train_dataset)
    padded_validation_dataset = PaddedEpisodeDataset(validation_dataset)
    train_loader = DataLoader(
        padded_train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        collate_fn=collate_robotwin_episodes,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    validation_loader = DataLoader(
        padded_validation_dataset,
        batch_size=args.batch_size,
        sampler=validation_sampler,
        collate_fn=collate_robotwin_episodes,
        num_workers=max(0, min(2, args.num_workers)),
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    # The samplers already shard by rank.  Preparing these loaders as well
    # would shard them a second time and silently drop most episodes.
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    normalizer = MeanStdActionNormalizer.from_stats_file(handoff.statistics)
    save_dir = Path(args.save_dir)
    manifest = _manifest(args, handoff, config, train_dataset)
    if accelerator.is_main_process:
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    accelerator.wait_for_everyone()

    sampler_epoch, batches_consumed = divmod(start_step, len(train_loader))
    train_sampler.set_epoch(sampler_epoch)
    iterator = iter(train_loader)
    for _ in range(batches_consumed):
        next(iterator)
    if resume_rng is not None:
        torch.set_rng_state(resume_rng["cpu"])
        if accelerator.device.type == "cuda":
            torch.cuda.set_rng_state(resume_rng["cuda"], accelerator.device)
    bar = tqdm.trange(
        start_step,
        args.steps,
        initial=start_step,
        total=args.steps,
        disable=not accelerator.is_local_main_process,
    )
    for step in bar:
        try:
            batch = next(iterator)
        except StopIteration:
            sampler_epoch += 1
            train_sampler.set_epoch(sampler_epoch)
            iterator = iter(train_loader)
            batch = next(iterator)
        batch = {
            key: value.to(accelerator.device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        valid_mask = batch["valid_mask"]
        episode_mask = batch["episode_mask"]
        encoder_mask = valid_mask.clone()
        # Padding-only rows still need one encoder token for shape/API
        # validity; their all-false loss mask ensures they contribute nothing.
        encoder_mask[:, 0] = True
        actions = normalizer.normalize(batch["actions"])
        model.train()
        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            batch["images_before"],
            actions,
            batch["images_after"],
            batch["goal_embedding"],
            valid_mask=encoder_mask,
        )
        augmented_outputs = model(
            augment_robotwin_images(
                batch["images_before"],
                noise_std=args.augmentation_noise,
                brightness=args.augmentation_brightness,
            ),
            actions,
            augment_robotwin_images(
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
        losses = compute_v2_losses(
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
            raise FloatingPointError(f"Non-finite Stage1 loss at step {step}; checkpoint not accepted.")
        accelerator.backward(losses["total"])
        accelerator.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        accelerator.unwrap_model(model).update_ema()
        completed = step + 1
        if completed % args.log_freq == 0:
            bar.set_postfix(
                loss=f"{float(losses['total'].detach()):.4f}",
                effect=f"{float(losses['effect'].detach()):.4f}",
                phase=f"{float(losses['phase_order'].detach()):.4f}",
            )
        if completed % args.save_freq == 0 or completed == args.steps:
            validation = evaluate_v2(model, validation_loader, normalizer, args, accelerator)
            accelerator.wait_for_everyone()
            local_rng = {
                "cpu": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state(accelerator.device) if accelerator.device.type == "cuda" else None,
            }
            rng_by_rank = [None] * accelerator.num_processes
            if dist.is_initialized():
                dist.all_gather_object(rng_by_rank, local_rng)
            else:
                rng_by_rank[0] = local_rng
            best_validation = min(best_validation, float(validation["validation_loss"]))
            if accelerator.is_main_process:
                unwrapped = accelerator.unwrap_model(model)
                checkpoint = {
                    "schema": "zeva-robotwin-zte-stage1-v2-checkpoint",
                    "step": completed,
                    "world_size": accelerator.num_processes,
                    "rng_by_rank": rng_by_rank,
                    "best_validation": min(best_validation, float(validation["validation_loss"])),
                    "validation": validation,
                    "model_state_dict": unwrapped.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "zte_config": dataclasses.asdict(config),
                    "manifest": manifest,
                    "action_normalization": normalizer.metadata(),
                }
                _save_checkpoint(checkpoint, save_dir / f"zte_v2_step_{completed:06d}.pth")
                _save_checkpoint(checkpoint, save_dir / "zte_v2_latest.pth")
                if float(validation["validation_loss"]) <= best_validation:
                    best_validation = float(validation["validation_loss"])
                    _save_checkpoint(checkpoint, save_dir / "zte_v2_best.pth")
                (save_dir / "probe_gate.json").write_text(
                    json.dumps(validation, indent=2, sort_keys=True) + "\n"
                )
            accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main(tyro.cli(Args))
