"""Stage 1: train the Zeva Transition Encoder on RoboTwin action-effect sequences."""

from __future__ import annotations

import bisect
import dataclasses
import hashlib
import inspect
import json
import math
from pathlib import Path
import subprocess
from typing import Any

from accelerate import Accelerator
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather as differentiable_all_gather
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Sampler
import tqdm
import tyro

from openpi.zeva.config import ZevaConfig
from openpi.zeva.robotwin_contract import ROBOTWIN_ACTION_DIM
from openpi.zeva.robotwin_contract import ROBOTWIN_ACTION_HORIZON
from openpi.zeva.robotwin_contract import ROBOTWIN_CAMERA_KEYS
from openpi.zeva.robotwin_contract import ROBOTWIN_IMAGE_SHAPE
from openpi.zeva.robotwin_contract import MeanStdActionNormalizer
from openpi.zeva.robotwin_contract import RobotWinHandoff
from openpi.zeva.robotwin_policy import robotwin_multiview_image
from openpi.zeva.transition_encoder import CausalTransitionEncoder


@dataclasses.dataclass
class Args:
    handoff_root: str = (
        "/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1"
    )
    dataset_root: str = "/data1/huangbingjia/robotwin-lerobot-sidney-eef16-v1/data"
    goal_embeddings: str = "/data1/dingxin/zeva-runs/robotwin-v5-h15-tasklang/pi05_task_embeddings.pt"
    save_dir: str = "/data1/dingxin/zeva-runs/robotwin-v5-h15-tasklang/stage1-zte"
    resume_checkpoint: str | None = None
    steps: int = 30_000
    batch_size: int = 1
    num_workers: int = 4
    transition_stride: int = 15
    effect_steps: int = 15
    executed_action_steps: int = 15
    learning_rate: float = 1e-4
    vision_learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    warmup_steps: int = 1_000
    action_loss_weight: float = 0.1
    effect_loss_weight: float = 1.0
    task_loss_weight: float = 2.0
    task_prototype_loss_weight: float = 0.1
    phase_loss_weight: float = 1.0
    phase_key_loss_weight: float = 1.0
    monotonic_loss_weight: float = 0.2
    contrastive_temperature: float = 0.03
    monotonic_margin: float = 0.01
    save_freq: int = 1_000
    eval_batches: int = 0
    log_freq: int = 10
    seed: int = 1000


class FFmpegRoboTwinDataset:
    """Exact handoff adapter with the host's offline AV1-capable FFmpeg decoder."""

    def __init__(self, manifest: str | Path, subset: str, decoder_threads: int = 1):
        from egoscale.data.robotwin.lerobot import RoboTwinLeRobotEEF16Dataset  # noqa: PLC0415

        if decoder_threads <= 0:
            raise ValueError(f"decoder_threads must be positive, got {decoder_threads}.")
        self.dataset = RoboTwinLeRobotEEF16Dataset(manifest, subset=subset)
        self.decoder_threads = int(decoder_threads)
        # Kinematics and action construction stay in the released adapter. Image
        # decoding is batched below to avoid reopening one AV1 video per frame.
        self.dataset._read_source_images = lambda _record, _frame: {}  # noqa: SLF001

    def read_images(self, record: dict[str, Any], frames: list[int]) -> dict[str, torch.Tensor]:
        episode_index = int(record["episode_index"])
        chunk = episode_index // 1000
        unique_frames = sorted(set(frames))
        select_filter = "select=" + "+".join(f"eq(n\\,{frame})" for frame in unique_frames)
        height, width = ROBOTWIN_IMAGE_SHAPE[:2]
        result = {}
        for output_key in ROBOTWIN_CAMERA_KEYS:
            path = (
                record["source_root"]
                / "videos"
                / f"chunk-{chunk:03d}"
                / output_key
                / f"episode_{episode_index:06d}.mp4"
            )
            process = subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    # AV1 decoding and the default filter graph otherwise each
                    # size themselves to all 192 host CPUs. With 32 persistent
                    # workers that creates thousands of runnable threads and
                    # leaves the H100s waiting for the next batch. Keep the
                    # decoder tunable for benchmarking, but always serialize
                    # the tiny select/rawvideo filter graph.
                    "-threads:v",
                    str(self.decoder_threads),
                    "-filter_threads",
                    "1",
                    "-filter_complex_threads",
                    "1",
                    "-i",
                    str(path),
                    "-vf",
                    select_filter,
                    "-vsync",
                    "0",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "pipe:1",
                ],
                check=True,
                capture_output=True,
            )
            expected = len(unique_frames) * height * width * 3
            if len(process.stdout) != expected:
                raise RuntimeError(
                    f"FFmpeg returned {len(process.stdout)} bytes instead of {expected} for {path}."
                )
            decoded = np.frombuffer(process.stdout, dtype=np.uint8).reshape(len(unique_frames), height, width, 3)
            decoded_by_frame = {frame: decoded[index] for index, frame in enumerate(unique_frames)}
            ordered = np.stack([decoded_by_frame[frame] for frame in frames]).copy()
            result[output_key] = torch.from_numpy(ordered).permute(0, 3, 1, 2)
        return result


class TorchCodecRoboTwinDataset:
    """Exact handoff adapter using LeRobot's cached TorchCodec decoders.

    Unlike :class:`FFmpegRoboTwinDataset`, this keeps one decoder per recently
    used video in each persistent DataLoader worker. Stage 2 requests a single
    decision frame at a time, so avoiding three process launches per sample is
    substantially more important than FFmpeg's multi-frame select filter.
    """

    def __init__(self, manifest: str | Path, subset: str):
        from egoscale.data.robotwin.lerobot import RoboTwinLeRobotEEF16Dataset  # noqa: PLC0415

        self.dataset = RoboTwinLeRobotEEF16Dataset(manifest, subset=subset)
        self.dataset.video_backend = "torchcodec"
        # Kinematics and action construction stay in the released adapter. The
        # images are decoded in one batched call per camera below, rather than
        # once in ``dataset.__getitem__`` and again in ``read_images``.
        self.dataset._read_source_images = lambda _record, _frame: {}  # noqa: SLF001

    def read_images(self, record: dict[str, Any], frames: list[int]) -> dict[str, torch.Tensor]:
        # The frozen LeRobot source uses Python 3.11 typing names while some
        # H100 images expose it through Python 3.10.
        import typing  # noqa: PLC0415

        import typing_extensions  # noqa: PLC0415

        for typing_name in ("Self", "Unpack", "NotRequired"):
            if not hasattr(typing, typing_name):
                setattr(typing, typing_name, getattr(typing_extensions, typing_name))
        # Import only video_utils. The aggregate datasets package pulls optional
        # Hub APIs from a newer dependency set that Stage 2 does not need.
        import sys  # noqa: PLC0415
        from types import ModuleType  # noqa: PLC0415

        import lerobot  # noqa: PLC0415

        if "lerobot.datasets" not in sys.modules:
            datasets_package = ModuleType("lerobot.datasets")
            datasets_package.__path__ = [str(Path(next(iter(lerobot.__path__))) / "datasets")]
            datasets_package.__package__ = "lerobot.datasets"
            sys.modules["lerobot.datasets"] = datasets_package
        from lerobot.datasets.video_utils import decode_video_frames  # noqa: PLC0415

        episode_index = int(record["episode_index"])
        chunk = episode_index // 1000
        unique_frames = sorted(set(frames))
        timestamps = [frame / self.dataset.source_fps for frame in unique_frames]
        result = {}
        for output_key in ROBOTWIN_CAMERA_KEYS:
            path = (
                record["source_root"]
                / "videos"
                / f"chunk-{chunk:03d}"
                / output_key
                / f"episode_{episode_index:06d}.mp4"
            )
            decoded = decode_video_frames(
                path,
                timestamps,
                tolerance_s=self.dataset.video_tolerance_s,
                backend="torchcodec",
                return_uint8=True,
            )
            if decoded.ndim != 4 or len(decoded) != len(unique_frames):
                raise RuntimeError(f"TorchCodec returned an invalid frame batch for {path}.")
            decoded_by_frame = {
                frame: decoded[index] for index, frame in enumerate(unique_frames)
            }
            result[output_key] = torch.stack([decoded_by_frame[frame] for frame in frames])
        return result


class EpisodeGoalEmbeddingTable:
    """Frozen per-episode PI0.5 prompt embeddings used as ZeVA goal g."""

    def __init__(self, path: str | Path, subset: str, records: list[dict[str, Any]]):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema") != "zeva-robotwin-pi05-goal-embeddings-v2":
            raise ValueError("Unsupported PI0.5 goal embedding table.")
        if payload.get("embedding_dim") != 2048:
            raise ValueError("PI0.5 goal embeddings must be 2048-dimensional.")
        split = payload["splits"][subset]
        expected_ids = tuple(
            f"{record['key'][0]}:{record['key'][1]}:{int(record['episode_index'])}" for record in records
        )
        if tuple(split["record_ids"]) != expected_ids:
            raise ValueError(f"Goal embedding record order differs from the {subset} dataset.")
        self.embeddings = torch.as_tensor(split["embeddings"], dtype=torch.float32)
        if self.embeddings.shape != (len(records), 2048):
            raise ValueError(f"Goal embedding table has wrong {subset} shape: {tuple(self.embeddings.shape)}")


class RobotWinZTESequenceDataset(Dataset):
    """Natural valid windows of chunk-aligned RoboTwin transitions."""

    def __init__(
        self,
        adapter_manifest: str | Path,
        *,
        subset: str,
        sequence_length: int,
        transition_stride: int,
        effect_steps: int,
    ):
        if sequence_length <= 0 or transition_stride <= 0 or effect_steps <= 0:
            raise ValueError("ZTE sequence parameters must be positive.")
        self.source = FFmpegRoboTwinDataset(adapter_manifest, subset)
        self.dataset = self.source.dataset
        self.sequence_length = sequence_length
        self.transition_stride = transition_stride
        self.effect_steps = effect_steps
        span = (sequence_length - 1) * transition_stride + effect_steps
        counts = np.asarray(
            [max(0, int(record["length"]) - span) for record in self.dataset._records],  # noqa: SLF001
            dtype=np.int64,
        )
        self._cumulative = np.concatenate(([0], np.cumsum(counts)))
        self._record_indices = np.flatnonzero(counts > 0)
        if len(self._record_indices) != len(counts):
            counts = counts[self._record_indices]
            self._cumulative = np.concatenate(([0], np.cumsum(counts)))
        task_names = sorted({record["key"][1] for record in self.dataset._records})  # noqa: SLF001
        self._task_ids = {name: index for index, name in enumerate(task_names)}
        self.task_count = len(task_names)

    def __len__(self) -> int:
        return int(self._cumulative[-1])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        sequence_record = bisect.bisect_right(self._cumulative, index) - 1
        record_index = int(self._record_indices[sequence_record])
        start_frame = index - int(self._cumulative[sequence_record])
        record = self.dataset._records[record_index]  # noqa: SLF001
        dataset_start = int(self.dataset._cumulative[record_index])  # noqa: SLF001
        actions = []
        progress = []
        current_frames = []
        effect_frames = []
        for sequence_index in range(self.sequence_length):
            frame = start_frame + sequence_index * self.transition_stride
            current = self.dataset[dataset_start + frame]
            current_frames.append(frame)
            effect_frames.append(frame + self.effect_steps)
            actions.append(current["action"])
            progress.append(frame / max(1, int(record["length"]) - 1))
        images = self.source.read_images(record, current_frames + effect_frames)
        current_batch = {key: value[: self.sequence_length] for key, value in images.items()}
        effect_batch = {key: value[self.sequence_length :] for key, value in images.items()}
        return {
            "images_before": robotwin_multiview_image(current_batch),
            "images_after": robotwin_multiview_image(effect_batch),
            "actions": torch.stack(actions),
            "progress": torch.tensor(progress, dtype=torch.float32),
            "task_id": torch.tensor(self._task_ids[record["key"][1]], dtype=torch.long),
        }


class TaskPairedDistributedSampler(Sampler[int]):
    """Give every DDP global batch cross-episode positives for each task."""

    def __init__(self, dataset: RobotWinZTEEpisodeDataset, replicas: int, rank: int, seed: int):
        if replicas <= 0 or not 0 <= rank < replicas:
            raise ValueError("Invalid distributed sampler topology.")
        self.dataset = dataset
        self.replicas = replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        self.steps_per_epoch = math.ceil(len(dataset) / replicas)

    def __len__(self) -> int:
        return self.steps_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        pair_count = self.replicas // 2
        for _ in range(self.steps_per_epoch):
            if pair_count:
                tasks = torch.randperm(self.dataset.task_count, generator=generator)[:pair_count].tolist()
                global_indices = []
                for task_id in tasks:
                    candidates = self.dataset.indices_by_task[task_id]
                    picks = torch.randint(len(candidates), (2,), generator=generator).tolist()
                    global_indices.extend(candidates[pick] for pick in picks)
            else:
                global_indices = []
            while len(global_indices) < self.replicas:
                task_id = int(torch.randint(self.dataset.task_count, (), generator=generator))
                candidates = self.dataset.indices_by_task[task_id]
                pick = int(torch.randint(len(candidates), (), generator=generator))
                global_indices.append(candidates[pick])
            yield global_indices[self.rank]


class RobotWinZTEEpisodeDataset(Dataset):
    """Complete episodes sampled at the H15 RoboTwin replanning boundary."""

    def __init__(
        self,
        adapter_manifest: str | Path,
        *,
        subset: str,
        transition_stride: int,
        effect_steps: int,
        executed_action_steps: int,
        goal_embeddings: str | Path,
    ):
        if transition_stride <= 0 or effect_steps <= 0 or executed_action_steps <= 0:
            raise ValueError("ZTE episode parameters must be positive.")
        if transition_stride != effect_steps or effect_steps != executed_action_steps:
            raise ValueError(
                "Deployment-aligned Stage 1 requires transition_stride=effect_steps="
                "executed_action_steps."
            )
        self.source = FFmpegRoboTwinDataset(adapter_manifest, subset)
        self.dataset = self.source.dataset
        self.transition_stride = transition_stride
        self.effect_steps = effect_steps
        self.executed_action_steps = executed_action_steps
        self.goals = EpisodeGoalEmbeddingTable(
            goal_embeddings,
            subset,
            self.dataset._records,  # noqa: SLF001
        )
        self._record_indices = [
            index
            for index, record in enumerate(self.dataset._records)  # noqa: SLF001
            if int(record["length"]) > effect_steps
        ]
        task_names = sorted({record["key"][1] for record in self.dataset._records})  # noqa: SLF001
        self.task_names = tuple(task_names)
        self._task_ids = {name: index for index, name in enumerate(task_names)}
        self.task_count = len(task_names)
        self.indices_by_task: tuple[tuple[int, ...], ...] = tuple(
            tuple(
                dataset_index
                for dataset_index, record_index in enumerate(self._record_indices)
                if self._task_ids[self.dataset._records[record_index]["key"][1]] == task_id  # noqa: SLF001
            )
            for task_id in range(self.task_count)
        )

    def __len__(self) -> int:
        return len(self._record_indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record_index = self._record_indices[index]
        record = self.dataset._records[record_index]  # noqa: SLF001
        length = int(record["length"])
        before_frames = list(range(0, length - self.effect_steps, self.transition_stride))
        after_frames = [frame + self.effect_steps for frame in before_frames]
        images = self.source.read_images(record, before_frames + after_frames)
        sequence_length = len(before_frames)
        before = {key: value[:sequence_length] for key, value in images.items()}
        after = {key: value[sequence_length:] for key, value in images.items()}
        dataset_start = int(self.dataset._cumulative[record_index])  # noqa: SLF001
        actions = torch.stack(
            [
                self.dataset[dataset_start + frame]["action"][: self.executed_action_steps]
                for frame in before_frames
            ]
        )
        denominator = max(1, length - 1)
        return {
            "images_before": robotwin_multiview_image(before),
            "images_after": robotwin_multiview_image(after),
            "actions": actions,
            # Transition outputs condition planning at the post-action state.
            "progress": torch.tensor([frame / denominator for frame in after_frames], dtype=torch.float32),
            "initial_progress": torch.tensor(before_frames[0] / denominator, dtype=torch.float32),
            "goal_embedding": self.goals.embeddings[record_index],
            "task_id": torch.tensor(self._task_ids[record["key"][1]], dtype=torch.long),
        }


def _masked_mean(loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.ndim < loss.ndim:
        mask = mask.unsqueeze(-1)
    expanded = mask.expand_as(loss).to(loss.dtype)
    return (loss * expanded).sum() / expanded.sum().clamp_min(1.0)


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    task_ids: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    # Task identity is episode-level.  v5 receives the explicit global key;
    # accepting a sequence here keeps this helper usable by the LIBERO path.
    features = embeddings.mean(dim=1) if embeddings.ndim == 3 else embeddings
    features = F.normalize(features, dim=-1)
    labels = task_ids.reshape(-1)
    # Per-device batch size is intentionally one because each sample decodes
    # six 480x640 AV1 frames. Gather across DDP ranks so task contrastive
    # learning still sees negatives from the full global batch. PyTorch's
    # differentiable gather sums each rank's contribution in backward; DDP's
    # gradient averaging then produces the gradient of this global loss.
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        local_size = torch.tensor([len(features)], dtype=torch.long, device=features.device)
        gathered_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
        dist.all_gather(gathered_sizes, local_size)
        sizes = [int(item.item()) for item in gathered_sizes]
        max_size = max(sizes)
        padded_features = F.pad(features, (0, 0, 0, max_size - len(features)))
        gathered_features = differentiable_all_gather(padded_features.contiguous())
        features = torch.cat(
            [item[:size] for item, size in zip(gathered_features, sizes, strict=True)], dim=0
        )
        padded_labels = F.pad(labels, (0, max_size - len(labels)), value=-1)
        gathered_labels = [torch.empty_like(padded_labels) for _ in range(world_size)]
        dist.all_gather(gathered_labels, padded_labels.contiguous())
        labels = torch.cat(
            [item[:size] for item, size in zip(gathered_labels, sizes, strict=True)], dim=0
        )
    logits = features @ features.T / temperature
    diagonal = torch.eye(len(features), dtype=torch.bool, device=features.device)
    logits = logits.masked_fill(diagonal, -torch.inf)
    positives = labels[:, None].eq(labels[None, :]) & ~diagonal
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    valid = positives.any(dim=1)
    contrastive = (
        -(log_prob.masked_fill(~positives, 0.0).sum(dim=1)[valid]
          / positives.sum(dim=1)[valid]).mean()
        if valid.any()
        else logits.new_zeros(())
    )
    return contrastive, features, labels


def phase_key_target(progress: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """Order-preserving RBF phase coordinate without Fourier wrap-around."""
    centers = torch.linspace(
        0.0,
        1.0,
        embedding_dim,
        device=progress.device,
        dtype=progress.dtype,
    )
    bandwidth = progress.new_tensor(0.08)
    target = torch.exp(-0.5 * ((progress.unsqueeze(-1) - centers) / bandwidth).square())
    return F.normalize(target, dim=-1)


def compute_losses(
    outputs,
    progress: torch.Tensor,
    task_ids: torch.Tensor,
    args: Args,
    initial_progress: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    mask = torch.ones_like(progress, dtype=torch.bool)
    effect = _masked_mean(F.mse_loss(outputs.predicted_effect, outputs.target_effect, reduction="none"), mask)
    predicted_action = outputs.predicted_action[..., : outputs.target_action.shape[-2], :]
    action = _masked_mean(F.mse_loss(predicted_action, outputs.target_action, reduction="none"), mask)
    phase = F.smooth_l1_loss(outputs.phase_progress, progress)
    phase_key = 1.0 - F.cosine_similarity(
        outputs.phase_token,
        phase_key_target(progress, outputs.phase_token.shape[-1]),
        dim=-1,
    ).mean()
    if initial_progress is not None:
        if outputs.initial_phase_token is None or outputs.initial_phase_progress is None:
            raise RuntimeError("Stage 1 output omitted its initial phase state.")
        phase = 0.5 * (
            phase + F.smooth_l1_loss(outputs.initial_phase_progress, initial_progress)
        )
        initial_phase_key = 1.0 - F.cosine_similarity(
            outputs.initial_phase_token,
            phase_key_target(initial_progress, outputs.initial_phase_token.shape[-1]),
            dim=-1,
        ).mean()
        phase_key = 0.5 * (phase_key + initial_phase_key)
    if outputs.global_task_embedding is None:
        raise RuntimeError("Stage 1 output omitted its episode-global task key.")
    task_contrastive, pooled_tasks, gathered_task_ids = supervised_contrastive_loss(
        outputs.global_task_embedding, task_ids, args.contrastive_temperature
    )
    task = task_contrastive
    if outputs.task_prototypes is not None:
        task = task + args.task_prototype_loss_weight * F.cross_entropy(
            pooled_tasks @ outputs.task_prototypes.T / args.contrastive_temperature,
            gathered_task_ids,
        )
    phase_sequence = outputs.phase_progress
    if outputs.initial_phase_progress is not None:
        phase_sequence = torch.cat([outputs.initial_phase_progress[:, None], phase_sequence], dim=1)
    differences = phase_sequence[:, 1:] - phase_sequence[:, :-1]
    monotonic = F.relu(args.monotonic_margin - differences).mean()
    total = (
        args.effect_loss_weight * effect
        + args.action_loss_weight * action
        + args.task_loss_weight * task
        + args.phase_loss_weight * phase
        + args.phase_key_loss_weight * phase_key
        + args.monotonic_loss_weight * monotonic
    )
    return {
        "total": total,
        "effect": effect,
        "action": action,
        "task": task,
        "phase": phase,
        "phase_key": phase_key,
        "monotonic": monotonic,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest(args: Args, handoff: RobotWinHandoff, config: ZevaConfig, task_count: int) -> dict[str, Any]:
    return {
        "schema": "zeva-robotwin-zte-stage1-v5",
        "handoff_root": str(handoff.root),
        "handoff_contract": str(handoff.contract),
        "statistics": str(handoff.statistics),
        "statistics_sha256": _sha256(handoff.statistics),
        "dataset_adapter": str((Path(args.dataset_root) / "adapter.json").resolve()),
        "dataset_subset": "train95",
        "goal_embeddings": str(Path(args.goal_embeddings).resolve()),
        "goal_embeddings_sha256": _sha256(Path(args.goal_embeddings).resolve()),
        "initial_state": "F_init(pi05-task-only-embedding-g,initial-three-view-visual-s0)",
        "goal_embedding_policy": "task-only-frozen-foundation-table-across-stage2",
        "vision_initialization": "torchvision-resnet18-imagenet1k-v1",
        "split_seed": args.seed,
        "camera_order": list(ROBOTWIN_CAMERA_KEYS),
        "state": "absolute-joint14",
        "action": "chunk-start-relative-eef16",
        "action_dim": ROBOTWIN_ACTION_DIM,
        "policy_action_horizon": ROBOTWIN_ACTION_HORIZON,
        "causal_transition_horizon": args.executed_action_steps,
        "phase_supervision": "strict-monotonic-positive-hazard-plus-rbf-phase-key",
        "temporal_sampling": "complete-episode-at-replan-boundaries-task-paired-ddp",
        "normalization": "frozen-mean-std",
        "task_count": task_count,
        "zte_config": dataclasses.asdict(config),
        "train_args": dataclasses.asdict(args),
        "source_sha256": {
            "trainer": _sha256(Path(__file__).resolve()),
            "transition_encoder": _sha256(Path(inspect.getfile(CausalTransitionEncoder)).resolve()),
        },
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    normalizer: MeanStdActionNormalizer,
    args: Args,
    accelerator: Accelerator,
) -> float:
    model.eval()
    totals = []
    for batch_index, batch in enumerate(loader):
        if args.eval_batches > 0 and batch_index >= args.eval_batches:
            break
        normalized_actions = normalizer.normalize(batch["actions"])
        outputs = model(
            batch["images_before"],
            normalized_actions,
            batch["images_after"],
            batch["goal_embedding"],
        )
        losses = compute_losses(
            outputs,
            batch["progress"],
            batch["task_id"],
            args,
            initial_progress=batch["initial_progress"],
        )
        totals.append(accelerator.gather_for_metrics(losses["total"].detach().reshape(1)))
    model.train()
    return float(torch.cat(totals).mean()) if totals else math.inf


def main(args: Args) -> None:
    if args.batch_size != 1:
        raise ValueError("Complete variable-length episode training currently requires batch_size=1 per GPU.")
    accelerator = Accelerator()
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
    config = ZevaConfig(
        action_dim=ROBOTWIN_ACTION_DIM,
        action_horizon=ROBOTWIN_ACTION_HORIZON,
        vision_pretrained=True,
        num_views=len(ROBOTWIN_CAMERA_KEYS),
        task_count=train_dataset.task_count,
    )
    model = CausalTransitionEncoder(config)
    vision_parameters = list(model.vision_encoder.parameters())
    vision_parameter_ids = {id(parameter) for parameter in vision_parameters}
    other_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in vision_parameter_ids
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
    if args.resume_checkpoint is not None:
        checkpoint = torch.load(args.resume_checkpoint, map_location="cpu")
        if checkpoint.get("schema") != "zeva-robotwin-zte-stage1-checkpoint-v5":
            raise ValueError("Stage 1 v5 can resume only a Stage 1 v5 checkpoint.")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_step = int(checkpoint["step"])
        best_validation = float(checkpoint.get("best_validation", math.inf))

    train_sampler = TaskPairedDistributedSampler(
        train_dataset,
        replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(0, min(2, args.num_workers)),
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    validation_loader = accelerator.prepare(validation_loader)
    normalizer = MeanStdActionNormalizer.from_stats_file(handoff.statistics)
    save_dir = Path(args.save_dir)
    manifest = _manifest(args, handoff, config, train_dataset.task_count)
    if accelerator.is_main_process:
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    accelerator.wait_for_everyone()

    sampler_epoch = 0
    train_sampler.set_epoch(sampler_epoch)
    iterator = iter(train_loader)
    progress_bar = tqdm.trange(
        start_step,
        args.steps,
        disable=not accelerator.is_local_main_process,
        initial=start_step,
        total=args.steps,
    )
    for step in progress_bar:
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
        model.train()
        optimizer.zero_grad(set_to_none=True)
        normalized_actions = normalizer.normalize(batch["actions"])
        outputs = model(
            batch["images_before"],
            normalized_actions,
            batch["images_after"],
            batch["goal_embedding"],
        )
        losses = compute_losses(
            outputs,
            batch["progress"],
            batch["task_id"],
            args,
            initial_progress=batch["initial_progress"],
        )
        accelerator.backward(losses["total"])
        accelerator.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        accelerator.unwrap_model(model).update_ema()

        completed = step + 1
        if completed % args.log_freq == 0:
            progress_bar.set_postfix(
                loss=f"{float(losses['total'].detach()):.4f}",
                effect=f"{float(losses['effect'].detach()):.4f}",
                action=f"{float(losses['action'].detach()):.4f}",
            )
        if completed % args.save_freq == 0 or completed == args.steps:
            validation = evaluate(model, validation_loader, normalizer, args, accelerator)
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped = accelerator.unwrap_model(model)
                checkpoint = {
                    "schema": "zeva-robotwin-zte-stage1-checkpoint-v5",
                    "step": completed,
                    "best_validation": min(best_validation, validation),
                    "validation_loss": validation,
                    "model_state_dict": unwrapped.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "zte_config": dataclasses.asdict(config),
                    "manifest": manifest,
                    "action_normalization": normalizer.metadata(),
                }
                torch.save(checkpoint, save_dir / f"zte_step_{completed:06d}.pth")
                torch.save(checkpoint, save_dir / "zte_latest.pth")
                if validation <= best_validation:
                    best_validation = validation
                    torch.save(checkpoint, save_dir / "zte_best.pth")
                (save_dir / "last_metrics.json").write_text(
                    json.dumps(
                        {"step": completed, "train_loss": float(losses["total"]), "validation_loss": validation},
                        indent=2,
                    )
                    + "\n"
                )
            accelerator.wait_for_everyone()

    accelerator.end_training()


if __name__ == "__main__":
    main(tyro.cli(Args))
