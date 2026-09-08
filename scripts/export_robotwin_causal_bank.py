"""Export the train95-only Zeva causal bank from a frozen Stage 1 ZTE."""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

from accelerate import Accelerator
from accelerate import DataLoaderConfiguration
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import tqdm
import tyro

from openpi.zeva.causal_bank import CAUSAL_BANK_SCHEMA
from openpi.zeva.config import ZevaConfig
from openpi.zeva.robotwin_contract import ROBOTWIN_ACTION_DIM
from openpi.zeva.robotwin_contract import MeanStdActionNormalizer
from openpi.zeva.robotwin_contract import RobotWinHandoff
from openpi.zeva.robotwin_policy import robotwin_multiview_image
from openpi.zeva.transition_encoder import CausalTransitionEncoder
try:
    from scripts.train_robotwin_zte import EpisodeGoalEmbeddingTable
    from scripts.train_robotwin_zte import FFmpegRoboTwinDataset
except ModuleNotFoundError:  # Direct `python scripts/...py` execution.
    from train_robotwin_zte import EpisodeGoalEmbeddingTable
    from train_robotwin_zte import FFmpegRoboTwinDataset


@dataclasses.dataclass
class Args:
    handoff_root: str = (
        "/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1"
    )
    dataset_root: str = "/data1/huangbingjia/robotwin-lerobot-sidney-eef16-v1/data"
    goal_embeddings: str = "/data1/dingxin/zeva-runs/robotwin-v5-h15-tasklang/pi05_task_embeddings.pt"
    zte_checkpoint: str = "/data1/dingxin/zeva-runs/robotwin-v5-h15-tasklang/stage1-zte/zte_best.pth"
    output_path: str = "/data1/dingxin/zeva-runs/robotwin-v5-h15-tasklang/stage1-zte/train_causal_bank.pt"
    phase_bins: int = 32
    samples_per_episode: int = 4
    transition_horizon: int = 15
    batch_size: int = 1
    num_workers: int = 4
    max_episodes: int | None = None
    seed: int = 1000


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class EpisodePhaseDataset(Dataset):
    """A complete H15 causal rollout plus evenly spaced bank collection points."""

    def __init__(
        self,
        adapter_manifest: str | Path,
        samples_per_episode: int,
        effect_steps: int = 15,
        executed_action_steps: int = 15,
        phase_bins: int = 32,
        max_episodes: int | None = None,
        goal_embeddings: str | Path | None = None,
    ):
        if samples_per_episode <= 0:
            raise ValueError("samples_per_episode must be positive.")
        if samples_per_episode > phase_bins:
            raise ValueError("samples_per_episode cannot exceed phase_bins.")
        self.source = FFmpegRoboTwinDataset(adapter_manifest, subset="train")
        self.dataset = self.source.dataset
        self.samples_per_episode = samples_per_episode
        self.effect_steps = effect_steps
        self.executed_action_steps = executed_action_steps
        self.phase_bins = phase_bins
        if goal_embeddings is None:
            raise ValueError("goal_embeddings is required for language-conditioned bank export.")
        self.goals = EpisodeGoalEmbeddingTable(
            goal_embeddings,
            "train",
            self.dataset._records,  # noqa: SLF001
        )
        task_names = sorted({record["key"][1] for record in self.dataset._records})  # noqa: SLF001
        self.task_names = tuple(task_names)
        self._task_ids = {name: index for index, name in enumerate(task_names)}
        record_count = len(self.dataset._records)  # noqa: SLF001
        if max_episodes and max_episodes < record_count:
            if max_episodes < len(task_names):
                raise ValueError("max_episodes must retain at least one episode per task.")
            grouped = {
                name: [
                    index
                    for index, record in enumerate(self.dataset._records)  # noqa: SLF001
                    if record["key"][1] == name
                ]
                for name in task_names
            }
            base, remainder = divmod(max_episodes, len(task_names))
            selected = []
            for task_index, name in enumerate(task_names):
                count = min(len(grouped[name]), base + int(task_index < remainder))
                positions = np.linspace(0, len(grouped[name]) - 1, count, dtype=np.int64)
                selected.extend(grouped[name][int(position)] for position in positions)
            self._record_indices = sorted(selected)
        else:
            self._record_indices = list(range(record_count))

    def __len__(self) -> int:
        return len(self._record_indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record_index = self._record_indices[index]
        record = self.dataset._records[record_index]  # noqa: SLF001
        dataset_start = int(self.dataset._cumulative[record_index])  # noqa: SLF001
        length = int(record["length"])
        before_frames = list(range(0, length - self.effect_steps, self.effect_steps))
        if not before_frames:
            raise ValueError(f"Episode {record['key']} is shorter than one action-effect transition.")
        after_frames = [frame + self.effect_steps for frame in before_frames]
        progress = torch.tensor(
            [frame / max(1, length - 1) for frame in after_frames], dtype=torch.float32
        )
        # Run Mamba over every consecutive transition from B0. Only collection
        # is sparse; the recurrent history is never shortened or reinitialized.
        phase_ids = sorted(
            {
                (record_index + round(item * self.phase_bins / self.samples_per_episode)) % self.phase_bins
                for item in range(self.samples_per_episode)
            }
        )
        desired_progress = torch.tensor(phase_ids, dtype=torch.float32) / max(1, self.phase_bins - 1)
        collection_indices: list[int] = []
        for desired in desired_progress:
            order = torch.argsort((progress - desired).abs()).tolist()
            collection_indices.append(next(item for item in order if item not in collection_indices))
        collection_mask = torch.zeros(len(before_frames), dtype=torch.bool)
        collection_mask[collection_indices] = True
        images = self.source.read_images(record, before_frames + after_frames)
        sequence_length = len(before_frames)
        before = {key: value[:sequence_length] for key, value in images.items()}
        after = {key: value[sequence_length:] for key, value in images.items()}
        actions = torch.stack(
            [
                self.dataset[dataset_start + frame]["action"][: self.executed_action_steps]
                for frame in before_frames
            ]
        )
        return {
            "images_before": robotwin_multiview_image(before),
            "images_after": robotwin_multiview_image(after),
            "actions": actions,
            # ZTE outputs describe the effect observation after the executed H15 prefix.
            "progress": progress,
            "collection_mask": collection_mask,
            "task_id": torch.tensor(self._task_ids[record["key"][1]], dtype=torch.long),
            "goal_embedding": self.goals.embeddings[record_index],
        }


def _all_reduce(value: torch.Tensor) -> None:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)


def _fill_empty_bins(values: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    result = values.clone()
    for task_id in range(len(result)):
        valid = torch.nonzero(counts[task_id] > 0, as_tuple=False).flatten()
        for bin_id in range(result.shape[1]):
            if counts[task_id, bin_id] == 0:
                nearest = valid[torch.argmin(torch.abs(valid - bin_id))]
                result[task_id, bin_id] = result[task_id, nearest]
    return result


def main(args: Args) -> None:
    # Bank aggregation must visit every train95 episode exactly once. The
    # default distributed loader pads to equal rank lengths, which would
    # duplicate up to world_size-1 episodes and bias their phase bins.
    accelerator = Accelerator(dataloader_config=DataLoaderConfiguration(even_batches=False))
    torch.manual_seed(args.seed + accelerator.process_index)
    handoff = RobotWinHandoff.from_root(args.handoff_root)
    checkpoint = torch.load(args.zte_checkpoint, map_location="cpu")
    if checkpoint.get("schema") != "zeva-robotwin-zte-stage1-checkpoint-v5":
        raise ValueError("The v5 causal bank requires a Stage 1 v5 checkpoint.")
    if checkpoint["manifest"]["statistics_sha256"] != _sha256(handoff.statistics):
        raise ValueError("Stage 1 checkpoint statistics differ from the selected PI0.5 handoff.")
    config = ZevaConfig(**checkpoint["zte_config"])
    model = CausalTransitionEncoder(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.requires_grad_(False).eval().to(accelerator.device)
    dataset = EpisodePhaseDataset(
        Path(args.dataset_root) / "adapter.json",
        samples_per_episode=args.samples_per_episode,
        effect_steps=args.transition_horizon,
        executed_action_steps=args.transition_horizon,
        phase_bins=args.phase_bins,
        max_episodes=args.max_episodes,
        goal_embeddings=args.goal_embeddings,
    )
    loader = accelerator.prepare(
        DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )
    )
    task_count = len(dataset.task_names)
    shape = (task_count, args.phase_bins)
    count = torch.zeros(shape, dtype=torch.float64, device=accelerator.device)
    progress_sum = torch.zeros(shape, dtype=torch.float64, device=accelerator.device)
    task_episode_count = torch.zeros(task_count, dtype=torch.float64, device=accelerator.device)
    task_episode_sum = torch.zeros((task_count, config.task_dim), dtype=torch.float64, device=accelerator.device)
    phase_sum = torch.zeros((*shape, config.phase_dim), dtype=torch.float64, device=accelerator.device)
    value_sum = torch.zeros((*shape, config.signal_dim), dtype=torch.float64, device=accelerator.device)
    normalizer = MeanStdActionNormalizer.from_stats_file(handoff.statistics)

    progress_bar = tqdm.tqdm(loader, disable=not accelerator.is_local_main_process, desc="export train95 bank")
    with torch.inference_mode():
        for batch in progress_bar:
            actions = normalizer.normalize(batch["actions"])
            outputs = model(
                batch["images_before"],
                actions,
                batch["images_after"],
                batch["goal_embedding"],
            )
            if outputs.global_task_embedding is None or outputs.initial_phase_token is None:
                raise RuntimeError("Stage 1 v5 omitted global task or B0 phase output.")
            episode_task_ids = batch["task_id"]
            task_episode_count.index_add_(
                0,
                episode_task_ids,
                torch.ones_like(episode_task_ids, dtype=torch.float64),
            )
            task_episode_sum.index_add_(
                0,
                episode_task_ids,
                outputs.global_task_embedding.to(torch.float64),
            )
            mask = batch["collection_mask"].bool()
            batch_size, sequence_length = mask.shape
            task_ids = batch["task_id"][:, None].expand(batch_size, sequence_length)[mask]
            progress = batch["progress"][mask]
            phase_tokens = outputs.phase_token[mask]
            causal_signals = outputs.causal_signal[mask]
            bin_ids = torch.round(progress * (args.phase_bins - 1)).to(torch.long)
            flat_ids = task_ids * args.phase_bins + bin_ids
            count.view(-1).index_add_(0, flat_ids, torch.ones_like(progress, dtype=torch.float64))
            progress_sum.view(-1).index_add_(0, flat_ids, progress.to(torch.float64))
            phase_sum.view(-1, config.phase_dim).index_add_(
                0, flat_ids, phase_tokens.to(torch.float64)
            )
            value_sum.view(-1, config.signal_dim).index_add_(
                0, flat_ids, causal_signals.to(torch.float64)
            )

            # B0 is a real deployment query and must be represented explicitly
            # instead of borrowing the first post-action phase entry.
            initial_flat_ids = episode_task_ids * args.phase_bins
            count.view(-1).index_add_(
                0,
                initial_flat_ids,
                torch.ones_like(episode_task_ids, dtype=torch.float64),
            )
            phase_sum.view(-1, config.phase_dim).index_add_(
                0,
                initial_flat_ids,
                outputs.initial_phase_token.to(torch.float64),
            )

    for tensor in (
        count,
        progress_sum,
        task_episode_count,
        task_episode_sum,
        phase_sum,
        value_sum,
    ):
        _all_reduce(tensor)
    if accelerator.is_main_process:
        divisor = count.clamp_min(1.0)
        task_prototype = F.normalize(
            (task_episode_sum / task_episode_count.clamp_min(1.0).unsqueeze(-1)).to(torch.float32),
            dim=-1,
        )
        phase_key = F.normalize((phase_sum / divisor.unsqueeze(-1)).to(torch.float32), dim=-1)
        causal_value = F.normalize((value_sum / divisor.unsqueeze(-1)).to(torch.float32), dim=-1)
        phase_key = _fill_empty_bins(phase_key, count)
        causal_value = _fill_empty_bins(causal_value, count)
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "schema": CAUSAL_BANK_SCHEMA,
            "split": "train95",
            "task_names": dataset.task_names,
            "phase_bins": args.phase_bins,
            "count": count.cpu().to(torch.long),
            "progress": (progress_sum / divisor).cpu().to(torch.float32),
            "task_prototype": task_prototype.cpu(),
            "phase_key": phase_key.cpu(),
            "causal_value": causal_value.cpu(),
            "manifest": {
                "stage1_checkpoint": str(Path(args.zte_checkpoint).resolve()),
                "stage1_checkpoint_sha256": _sha256(args.zte_checkpoint),
                "stage1_step": int(checkpoint["step"]),
                "stage1_manifest": checkpoint["manifest"],
                "handoff_root": str(handoff.root),
                "statistics_sha256": _sha256(handoff.statistics),
                "dataset_adapter": str((Path(args.dataset_root) / "adapter.json").resolve()),
                "samples_per_episode": args.samples_per_episode,
                "recurrent_history": "complete-consecutive-H15-rollout-from-B0",
                "causal_transition_horizon": args.transition_horizon,
                "goal_embeddings_sha256": _sha256(args.goal_embeddings),
                "episodes": len(dataset),
                "transitions": int(count.sum()),
                "source_sha256": {
                    "exporter": _sha256(Path(__file__).resolve()),
                    "transition_encoder": _sha256(Path(inspect.getfile(CausalTransitionEncoder)).resolve()),
                },
            },
        }
        torch.save(payload, output_path)
        (output_path.with_suffix(".json")).write_text(
            json.dumps(
                {
                    "schema": payload["schema"],
                    "split": payload["split"],
                    "task_count": len(dataset.task_names),
                    "phase_bins": args.phase_bins,
                    "transitions": int(count.sum()),
                    "nonempty_bins": int((count > 0).sum()),
                    "manifest": payload["manifest"],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main(tyro.cli(Args))
