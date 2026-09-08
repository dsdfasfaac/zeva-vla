"""Export the official-train LIBERO causal bank from the selected ZTE."""

from __future__ import annotations

import dataclasses
import inspect
import json
from pathlib import Path
from typing import Any

from accelerate import Accelerator
from accelerate import DataLoaderConfiguration
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import tqdm
import tyro

from openpi.zeva.config import ZevaConfig
from openpi.zeva.libero_bank import LIBERO_CAUSAL_BANK_SCHEMA
from openpi.zeva.libero_contract import LIBERO_EXECUTION_HORIZON
from openpi.zeva.libero_contract import LiberoHandoff
from openpi.zeva.libero_contract import QuantileActionNormalizer
from openpi.zeva.libero_contract import sha256
from openpi.zeva.libero_data import LiberoZTEEpisodeDataset
from openpi.zeva.transition_encoder import CausalTransitionEncoder


@dataclasses.dataclass
class Args:
    handoff_root: str = "/data1/dingxin/libero-memory-baseline-v1"
    dataset_root: str = "/data1/dingxin/libero-memory-baseline-v1/data"
    goal_embeddings: str = "/data1/dingxin/zeva-runs/libero-v1-h5-lang/pi05_goal_embeddings.pt"
    zte_checkpoint: str = "/data1/dingxin/zeva-runs/libero-v3-h5-lang/stage1-zte/zte_best.pth"
    output_path: str = "/data1/dingxin/zeva-runs/libero-v3-h5-lang/stage1-zte/train_causal_bank.pt"
    phase_bins: int = 32
    transition_horizon: int = LIBERO_EXECUTION_HORIZON
    batch_size: int = 1
    num_workers: int = 4
    max_episodes: int | None = None
    seed: int = 1000


class EpisodePhaseDataset(Dataset):
    def __init__(self, args: Args):
        self.source = LiberoZTEEpisodeDataset(
            args.dataset_root,
            subset="train",
            goal_embeddings=args.goal_embeddings,
        )
        count = len(self.source)
        self.indices = list(range(min(count, args.max_episodes or count)))
        self.task_names = self.source.task_names

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.source[self.indices[index]]


def _reduce(value):
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)


def _fill(values, counts):
    result = values.clone()
    for task in range(len(result)):
        valid = torch.nonzero(counts[task] > 0).flatten()
        if not len(valid):
            raise ValueError(f"LIBERO task {task} has no bank entries.")
        for phase in range(result.shape[1]):
            if counts[task, phase] == 0:
                nearest = valid[torch.argmin(torch.abs(valid - phase))]
                result[task, phase] = result[task, nearest]
    return result


def main(args: Args) -> None:
    if args.transition_horizon != LIBERO_EXECUTION_HORIZON:
        raise ValueError("Formal LIBERO bank export requires H5.")
    accelerator = Accelerator(dataloader_config=DataLoaderConfiguration(even_batches=False))
    handoff = LiberoHandoff.from_root(args.handoff_root)
    checkpoint = torch.load(args.zte_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "zeva-libero-zte-stage1-checkpoint-v3":
        raise ValueError("LIBERO bank export requires a formal LIBERO Stage 1 checkpoint.")
    if checkpoint["manifest"]["statistics_sha256"] != sha256(handoff.statistics):
        raise ValueError("LIBERO ZTE and normalizer hashes differ.")
    config = ZevaConfig(**checkpoint["zte_config"])
    model = CausalTransitionEncoder(config).to(accelerator.device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.requires_grad_(False).eval()
    dataset = EpisodePhaseDataset(args)
    loader = accelerator.prepare(
        DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True,
                   persistent_workers=args.num_workers > 0)
    )
    shape = (len(dataset.task_names), args.phase_bins)
    count = torch.zeros(shape, dtype=torch.float64, device=accelerator.device)
    progress_sum = torch.zeros(shape, dtype=torch.float64, device=accelerator.device)
    task_episode_count = torch.zeros(len(dataset.task_names), dtype=torch.float64, device=accelerator.device)
    task_episode_sum = torch.zeros(
        (len(dataset.task_names), config.task_dim), dtype=torch.float64, device=accelerator.device
    )
    phase_sum = torch.zeros((*shape, config.phase_dim), dtype=torch.float64, device=accelerator.device)
    value_sum = torch.zeros((*shape, config.signal_dim), dtype=torch.float64, device=accelerator.device)
    normalizer = QuantileActionNormalizer.from_stats_file(handoff.statistics)
    with torch.inference_mode():
        for batch in tqdm.tqdm(loader, disable=not accelerator.is_local_main_process, desc="export LIBERO bank"):
            output = model(batch["images_before"], normalizer.normalize(batch["actions"]),
                           batch["images_after"], batch["goal_embedding"])
            if output.global_task_embedding is None or output.initial_phase_token is None:
                raise RuntimeError("LIBERO Stage 1 v3 omitted its global task or B0 phase output.")
            episode_tasks = batch["task_id"]
            task_episode_count.index_add_(
                0, episode_tasks, torch.ones_like(episode_tasks, dtype=torch.float64)
            )
            task_episode_sum.index_add_(
                0, episode_tasks, output.global_task_embedding.to(torch.float64)
            )
            batch_size, sequence = output.phase_token.shape[:2]
            tasks = batch["task_id"][:, None].expand(batch_size, sequence).reshape(-1)
            progress = batch["progress"].reshape(-1).to(accelerator.device)
            bins = torch.round(progress * (args.phase_bins - 1)).long()
            flat = tasks * args.phase_bins + bins
            count.view(-1).index_add_(0, flat, torch.ones_like(progress, dtype=torch.float64))
            progress_sum.view(-1).index_add_(0, flat, progress.double())
            phase_sum.view(-1, config.phase_dim).index_add_(0, flat, output.phase_token.reshape(-1, config.phase_dim).double())
            value_sum.view(-1, config.signal_dim).index_add_(0, flat, output.causal_signal.reshape(-1, config.signal_dim).double())

            # B0 is the real deployment query before the first H5 action.
            initial_flat = episode_tasks * args.phase_bins
            count.view(-1).index_add_(
                0, initial_flat, torch.ones_like(episode_tasks, dtype=torch.float64)
            )
            phase_sum.view(-1, config.phase_dim).index_add_(
                0, initial_flat, output.initial_phase_token.to(torch.float64)
            )
    for value in (count, progress_sum, task_episode_count, task_episode_sum, phase_sum, value_sum):
        _reduce(value)
    if accelerator.is_main_process:
        divisor = count.clamp_min(1.0)
        task_prototype = F.normalize(
            (task_episode_sum / task_episode_count.clamp_min(1.0).unsqueeze(-1)).float(), dim=-1
        )
        phase_key = _fill(F.normalize((phase_sum / divisor[..., None]).float(), dim=-1), count)
        causal_value = _fill(F.normalize((value_sum / divisor[..., None]).float(), dim=-1), count)
        manifest = {
            "stage1_checkpoint": str(Path(args.zte_checkpoint).resolve()),
            "stage1_checkpoint_sha256": sha256(args.zte_checkpoint),
            "stage1_step": int(checkpoint["step"]),
            "stage1_manifest": checkpoint["manifest"],
            "statistics_sha256": sha256(handoff.statistics),
            "goal_embeddings_sha256": sha256(args.goal_embeddings),
            "dataset_root": str(Path(args.dataset_root).resolve()),
            "causal_transition_horizon": args.transition_horizon,
            "recurrent_history": "complete-consecutive-H5-rollout-from-B0",
            "episodes": len(dataset),
            "transitions": int(count.sum()),
            "source_sha256": {"exporter": sha256(Path(__file__).resolve()),
                              "transition_encoder": sha256(Path(inspect.getfile(CausalTransitionEncoder)).resolve())},
        }
        payload: dict[str, Any] = {
            "schema": LIBERO_CAUSAL_BANK_SCHEMA,
            "split": "official-train-1614",
            "task_names": dataset.task_names,
            "phase_bins": args.phase_bins,
            "count": count.cpu().long(),
            "progress": (progress_sum / divisor).cpu().float(),
            "task_prototype": task_prototype.cpu(),
            "phase_key": phase_key.cpu(),
            "causal_value": causal_value.cpu(),
            "manifest": manifest,
        }
        output = Path(args.output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, output)
        output.with_suffix(".json").write_text(json.dumps({"schema": payload["schema"], "task_count": len(dataset.task_names),
                                                            "transitions": int(count.sum()), "manifest": manifest}, indent=2) + "\n")
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main(tyro.cli(Args))
