"""Export deployment-equivalent recurrent ZTE phase queries at every H15 decision."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

from accelerate import Accelerator
from accelerate import DataLoaderConfiguration
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import tqdm
import tyro

from openpi.zeva.config import ZevaConfig
from openpi.zeva.robotwin_contract import MeanStdActionNormalizer
from openpi.zeva.robotwin_contract import RobotWinHandoff
from openpi.zeva.robotwin_policy import robotwin_multiview_image
from openpi.zeva.transition_encoder import CausalTransitionEncoder
try:
    from scripts.train_robotwin_zte import EpisodeGoalEmbeddingTable
    from scripts.train_robotwin_zte import FFmpegRoboTwinDataset
except ModuleNotFoundError:
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
    output: str = "/data1/dingxin/zeva-runs/robotwin-v5-h15-tasklang/stage1-zte/live_queries_h15.pt"
    transition_horizon: int = 15
    num_workers: int = 4


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class EpisodeQueryDataset(Dataset):
    def __init__(self, adapter_manifest: Path, goal_embeddings: str, subset: str):
        self.source = FFmpegRoboTwinDataset(adapter_manifest, subset=subset)
        self.dataset = self.source.dataset
        self.goals = EpisodeGoalEmbeddingTable(
            goal_embeddings, subset, self.dataset._records  # noqa: SLF001
        )
        names = sorted({record["key"][1] for record in self.dataset._records})  # noqa: SLF001
        self.task_names = tuple(names)
        self.task_ids = {name: index for index, name in enumerate(names)}

    def __len__(self) -> int:
        return len(self.dataset._records)  # noqa: SLF001

    def __getitem__(self, record_index: int) -> dict[str, torch.Tensor]:
        record = self.dataset._records[record_index]  # noqa: SLF001
        start = int(self.dataset._cumulative[record_index])  # noqa: SLF001
        length = int(record["length"])
        before_frames = list(range(0, length - 15, 15))
        if not before_frames:
            raise ValueError(f"Episode {record['key']} is shorter than H15.")
        after_frames = [frame + 15 for frame in before_frames]
        images = self.source.read_images(record, before_frames + after_frames)
        sequence_length = len(before_frames)
        before = {key: value[:sequence_length] for key, value in images.items()}
        after = {key: value[sequence_length:] for key, value in images.items()}
        actions = torch.stack([self.dataset[start + frame]["action"][:15] for frame in before_frames])
        return {
            "record_index": torch.tensor(record_index),
            "task_id": torch.tensor(self.task_ids[record["key"][1]]),
            "decision_frames": torch.tensor([0, *after_frames]),
            "images_before": robotwin_multiview_image(before),
            "images_after": robotwin_multiview_image(after),
            "actions": actions,
            "goal_embedding": self.goals.embeddings[record_index],
        }


def _export_split(
    subset: str,
    model: CausalTransitionEncoder,
    normalizer: MeanStdActionNormalizer,
    args: Args,
    accelerator: Accelerator,
    shard_dir: Path,
) -> tuple[str, ...]:
    dataset = EpisodeQueryDataset(Path(args.dataset_root) / "adapter.json", args.goal_embeddings, subset)
    loader = accelerator.prepare(
        DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )
    )
    records = []
    progress = tqdm.tqdm(loader, desc=f"live-{subset}", disable=not accelerator.is_local_main_process)
    with torch.inference_mode():
        for batch in progress:
            outputs = model(
                batch["images_before"],
                normalizer.normalize(batch["actions"]),
                batch["images_after"],
                batch["goal_embedding"],
            )
            phases = torch.cat([outputs.initial_phase_token[:, None], outputs.phase_token], dim=1)
            records.append(
                {
                    "record_index": int(batch["record_index"].item()),
                    "task_id": int(batch["task_id"].item()),
                    "decision_frames": batch["decision_frames"][0].cpu().to(torch.int32),
                    "phase_queries": phases[0].cpu().to(torch.float16),
                    "causal_signals": outputs.causal_signal[0].cpu().to(torch.float16),
                }
            )
    torch.save(
        {"task_names": dataset.task_names, "records": records},
        shard_dir / f"{subset}-rank{accelerator.process_index:02d}.pt",
    )
    return dataset.task_names


def main(args: Args) -> None:
    accelerator = Accelerator(dataloader_config=DataLoaderConfiguration(even_batches=False))
    handoff = RobotWinHandoff.from_root(args.handoff_root)
    checkpoint = torch.load(args.zte_checkpoint, map_location="cpu")
    if checkpoint.get("schema") != "zeva-robotwin-zte-stage1-checkpoint-v5":
        raise ValueError("Live query export requires the passed Stage 1 v5 checkpoint.")
    config = ZevaConfig(**checkpoint["zte_config"])
    model = CausalTransitionEncoder(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.requires_grad_(False).eval().to(accelerator.device)
    normalizer = MeanStdActionNormalizer.from_stats_file(handoff.statistics)
    output = Path(args.output)
    shard_dir = output.parent / f".{output.stem}-shards"
    if accelerator.is_main_process:
        shard_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    task_names = None
    for subset in ("train", "validation"):
        task_names = _export_split(subset, model, normalizer, args, accelerator, shard_dir)
        accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        splits = {}
        for subset in ("train", "validation"):
            shards = [
                torch.load(shard_dir / f"{subset}-rank{rank:02d}.pt", map_location="cpu")
                for rank in range(accelerator.num_processes)
            ]
            if {tuple(item["task_names"]) for item in shards} != {tuple(task_names)}:
                raise RuntimeError("Live-query shard task ordering differs.")
            records = sorted(
                [record for shard in shards for record in shard["records"]],
                key=lambda item: item["record_index"],
            )
            if [item["record_index"] for item in records] != list(range(len(records))):
                raise RuntimeError(f"{subset} live-query cache missed or duplicated records.")
            splits[subset] = records
        payload = {
            "schema": "zeva-robotwin-live-queries-h15-v1",
            "transition_horizon": args.transition_horizon,
            "task_names": task_names,
            "zte_checkpoint_sha256": _sha256(args.zte_checkpoint),
            "statistics_sha256": _sha256(handoff.statistics),
            "splits": splits,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, output)
        output.with_suffix(".json").write_text(
            json.dumps(
                {
                    "schema": payload["schema"],
                    "transition_horizon": args.transition_horizon,
                    "records": {key: len(value) for key, value in splits.items()},
                    "zte_checkpoint_sha256": payload["zte_checkpoint_sha256"],
                    "statistics_sha256": payload["statistics_sha256"],
                },
                indent=2,
            )
            + "\n"
        )
        print(f"saved={output}")
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main(tyro.cli(Args))
