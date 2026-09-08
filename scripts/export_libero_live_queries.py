"""Export deployment-recurrent LIBERO ZTE phase queries at every H5 boundary."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from accelerate import Accelerator, DataLoaderConfiguration
import torch
from torch.utils.data import DataLoader, Dataset
import tqdm
import tyro

from openpi.zeva.config import ZevaConfig
from openpi.zeva.libero_contract import LIBERO_EXECUTION_HORIZON, LiberoHandoff
from openpi.zeva.libero_contract import QuantileActionNormalizer, sha256
from openpi.zeva.libero_data import LiberoZTEEpisodeDataset
from openpi.zeva.transition_encoder import CausalTransitionEncoder


@dataclasses.dataclass
class Args:
    handoff_root: str = "/data1/dingxin/libero-memory-baseline-v1"
    dataset_root: str = "/data1/dingxin/libero-memory-baseline-v1/data"
    goal_embeddings: str = "/data1/dingxin/zeva-runs/libero-v1-h5-lang/pi05_goal_embeddings.pt"
    zte_checkpoint: str = "/data1/dingxin/zeva-runs/libero-v3-h5-lang/stage1-zte/zte_best.pth"
    output: str = "/data1/dingxin/zeva-runs/libero-v3-h5-lang/stage1-zte/live_queries_h5.pt"
    transition_horizon: int = LIBERO_EXECUTION_HORIZON
    num_workers: int = 4


class EpisodeQueryDataset(Dataset):
    def __init__(self, args: Args, subset: str):
        self.source = LiberoZTEEpisodeDataset(
            args.dataset_root,
            subset=subset,
            goal_embeddings=args.goal_embeddings,
            transition_horizon=args.transition_horizon,
        )
        self.task_names = self.source.task_names

    def __len__(self):
        return len(self.source)

    def __getitem__(self, record_index):
        result = self.source[record_index]
        length = int(self.source.table.episodes[record_index]["length"])
        transition_frames = list(
            range(0, length - self.source.transition_horizon, self.source.transition_horizon)
        )
        result["record_index"] = torch.tensor(record_index)
        result["decision_frames"] = torch.tensor(
            [0, *[frame + self.source.transition_horizon for frame in transition_frames]],
            dtype=torch.int32,
        )
        return result


def _export_split(subset, model, normalizer, args, accelerator, shard_dir):
    dataset = EpisodeQueryDataset(args, subset)
    loader = accelerator.prepare(DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    ))
    records = []
    with torch.inference_mode():
        for batch in tqdm.tqdm(loader, desc=f"live-{subset}", disable=not accelerator.is_local_main_process):
            output = model(
                batch["images_before"],
                normalizer.normalize(batch["actions"]),
                batch["images_after"],
                batch["goal_embedding"],
            )
            records.append({
                "record_index": int(batch["record_index"].item()),
                "task_id": int(batch["task_id"].item()),
                "decision_frames": batch["decision_frames"][0].cpu(),
                "phase_queries": torch.cat(
                    (output.initial_phase_token[:, None], output.phase_token), dim=1
                )[0].cpu().half(),
                "causal_signals": output.causal_signal[0].cpu().half(),
            })
    torch.save(
        {"task_names": dataset.task_names, "records": records},
        shard_dir / f"{subset}-rank{accelerator.process_index:02d}.pt",
    )
    return dataset.task_names


def main(args: Args):
    if args.transition_horizon != LIBERO_EXECUTION_HORIZON:
        raise ValueError("Formal LIBERO live queries require H5.")
    accelerator = Accelerator(dataloader_config=DataLoaderConfiguration(even_batches=False))
    handoff = LiberoHandoff.from_root(args.handoff_root)
    checkpoint = torch.load(args.zte_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "zeva-libero-zte-stage1-checkpoint-v3":
        raise ValueError("LIBERO live queries require Stage 1 v3.")
    model = CausalTransitionEncoder(ZevaConfig(**checkpoint["zte_config"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.requires_grad_(False).eval().to(accelerator.device)
    normalizer = QuantileActionNormalizer.from_stats_file(handoff.statistics)
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
            shards = [torch.load(
                shard_dir / f"{subset}-rank{rank:02d}.pt", map_location="cpu", weights_only=False
            ) for rank in range(accelerator.num_processes)]
            if {tuple(item["task_names"]) for item in shards} != {tuple(task_names)}:
                raise RuntimeError("LIBERO live-query shard task ordering differs.")
            records = sorted(
                (record for shard in shards for record in shard["records"]),
                key=lambda item: item["record_index"],
            )
            if [item["record_index"] for item in records] != list(range(len(records))):
                raise RuntimeError(f"{subset} live-query cache missed or duplicated episodes.")
            splits[subset] = records
        payload = {
            "schema": "zeva-libero-live-queries-h5-v1",
            "transition_horizon": args.transition_horizon,
            "task_names": task_names,
            "zte_checkpoint_sha256": sha256(args.zte_checkpoint),
            "statistics_sha256": sha256(handoff.statistics),
            "goal_embeddings_sha256": sha256(args.goal_embeddings),
            "splits": splits,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, output)
        output.with_suffix(".json").write_text(json.dumps({
            "schema": payload["schema"],
            "transition_horizon": args.transition_horizon,
            "records": {key: len(value) for key, value in splits.items()},
            "zte_checkpoint_sha256": payload["zte_checkpoint_sha256"],
            "statistics_sha256": payload["statistics_sha256"],
            "goal_embeddings_sha256": payload["goal_embeddings_sha256"],
        }, indent=2) + "\n")
        print(f"saved={output}")
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main(tyro.cli(Args))
