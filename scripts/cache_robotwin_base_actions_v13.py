"""Cache untouched PI0.5 H50 actions for v13's direct output-correction training."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

from accelerate import Accelerator
from accelerate import DataLoaderConfiguration
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import tqdm
import tyro

from openpi.zeva.robotwin_policy import RobotWinZevaPolicy
from scripts.train_robotwin_stage2 import RobotWinStage2Dataset
from scripts.train_robotwin_stage2 import _load_task_subset
from scripts.train_robotwin_stage2 import _preprocess_with_task_only_goal


@dataclasses.dataclass
class Args:
    handoff_root: str = "/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1"
    foundation_checkpoint: str = "/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1/checkpoint/pretrained_model-best-v1"
    dataset_root: str = "/data1/dingxin/robotwin-lerobot-sidney-eef16-v1/data"
    goal_embedding_checkpoint: str = "/mnt/100T/users/dingxin/VLA/runtime/pretrained_model-stage1-language-v1"
    zte_checkpoint: str = "/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage1-artifacts-v1/stage1-zte/zte_best.pth"
    live_queries: str = "/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage1-artifacts-v1/stage1-zte/live_queries_h15.pt"
    task_subset: str = "configs/robotwin_zeva_advantage10.json"
    output: str = "/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/advantage10-output-correction-v13/base_action_cache.pt"
    train_samples: int = 8_192
    validation_samples: int = 4_096
    batch_size: int = 32
    num_workers: int = 4
    seed: int = 1000


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _selected_indices(dataset: RobotWinStage2Dataset, limit: int, seed: int) -> list[int]:
    grouped: dict[str, list[int]] = {name: [] for name in dataset.selected_task_names}
    for sample_index, (record_index, _, _) in enumerate(dataset._samples):  # noqa: SLF001
        record = dataset.dataset._records[record_index]  # noqa: SLF001
        grouped[record["key"][1]].append(sample_index)
    rng = np.random.default_rng(seed)
    per_task = max(1, limit // len(grouped))
    selected = []
    for name in dataset.selected_task_names:
        values = np.asarray(grouped[name], dtype=np.int64)
        count = min(per_task, len(values))
        selected.extend(rng.choice(values, size=count, replace=False).tolist())
    rng.shuffle(selected)
    return selected[:limit]


def _cache_split(
    split: str,
    limit: int,
    policy: RobotWinZevaPolicy,
    args: Args,
    accelerator: Accelerator,
    shard_root: Path,
) -> tuple[str, ...]:
    dataset = RobotWinStage2Dataset(
        Path(args.dataset_root) / "adapter.json",
        args.live_queries,
        subset=split,
        config=policy.zeva_config,
        selected_tasks=_load_task_subset(args.task_subset),
        video_backend="torchcodec",
    )
    indices = _selected_indices(dataset, min(limit, len(dataset)), args.seed + (split == "validation"))
    loader = accelerator.prepare(
        DataLoader(
            Subset(dataset, indices),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )
    )
    cached_indices = []
    cached_actions = []
    progress = tqdm.tqdm(loader, desc=f"base-cache-{split}", disable=not accelerator.is_local_main_process)
    with torch.inference_mode():
        for raw_batch in progress:
            sample_indices = raw_batch.pop("zeva.sample_index")
            for key in (
                "zeva.task_id",
                "zeva.phase_query",
                "zeva.live_brief",
                "zeva.live_brief_mask",
                "zeva.live_retrieved",
                "zeva.live_retrieved_mask",
            ):
                raw_batch.pop(key)
            processed = _preprocess_with_task_only_goal(policy, policy.preprocessor, raw_batch)
            actions = policy.foundation.predict_action_chunk(processed)
            cached_indices.append(sample_indices.cpu().to(torch.int32))
            cached_actions.append(actions.cpu().to(torch.float16))
    torch.save(
        {
            "sample_indices": (
                torch.cat(cached_indices)
                if cached_indices
                else torch.empty(0, dtype=torch.int32)
            ),
            "base_actions": (
                torch.cat(cached_actions)
                if cached_actions
                else torch.empty(0, 50, 16, dtype=torch.float16)
            ),
        },
        shard_root / f"{split}-rank{accelerator.process_index:02d}.pt",
    )
    return dataset.task_names


def main(args: Args) -> None:
    accelerator = Accelerator(dataloader_config=DataLoaderConfiguration(even_batches=False))
    torch.manual_seed(args.seed + accelerator.process_index)
    torch.cuda.manual_seed_all(args.seed + accelerator.process_index)
    policy = RobotWinZevaPolicy.from_handoff(
        args.handoff_root,
        device=str(accelerator.device),
        foundation_checkpoint=args.foundation_checkpoint,
        goal_embedding_checkpoint=args.goal_embedding_checkpoint,
        zte_checkpoint=args.zte_checkpoint,
    )
    policy.requires_grad_(False).eval()
    output = Path(args.output)
    shard_root = output.parent / f".{output.stem}-shards"
    if accelerator.is_main_process:
        shard_root.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    task_names = None
    limits = {"train": args.train_samples, "validation": args.validation_samples}
    for split, limit in limits.items():
        task_names = _cache_split(split, limit, policy, args, accelerator, shard_root)
        accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        splits = {}
        for split in limits:
            shards = [
                torch.load(shard_root / f"{split}-rank{rank:02d}.pt", map_location="cpu")
                for rank in range(accelerator.num_processes)
            ]
            indices = torch.cat([item["sample_indices"] for item in shards])
            actions = torch.cat([item["base_actions"] for item in shards])
            order = indices.argsort()
            indices, actions = indices[order], actions[order]
            if len(indices) != len(indices.unique()):
                raise RuntimeError(f"{split} Base cache contains duplicate sample indices.")
            splits[split] = {"sample_indices": indices, "base_actions": actions}
        payload = {
            "schema": "zeva-robotwin-untouched-base-action-cache-v1",
            "foundation_checkpoint": str(Path(args.foundation_checkpoint).resolve()),
            "foundation_model_sha256": _sha256(Path(args.foundation_checkpoint) / "model.safetensors"),
            "dataset_adapter": str((Path(args.dataset_root) / "adapter.json").resolve()),
            "dataset_adapter_sha256": _sha256(Path(args.dataset_root) / "adapter.json"),
            "live_queries_sha256": _sha256(args.live_queries),
            "task_subset_sha256": _sha256(args.task_subset),
            "task_names": list(task_names or ()),
            "selected_task_names": list(_load_task_subset(args.task_subset) or ()),
            "model_rng": "continuous_per_rank_seed_1000_plus_rank",
            "splits": splits,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, output)
        output.with_suffix(".json").write_text(
            json.dumps(
                {
                    key: value
                    for key, value in payload.items()
                    if key != "splits"
                }
                | {"counts": {name: len(value["sample_indices"]) for name, value in splits.items()}},
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main(tyro.cli(Args))
