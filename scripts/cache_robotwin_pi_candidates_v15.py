"""Cache K independent frozen-PI0.5 H50 candidates for v15 oracle analysis.

This script intentionally does not train or rank candidates.  Candidate zero
is the ordinary Base draw; candidates 1..K-1 are consecutive independent
diffusion draws from the same untouched PI0.5 checkpoint.  The cache contains
the normalized H50 expert target so the accompanying probe can measure the
oracle upper bound without decoding videos or loading a ZeVA corrector.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

from accelerate import Accelerator
from accelerate import DataLoaderConfiguration
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import tqdm
import tyro

from openpi.zeva.robotwin_policy import RobotWinZevaPolicy
from openpi.zeva.robotwin_policy import robotwin_multiview_image
from scripts.train_robotwin_stage2 import RobotWinStage2Dataset
from scripts.train_robotwin_stage2 import _load_task_subset
from scripts.train_robotwin_stage2 import _preprocess_with_task_only_goal


@dataclasses.dataclass
class Args:
    handoff_root: str = (
        "/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/"
        "robotwin-memory-baseline-v1"
    )
    foundation_checkpoint: str = (
        "/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/"
        "robotwin-memory-baseline-v1/checkpoint/pretrained_model-best-v1"
    )
    dataset_root: str = "/data1/dingxin/robotwin-lerobot-sidney-eef16-v1/data"
    goal_embedding_checkpoint: str = (
        "/mnt/100T/users/dingxin/VLA/runtime/pretrained_model-stage1-language-v1"
    )
    zte_checkpoint: str = (
        "/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/"
        "stage1-artifacts-v1/stage1-zte/zte_best.pth"
    )
    live_queries: str = (
        "/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/"
        "stage1-artifacts-v1/stage1-zte/live_queries_h15.pt"
    )
    task_subset: str = "configs/robotwin_zeva_advantage10.json"
    output: str = (
        "/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/"
        "advantage10-pi-candidates-v15/pi_candidates_k4.pt"
    )
    k_candidates: int = 4
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


def _file_identity(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _foundation_identity(path: str | Path) -> dict[str, Any]:
    root = Path(path).resolve()
    model = _file_identity(root / "model.safetensors")
    return {
        "checkpoint": str(root),
        "model": model,
        "config": _file_identity(root / "config.json"),
        "tokenizer": _file_identity(root / "tokenizer" / "tokenizer.json"),
    }


def _selected_indices(dataset: RobotWinStage2Dataset, limit: int, seed: int) -> list[int]:
    grouped: dict[str, list[int]] = {name: [] for name in dataset.selected_task_names}
    for sample_index, (record_index, _, _) in enumerate(dataset._samples):  # noqa: SLF001
        record = dataset.dataset._records[record_index]  # noqa: SLF001
        grouped[record["key"][1]].append(sample_index)
    rng = np.random.default_rng(seed)
    per_task = max(1, limit // len(grouped))
    selected: list[int] = []
    for name in dataset.selected_task_names:
        values = np.asarray(grouped[name], dtype=np.int64)
        count = min(per_task, len(values))
        selected.extend(rng.choice(values, size=count, replace=False).tolist())
    rng.shuffle(selected)
    return selected[:limit]


def _normalized_expert_target(
    policy: RobotWinZevaPolicy, raw_actions: torch.Tensor
) -> torch.Tensor:
    """Normalize raw EEF16 expert actions with the handoff's frozen statistics."""
    normalized = policy.action_normalizer.normalize(raw_actions)
    if normalized.ndim != 3 or normalized.shape[-1] != 16:
        raise ValueError(
            "Expected normalized H50 EEF16 expert actions, got "
            f"{tuple(normalized.shape)}."
        )
    return normalized


def _cache_split(
    split: str,
    limit: int,
    policy: RobotWinZevaPolicy,
    args: Args,
    accelerator: Accelerator,
    shard_root: Path,
) -> tuple[tuple[str, ...], dict[str, torch.Tensor]]:
    dataset = RobotWinStage2Dataset(
        Path(args.dataset_root) / "adapter.json",
        args.live_queries,
        subset=split,
        config=policy.zeva_config,
        selected_tasks=_load_task_subset(args.task_subset),
        video_backend="torchcodec",
    )
    indices = _selected_indices(
        dataset,
        min(limit, len(dataset)),
        args.seed + int(split == "validation"),
    )
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
    cached_indices: list[torch.Tensor] = []
    cached_task_ids: list[torch.Tensor] = []
    cached_candidates: list[torch.Tensor] = []
    cached_targets: list[torch.Tensor] = []
    cached_phase_queries: list[torch.Tensor] = []
    cached_visual_features: list[torch.Tensor] = []
    cached_pi_vlm_features: list[torch.Tensor] = []
    progress = tqdm.tqdm(
        loader,
        desc=f"pi-candidates-{split}-k{args.k_candidates}",
        disable=not accelerator.is_local_main_process,
    )
    with torch.inference_mode():
        for raw_batch in progress:
            sample_indices = raw_batch.pop("zeva.sample_index")
            task_ids = raw_batch.pop("zeva.task_id")
            phase_queries = raw_batch.pop("zeva.phase_query")
            for key in (
                "zeva.live_brief",
                "zeva.live_brief_mask",
                "zeva.live_retrieved",
                "zeva.live_retrieved_mask",
            ):
                raw_batch.pop(key)
            raw_actions = raw_batch["action"]
            target = _normalized_expert_target(policy, raw_actions)
            # Preserve the fine-grained current-state geometry that the
            # normalized phase head can discard.  This is the frozen Stage1
            # EMA visual encoder, not a new trainable VLM or an oracle feature.
            cte = policy.causal_transition_encoder
            current_images = robotwin_multiview_image(raw_batch)
            current_images = cte._normalize_images(cte._ensure_bchw(current_images))  # noqa: SLF001
            current_images = cte._split_and_resize_views(current_images)  # noqa: SLF001
            visual_features = cte.target_vision_encoder(current_images)
            processed = _preprocess_with_task_only_goal(
                policy, policy.preprocessor, raw_batch
            )
            # The critic must judge samples in the same observation space that
            # produced them.  Cache the untouched PI0.5 VLM EOS state instead
            # of asking the lightweight ZTE phase bottleneck to retain all
            # object/EEF geometry.
            pi_vlm_features = policy.extract_vlm_features(processed)
            # Do not reseed between calls: these are consecutive independent
            # diffusion draws from the frozen policy's normal RNG stream.
            candidates = [policy.foundation.predict_action_chunk(processed)]
            for _ in range(args.k_candidates - 1):
                candidates.append(policy.foundation.predict_action_chunk(processed))
            candidates_tensor = torch.stack(candidates, dim=1)
            if candidates_tensor.shape != (
                target.shape[0],
                args.k_candidates,
                50,
                16,
            ):
                raise ValueError(
                    "PI candidate contract drifted: "
                    f"candidates={tuple(candidates_tensor.shape)}, target={tuple(target.shape)}."
                )
            cached_indices.append(sample_indices.cpu().to(torch.int32))
            cached_task_ids.append(task_ids.cpu().to(torch.int16))
            cached_candidates.append(candidates_tensor.cpu().to(torch.float16))
            cached_targets.append(target.cpu().to(torch.float16))
            cached_phase_queries.append(phase_queries.cpu().to(torch.float16))
            cached_visual_features.append(visual_features.cpu().to(torch.float16))
            cached_pi_vlm_features.append(pi_vlm_features.cpu().to(torch.float16))
    shard = {
        "sample_indices": (
            torch.cat(cached_indices)
            if cached_indices
            else torch.empty(0, dtype=torch.int32)
        ),
        "task_ids": (
            torch.cat(cached_task_ids)
            if cached_task_ids
            else torch.empty(0, dtype=torch.int16)
        ),
        "candidate_actions": (
            torch.cat(cached_candidates)
            if cached_candidates
            else torch.empty(0, args.k_candidates, 50, 16, dtype=torch.float16)
        ),
        "normalized_expert_actions": (
            torch.cat(cached_targets)
            if cached_targets
            else torch.empty(0, 50, 16, dtype=torch.float16)
        ),
        "phase_queries": (
            torch.cat(cached_phase_queries)
            if cached_phase_queries
            else torch.empty(0, policy.zeva_config.phase_dim, dtype=torch.float16)
        ),
        "current_visual_features": (
            torch.cat(cached_visual_features)
            if cached_visual_features
            else torch.empty(0, policy.zeva_config.model_dim, dtype=torch.float16)
        ),
        "current_pi_vlm_features": (
            torch.cat(cached_pi_vlm_features)
            if cached_pi_vlm_features
            else torch.empty(0, 2048, dtype=torch.float16)
        ),
    }
    torch.save(shard, shard_root / f"{split}-rank{accelerator.process_index:02d}.pt")
    return dataset.task_names, shard


def _merge_split(
    split: str,
    world_size: int,
    shard_root: Path,
) -> dict[str, torch.Tensor]:
    shards = [
        torch.load(
            shard_root / f"{split}-rank{rank:02d}.pt",
            map_location="cpu",
            weights_only=False,
        )
        for rank in range(world_size)
    ]
    indices = torch.cat([item["sample_indices"] for item in shards])
    task_ids = torch.cat([item["task_ids"] for item in shards])
    candidates = torch.cat([item["candidate_actions"] for item in shards])
    targets = torch.cat([item["normalized_expert_actions"] for item in shards])
    phase_queries = torch.cat([item["phase_queries"] for item in shards])
    visual_features = torch.cat([item["current_visual_features"] for item in shards])
    pi_vlm_features = torch.cat([item["current_pi_vlm_features"] for item in shards])
    order = indices.argsort()
    indices, task_ids = indices[order], task_ids[order]
    candidates, targets = candidates[order], targets[order]
    phase_queries, visual_features = phase_queries[order], visual_features[order]
    pi_vlm_features = pi_vlm_features[order]
    if len(indices) != len(indices.unique()):
        raise RuntimeError(f"{split} candidate cache contains duplicate sample indices.")
    if candidates.shape[0] != targets.shape[0] or task_ids.shape[0] != indices.shape[0]:
        raise RuntimeError(f"{split} candidate cache fields have inconsistent lengths.")
    return {
        "sample_indices": indices,
        "task_ids": task_ids,
        "candidate_actions": candidates,
        "normalized_expert_actions": targets,
        "phase_queries": phase_queries,
        "current_visual_features": visual_features,
        "current_pi_vlm_features": pi_vlm_features,
    }


def main(args: Args) -> None:
    if args.k_candidates != 4:
        raise ValueError("v15 oracle cache is defined for exactly K=4 candidates.")
    accelerator = Accelerator(
        dataloader_config=DataLoaderConfiguration(even_batches=False)
    )
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
    output = Path(args.output).resolve()
    shard_root = output.parent / f".{output.stem}-shards"
    if accelerator.is_main_process:
        shard_root.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    task_names: tuple[str, ...] | None = None
    limits = {"train": args.train_samples, "validation": args.validation_samples}
    for split, limit in limits.items():
        task_names, _ = _cache_split(split, limit, policy, args, accelerator, shard_root)
        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        splits = {
            split: _merge_split(split, accelerator.num_processes, shard_root)
            for split in limits
        }
        foundation_identity = _foundation_identity(args.foundation_checkpoint)
        dataset_identity = _file_identity(Path(args.dataset_root) / "adapter.json")
        live_identity = _file_identity(args.live_queries)
        zte_identity = _file_identity(args.zte_checkpoint)
        task_subset_identity = _file_identity(args.task_subset)
        live_payload = torch.load(args.live_queries, map_location="cpu", weights_only=False)
        payload = {
            "schema": "zeva-robotwin-pi-candidate-cache-v15",
            "schema_version": 3,
            "candidate_count": args.k_candidates,
            "candidate_contract": {
                "candidate_0": "untouched_PI0.5_Base_draw",
                "candidate_1_to_k_minus_1": "consecutive_independent_frozen_diffusion_draws",
                "shape": [args.k_candidates, 50, 16],
                "domain": "normalized_relative_eef16",
                "selection": "offline_oracle_only_no_ranker_or_deployment_logic",
                "conditioning": (
                    "frozen_pi_vlm_state_plus_stage1_task_phase_causal_and_ema_visual"
                ),
            },
            "foundation": foundation_identity,
            "goal_embedding_checkpoint": _foundation_identity(args.goal_embedding_checkpoint),
            "dataset_adapter": dataset_identity,
            "dataset_root": str(Path(args.dataset_root).resolve()),
            "live_queries": live_identity,
            "live_queries_schema": live_payload.get("schema"),
            "live_queries_zte_checkpoint_sha256": live_payload.get(
                "zte_checkpoint_sha256"
            ),
            "zte_checkpoint": zte_identity,
            "task_subset": task_subset_identity,
            "task_names": list(task_names or ()),
            "selected_task_names": list(_load_task_subset(args.task_subset) or ()),
            "model_rng": {
                "initial_seed": args.seed,
                "per_rank_seed": "seed_plus_accelerator_process_index",
                "draw_schedule": "candidate0_then_candidate1_to_k_minus_1_without_reseeding",
            },
            "sampling": {
                "train_limit": args.train_samples,
                "validation_limit": args.validation_samples,
                "selection": "stratified_per_task_numpy_default_rng",
                "selection_seed_train": args.seed,
                "selection_seed_validation": args.seed + 1,
            },
            "splits": splits,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, output)
        output.with_suffix(".json").write_text(
            json.dumps(
                {
                    key: value for key, value in payload.items() if key != "splits"
                }
                | {
                    "counts": {
                        name: int(value["sample_indices"].numel())
                        for name, value in splits.items()
                    },
                    "split_fields": {
                        name: sorted(value.keys()) for name, value in splits.items()
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main(tyro.cli(Args))
