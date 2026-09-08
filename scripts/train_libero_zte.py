"""Stage 1: train language-conditioned ZeVA ZTE on official LIBERO episodes."""

from __future__ import annotations

import dataclasses
import inspect
import json
import math
from pathlib import Path

from accelerate import Accelerator
import torch
from torch import nn
from torch.utils.data import DataLoader
import tqdm
import tyro

from openpi.zeva.config import ZevaConfig
from openpi.zeva.libero_contract import LIBERO_ACTION_DIM
from openpi.zeva.libero_contract import LIBERO_CHECKPOINT_SHA256
from openpi.zeva.libero_contract import LIBERO_CAMERA_KEYS
from openpi.zeva.libero_contract import LIBERO_EXECUTION_HORIZON
from openpi.zeva.libero_contract import LIBERO_POLICY_HORIZON
from openpi.zeva.libero_contract import LiberoHandoff
from openpi.zeva.libero_contract import QuantileActionNormalizer
from openpi.zeva.libero_contract import sha256
from openpi.zeva.libero_data import LiberoZTEEpisodeDataset
from openpi.zeva.transition_encoder import CausalTransitionEncoder
from scripts.train_robotwin_zte import compute_losses
from scripts.train_robotwin_zte import TaskPairedDistributedSampler


@dataclasses.dataclass
class Args:
    handoff_root: str = "/data1/dingxin/libero-memory-baseline-v1"
    dataset_root: str = "/data1/dingxin/libero-memory-baseline-v1/data"
    goal_embeddings: str = "/data1/dingxin/zeva-runs/libero-v1-h5-lang/pi05_goal_embeddings.pt"
    save_dir: str = "/data1/dingxin/zeva-runs/libero-v3-h5-lang/stage1-zte"
    resume_checkpoint: str | None = None
    steps: int = 30_000
    batch_size: int = 1
    num_workers: int = 4
    transition_stride: int = LIBERO_EXECUTION_HORIZON
    effect_steps: int = LIBERO_EXECUTION_HORIZON
    executed_action_steps: int = LIBERO_EXECUTION_HORIZON
    learning_rate: float = 1e-4
    vision_learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    warmup_steps: int = 1_000
    action_loss_weight: float = 0.1
    effect_loss_weight: float = 0.2
    task_loss_weight: float = 2.0
    task_prototype_loss_weight: float = 0.1
    phase_loss_weight: float = 1.0
    phase_key_loss_weight: float = 1.0
    monotonic_loss_weight: float = 0.2
    contrastive_temperature: float = 0.1
    monotonic_margin: float = 0.01
    save_freq: int = 1_000
    eval_batches: int = 0
    log_freq: int = 10
    seed: int = 1000


def _manifest(args: Args, handoff: LiberoHandoff, config: ZevaConfig, task_count: int) -> dict:
    return {
        "schema": "zeva-libero-zte-stage1-v3",
        "handoff_root": str(handoff.root),
        "handoff_contract": str(handoff.contract),
        "foundation_checkpoint": str(handoff.checkpoint / "model.safetensors"),
        "foundation_checkpoint_sha256": LIBERO_CHECKPOINT_SHA256,
        "statistics": str(handoff.statistics),
        "statistics_sha256": sha256(handoff.statistics),
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "dataset_subset": "official-train-1614",
        "goal_embeddings": str(Path(args.goal_embeddings).resolve()),
        "goal_embeddings_sha256": sha256(args.goal_embeddings),
        "initial_state": "F_init(pi05-task-only-embedding-g,initial-two-view-visual-s0)",
        "goal_embedding_policy": "frozen-foundation-table-across-stage2",
        "vision_initialization": "torchvision-resnet18-imagenet1k-v1",
        "camera_order": list(LIBERO_CAMERA_KEYS),
        "source_image_size": 256,
        "model_image_size": 224,
        "state": "single-arm-eef-in-fixed-agent-camera-frame-padded-to-16d",
        "action": "chunk-start-relative-eef16-official-gripper-command",
        "action_dim": LIBERO_ACTION_DIM,
        "policy_action_horizon": LIBERO_POLICY_HORIZON,
        "causal_transition_horizon": LIBERO_EXECUTION_HORIZON,
        "phase_supervision": "strict-monotonic-positive-hazard-plus-rbf-phase-key",
        "temporal_sampling": "complete-episode-at-replan-boundaries-task-paired-ddp",
        "normalization": "frozen-quantile",
        "task_count": task_count,
        "split": {"train_episodes": 1614, "validation_episodes": 79},
        "zte_config": dataclasses.asdict(config),
        "train_args": dataclasses.asdict(args),
        "source_sha256": {
            "trainer": sha256(Path(__file__).resolve()),
            "transition_encoder": sha256(Path(inspect.getfile(CausalTransitionEncoder)).resolve()),
            "losses_and_sampler": sha256(Path(inspect.getfile(compute_losses)).resolve()),
            "config": sha256(Path(inspect.getfile(ZevaConfig)).resolve()),
        },
    }


@torch.no_grad()
def evaluate(model, loader, normalizer, args, accelerator) -> float:
    model.eval()
    totals = []
    for batch_index, batch in enumerate(loader):
        if args.eval_batches > 0 and batch_index >= args.eval_batches:
            break
        outputs = model(
            batch["images_before"],
            normalizer.normalize(batch["actions"]),
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
        raise ValueError("Complete LIBERO episode training requires batch size one per GPU.")
    if not (
        args.transition_stride
        == args.effect_steps
        == args.executed_action_steps
        == LIBERO_EXECUTION_HORIZON
    ):
        raise ValueError("Formal LIBERO Stage 1 requires stride=effect=executed=5.")
    accelerator = Accelerator()
    torch.manual_seed(args.seed + accelerator.process_index)
    handoff = LiberoHandoff.from_root(args.handoff_root)
    train_dataset = LiberoZTEEpisodeDataset(
        args.dataset_root,
        subset="train",
        goal_embeddings=args.goal_embeddings,
    )
    validation_dataset = LiberoZTEEpisodeDataset(
        args.dataset_root,
        subset="validation",
        goal_embeddings=args.goal_embeddings,
    )
    if train_dataset.task_names != validation_dataset.task_names or train_dataset.task_count != 40:
        raise ValueError("LIBERO train/validation task table must contain the same 40 tasks.")
    config = ZevaConfig(
        action_dim=LIBERO_ACTION_DIM,
        action_horizon=LIBERO_POLICY_HORIZON,
        vision_pretrained=True,
        num_views=len(LIBERO_CAMERA_KEYS),
        task_count=train_dataset.task_count,
    )
    model = CausalTransitionEncoder(config)
    vision_parameters = list(model.vision_encoder.parameters())
    vision_ids = {id(parameter) for parameter in vision_parameters}
    other_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad and id(parameter) not in vision_ids
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
    if args.resume_checkpoint:
        checkpoint = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        if checkpoint.get("schema") != "zeva-libero-zte-stage1-checkpoint-v3":
            raise ValueError("Cannot resume a non-LIBERO Stage 1 checkpoint.")
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
        batch_size=1,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=max(0, min(2, args.num_workers)),
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    validation_loader = accelerator.prepare(validation_loader)
    normalizer = QuantileActionNormalizer.from_stats_file(handoff.statistics)
    manifest = _manifest(args, handoff, config, train_dataset.task_count)
    save_dir = Path(args.save_dir)
    if accelerator.is_main_process:
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    accelerator.wait_for_everyone()

    sampler_epoch = 0
    train_sampler.set_epoch(sampler_epoch)
    iterator = iter(train_loader)
    progress = tqdm.trange(
        start_step,
        args.steps,
        initial=start_step,
        total=args.steps,
        disable=not accelerator.is_local_main_process,
    )
    for step in progress:
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
        outputs = model(
            batch["images_before"],
            normalizer.normalize(batch["actions"]),
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
            progress.set_postfix(
                loss=f"{float(losses['total'].detach()):.4f}",
                effect=f"{float(losses['effect'].detach()):.4f}",
                action=f"{float(losses['action'].detach()):.4f}",
            )
        if completed % args.save_freq == 0 or completed == args.steps:
            validation = evaluate(model, validation_loader, normalizer, args, accelerator)
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped: nn.Module = accelerator.unwrap_model(model)
                payload = {
                    "schema": "zeva-libero-zte-stage1-checkpoint-v3",
                    "step": completed,
                    "validation_loss": validation,
                    "best_validation": min(best_validation, validation),
                    "model_state_dict": unwrapped.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "zte_config": dataclasses.asdict(config),
                    "manifest": manifest,
                    "action_normalization": normalizer.metadata(),
                }
                torch.save(payload, save_dir / f"zte_step_{completed:06d}.pth")
                torch.save(payload, save_dir / "zte_latest.pth")
                if validation <= best_validation:
                    best_validation = validation
                    torch.save(payload, save_dir / "zte_best.pth")
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
