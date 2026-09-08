"""Train Zeva's Mamba Causal Transition Encoder on action-effect rollouts."""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import tqdm
import tyro

from openpi.zeva.config import ZevaConfig
from openpi.zeva.normalization import QuantileActionNormalizer
from openpi.zeva.transition_encoder import CausalTransitionEncoder


@dataclasses.dataclass
class Args:
    rollout_dir: str
    norm_stats_dir: str
    save_dir: str = "checkpoints/zeva_causal_encoder"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    epochs: int = 80
    batch_size: int = 8
    sequence_length: int = 16
    learning_rate: float = 1e-4
    vision_learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    num_workers: int = 4
    seed: int = 7
    use_effect_stream: bool = True


class CausalRolloutDataset(Dataset):
    def __init__(self, rollout_dir: str, normalizer: QuantileActionNormalizer, sequence_length: int):
        self.normalizer = normalizer
        self.sequence_length = sequence_length
        records = []
        for path in sorted(Path(rollout_dir).glob("transition_*.npz")):
            with np.load(path) as item:
                records.append(
                    {
                        "path": path,
                        "task_id": int(item["task_id"]),
                        "episode_id": int(item["episode_id"]),
                        "attempt_id": int(item["attempt_id"]),
                    }
                )
        if not records:
            raise FileNotFoundError(f"No transition_*.npz files found in {rollout_dir}.")

        groups: dict[tuple[int, int, int], list[Path]] = {}
        for record in records:
            key = (record["task_id"], record["episode_id"], record["attempt_id"])
            groups.setdefault(key, []).append(record["path"])
        self.windows = []
        for key, paths in groups.items():
            for start in range(0, len(paths), sequence_length):
                self.windows.append((key, paths[start : start + sequence_length]))

    def __len__(self):
        return len(self.windows)

    @staticmethod
    def _image(value: np.ndarray) -> torch.Tensor:
        image = torch.from_numpy(np.asarray(value).copy()).to(torch.float32)
        return image.permute(2, 0, 1) / 127.5 - 1.0

    def __getitem__(self, index):
        (task_id, _, _), paths = self.windows[index]
        before, after, actions, progress = [], [], [], []
        for path in paths:
            with np.load(path) as item:
                before.append(self._image(item["image_before"]))
                after.append(self._image(item["image_after"]))
                action_chunk = torch.from_numpy(item["executed_actions"].copy()).to(torch.float32)
                actions.append(self.normalizer.normalize(action_chunk[..., :7]).mean(dim=0))
                progress.append(float(item["progress"]))

        valid_length = len(paths)
        while len(before) < self.sequence_length:
            before.append(before[-1].clone())
            after.append(after[-1].clone())
            actions.append(torch.zeros_like(actions[-1]))
            progress.append(progress[-1])
        mask = torch.arange(self.sequence_length) < valid_length
        return {
            "images_before": torch.stack(before),
            "images_after": torch.stack(after),
            "actions": torch.stack(actions),
            "progress": torch.tensor(progress, dtype=torch.float32),
            "mask": mask,
            "task_id": torch.tensor(task_id, dtype=torch.long),
        }


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(value.dtype)
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(value)
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def _task_contrastive_loss(embeddings: torch.Tensor, task_ids: torch.Tensor, temperature: float = 0.07):
    embeddings = F.normalize(embeddings, dim=-1)
    logits = embeddings @ embeddings.T / temperature
    eye = torch.eye(len(task_ids), dtype=torch.bool, device=task_ids.device)
    positives = task_ids[:, None].eq(task_ids[None, :]) & ~eye
    if not positives.any():
        return logits.new_zeros(())
    logits = logits.masked_fill(eye, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    valid = positives.any(dim=1)
    return -(log_prob.masked_fill(~positives, 0.0).sum(dim=1)[valid] / positives.sum(dim=1)[valid]).mean()


def compute_loss(outputs, batch, device):
    mask = batch["mask"].to(device)
    progress_target = batch["progress"].to(device)
    effect_loss = _masked_mean((outputs.predicted_effect - outputs.target_effect).square(), mask)
    progress_loss = _masked_mean(F.smooth_l1_loss(outputs.phase_progress, progress_target, reduction="none"), mask)

    adjacent_mask = mask[:, 1:] & mask[:, :-1]
    progress_delta = outputs.phase_progress[:, 1:] - outputs.phase_progress[:, :-1]
    monotonic_loss = _masked_mean(F.relu(0.01 - progress_delta), adjacent_mask)

    mask_float = mask.to(outputs.task_embedding.dtype).unsqueeze(-1)
    task_embedding = (outputs.task_embedding * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp_min(1.0)
    task_loss = _task_contrastive_loss(task_embedding, batch["task_id"].to(device))
    total = effect_loss + task_loss + progress_loss + monotonic_loss
    return total, {
        "effect": effect_loss.detach(),
        "task": task_loss.detach(),
        "phase": (progress_loss + monotonic_loss).detach(),
    }


def main(args: Args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    normalizer = QuantileActionNormalizer.from_source(args.norm_stats_dir, action_dim=7)
    dataset = CausalRolloutDataset(args.rollout_dir, normalizer, args.sequence_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    config = ZevaConfig(action_dim=7, vision_pretrained=True, use_effect_stream=args.use_effect_stream)
    model = CausalTransitionEncoder(config).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.vision_encoder.parameters(), "lr": args.vision_learning_rate},
            {
                "params": [parameter for name, parameter in model.named_parameters() if "vision_encoder" not in name],
                "lr": args.learning_rate,
            },
        ],
        weight_decay=args.weight_decay,
    )

    best_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        progress = tqdm.tqdm(loader, desc=f"CTE epoch {epoch + 1}/{args.epochs}")
        for batch in progress:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                batch["images_before"].to(device),
                batch["actions"].to(device),
                batch["images_after"].to(device),
            )
            loss, metrics = compute_loss(outputs, batch, device)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.update_ema()
            running += float(loss)
            progress.set_postfix(loss=f"{float(loss):.4f}", effect=f"{float(metrics['effect']):.4f}")

        average_loss = running / max(1, len(loader))
        payload = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "model_config": dataclasses.asdict(config),
            "action_normalization": normalizer.metadata(),
            "loss": average_loss,
        }
        torch.save(payload, save_dir / f"causal_encoder_epoch_{epoch + 1:03d}.pth")
        if average_loss <= best_loss:
            best_loss = average_loss
            torch.save(payload, save_dir / "best_model.pth")
        logging.info("epoch=%d loss=%.6f best=%.6f", epoch + 1, average_loss, best_loss)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
