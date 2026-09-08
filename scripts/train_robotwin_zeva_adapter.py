"""Train Zeva causal adapters on the frozen RoboTwin LeRobot PI0.5 checkpoint."""

from __future__ import annotations

import bisect
import dataclasses
from pathlib import Path

from accelerate import Accelerator
import torch
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import DistributedSampler
import tqdm
import tyro

from openpi.zeva.robotwin_contract import ROBOTWIN_CAMERA_KEYS
from openpi.zeva.robotwin_policy import RobotWinZevaPolicy
from openpi.zeva.robotwin_policy import robotwin_multiview_image


@dataclasses.dataclass
class Args:
    handoff_root: str = (
        "/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1"
    )
    dataset_root: str = "/data1/huangbingjia/robotwin-lerobot-sidney-eef16-v1/data"
    adapter_manifest: str | None = None
    save_dir: str = "/data1/dingxin/zeva-runs/robotwin-adapter-v1"
    adapter_checkpoint: str | None = None
    execution_steps: int = 50
    steps: int = 30_000
    batch_size: int = 2
    num_workers: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    effect_loss_weight: float = 1.0
    progress_loss_weight: float = 0.2
    save_freq: int = 1_000
    seed: int = 1000


class RobotWinCausalPairDataset(Dataset):
    """Current PI0.5 sample paired with a later effect image in the same episode."""

    def __init__(self, adapter_manifest: str, execution_steps: int):
        try:
            from egoscale.data.robotwin.lerobot import RoboTwinLeRobotEEF16Dataset  # noqa: PLC0415
        except ImportError as error:  # pragma: no cover - requires the released runtime.
            raise RuntimeError("RoboTwin adapter training requires the handoff runtime on PYTHONPATH.") from error
        self.dataset = RoboTwinLeRobotEEF16Dataset(adapter_manifest, subset="train")
        self.execution_steps = execution_steps
        if execution_steps <= 0:
            raise ValueError("execution_steps must be positive.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        cumulative = self.dataset._cumulative  # noqa: SLF001 - frozen handoff dataset boundary.
        episode = bisect.bisect_right(cumulative, index) - 1
        episode_start = int(cumulative[episode])
        episode_end = int(cumulative[episode + 1])
        effect_index = min(index + self.execution_steps, episode_end - 1)
        current = self.dataset[index]
        effect = self.dataset[effect_index]
        result = dict(current)
        for camera_index, key in enumerate(ROBOTWIN_CAMERA_KEYS):
            result[f"zeva.effect_image.{camera_index}"] = effect[key]
        result["zeva.progress"] = torch.tensor(
            (index - episode_start) / max(1, episode_end - episode_start - 1),
            dtype=torch.float32,
        )
        return result


def main(args: Args) -> None:
    accelerator = Accelerator()
    torch.manual_seed(args.seed + accelerator.process_index)
    policy = RobotWinZevaPolicy.from_handoff(
        args.handoff_root,
        device=str(accelerator.device),
        adapter_checkpoint=args.adapter_checkpoint,
    )
    adapter_manifest = args.adapter_manifest or str(Path(args.dataset_root) / "adapter.json")
    dataset = RobotWinCausalPairDataset(adapter_manifest, args.execution_steps)
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            shuffle=True,
            seed=args.seed,
        )
        if accelerator.num_processes > 1
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    policy, optimizer, loader = accelerator.prepare(policy, optimizer, loader)
    unwrapped = accelerator.unwrap_model(policy)
    save_dir = Path(args.save_dir)
    if accelerator.is_main_process:
        save_dir.mkdir(parents=True, exist_ok=True)

    iterator = iter(loader)
    progress = tqdm.trange(args.steps, disable=not accelerator.is_local_main_process)
    for step in progress:
        try:
            raw_batch = next(iterator)
        except StopIteration:
            if sampler is not None:
                sampler.set_epoch(step)
            iterator = iter(loader)
            raw_batch = next(iterator)

        effect_views = {
            key: raw_batch.pop(f"zeva.effect_image.{camera_index}").to(accelerator.device)
            for camera_index, key in enumerate(ROBOTWIN_CAMERA_KEYS)
        }
        effect_image = robotwin_multiview_image(effect_views)
        progress_target = raw_batch.pop("zeva.progress").to(accelerator.device)
        previous_image = robotwin_multiview_image(
            {key: raw_batch[key].to(accelerator.device) for key in ROBOTWIN_CAMERA_KEYS}
        )
        executed_actions = raw_batch["action"].to(accelerator.device)
        processed = unwrapped.preprocessor(raw_batch)

        policy.train()
        optimizer.zero_grad(set_to_none=True)
        (foundation_output, causal) = policy(
            processed,
            previous_image=previous_image,
            effect_image=effect_image,
            executed_actions=executed_actions,
            return_causal=True,
        )
        foundation_loss = foundation_output[0] if isinstance(foundation_output, tuple) else foundation_output
        effect_loss = F.mse_loss(causal.predicted_effect, causal.target_effect)
        progress_loss = F.smooth_l1_loss(causal.phase_progress[:, -1], progress_target)
        loss = (
            foundation_loss
            + args.effect_loss_weight * effect_loss
            + args.progress_loss_weight * progress_loss
        )
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        unwrapped.causal_transition_encoder.update_ema()

        progress.set_postfix(
            loss=f"{float(loss.detach()):.4f}",
            flow=f"{float(foundation_loss.detach()):.4f}",
            effect=f"{float(effect_loss.detach()):.4f}",
        )
        completed = step + 1
        if completed % args.save_freq == 0 or completed == args.steps:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped.save_adapter(save_dir / f"adapter_step_{completed:06d}.pth")
                unwrapped.save_adapter(save_dir / "latest.pth")


if __name__ == "__main__":
    main(tyro.cli(Args))
