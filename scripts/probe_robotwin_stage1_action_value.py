"""Probe whether frozen Stage1 phase/causal features add held-out H15 action signal."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812
import tyro

from openpi.zeva.robotwin_contract import MeanStdActionNormalizer
from openpi.zeva.robotwin_contract import RobotWinHandoff


@dataclasses.dataclass
class Args:
    handoff_root: str = "/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1"
    dataset_root: str = "/data1/dingxin/robotwin-lerobot-sidney-eef16-v1/data"
    live_queries: str = "/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage1-artifacts-v1/stage1-zte/live_queries_h15.pt"
    task_subset: str = "configs/robotwin_zeva_advantage10.json"
    output: str = "/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage1-artifacts-v1/stage1-zte/action_value_probe_v1.json"
    train_samples: int = 32_768
    validation_samples: int = 5_874
    steps: int = 1_000
    batch_size: int = 1_024
    learning_rate: float = 3e-4
    seed: int = 1000
    device: str = "cuda"


class Probe(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 15 * 16),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value).view(-1, 15, 16)


def _select(indices_by_task: dict[int, list[int]], limit: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    task_ids = sorted(indices_by_task)
    per_task = max(1, limit // len(task_ids))
    selected: list[int] = []
    for task_id in task_ids:
        values = np.asarray(indices_by_task[task_id], dtype=np.int64)
        count = min(per_task, len(values))
        selected.extend(rng.choice(values, size=count, replace=False).tolist())
    rng.shuffle(selected)
    return np.asarray(selected[:limit], dtype=np.int64)


def _build_split(
    dataset,
    cached_records,
    task_names: list[str],
    selected_tasks: set[str],
    normalizer,
    limit: int,
    seed: int,
):
    rows = []
    indices_by_task: dict[int, list[int]] = {}
    task_count = len(task_names)
    for record_index, cached in enumerate(cached_records):
        task_id = int(cached["task_id"])
        if task_names[task_id] not in selected_tasks:
            continue
        start = int(dataset._cumulative[record_index])  # noqa: SLF001
        running = torch.zeros(256, dtype=torch.float32)
        for decision_index, frame in enumerate(cached["decision_frames"].tolist()):
            if decision_index:
                signal = cached["causal_signals"][decision_index - 1].float()
                running = running + (signal - running) / decision_index
                last = signal
            else:
                last = torch.zeros_like(running)
            target = dataset[start + int(frame)]["action"][:15].float()
            task = F.one_hot(torch.tensor(task_id), num_classes=task_count).float()
            phase = cached["phase_queries"][decision_index].float()
            rows.append((task, phase, last, running.clone(), target, task_id))
            indices_by_task.setdefault(task_id, []).append(len(rows) - 1)
    picked = _select(indices_by_task, min(limit, len(rows)), seed)
    task = torch.stack([rows[i][0] for i in picked])
    phase = torch.stack([rows[i][1] for i in picked])
    last = torch.stack([rows[i][2] for i in picked])
    running = torch.stack([rows[i][3] for i in picked])
    target = normalizer.normalize(torch.stack([rows[i][4] for i in picked]))
    task_ids = torch.tensor([rows[i][5] for i in picked], dtype=torch.long)
    return torch.cat([task, phase, last, running], dim=-1), target, task_ids


def main(args: Args) -> None:
    torch.manual_seed(args.seed)
    from egoscale.data.robotwin.lerobot import RoboTwinLeRobotEEF16Dataset  # noqa: PLC0415

    live = torch.load(args.live_queries, map_location="cpu", weights_only=False)
    tasks = set(json.loads(Path(args.task_subset).read_text())["task_names"])
    handoff = RobotWinHandoff.from_root(args.handoff_root)
    normalizer = MeanStdActionNormalizer.from_stats_file(handoff.statistics)
    datasets = {}
    for split in ("train", "validation"):
        dataset = RoboTwinLeRobotEEF16Dataset(
            Path(args.dataset_root) / "adapter.json", subset=split
        )
        dataset._read_source_images = lambda _record, _frame: {}  # noqa: SLF001
        datasets[split] = dataset
    train_x, train_y, _ = _build_split(
        datasets["train"], live["splits"]["train"], live["task_names"], tasks,
        normalizer, args.train_samples, args.seed
    )
    val_x, val_y, val_task = _build_split(
        datasets["validation"], live["splits"]["validation"], live["task_names"], tasks, normalizer,
        args.validation_samples, args.seed + 1,
    )
    device = torch.device(args.device)
    train_x, train_y = train_x.to(device), train_y.to(device)
    val_x, val_y = val_x.to(device), val_y.to(device)
    boundaries = {
        "task_only": 50,
        "task_plus_phase": 50 + 128,
        "task_plus_phase_plus_causal": train_x.shape[1],
    }
    results = {}
    for name, visible in boundaries.items():
        torch.manual_seed(args.seed)
        generator = torch.Generator(device=device).manual_seed(args.seed)
        model = Probe(train_x.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        mask = train_x.new_zeros(train_x.shape[1])
        mask[:visible] = 1
        for _ in range(args.steps):
            index = torch.randint(len(train_x), (args.batch_size,), generator=generator, device=device)
            prediction = model(train_x[index] * mask)
            loss = F.mse_loss(prediction, train_y[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            squared = (model(val_x * mask) - val_y).square().mean(dim=(1, 2)).cpu()
        per_task = {}
        for task_id in val_task.unique().tolist():
            task_name = live["task_names"][task_id]
            per_task[task_name] = float(squared[val_task == task_id].mean())
        results[name] = {"mse": float(squared.mean()), "per_task_mse": per_task}
    task_mse = results["task_only"]["mse"]
    full_mse = results["task_plus_phase_plus_causal"]["mse"]
    payload = {
        "schema": "zeva-robotwin-stage1-action-value-probe-v1",
        "split": "episode-heldout-validation5",
        "task_names": sorted(tasks),
        "train_samples": len(train_x),
        "validation_samples": len(val_x),
        "features": "same-capacity probe; unavailable feature groups are exactly zeroed",
        "results": results,
        "full_relative_improvement_over_task_only": 1.0 - full_mse / task_mse,
        "stage1_adds_action_information": full_mse < task_mse,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(tyro.cli(Args))
