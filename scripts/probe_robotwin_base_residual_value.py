"""Test whether frozen ZTE features predict held-out Base H15 action errors.

This is a probe, not a deployable policy.  Both arms see the exact cached Base
H15 chunk and oracle task one-hot; only the full arm sees causal features from
the deployment-recurrent Stage1 cache.  The validation episodes are held out
from fitting, and no RoboTwin test success labels are read.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812
import tyro

from openpi.zeva.robotwin_contract import MeanStdActionNormalizer
from openpi.zeva.robotwin_contract import RobotWinHandoff


@dataclasses.dataclass
class Args:
    handoff_root: str = "/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1"
    dataset_root: str = "/data1/huangbingjia/robotwin-lerobot-sidney-eef16-v1/data"
    live_queries: str = "/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage1-zte-v2-artifacts-scheduler-repaired-20260911/live_queries_h15.pt"
    task_subset: str = "configs/robotwin_zeva_advantage10.json"
    base_action_cache: str = ""
    expected_base_model_sha256: str = "bcf1d4f5f3e77926b8b7f798bb983378ad096e3892eed1a66058c5d98ed9bc17"
    output: str = ""
    steps: int = 1000
    batch_size: int = 512
    learning_rate: float = 3e-4
    seed: int = 1000
    device: str = "cuda"


class ResidualProbe(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 15 * 16),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value).reshape(-1, 15, 16)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _build_split(dataset, live_records, task_names, selected_tasks, cached, normalizer):
    """Reconstruct exactly the cache script's selected-sample coordinate."""
    indices = torch.as_tensor(cached["sample_indices"], dtype=torch.long)
    base_actions = torch.as_tensor(cached["base_actions"], dtype=torch.float32)
    if len(indices) != len(base_actions) or len(indices) != len(indices.unique()):
        raise ValueError("Base cache indices/actions are missing or duplicated")
    if tuple(base_actions.shape[1:]) != (50, 16):
        raise ValueError(f"Base cache action shape mismatch: {tuple(base_actions.shape)}")
    if not bool(torch.all(indices[1:] > indices[:-1])):
        raise ValueError("Base cache sample indices must be sorted")
    index_to_cache = {int(index): offset for offset, index in enumerate(indices)}
    dataset._read_source_images = lambda _record, _frame: {}  # noqa: SLF001
    rows = []
    selected_sample_index = 0
    for record_index, live in enumerate(live_records):
        record = dataset._records[record_index]  # noqa: SLF001
        task_name = record["key"][1]
        if task_name not in selected_tasks:
            continue
        task_id = task_names.index(task_name)
        episode_start = int(dataset._cumulative[record_index])  # noqa: SLF001
        running = torch.zeros(256, dtype=torch.float32)
        for decision_index, frame in enumerate(live["decision_frames"].tolist()):
            if decision_index:
                last = live["causal_signals"][decision_index - 1].float()
                running = running + (last - running) / decision_index
            else:
                last = torch.zeros_like(running)
            cache_offset = index_to_cache.get(selected_sample_index)
            selected_sample_index += 1
            if cache_offset is None:
                continue
            target = dataset[episode_start + int(frame)]["action"][:15].float()
            base = base_actions[cache_offset, :15]
            phase = live["phase_queries"][decision_index].float()
            task = F.one_hot(torch.tensor(task_id), num_classes=len(task_names)).float()
            features = torch.cat((task, base.flatten(), phase, last, running.clone()))
            rows.append((cache_offset, features, base, target, task_id))
    if len(rows) != len(indices):
        raise ValueError(f"Found {len(rows)} of {len(indices)} cached decisions")
    rows.sort(key=lambda item: item[0])
    if [item[0] for item in rows] != list(range(len(indices))):
        raise ValueError("Base cache sample coordinates do not match live-query decisions")
    features = torch.stack([item[1] for item in rows])
    base = torch.stack([item[2] for item in rows])
    target = normalizer.normalize(torch.stack([item[3] for item in rows]))
    task_ids = torch.tensor([item[4] for item in rows], dtype=torch.long)
    return features, base, target, task_ids


def _shuffle_zte_within_task(
    features: torch.Tensor,
    task_ids: torch.Tensor,
    *,
    visible_base: int,
    seed: int,
) -> tuple[torch.Tensor, int]:
    """Preserve Base/action/task and ZTE marginals, break state alignment."""
    source = torch.arange(len(features))
    for task_id in task_ids.unique().tolist():
        positions = torch.nonzero(task_ids == task_id, as_tuple=False).flatten()
        generator = torch.Generator().manual_seed(seed + 104729 * int(task_id))
        source[positions] = positions[torch.randperm(len(positions), generator=generator)]
    shuffled = features.clone()
    shuffled[:, visible_base:] = features[source, visible_base:]
    return shuffled, int((source != torch.arange(len(source))).sum())


def main(args: Args) -> None:
    if not args.base_action_cache or not args.output:
        raise ValueError("Pass an explicit Base action cache and output report path")
    if args.steps <= 0 or args.batch_size <= 0:
        raise ValueError("Probe steps and batch size must be positive")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    from egoscale.data.robotwin.lerobot import RoboTwinLeRobotEEF16Dataset  # noqa: PLC0415

    torch.manual_seed(args.seed)
    live = torch.load(args.live_queries, map_location="cpu", weights_only=False)
    cache = torch.load(args.base_action_cache, map_location="cpu", weights_only=False)
    if cache.get("schema") != "zeva-robotwin-stage2-base-action-cache-v2":
        raise ValueError("Probe requires the frozen Stage2 Base cache v2")
    if cache.get("selected_model_sha256") != args.expected_base_model_sha256:
        raise ValueError("Base cache model SHA differs from the designated Base1000")
    if cache.get("goal_embedding_model_sha256") != cache.get("foundation_model_sha256"):
        raise ValueError("Base cache task-language embedding differs from the deployed foundation")
    if cache.get("live_queries_sha256") != _sha256(args.live_queries):
        raise ValueError("Base cache and ZTE live queries differ")
    if cache.get("task_subset_sha256") != _sha256(args.task_subset):
        raise ValueError("Base cache and task subset differ")
    if cache.get("dataset_adapter_sha256") != _sha256(Path(args.dataset_root) / "adapter.json"):
        raise ValueError("Base cache and dataset adapter differ")
    task_names = tuple(live["task_names"])
    if task_names != tuple(cache["task_names"]):
        raise ValueError("Base cache task ordering differs from ZTE live queries")
    selected_tasks = set(json.loads(Path(args.task_subset).read_text())["task_names"])
    if selected_tasks != set(cache["selected_task_names"]):
        raise ValueError("Base cache task subset differs from the probe")
    handoff = RobotWinHandoff.from_root(args.handoff_root)
    normalizer = MeanStdActionNormalizer.from_stats_file(handoff.statistics)
    splits = {}
    for split in ("train", "validation"):
        dataset = RoboTwinLeRobotEEF16Dataset(Path(args.dataset_root) / "adapter.json", subset=split)
        if len(live["splits"][split]) != len(dataset._records):  # noqa: SLF001
            raise ValueError(f"{split} live-query record count differs from dataset")
        splits[split] = _build_split(
            dataset, live["splits"][split], task_names, selected_tasks,
            cache["splits"][split], normalizer,
        )

    device = torch.device(args.device)
    train_x, train_base, train_target, _ = splits["train"]
    val_x, val_base, val_target, val_task = splits["validation"]
    train_x = train_x.to(device)
    train_base = train_base.to(device)
    train_target = train_target.to(device)
    val_x = val_x.to(device)
    val_base = val_base.to(device)
    val_target = val_target.to(device)
    if train_x.shape[1] != len(task_names) + 240 + 128 + 256 + 256:
        raise ValueError(f"Unexpected probe feature shape {tuple(train_x.shape)}")
    visible_base = len(task_names) + 240
    shuffled_train_x, shuffled_train_rows = _shuffle_zte_within_task(
        train_x.cpu(), splits["train"][3], visible_base=visible_base, seed=args.seed + 10_000
    )
    shuffled_val_x, shuffled_val_rows = _shuffle_zte_within_task(
        val_x.cpu(), val_task, visible_base=visible_base, seed=args.seed + 20_000
    )
    shuffled_train_x = shuffled_train_x.to(device)
    shuffled_val_x = shuffled_val_x.to(device)
    results = {}
    for arm, visible in (("base_action_and_task", visible_base),
                         ("base_action_task_and_zte", train_x.shape[1]),
                         ("base_action_task_and_shuffled_zte", train_x.shape[1])):
        torch.manual_seed(args.seed)
        generator = torch.Generator(device=device).manual_seed(args.seed)
        model = ResidualProbe(train_x.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        feature_mask = train_x.new_zeros(train_x.shape[1])
        feature_mask[:visible] = 1
        arm_train_x = shuffled_train_x if arm.endswith("shuffled_zte") else train_x
        arm_val_x = shuffled_val_x if arm.endswith("shuffled_zte") else val_x
        for _ in range(args.steps):
            positions = torch.randint(
                len(train_x), (args.batch_size,), generator=generator, device=device
            )
            predicted_delta = model(arm_train_x[positions] * feature_mask)
            loss = F.mse_loss(predicted_delta, train_target[positions] - train_base[positions])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            corrected = val_base + model(arm_val_x * feature_mask)
            squared = (corrected - val_target).square().mean(dim=(1, 2)).cpu()
        per_task = {}
        for task_id in val_task.unique().tolist():
            per_task[task_names[task_id]] = float(squared[val_task == task_id].mean())
        results[arm] = {"corrected_mse": float(squared.mean()), "per_task_mse": per_task}
    with torch.no_grad():
        base_squared = (val_base - val_target).square().mean(dim=(1, 2)).cpu()
    base_only = results["base_action_and_task"]["corrected_mse"]
    full = results["base_action_task_and_zte"]["corrected_mse"]
    shuffled_control = results["base_action_task_and_shuffled_zte"]["corrected_mse"]
    payload = {
        "schema": "zeva-robotwin-base1000-residual-probe-v2",
        "split": "episode-heldout-validation5",
        "diagnostic_only": True,
        "oracle_task_identity_used_equally_in_both_arms": True,
        "deployment_policy_changed": False,
        "base_model_sha256": args.expected_base_model_sha256,
        "base_action_cache_sha256": _sha256(args.base_action_cache),
        "live_queries_sha256": _sha256(args.live_queries),
        "task_subset_sha256": _sha256(args.task_subset),
        "train_samples": len(train_x),
        "validation_samples": len(val_x),
        "training_steps": args.steps,
        "seed": args.seed,
        "base_uncorrected_mse": float(base_squared.mean()),
        "results": results,
        "shuffled_zte_control": {
            "scheme": "independent deterministic within-task permutations of train and validation ZTE groups only",
            "train_rows_permuted": shuffled_train_rows,
            "validation_rows_permuted": shuffled_val_rows,
        },
        "zte_incremental_relative_improvement_over_base_action_task": 1.0 - full / base_only,
        "zte_incremental_relative_improvement_over_shuffled_control": 1.0 - full / shuffled_control,
        "zte_adds_base_residual_information": full < min(base_only, shuffled_control),
        "caveat": "Probe on expert states, not on-policy success; positive residual MSE gain cannot by itself justify deployment.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(tyro.cli(Args))
