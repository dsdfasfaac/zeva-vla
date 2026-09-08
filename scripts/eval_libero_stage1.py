"""Capability and exact H5 deployment gate for a trained LIBERO ZTE."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset
import tqdm
import tyro

from openpi.zeva.config import ZevaConfig
from openpi.zeva.libero_bank import LiberoCausalBank
from openpi.zeva.libero_contract import LIBERO_CAMERA_KEYS
from openpi.zeva.libero_contract import LIBERO_EXECUTION_HORIZON
from openpi.zeva.libero_contract import LiberoHandoff
from openpi.zeva.libero_contract import QuantileActionNormalizer
from openpi.zeva.libero_contract import sha256
from openpi.zeva.libero_data import LiberoEpisodeTable
from openpi.zeva.libero_data import _decode_png
from openpi.zeva.libero_data import absolute_pose_command_horizon_to_actions
from openpi.zeva.libero_data import libero_multiview_image
from openpi.zeva.transition_encoder import CausalTransitionEncoder
from scripts.eval_robotwin_stage1 import MetricAccumulator
from scripts.eval_robotwin_stage1 import _attach_bank_progress


@dataclasses.dataclass
class Args:
    handoff_root: str = "/data1/dingxin/libero-memory-baseline-v1"
    dataset_root: str = "/data1/dingxin/libero-memory-baseline-v1/data"
    goal_embeddings: str = "/data1/dingxin/zeva-runs/libero-v1-h5-lang/pi05_goal_embeddings.pt"
    zte_checkpoint: str = "/data1/dingxin/zeva-runs/libero-v3-h5-lang/stage1-zte/zte_best.pth"
    causal_bank: str = "/data1/dingxin/zeva-runs/libero-v3-h5-lang/stage1-zte/train_causal_bank.pt"
    output_path: str = "/data1/dingxin/zeva-runs/libero-v3-h5-lang/stage1-zte/stage1_gate.json"
    training_horizon: int = LIBERO_EXECUTION_HORIZON
    deployment_horizon: int = LIBERO_EXECUTION_HORIZON
    max_episodes_per_task: int | None = None
    batch_size: int = 1
    num_workers: int = 4
    device: str = "cuda"
    seed: int = 1000


class BalancedGateDataset(Dataset):
    def __init__(self, args: Args, task_names: tuple[str, ...]):
        self.args = args
        self.table = LiberoEpisodeTable(args.dataset_root, "validation")
        self.task_names = task_names
        self.task_ids = {task: index for index, task in enumerate(task_names)}
        goals = torch.load(args.goal_embeddings, map_location="cpu", weights_only=False)["splits"]["validation"]
        expected = tuple(self.table.record_id(index) for index in range(len(self.table.episodes)))
        if tuple(goals["record_ids"]) != expected:
            raise ValueError("LIBERO validation goal table differs from the dataset.")
        self.goals = torch.as_tensor(goals["embeddings"], dtype=torch.float32)
        grouped = {task: [] for task in task_names}
        minimum = 2 * LIBERO_EXECUTION_HORIZON + 1
        for index, episode in enumerate(self.table.episodes):
            task = self.table.task(index)
            if int(episode["length"]) >= minimum:
                grouped[task].append(index)
        self.validation_episode_count = len(self.table.episodes)
        self.eligible_episode_count = sum(map(len, grouped.values()))
        selected = []
        for task in task_names:
            indices = grouped[task]
            if not indices:
                raise ValueError(f"No eligible LIBERO validation episode for {task!r}.")
            if args.max_episodes_per_task and len(indices) > args.max_episodes_per_task:
                positions = np.linspace(0, len(indices) - 1, args.max_episodes_per_task, dtype=np.int64)
                indices = [indices[int(position)] for position in positions]
            selected.extend(indices)
        rng = np.random.default_rng(args.seed)
        self.indices = np.asarray(selected, dtype=np.int64)
        rng.shuffle(self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        record = int(self.indices[item])
        table = pq.read_table(self.table.paths[record], columns=["image", "wrist_image", "state", "actions"])
        length = len(table)
        before_frames = list(range(0, length - self.args.deployment_horizon, self.args.deployment_horizon))
        after_frames = [frame + self.args.deployment_horizon for frame in before_frames]
        frames = before_frames + after_frames
        agent = torch.stack([_decode_png(table["image"][frame].as_py()) for frame in frames])
        wrist = torch.stack([_decode_png(table["wrist_image"][frame].as_py()) for frame in frames])
        states = np.asarray(table["state"].to_pylist(), dtype=np.float32)
        stored = np.asarray(table["actions"].to_pylist(), dtype=np.float32)

        def images(start, count):
            return libero_multiview_image({LIBERO_CAMERA_KEYS[0]: agent[start : start + count],
                                           LIBERO_CAMERA_KEYS[1]: wrist[start : start + count]})

        def actions(starts, horizon):
            return torch.from_numpy(np.stack([absolute_pose_command_horizon_to_actions(
                states[frame], stored[frame : frame + horizon]) for frame in starts]))

        sequence = len(before_frames)
        denominator = max(1, length - 1)
        task = self.table.task(record)
        return {
            "task_id": torch.tensor(self.task_ids[task]),
            "goal_embedding": self.goals[record],
            "images_before": images(0, sequence),
            "images_after": images(sequence, sequence),
            "actions": actions(before_frames, self.args.deployment_horizon),
            "start_progress": torch.tensor([frame / denominator for frame in before_frames]),
            "end_progress": torch.tensor([frame / denominator for frame in after_frames]),
        }


def _gate(deployment):
    checks = {
        "task_macro_r1_ge_0.90": deployment["task_retrieval"]["macro_recall_at_1"] >= 0.90,
        "task_min_r1_ge_0.80": deployment["task_retrieval"]["min_task_recall_at_1"] >= 0.80,
        "phase_end_mae_le_0.10": deployment["phase_progress"]["mae_vs_transition_end"] <= 0.10,
        "phase_end_spearman_ge_0.80": deployment["phase_progress"]["spearman_vs_end"] >= 0.80,
        "phase_monotonic_violations_le_0.05": deployment["phase_progress"]["monotonic_violation_rate"] <= 0.05,
        "bank_phase_end_mae_le_0.10": deployment["bank_phase_retrieval"]["mae_vs_transition_end"] <= 0.10,
        "action_improvement_ge_0.20": deployment["action_reconstruction"]["relative_improvement"] >= 0.20,
        "effect_improvement_ge_0.20": deployment["effect_prediction"]["relative_improvement"] >= 0.20,
    }
    return {"passed": all(checks.values()), "checks": checks}


@torch.inference_mode()
def main(args: Args) -> None:
    if args.training_horizon != 5 or args.deployment_horizon != 5:
        raise ValueError("Formal LIBERO gate requires offline H5 and online H5.")
    device = torch.device(args.device)
    handoff = LiberoHandoff.from_root(args.handoff_root)
    checkpoint = torch.load(args.zte_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "zeva-libero-zte-stage1-checkpoint-v3":
        raise ValueError("LIBERO gate requires a formal LIBERO ZTE checkpoint.")
    bank_payload = torch.load(args.causal_bank, map_location="cpu", weights_only=False)
    if bank_payload["manifest"]["stage1_checkpoint_sha256"] != sha256(args.zte_checkpoint):
        raise ValueError("LIBERO causal bank was exported from another ZTE.")
    bank = LiberoCausalBank(bank_payload, device=device)
    _attach_bank_progress(bank, bank_payload)
    config = ZevaConfig(**checkpoint["zte_config"])
    model = CausalTransitionEncoder(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.requires_grad_(False).eval()
    normalizer = QuantileActionNormalizer.from_stats_file(handoff.statistics)
    dataset = BalancedGateDataset(args, bank.task_names)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=True, persistent_workers=args.num_workers > 0)
    offline = MetricAccumulator(len(bank.task_names))
    online = MetricAccumulator(len(bank.task_names))
    initial_errors = []
    parity_max_abs = {
        name: 0.0
        for name in (
            "phase_token", "causal_signal", "phase_progress", "task_embedding",
            "global_task_embedding", "predicted_effect", "target_effect",
            "predicted_action", "target_action",
        )
    }
    for batch in tqdm.tqdm(loader, desc="LIBERO Stage 1 gate"):
        task = batch["task_id"].to(device)
        goal = batch["goal_embedding"].to(device)
        before = batch["images_before"].to(device)
        after = batch["images_after"].to(device)
        actions = normalizer.normalize(batch["actions"].to(device))
        start = batch["start_progress"].to(device)
        end = batch["end_progress"].to(device)
        offline_output = model(before, actions, after, goal)
        offline.add(offline_output, task, start, end, bank)
        initial, cache = model.initialize_phase_state(before[:, 0], goal)
        scores = torch.einsum("bpd,bd->bp", bank.phase_key[task], initial).masked_fill(bank.count[task] == 0, -torch.inf)
        initial_progress = bank.manifest_progress[task, scores.argmax(1)]
        initial_errors.append((initial_progress - start[:, 0]).abs().cpu())
        outputs = []
        for step in range(before.shape[1]):
            output, cache = model.step(before[:, step], actions[:, step], after[:, step], cache)
            outputs.append(output)
        stacked = dataclasses.replace(
            outputs[0],
            phase_token=torch.stack([value.phase_token for value in outputs], 1),
            causal_signal=torch.stack([value.causal_signal for value in outputs], 1),
            phase_progress=torch.stack([value.phase_progress for value in outputs], 1),
            task_embedding=torch.stack([value.task_embedding for value in outputs], 1),
            predicted_effect=torch.stack([value.predicted_effect for value in outputs], 1),
            target_effect=torch.stack([value.target_effect for value in outputs], 1),
            predicted_action=torch.stack([value.predicted_action for value in outputs], 1),
            target_action=torch.stack([value.target_action for value in outputs], 1),
            global_task_embedding=model.pool_global_task_embedding(
                torch.stack([value.task_embedding for value in outputs], 1), goal
            ),
        )
        for name in parity_max_abs:
            error = float((getattr(offline_output, name) - getattr(stacked, name)).abs().max())
            parity_max_abs[name] = max(parity_max_abs[name], error)
        online.add(stacked, task, start, end, bank)
    deployment = online.summarize(bank.task_names)
    metrics = {
        "schema": "zeva-libero-stage1-gate-v2",
        "checkpoint": str(Path(args.zte_checkpoint).resolve()),
        "checkpoint_sha256": sha256(args.zte_checkpoint),
        "checkpoint_step": int(checkpoint["step"]),
        "causal_bank": str(Path(args.causal_bank).resolve()),
        "split": "official-validation-79",
        "episodes": len(dataset),
        "validation_episodes_total": dataset.validation_episode_count,
        "validation_episodes_gate_eligible": dataset.eligible_episode_count,
        "policy_horizon": 10,
        "execution_horizon": 5,
        "offline_h5": offline.summarize(bank.task_names),
        "deployment_h5_online": deployment,
        "offline_online_parity_max_abs": parity_max_abs,
        "initial_phase_bank_mae": float(torch.cat(initial_errors).mean()),
        "gate": _gate(deployment),
    }
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(tyro.cli(Args))
