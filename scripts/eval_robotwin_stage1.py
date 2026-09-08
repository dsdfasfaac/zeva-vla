"""Capability and deployment-alignment gate for a trained RoboTwin ZeVA ZTE."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import tqdm
import tyro

from openpi.zeva.causal_bank import RobotWinCausalBank
from openpi.zeva.config import ZevaConfig
from openpi.zeva.robotwin_contract import MeanStdActionNormalizer
from openpi.zeva.robotwin_contract import RobotWinHandoff
from openpi.zeva.robotwin_policy import robotwin_multiview_image
from openpi.zeva.transition_encoder import CausalTransitionEncoder
from train_robotwin_zte import FFmpegRoboTwinDataset
from train_robotwin_zte import EpisodeGoalEmbeddingTable


@dataclasses.dataclass
class Args:
    handoff_root: str = (
        "/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1"
    )
    dataset_root: str = "/data1/huangbingjia/robotwin-lerobot-sidney-eef16-v1/data"
    goal_embeddings: str = "/data1/dingxin/zeva-runs/robotwin-v5-h15-tasklang/pi05_task_embeddings.pt"
    zte_checkpoint: str = "/data1/dingxin/zeva-runs/robotwin-v5-h15-tasklang/stage1-zte/zte_best.pth"
    causal_bank: str = "/data1/dingxin/zeva-runs/robotwin-v5-h15-tasklang/stage1-zte/train_causal_bank.pt"
    output_path: str = "/data1/dingxin/zeva-runs/robotwin-v5-h15-tasklang/stage1-zte/stage1_gate.json"
    training_horizon: int = 15
    deployment_horizon: int = 15
    max_episodes_per_task: int | None = None
    batch_size: int = 1
    num_workers: int = 4
    device: str = "cuda"
    seed: int = 1000


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class BalancedStage1GateDataset(Dataset):
    """Every validation episode as one complete consecutive H15 rollout."""

    def __init__(
        self,
        adapter_manifest: str | Path,
        *,
        task_names: tuple[str, ...],
        training_horizon: int,
        deployment_horizon: int,
        max_episodes_per_task: int | None,
        seed: int,
        goal_embeddings: str | Path,
    ):
        self.source = FFmpegRoboTwinDataset(adapter_manifest, subset="validation")
        self.dataset = self.source.dataset
        self.deployment_horizon = deployment_horizon
        self.training_horizon = training_horizon
        self.task_names = task_names
        self.goals = EpisodeGoalEmbeddingTable(
            goal_embeddings,
            "validation",
            self.dataset._records,  # noqa: SLF001
        )
        self._task_ids = {name: index for index, name in enumerate(task_names)}
        grouped: dict[str, list[int]] = {name: [] for name in task_names}
        self.validation_episode_count = len(self.dataset._records)  # noqa: SLF001
        if training_horizon != deployment_horizon:
            raise ValueError("The Stage 1 gate requires identical training and deployment horizons.")
        minimum_length = deployment_horizon + 1
        for index, record in enumerate(self.dataset._records):  # noqa: SLF001
            name = record["key"][1]
            if name not in grouped:
                raise ValueError(f"Validation task {name!r} is absent from the causal bank.")
            # Require one training transition and a full online H15 sequence.
            if int(record["length"]) >= minimum_length:
                grouped[name].append(index)
        self.eligible_episode_count = sum(len(indices) for indices in grouped.values())
        selected = []
        for name in task_names:
            indices = grouped[name]
            if not indices:
                raise ValueError(f"No validation episode found for {name!r}.")
            if max_episodes_per_task is not None and len(indices) > max_episodes_per_task:
                positions = np.linspace(0, len(indices) - 1, max_episodes_per_task, dtype=np.int64)
                indices = [indices[int(position)] for position in positions]
            selected.extend(indices)
        rng = np.random.default_rng(seed)
        # Keep the sample set deterministic while avoiding long runs of one task.
        self._record_indices = np.asarray(selected, dtype=np.int64)
        rng.shuffle(self._record_indices)

    def __len__(self) -> int:
        return len(self._record_indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record_index = int(self._record_indices[index])
        record = self.dataset._records[record_index]  # noqa: SLF001
        length = int(record["length"])
        before_frames = list(range(0, length - self.deployment_horizon, self.deployment_horizon))
        after_frames = [frame + self.deployment_horizon for frame in before_frames]
        images = self.source.read_images(record, before_frames + after_frames)
        sequence_length = len(before_frames)
        before = robotwin_multiview_image(
            {key: value[:sequence_length] for key, value in images.items()}
        )
        after = robotwin_multiview_image(
            {key: value[sequence_length:] for key, value in images.items()}
        )

        dataset_start = int(self.dataset._cumulative[record_index])  # noqa: SLF001

        denominator = max(1, length - 1)
        return {
            "task_id": torch.tensor(self._task_ids[record["key"][1]], dtype=torch.long),
            "goal_embedding": self.goals.embeddings[record_index],
            "images_before": before,
            "images_after": after,
            "actions": torch.stack(
                [
                    self.dataset[dataset_start + frame]["action"][: self.deployment_horizon]
                    for frame in before_frames
                ]
            ),
            "start_progress": torch.tensor([frame / denominator for frame in before_frames]),
            "end_progress": torch.tensor([frame / denominator for frame in after_frames]),
        }


class MetricAccumulator:
    def __init__(self, task_count: int):
        self.task_correct = torch.zeros(task_count, dtype=torch.long)
        self.task_total = torch.zeros(task_count, dtype=torch.long)
        self.phase_progress: list[torch.Tensor] = []
        self.phase_start: list[torch.Tensor] = []
        self.phase_end: list[torch.Tensor] = []
        self.bank_progress: list[torch.Tensor] = []
        self.action_squared_error = 0.0
        self.action_zero_squared_error = 0.0
        self.action_elements = 0
        self.effect_squared_error = 0.0
        self.effect_zero_squared_error = 0.0
        self.effect_elements = 0
        self.monotonic_violations = 0
        self.monotonic_pairs = 0
        self.bank_monotonic_violations = 0
        self.bank_monotonic_pairs = 0

    def add(
        self,
        outputs,
        task_ids: torch.Tensor,
        start_progress: torch.Tensor,
        end_progress: torch.Tensor,
        bank: RobotWinCausalBank,
    ) -> None:
        batch, sequence = outputs.task_embedding.shape[:2]
        if outputs.global_task_embedding is None:
            raise RuntimeError("Stage 1 v5 gate requires an episode-global task key.")
        labels = task_ids.reshape(-1)
        predictions = (outputs.global_task_embedding @ bank.task_prototype.T).argmax(1)
        for task_id in labels.unique():
            mask = labels == task_id
            self.task_total[int(task_id)] += int(mask.sum())
            self.task_correct[int(task_id)] += int(predictions[mask].eq(task_id).sum())

        step_labels = task_ids[:, None].expand(batch, sequence).reshape(-1)
        flat_phase = outputs.phase_token.reshape(-1, outputs.phase_token.shape[-1])
        table = bank.phase_key[step_labels]
        valid = bank.count[step_labels] > 0
        scores = torch.einsum("bpd,bd->bp", table, flat_phase).masked_fill(~valid, -torch.inf)
        bins = scores.argmax(1)
        retrieved_progress = bank.manifest_progress[step_labels, bins].reshape(batch, sequence)
        self.bank_progress.append(retrieved_progress.reshape(-1).detach().cpu())
        self.phase_progress.append(outputs.phase_progress.reshape(-1).detach().cpu())
        self.phase_start.append(start_progress.reshape(-1).detach().cpu())
        self.phase_end.append(end_progress.reshape(-1).detach().cpu())

        differences = outputs.phase_progress[:, 1:] - outputs.phase_progress[:, :-1]
        self.monotonic_violations += int((differences < 0).sum())
        self.monotonic_pairs += differences.numel()
        bank_differences = retrieved_progress[:, 1:] - retrieved_progress[:, :-1]
        self.bank_monotonic_violations += int((bank_differences < 0).sum())
        self.bank_monotonic_pairs += bank_differences.numel()

        action_target = outputs.target_action
        action_prediction = outputs.predicted_action[..., : action_target.shape[-2], :]
        self.action_squared_error += float(F.mse_loss(action_prediction, action_target, reduction="sum"))
        self.action_zero_squared_error += float(action_target.square().sum())
        self.action_elements += action_target.numel()
        self.effect_squared_error += float(F.mse_loss(outputs.predicted_effect, outputs.target_effect, reduction="sum"))
        self.effect_zero_squared_error += float(outputs.target_effect.square().sum())
        self.effect_elements += outputs.target_effect.numel()

    @staticmethod
    def _spearman(prediction: torch.Tensor, target: torch.Tensor) -> float:
        prediction_rank = prediction.argsort().argsort().float()
        target_rank = target.argsort().argsort().float()
        prediction_rank -= prediction_rank.mean()
        target_rank -= target_rank.mean()
        denominator = prediction_rank.norm() * target_rank.norm()
        return float((prediction_rank @ target_rank / denominator).item()) if denominator > 0 else math.nan

    def summarize(self, task_names: tuple[str, ...]) -> dict[str, Any]:
        predicted = torch.cat(self.phase_progress)
        start = torch.cat(self.phase_start)
        end = torch.cat(self.phase_end)
        bank_progress = torch.cat(self.bank_progress)
        rates = self.task_correct.float() / self.task_total.clamp_min(1)
        action_mse = self.action_squared_error / self.action_elements
        action_zero_mse = self.action_zero_squared_error / self.action_elements
        effect_mse = self.effect_squared_error / self.effect_elements
        effect_zero_mse = self.effect_zero_squared_error / self.effect_elements
        return {
            "samples": int(len(predicted)),
            "task_retrieval": {
                "micro_recall_at_1": float(self.task_correct.sum() / self.task_total.sum()),
                "macro_recall_at_1": float(rates.mean()),
                "min_task_recall_at_1": float(rates.min()),
                "per_task": {name: float(rate) for name, rate in zip(task_names, rates, strict=True)},
            },
            "phase_progress": {
                "mae_vs_transition_start": float((predicted - start).abs().mean()),
                "mae_vs_transition_end": float((predicted - end).abs().mean()),
                "spearman_vs_start": self._spearman(predicted, start),
                "spearman_vs_end": self._spearman(predicted, end),
                "monotonic_violation_rate": self.monotonic_violations / max(1, self.monotonic_pairs),
            },
            "bank_phase_retrieval": {
                "mae_vs_transition_start": float((bank_progress - start).abs().mean()),
                "mae_vs_transition_end": float((bank_progress - end).abs().mean()),
                "monotonic_violation_rate": (
                    self.bank_monotonic_violations / max(1, self.bank_monotonic_pairs)
                ),
            },
            "action_reconstruction": {
                "mse": action_mse,
                "zero_baseline_mse": action_zero_mse,
                "relative_improvement": 1.0 - action_mse / max(action_zero_mse, 1e-12),
            },
            "effect_prediction": {
                "mse": effect_mse,
                "zero_change_baseline_mse": effect_zero_mse,
                "relative_improvement": 1.0 - effect_mse / max(effect_zero_mse, 1e-12),
            },
        }


def _attach_bank_progress(bank: RobotWinCausalBank, payload: dict[str, Any]) -> None:
    progress = torch.as_tensor(payload["progress"], dtype=torch.float32, device=bank.count.device)
    if progress.shape != bank.count.shape:
        raise ValueError("Causal-bank progress and count tables disagree.")
    # Empty bins are never selected because retrieval masks them out.
    bank.manifest_progress = progress


def _gate(metrics: dict[str, Any]) -> dict[str, Any]:
    deployment = metrics["deployment_h15_online"]
    parity = metrics["offline_online_parity_max_abs"]
    latent_names = (
        "phase_token",
        "causal_signal",
        "phase_progress",
        "task_embedding",
        "global_task_embedding",
    )
    auxiliary_names = ("predicted_effect", "target_effect", "predicted_action", "target_action")
    checks = {
        "offline_online_latent_max_abs_le_5e-3": max(parity[name] for name in latent_names) <= 5e-3,
        "offline_online_aux_max_abs_le_2e-2": max(parity[name] for name in auxiliary_names) <= 2e-2,
        "task_global_macro_r1_ge_0.95": deployment["task_retrieval"]["macro_recall_at_1"] >= 0.95,
        "phase_end_mae_le_0.10": deployment["phase_progress"]["mae_vs_transition_end"] <= 0.10,
        "phase_end_spearman_ge_0.80": deployment["phase_progress"]["spearman_vs_end"] >= 0.80,
        "phase_monotonic_violations_le_0.05": (
            deployment["phase_progress"]["monotonic_violation_rate"] <= 0.05
        ),
        "bank_phase_end_mae_le_0.10": deployment["bank_phase_retrieval"]["mae_vs_transition_end"] <= 0.10,
        "bank_phase_monotonic_violations_le_0.05": (
            deployment["bank_phase_retrieval"]["monotonic_violation_rate"] <= 0.05
        ),
        "initial_phase_bank_mae_le_0.05": metrics["initial_phase_bank_mae"] <= 0.05,
        "action_improvement_ge_0.20": deployment["action_reconstruction"]["relative_improvement"] >= 0.20,
        "effect_improvement_ge_0.20": deployment["effect_prediction"]["relative_improvement"] >= 0.20,
    }
    return {"passed": all(checks.values()), "checks": checks}


@torch.inference_mode()
def main(args: Args) -> None:
    if args.training_horizon != args.deployment_horizon:
        raise ValueError("The Stage 1 gate requires training_horizon == deployment_horizon.")
    device = torch.device(args.device)
    handoff = RobotWinHandoff.from_root(args.handoff_root)
    checkpoint = torch.load(args.zte_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "zeva-robotwin-zte-stage1-checkpoint-v5":
        raise ValueError("The aligned gate requires a Stage 1 v5 checkpoint.")
    bank_payload = torch.load(args.causal_bank, map_location="cpu", weights_only=False)
    if bank_payload["manifest"]["stage1_checkpoint_sha256"] != _sha256(args.zte_checkpoint):
        raise ValueError("The causal bank was not exported from this Stage 1 checkpoint.")
    bank = RobotWinCausalBank(bank_payload, device=device)
    _attach_bank_progress(bank, bank_payload)
    config = ZevaConfig(**checkpoint["zte_config"])
    model = CausalTransitionEncoder(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.requires_grad_(False).eval().to(device)
    normalizer = MeanStdActionNormalizer.from_stats_file(handoff.statistics)
    dataset = BalancedStage1GateDataset(
        Path(args.dataset_root) / "adapter.json",
        task_names=bank.task_names,
        training_horizon=args.training_horizon,
        deployment_horizon=args.deployment_horizon,
        max_episodes_per_task=args.max_episodes_per_task,
        seed=args.seed,
        goal_embeddings=args.goal_embeddings,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    offline = MetricAccumulator(len(bank.task_names))
    online = MetricAccumulator(len(bank.task_names))
    initial_bank_errors = []
    parity_max_abs = {
        name: 0.0
        for name in (
            "phase_token",
            "causal_signal",
            "phase_progress",
            "task_embedding",
            "global_task_embedding",
            "predicted_effect",
            "target_effect",
            "predicted_action",
            "target_action",
        )
    }
    for batch in tqdm.tqdm(loader, desc="stage1 gate"):
        task_ids = batch["task_id"].to(device)
        goal_embedding = batch["goal_embedding"].to(device, non_blocking=True)
        h15_before = batch["images_before"].to(device, non_blocking=True)
        h15_after = batch["images_after"].to(device, non_blocking=True)
        h15_actions = normalizer.normalize(batch["actions"].to(device, non_blocking=True))
        h15_start = batch["start_progress"].to(device)
        h15_end = batch["end_progress"].to(device)
        training_outputs = model(h15_before, h15_actions, h15_after, goal_embedding)
        offline.add(training_outputs, task_ids, h15_start, h15_end, bank)
        initial_phase, inference_params = model.initialize_phase_state(
            h15_before[:, 0],
            goal_embedding,
        )
        initial_table = bank.phase_key[task_ids]
        initial_valid = bank.count[task_ids] > 0
        initial_scores = torch.einsum("bpd,bd->bp", initial_table, initial_phase).masked_fill(
            ~initial_valid, -torch.inf
        )
        initial_bins = initial_scores.argmax(1)
        initial_progress = bank.manifest_progress[task_ids, initial_bins]
        initial_bank_errors.append((initial_progress - h15_start[:, 0]).abs().cpu())

        step_outputs = []
        for step in range(h15_before.shape[1]):
            output, inference_params = model.step(
                h15_before[:, step], h15_actions[:, step], h15_after[:, step], inference_params
            )
            step_outputs.append(output)
        stacked = dataclasses.replace(
            step_outputs[0],
            phase_token=torch.stack([item.phase_token for item in step_outputs], dim=1),
            causal_signal=torch.stack([item.causal_signal for item in step_outputs], dim=1),
            phase_progress=torch.stack([item.phase_progress for item in step_outputs], dim=1),
            task_embedding=torch.stack([item.task_embedding for item in step_outputs], dim=1),
            predicted_effect=torch.stack([item.predicted_effect for item in step_outputs], dim=1),
            target_effect=torch.stack([item.target_effect for item in step_outputs], dim=1),
            predicted_action=torch.stack([item.predicted_action for item in step_outputs], dim=1),
            target_action=torch.stack([item.target_action for item in step_outputs], dim=1),
            global_task_embedding=model.pool_global_task_embedding(
                torch.stack([item.task_embedding for item in step_outputs], dim=1),
                goal_embedding,
            ),
        )
        for name in parity_max_abs:
            difference = float((getattr(training_outputs, name) - getattr(stacked, name)).abs().max())
            parity_max_abs[name] = max(parity_max_abs[name], difference)
        online.add(stacked, task_ids, h15_start, h15_end, bank)

    metrics = {
        "schema": "zeva-robotwin-stage1-gate-v4",
        "checkpoint": str(Path(args.zte_checkpoint).resolve()),
        "checkpoint_sha256": _sha256(args.zte_checkpoint),
        "checkpoint_step": int(checkpoint["step"]),
        "causal_bank": str(Path(args.causal_bank).resolve()),
        "split": "validation5",
        "episodes": len(dataset),
        "validation_episodes_total": dataset.validation_episode_count,
        "validation_episodes_gate_eligible": dataset.eligible_episode_count,
        "episodes_per_task": args.max_episodes_per_task,
        "sequence_protocol": "complete-consecutive-H15-rollout-from-frame0",
        "training_horizon": args.training_horizon,
        "training_h15_offline": offline.summarize(bank.task_names),
        "deployment_h15_online": online.summarize(bank.task_names),
        "offline_online_parity_max_abs": parity_max_abs,
        "initial_phase_bank_mae": float(torch.cat(initial_bank_errors).mean()),
    }
    metrics["gate"] = _gate(metrics)
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(tyro.cli(Args))
