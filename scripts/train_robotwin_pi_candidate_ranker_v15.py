"""Train and audit the v15 ZeVA frozen-PI candidate value head.

This is deliberately an offline gate before any closed-loop rollout.  PI0.5,
ZTE and the causal bank are immutable.  The only learned component predicts
which of K frozen-PI H15 candidates is closest to the expert action.  A
calibration half of validation chooses a conservative Base-fallback margin;
the disjoint audit half decides whether the method is allowed into rollout.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader, TensorDataset
import tyro

from openpi.zeva.causal_bank import RobotWinCausalBank
from openpi.zeva.config import ZevaConfig
from openpi.zeva.pi_candidate_ranker import RobotWinPICandidateRanker
from scripts.train_robotwin_stage2 import RobotWinStage2Dataset
from scripts.train_robotwin_stage2 import _load_task_subset


@dataclasses.dataclass
class Args:
    cache: str = (
        "/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/"
        "advantage10-pi-candidates-v15/pi_candidates_k4.pt"
    )
    live_queries: str = (
        "/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/"
        "stage1-artifacts-v1/stage1-zte/live_queries_h15.pt"
    )
    causal_bank: str = (
        "/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/"
        "stage1-artifacts-v1/stage1-zte/train_causal_bank.pt"
    )
    zte_checkpoint: str = (
        "/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/"
        "stage1-artifacts-v1/stage1-zte/zte_best.pth"
    )
    dataset_root: str = "/data1/dingxin/robotwin-lerobot-sidney-eef16-v1/data"
    task_subset: str = "configs/robotwin_zeva_advantage10.json"
    output_dir: str = (
        "/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/"
        "advantage10-pi-candidate-ranker-v15"
    )
    horizon: int = 15
    hidden_dim: int = 256
    dropout: float = 0.1
    batch_size: int = 256
    epochs: int = 80
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    listwise_weight: float = 0.5
    seed: int = 20260911


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _phase_queries_for_indices(
    *,
    dataset_root: str,
    live_queries: str,
    task_subset: str,
    zte_config: ZevaConfig,
    split: str,
    sample_indices: torch.Tensor,
) -> torch.Tensor:
    """Resolve Stage1 recurrent queries without decoding a single video frame."""
    dataset = RobotWinStage2Dataset(
        Path(dataset_root) / "adapter.json",
        live_queries,
        subset=split,
        config=zte_config,
        selected_tasks=_load_task_subset(task_subset),
        video_backend="torchcodec",
    )
    result = []
    for index in sample_indices.tolist():
        record_index, decision_index, _ = dataset._samples[int(index)]  # noqa: SLF001
        result.append(dataset.live_records[record_index]["phase_queries"][decision_index].float())
    return torch.stack(result)


def _stage1_static_features(
    bank: RobotWinCausalBank,
    task_ids: torch.Tensor,
    phase_queries: torch.Tensor,
    *,
    brief_size: int,
    retrieval_top_k: int,
) -> torch.Tensor:
    """Frozen task schema + real H15 phase + phase-indexed causal values."""
    retrieved = bank.retrieve(
        task_ids,
        phase_queries,
        brief_size=brief_size,
        retrieval_top_k=retrieval_top_k,
    )
    task = bank.task_prototype[task_ids]
    brief_mean = retrieved.brief_signals.mean(dim=1)
    brief_std = retrieved.brief_signals.std(dim=1, unbiased=False)
    retrieved_mean = retrieved.retrieved_signals.mean(dim=1)
    retrieved_std = retrieved.retrieved_signals.std(dim=1, unbiased=False)
    return torch.cat(
        [
            task,
            phase_queries,
            retrieved.phase_token,
            brief_mean,
            brief_std,
            retrieved_mean,
            retrieved_std,
        ],
        dim=-1,
    ).float()


def _make_split(
    payload: dict[str, Any],
    split: str,
    args: Args,
    zte_config: ZevaConfig,
    bank: RobotWinCausalBank,
) -> dict[str, torch.Tensor]:
    cached = payload["splits"][split]
    indices = cached["sample_indices"].long()
    task_ids = cached["task_ids"].long()
    candidates = cached["candidate_actions"].float()[:, :, : args.horizon]
    targets = cached["normalized_expert_actions"].float()[:, : args.horizon]
    if "phase_queries" in cached:
        phase = cached["phase_queries"].float()
    else:
        phase = _phase_queries_for_indices(
            dataset_root=args.dataset_root,
            live_queries=args.live_queries,
            task_subset=args.task_subset,
            zte_config=zte_config,
            split=split,
            sample_indices=indices,
        )
    static = _stage1_static_features(
        bank,
        task_ids,
        phase,
        brief_size=zte_config.brief_memory_size,
        retrieval_top_k=zte_config.retrieval_top_k,
    )
    visual = cached.get("current_visual_features")
    if visual is not None:
        static = torch.cat([static, visual.float()], dim=-1)
    pi_vlm = cached.get("current_pi_vlm_features")
    if pi_vlm is not None:
        static = torch.cat([static, pi_vlm.float()], dim=-1)
    errors = (candidates - targets[:, None]).square().mean(dim=(2, 3))
    return {
        "sample_indices": indices,
        "task_ids": task_ids,
        "static": static,
        "candidates": candidates,
        "errors": errors,
    }


def _metrics(
    predicted_log_errors: torch.Tensor,
    errors: torch.Tensor,
    task_ids: torch.Tensor,
    task_names: list[str],
    *,
    threshold: float,
) -> dict[str, Any]:
    base = errors[:, 0]
    oracle = errors.min(dim=1).values
    best_nonbase_score, best_nonbase_index = predicted_log_errors[:, 1:].min(dim=1)
    best_nonbase_index = best_nonbase_index + 1
    predicted_gain = predicted_log_errors[:, 0] - best_nonbase_score
    choose_nonbase = predicted_gain > threshold
    selected_index = torch.where(choose_nonbase, best_nonbase_index, torch.zeros_like(best_nonbase_index))
    selected = errors.gather(1, selected_index[:, None]).squeeze(1)
    improvement = base.mean() - selected.mean()
    oracle_room = (base.mean() - oracle.mean()).clamp_min(1e-12)
    per_task = {}
    for task_id, name in enumerate(task_names):
        mask = task_ids == task_id
        if not bool(mask.any()):
            continue
        task_base = base[mask].mean()
        task_selected = selected[mask].mean()
        per_task[name] = {
            "count": int(mask.sum()),
            "base_mse_h15": float(task_base),
            "selected_mse_h15": float(task_selected),
            "absolute_improvement": float(task_base - task_selected),
            "nonbase_fraction": float(choose_nonbase[mask].float().mean()),
        }
    return {
        "count": int(len(base)),
        "threshold": float(threshold),
        "base_mse_h15": float(base.mean()),
        "selected_mse_h15": float(selected.mean()),
        "oracle_mse_h15": float(oracle.mean()),
        "absolute_improvement": float(improvement),
        "relative_improvement": float(improvement / base.mean().clamp_min(1e-12)),
        "oracle_fraction_captured": float(improvement / oracle_room),
        "nonbase_fraction": float(choose_nonbase.float().mean()),
        "top1_oracle_accuracy": float(
            (predicted_log_errors.argmin(dim=1) == errors.argmin(dim=1)).float().mean()
        ),
        "per_task": per_task,
    }


@torch.inference_mode()
def _predict(
    model: nn.Module,
    split: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    model.eval()
    outputs = []
    for start in range(0, len(split["static"]), batch_size):
        stop = start + batch_size
        outputs.append(
            model(
                split["static"][start:stop].to(device),
                split["candidates"][start:stop].to(device),
            ).cpu()
        )
    return torch.cat(outputs)


def _choose_threshold(
    predicted: torch.Tensor,
    errors: torch.Tensor,
) -> tuple[float, list[dict[str, float]]]:
    best_nonbase = predicted[:, 1:].min(dim=1).values
    gain = predicted[:, 0] - best_nonbase
    finite = gain[torch.isfinite(gain)]
    candidates = torch.unique(
        torch.cat(
            [
                torch.tensor([float("-inf"), float("inf")]),
                torch.quantile(finite, torch.linspace(0, 1, 101)),
            ]
        )
    )
    base_mean = errors[:, 0].mean()
    rows = []
    best = (float("inf"), float("inf"))
    for value in candidates.tolist():
        metrics = _metrics(predicted, errors, torch.zeros(len(errors), dtype=torch.long), ["all"], threshold=value)
        selected = metrics["selected_mse_h15"]
        nonbase = metrics["nonbase_fraction"]
        rows.append({"threshold": float(value), "mse": selected, "nonbase_fraction": nonbase})
        # Require calibration non-regression.  Among safe thresholds choose the
        # lowest MSE, breaking ties toward less intervention.
        if selected <= float(base_mean) + 1e-12 and (selected, nonbase) < best:
            best = (selected, nonbase)
            threshold = float(value)
    if not math.isfinite(best[0]):
        threshold = float("inf")
    return threshold, rows


def main(args: Args) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("v15 ranker training requires CUDA.")
    device = torch.device("cuda")
    cache_path = Path(args.cache).resolve()
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "zeva-robotwin-pi-candidate-cache-v15":
        raise ValueError("Unsupported candidate cache.")
    zte_payload = torch.load(args.zte_checkpoint, map_location="cpu", weights_only=False)
    zte_config = ZevaConfig(**zte_payload["zte_config"])
    bank = RobotWinCausalBank.load(args.causal_bank, device="cpu")
    if tuple(bank.task_names) != tuple(payload["task_names"]):
        raise ValueError("Candidate cache and Stage1 causal bank task ordering differ.")
    train = _make_split(payload, "train", args, zte_config, bank)
    validation = _make_split(payload, "validation", args, zte_config, bank)

    audit_mask = validation["sample_indices"].remainder(2) == 1
    calibration_mask = ~audit_mask
    if not bool(audit_mask.any()) or not bool(calibration_mask.any()):
        raise RuntimeError("Validation cannot be divided into calibration and audit halves.")
    model = RobotWinPICandidateRanker(
        static_dim=train["static"].shape[-1],
        horizon=args.horizon,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loader = DataLoader(
        TensorDataset(train["static"], train["candidates"], train["errors"]),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        pin_memory=True,
    )
    best_state = None
    best_epoch = -1
    best_validation = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = count = 0
        for static, candidates, errors in loader:
            static = static.to(device, non_blocking=True)
            candidates = candidates.to(device, non_blocking=True)
            errors = errors.to(device, non_blocking=True)
            predicted = model(static, candidates)
            log_errors = errors.clamp_min(1e-8).log()
            regression = F.smooth_l1_loss(predicted, log_errors)
            listwise = F.cross_entropy(-predicted, errors.argmin(dim=1))
            loss = regression + args.listwise_weight * listwise
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(static)
            count += len(static)
        validation_predicted = _predict(model, validation, device, args.batch_size)
        validation_regression = F.smooth_l1_loss(
            validation_predicted, validation["errors"].clamp_min(1e-8).log()
        ).item()
        history.append(
            {
                "epoch": epoch,
                "train_loss": total / count,
                "validation_log_mse_smooth_l1": validation_regression,
            }
        )
        if validation_regression < best_validation:
            best_validation = validation_regression
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("Ranker training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    predicted = _predict(model, validation, device, args.batch_size)
    threshold, threshold_search = _choose_threshold(
        predicted[calibration_mask], validation["errors"][calibration_mask]
    )
    calibration = _metrics(
        predicted[calibration_mask],
        validation["errors"][calibration_mask],
        validation["task_ids"][calibration_mask],
        payload["task_names"],
        threshold=threshold,
    )
    audit = _metrics(
        predicted[audit_mask],
        validation["errors"][audit_mask],
        validation["task_ids"][audit_mask],
        payload["task_names"],
        threshold=threshold,
    )
    positive_tasks = sum(
        item["absolute_improvement"] >= 0 for item in audit["per_task"].values()
    )
    passed = audit["absolute_improvement"] > 0 and positive_tasks >= 8
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "zeva-robotwin-pi-candidate-ranker-v15",
        "method": "frozen_pi_k4_generation_plus_stage1_causal_value_selection",
        "best_epoch": best_epoch,
        "best_validation_log_mse_smooth_l1": best_validation,
        "base_fallback_threshold": threshold,
        "audit_gate": {
            "passed": passed,
            "requirements": "audit improvement > 0 and >=8/10 non-degrading tasks",
            "positive_or_equal_tasks": positive_tasks,
        },
        "calibration": calibration,
        "audit": audit,
        "frozen_components": ["PI0.5", "Stage1_ZTE", "causal_bank"],
        "task_identity_at_deployment": "task-language retrieval; no episode_index lookup",
        "phase_contract": "real H15 recurrent Stage1 phase query",
        "cache": str(cache_path),
        "cache_sha256": _sha256(cache_path),
        "zte_checkpoint": str(Path(args.zte_checkpoint).resolve()),
        "zte_checkpoint_sha256": _sha256(args.zte_checkpoint),
        "causal_bank": str(Path(args.causal_bank).resolve()),
        "causal_bank_sha256": _sha256(args.causal_bank),
        "train_args": dataclasses.asdict(args),
        "history": history,
        "threshold_search": threshold_search,
    }
    torch.save(
        {
            "schema": manifest["schema"],
            "model_state_dict": best_state,
            "model_config": {
                "static_dim": train["static"].shape[-1],
                "horizon": args.horizon,
                "action_dim": 16,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
            },
            "base_fallback_threshold": threshold,
            "manifest": manifest,
        },
        output / "ranker_best.pt",
    )
    (output / "ranker_audit.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"passed": passed, "best_epoch": best_epoch, "calibration": calibration, "audit": audit}, indent=2))


if __name__ == "__main__":
    main(tyro.cli(Args))
