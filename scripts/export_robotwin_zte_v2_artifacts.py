"""Export v2 Stage-1 train95 bank and deployment live queries.

The exporter deliberately uses the complete-episode v2 dataset contract.  A
goal-conditioned forward rolls the three causal streams from B0 through every
H15 transition; a second goal-masked forward supplies the global task
prototype.  No episode index is used for retrieval: ``record_index`` is kept
only as immutable source identity in the live cache.

``--max-episodes`` is a bounded smoke option.  Such artifacts preserve source
ordering but are marked ``incomplete=true`` and ``usable_for_training=false``
in both manifests, so they cannot be mistaken for a train95 bank.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import math
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F  # noqa: N812


LIVE_QUERY_SCHEMA = "zeva-robotwin-live-queries-h15-v1"
V2_STAGE1_SCHEMA = "zeva-robotwin-zte-stage1-v2-checkpoint"
HORIZON = 15


@dataclasses.dataclass
class Args:
    handoff_root: str = (
        "/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1"
    )
    dataset_root: str = "/data1/huangbingjia/robotwin-lerobot-sidney-eef16-v1/data"
    goal_embeddings: str = "/data1/dingxin/zeva-runs/robotwin-v5-h15-tasklang/pi05_task_embeddings.pt"
    checkpoint: str = (
        "/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/"
        "stage1-zte-v2-phase-vector-mse-4096-20260911h/zte_v2_step_000512.pth"
    )
    output_dir: str = (
        "/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/"
        "stage1-zte-v2-artifacts-20260911"
    )
    phase_bins: int = 32
    samples_per_episode: int = 4
    batch_size: int = 1
    num_workers: int = 0
    max_episodes: int | None = None
    device: str = "cuda"
    seed: int = 1000


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_decision_frames(length: int, horizon: int = HORIZON) -> tuple[int, ...]:
    before = list(range(0, int(length) - horizon, horizon))
    if not before:
        raise ValueError(f"Episode length {length} is shorter than H{horizon}.")
    return tuple([0, *[frame + horizon for frame in before]])


def validate_export_args(args: Args) -> None:
    if args.phase_bins <= 0:
        raise ValueError("phase_bins must be positive.")
    if args.samples_per_episode <= 0 or args.samples_per_episode > args.phase_bins:
        raise ValueError("samples_per_episode must be in [1, phase_bins].")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative.")
    if args.max_episodes is not None and args.max_episodes <= 0:
        raise ValueError("max_episodes must be positive when supplied.")


def _select_record_indices(records: Sequence[dict[str, Any]], max_episodes: int | None) -> list[int]:
    valid = [index for index, record in enumerate(records) if int(record["length"]) > HORIZON]
    if max_episodes is not None:
        valid = valid[:max_episodes]
    if not valid:
        raise ValueError("No complete H15 episodes are available for export.")
    return valid


def _collection_mask(
    progress: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    phase_bins: int,
    samples_per_episode: int,
) -> torch.Tensor:
    """Pick sparse phase-bin observations without consulting episode IDs."""

    if progress.ndim != 2 or valid_mask.shape != progress.shape:
        raise ValueError("progress and valid_mask must both be [B,T].")
    result = torch.zeros_like(valid_mask, dtype=torch.bool)
    desired = torch.linspace(0.0, 1.0, samples_per_episode, device=progress.device)
    for row in range(progress.shape[0]):
        available = torch.nonzero(valid_mask[row], as_tuple=False).flatten()
        if not len(available):
            continue
        chosen: list[int] = []
        for target in desired:
            order = torch.argsort((progress[row, available] - target).abs()).tolist()
            candidate = next((int(available[position]) for position in order if int(available[position]) not in chosen), None)
            if candidate is not None:
                chosen.append(candidate)
        if chosen:
            result[row, torch.as_tensor(chosen, device=result.device)] = True
    return result


def _fill_empty_bins(values: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Fill only bins of tasks with data; leave absent smoke tasks zero."""

    result = values.clone()
    for task_id in range(result.shape[0]):
        valid = torch.nonzero(counts[task_id] > 0, as_tuple=False).flatten()
        if not len(valid):
            continue
        for bin_id in range(result.shape[1]):
            if counts[task_id, bin_id] == 0:
                nearest = valid[torch.argmin(torch.abs(valid - bin_id))]
                result[task_id, bin_id] = result[task_id, nearest]
    return result


def aggregate_bank(
    *,
    task_count: int,
    phase_bins: int,
    task_dim: int,
    phase_dim: int,
    signal_dim: int,
    task_ids: torch.Tensor,
    progress: torch.Tensor,
    phase: torch.Tensor,
    initial_phase: torch.Tensor,
    causal: torch.Tensor,
    global_prompt_masked: torch.Tensor,
    valid_mask: torch.Tensor,
    samples_per_episode: int,
) -> dict[str, torch.Tensor]:
    """Aggregate one or more complete episode batches into bank tensors."""

    if not (progress.shape == valid_mask.shape == phase.shape[:2] == causal.shape[:2]):
        raise ValueError("Bank batch time dimensions disagree.")
    if phase.shape[-1] != phase_dim or initial_phase.shape != (len(task_ids), phase_dim):
        raise ValueError("Phase representations disagree with config.")
    if causal.shape[-1] != signal_dim:
        raise ValueError("Bank representation dimensions disagree with config.")
    if global_prompt_masked.shape != (len(task_ids), task_dim):
        raise ValueError("Masked global prompt must be [B, task_dim].")
    count = torch.zeros(task_count, phase_bins, dtype=torch.float64, device=progress.device)
    progress_sum = torch.zeros_like(count)
    phase_sum = torch.zeros(task_count, phase_bins, phase_dim, dtype=torch.float64, device=progress.device)
    value_sum = torch.zeros(task_count, phase_bins, signal_dim, dtype=torch.float64, device=progress.device)
    task_count_sum = torch.zeros(task_count, dtype=torch.float64, device=progress.device)
    task_sum = torch.zeros(task_count, task_dim, dtype=torch.float64, device=progress.device)
    collect = _collection_mask(
        progress,
        valid_mask,
        phase_bins=phase_bins,
        samples_per_episode=samples_per_episode,
    )
    for row, task_id in enumerate(task_ids.to(torch.long).tolist()):
        if task_id < 0 or task_id >= task_count:
            raise ValueError(f"Task id {task_id} is outside the bank task table.")
        task_count_sum[task_id] += 1
        task_sum[task_id] += global_prompt_masked[row].to(torch.float64)
        # B0 has a real phase query but no causal signal.  Its count and phase
        # key occupy bin zero exactly as deployment starts from B0.
        count[task_id, 0] += 1
        phase_sum[task_id, 0] += initial_phase[row].to(torch.float64)
        for time in torch.nonzero(collect[row], as_tuple=False).flatten().tolist():
            bin_id = int(torch.round(progress[row, time] * (phase_bins - 1)).clamp(0, phase_bins - 1).item())
            count[task_id, bin_id] += 1
            progress_sum[task_id, bin_id] += progress[row, time].double()
            phase_sum[task_id, bin_id] += phase[row, time].double()
            value_sum[task_id, bin_id] += causal[row, time].double()
    divisor = count.clamp_min(1.0)
    task_prototype = F.normalize((task_sum / task_count_sum.clamp_min(1.0).unsqueeze(-1)).float(), dim=-1)
    phase_key = F.normalize((phase_sum / divisor.unsqueeze(-1)).float(), dim=-1)
    causal_value = F.normalize((value_sum / divisor.unsqueeze(-1)).float(), dim=-1)
    phase_key = _fill_empty_bins(phase_key, count)
    causal_value = _fill_empty_bins(causal_value, count)
    return {
        "count": count.to(torch.long),
        "progress": (progress_sum / divisor).float(),
        "task_prototype": task_prototype,
        "phase_key": phase_key,
        "causal_value": causal_value,
    }


def validate_live_record(record: dict[str, Any], *, length: int, task_id: int) -> None:
    expected = expected_decision_frames(length)
    if int(record.get("task_id", -1)) != task_id:
        raise ValueError("Live record task_id does not match source dataset.")
    frames = tuple(int(value) for value in torch.as_tensor(record["decision_frames"]).tolist())
    if frames != expected:
        raise ValueError(f"Live record frame sequence {frames} != {expected}.")
    phases = torch.as_tensor(record["phase_queries"])
    signals = torch.as_tensor(record["causal_signals"])
    if phases.shape != (len(expected), 128):
        raise ValueError("Live phase_queries must contain B0 plus every H15 phase.")
    if signals.shape != (len(expected) - 1, 256):
        raise ValueError("Live causal_signals must contain one signal per after-frame transition.")


def validate_live_payload(payload: dict[str, Any], *, task_names: Sequence[str], complete: bool) -> None:
    if payload.get("schema") != LIVE_QUERY_SCHEMA:
        raise ValueError("Unsupported v2 live-query schema.")
    if payload.get("stage1_checkpoint_schema") != V2_STAGE1_SCHEMA:
        raise ValueError("v2 live queries must declare the v2 Stage1 checkpoint schema.")
    if tuple(payload.get("task_names", ())) != tuple(task_names):
        raise ValueError("Live task ordering differs from dataset task ordering.")
    if bool(payload.get("incomplete")) == complete:
        raise ValueError("Live incomplete flag disagrees with expected completeness.")
    for split in ("train", "validation"):
        records = payload.get("splits", {}).get(split)
        if not isinstance(records, list):
            raise ValueError(f"Live payload has no {split} record list.")


def _manifest(
    *,
    args: Args,
    checkpoint: dict[str, Any],
    checkpoint_sha256: str,
    stats_sha256: str,
    goal_sha256: str,
    adapter_sha256: str,
    exporter_path: Path,
    encoder_path: Path,
    split: str,
    exported_episodes: int,
    total_episodes: int,
) -> dict[str, Any]:
    incomplete = exported_episodes != total_episodes
    return {
        "schema": V2_STAGE1_SCHEMA,
        "stage1_checkpoint_schema": V2_STAGE1_SCHEMA,
        "stage1_checkpoint_sha256": checkpoint_sha256,
        "stage1_step": int(checkpoint["step"]),
        "statistics_sha256": stats_sha256,
        "goal_embeddings_sha256": goal_sha256,
        "dataset_adapter_sha256": adapter_sha256,
        "split": split,
        "episodes": exported_episodes,
        "source_episodes": total_episodes,
        "incomplete": incomplete,
        "usable_for_training": not incomplete,
        "causal_transition_horizon": HORIZON,
        "recurrent_history": "complete-consecutive-H15-rollout-from-B0",
        "goal_conditioning": "phase_and_causal_goal_conditioned; global_prompt_from_separate_goal_masked_forward",
        "episode_index_oracle": False,
        "export_args": dataclasses.asdict(args),
        "source_files": {
            "exporter": sha256_file(exporter_path),
            "transition_encoder": sha256_file(encoder_path),
        },
    }


def _checkpoint(args: Args, handoff, device: torch.device):
    from openpi.zeva.stage1_checkpoint import V2_SCHEMA, load_stage1_encoder

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") != V2_SCHEMA:
        raise ValueError(f"v2 artifact export requires schema {V2_SCHEMA!r}.")
    if int(payload.get("zte_config", {}).get("executed_action_steps", -1)) != HORIZON:
        raise ValueError("v2 checkpoint is not an H15 encoder.")
    stats_sha = sha256_file(handoff.statistics)
    manifest = payload.get("manifest", {})
    if manifest.get("statistics_sha256") != stats_sha:
        raise ValueError("v2 checkpoint statistics do not match the handoff statistics.")
    encoder = load_stage1_encoder(payload, device=device).eval().requires_grad_(False)
    return payload, encoder, sha256_file(args.checkpoint), stats_sha


def _dataset(args: Args, subset: str):
    from scripts.train_robotwin_zte_v2 import RobotWinZTEEpisodeDataset, use_torchcodec_source

    manifest = Path(args.dataset_root) / "adapter.json"
    dataset = RobotWinZTEEpisodeDataset(
        manifest,
        subset=subset,
        transition_stride=HORIZON,
        effect_steps=HORIZON,
        executed_action_steps=HORIZON,
        goal_embeddings=args.goal_embeddings,
    )
    return use_torchcodec_source(dataset, manifest, subset)


def _export_split(
    *,
    args: Args,
    subset: str,
    dataset,
    encoder,
    normalizer,
    device: torch.device,
    task_names: Sequence[str],
    checkpoint: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    from torch.utils.data import DataLoader
    from scripts.train_robotwin_zte_v2 import PaddedEpisodeDataset, collate_robotwin_episodes

    original_record_indices = list(dataset._record_indices)  # noqa: SLF001
    selected = _select_record_indices(dataset.dataset._records, args.max_episodes)
    selected_set = set(selected)
    # The v2 dataset indexes only records with at least one H15 transition.
    dataset._record_indices = [index for index in original_record_indices if index in selected_set]  # noqa: SLF001
    if len(dataset) != len(selected):
        raise RuntimeError("Selected record count disagrees with the v2 complete-episode dataset.")
    if [int(value) for value in dataset._record_indices] != selected:  # noqa: SLF001
        raise RuntimeError("Selected records are not in source record order.")
    loader = DataLoader(
        PaddedEpisodeDataset(dataset),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_robotwin_episodes,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    from openpi.zeva.robotwin_contract import MeanStdActionNormalizer

    count = len(dataset)
    live_records: list[dict[str, Any]] = []
    bank_parts: list[dict[str, torch.Tensor]] = []
    source_position = 0
    with torch.inference_mode():
        for raw_batch in loader:
            batch = {
                key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in raw_batch.items()
            }
            valid_mask = batch["valid_mask"]
            encoder_mask = valid_mask.clone()
            encoder_mask[:, 0] = True
            actions = normalizer.normalize(batch["actions"])
            outputs = encoder(
                batch["images_before"], actions, batch["images_after"], batch["goal_embedding"],
                valid_mask=encoder_mask,
            )
            # This is intentionally an independent pass.  The global task
            # prototype must not be a goal-conditioned representation.
            masked = encoder(
                batch["images_before"], actions, batch["images_after"], batch["goal_embedding"],
                # Keep this explicit rather than relying on ``None`` to mean
                # zero goal: it makes the language-masked prototype contract
                # visible to future encoder implementations and audits.
                goal_mask=torch.ones(
                    batch["goal_embedding"].shape[0], device=device, dtype=torch.bool
                ),
                valid_mask=encoder_mask,
            )
            real_rows = batch["episode_mask"].bool()
            for local_row in torch.nonzero(real_rows, as_tuple=False).flatten().tolist():
                source_index = int(dataset._record_indices[source_position])  # noqa: SLF001
                source_position += 1
                record = dataset.dataset._records[source_index]  # noqa: SLF001
                task_id = int(batch["task_id"][local_row].item())
                length = int(record["length"])
                time_mask = valid_mask[local_row]
                time_count = int(time_mask.sum().item())
                decision_frames = expected_decision_frames(length)
                phases = torch.cat(
                    [outputs.initial_phase_token[local_row:local_row + 1], outputs.phase_token[local_row]], dim=0
                )
                causal = outputs.causal_signal[local_row, :time_count]
                if phases.shape[0] != len(decision_frames) or causal.shape[0] != len(decision_frames) - 1:
                    raise RuntimeError("Encoder output length does not match source H15 frame sequence.")
                live = {
                    "record_index": source_index,
                    "task_id": task_id,
                    "decision_frames": torch.tensor(decision_frames, dtype=torch.int32),
                    "phase_queries": phases.cpu().to(torch.float16),
                    "causal_signals": causal.cpu().to(torch.float16),
                }
                validate_live_record(live, length=length, task_id=task_id)
                live_records.append(live)
            bank_parts.append(
                {
                    "task_ids": batch["task_id"][real_rows].detach(),
                    "progress": batch["progress"][real_rows].detach(),
                    "phase": outputs.phase_token[real_rows].detach(),
                    "initial_phase": outputs.initial_phase_token[real_rows].detach(),
                    "causal": outputs.causal_signal[real_rows].detach(),
                    "global_prompt_masked": masked.global_prompt[real_rows].detach(),
                    "valid_mask": valid_mask[real_rows].detach(),
                }
            )
    if source_position != count:
        raise RuntimeError(f"Exported {source_position} records, expected {count}.")
    live_records.sort(key=lambda row: int(row["record_index"]))
    if [int(row["record_index"]) for row in live_records] != selected:
        raise RuntimeError("Live records are missing or duplicated source record indices.")
    merged = {
        key: torch.cat([part[key] for part in bank_parts], dim=0)
        for key in bank_parts[0]
    }
    return {
        "schema": LIVE_QUERY_SCHEMA,
        "transition_horizon": HORIZON,
        "stage1_checkpoint_schema": V2_STAGE1_SCHEMA,
        "task_names": tuple(task_names),
        "records": live_records,
    }, merged


def run(args: Args) -> tuple[dict[str, Any], dict[str, Any]]:
    validate_export_args(args)
    torch.manual_seed(args.seed)
    from openpi.zeva.robotwin_contract import MeanStdActionNormalizer, RobotWinHandoff
    from openpi.zeva.transition_encoder_v2 import CausalTransitionEncoderV2

    handoff = RobotWinHandoff.from_root(args.handoff_root)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    checkpoint, encoder, checkpoint_sha, stats_sha = _checkpoint(args, handoff, device)
    goal_sha = sha256_file(args.goal_embeddings)
    adapter_manifest = Path(args.dataset_root) / "adapter.json"
    adapter_sha = sha256_file(adapter_manifest)
    normalizer = MeanStdActionNormalizer.from_stats_file(handoff.statistics)
    source_by_split = {split: _dataset(args, split) for split in ("train", "validation")}
    # ``RobotWinZTEEpisodeDataset`` excludes records too short for a complete
    # H15 transition.  Those are not exportable source episodes, so capture
    # the valid counts before _export_split narrows each dataset for a smoke.
    source_episode_counts = {split: len(dataset) for split, dataset in source_by_split.items()}
    task_names = source_by_split["train"].task_names
    if source_by_split["validation"].task_names != task_names:
        raise ValueError("Train and validation task ordering differ.")
    split_payloads: dict[str, dict[str, Any]] = {}
    train_bank_inputs: dict[str, torch.Tensor] | None = None
    for split, dataset in source_by_split.items():
        payload, bank_inputs = _export_split(
            args=args,
            subset=split,
            dataset=dataset,
            encoder=encoder,
            normalizer=normalizer,
            device=device,
            task_names=task_names,
            checkpoint=checkpoint,
        )
        split_payloads[split] = payload
        if split == "train":
            train_bank_inputs = bank_inputs
    assert train_bank_inputs is not None
    train_total = source_episode_counts["train"]
    bank_manifest = _manifest(
        args=args,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha,
        stats_sha256=stats_sha,
        goal_sha256=goal_sha,
        adapter_sha256=adapter_sha,
        exporter_path=Path(__file__).resolve(),
        encoder_path=Path(inspect.getfile(CausalTransitionEncoderV2)).resolve(),
        split="train95",
        exported_episodes=len(split_payloads["train"]["records"]),
        total_episodes=train_total,
    )
    bank_tensors = aggregate_bank(
        task_count=len(task_names),
        phase_bins=args.phase_bins,
        task_dim=checkpoint["zte_config"]["task_dim"],
        phase_dim=checkpoint["zte_config"]["phase_dim"],
        signal_dim=checkpoint["zte_config"]["signal_dim"],
        samples_per_episode=args.samples_per_episode,
        **train_bank_inputs,
    )
    bank_payload = {
        "schema": "zeva-robotwin-train-causal-bank-v2",
        "split": "train95",
        "stage1_checkpoint_schema": V2_STAGE1_SCHEMA,
        "incomplete": bank_manifest["incomplete"],
        "usable_for_training": bank_manifest["usable_for_training"],
        "task_names": tuple(task_names),
        "phase_bins": args.phase_bins,
        **bank_tensors,
        "manifest": bank_manifest,
    }
    live_complete = all(
        len(split_payloads[split]["records"]) == source_episode_counts[split]
        for split in source_by_split
    )
    live_payload = {
        "schema": LIVE_QUERY_SCHEMA,
        "transition_horizon": HORIZON,
        "stage1_checkpoint_schema": V2_STAGE1_SCHEMA,
        "zte_checkpoint_schema": V2_STAGE1_SCHEMA,
        "zte_checkpoint_sha256": checkpoint_sha,
        "statistics_sha256": stats_sha,
        "goal_embeddings_sha256": goal_sha,
        "task_names": tuple(task_names),
        "incomplete": not live_complete,
        "usable_for_training": live_complete,
        "source_record_counts": {
            split: source_episode_counts[split] for split in source_by_split
        },
        "exported_record_counts": {split: len(payload["records"]) for split, payload in split_payloads.items()},
        "splits": {split: payload["records"] for split, payload in split_payloads.items()},
        "manifest": {
            "stage1_checkpoint_schema": V2_STAGE1_SCHEMA,
            "stage1_checkpoint_sha256": checkpoint_sha,
            "stage1_step": int(checkpoint["step"]),
            "statistics_sha256": stats_sha,
            "goal_embeddings_sha256": goal_sha,
            "dataset_adapter_sha256": adapter_sha,
            "incomplete": not live_complete,
            "usable_for_training": live_complete,
            "causal_transition_horizon": HORIZON,
            "recurrent_history": "complete-consecutive-H15-rollout-from-B0",
            "source_files": {
                "exporter": sha256_file(Path(__file__).resolve()),
                "transition_encoder": sha256_file(Path(inspect.getfile(CausalTransitionEncoderV2)).resolve()),
            },
        },
    }
    validate_live_payload(live_payload, task_names=task_names, complete=live_complete)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bank_path = output_dir / "train_causal_bank.pt"
    live_path = output_dir / "live_queries_h15.pt"
    torch.save(bank_payload, bank_path)
    torch.save(live_payload, live_path)
    summary = {
        "schema": "zeva-robotwin-zte-v2-artifacts-v1",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "stage1_checkpoint_schema": V2_STAGE1_SCHEMA,
        "bank": str(bank_path),
        "live_queries": str(live_path),
        "incomplete": not live_complete,
        "usable_for_training": live_complete,
        "train_records": len(split_payloads["train"]["records"]),
        "validation_records": len(split_payloads["validation"]["records"]),
    }
    (output_dir / "export_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return bank_payload, live_payload


def main() -> None:
    import tyro

    bank, live = run(tyro.cli(Args))
    print(json.dumps({
        "bank_schema": bank["schema"],
        "live_schema": live["schema"],
        "incomplete": live["incomplete"],
        "usable_for_training": live["usable_for_training"],
    }))


if __name__ == "__main__":
    main()
