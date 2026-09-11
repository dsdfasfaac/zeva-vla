"""Export v2 Stage-1 train95 bank and deployment live queries.

The exporter deliberately uses the complete-episode v2 dataset contract.  A
goal-conditioned forward rolls the three causal streams from B0 through every
H15 transition; a second goal-masked forward supplies the global task
prototype.  No episode index is used for retrieval: ``record_index`` is kept
only as immutable source identity in the live cache.

``--max-episodes`` is a bounded smoke option.  Such artifacts preserve source
ordering but are marked ``incomplete=true`` and ``usable_for_training=false``
in both manifests, so they cannot be mistaken for a train95 bank.

For large exports, ``--world-size N --rank R`` writes deterministic source
shards containing CPU bank sums and live rows.  Rank 0 can later run with
``--merge-shards`` to verify hashes/configuration, reject gaps or overlaps,
and materialize the final bank/cache without re-running the encoder.
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
PARTIAL_SCHEMA = "zeva-robotwin-zte-v2-artifact-shard-v1"
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
    log_every: int = 100
    rank: int = 0
    world_size: int = 1
    merge_shards: bool = False


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
    if args.log_every <= 0:
        raise ValueError("log_every must be positive.")
    if args.world_size <= 0 or not 0 <= args.rank < args.world_size:
        raise ValueError("rank must satisfy 0 <= rank < world_size, with world_size positive.")
    if args.merge_shards and args.rank != 0:
        raise ValueError("Only rank 0 may merge artifact shards.")


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


def _new_bank_accumulator(
    *,
    task_count: int,
    phase_bins: int,
    task_dim: int,
    phase_dim: int,
    signal_dim: int,
) -> dict[str, torch.Tensor]:
    """Allocate fixed-size CPU sums so episode lengths never enter storage shape."""

    return {
        "count": torch.zeros(task_count, phase_bins, dtype=torch.float64),
        "progress_sum": torch.zeros(task_count, phase_bins, dtype=torch.float64),
        "phase_sum": torch.zeros(task_count, phase_bins, phase_dim, dtype=torch.float64),
        "value_sum": torch.zeros(task_count, phase_bins, signal_dim, dtype=torch.float64),
        "task_count_sum": torch.zeros(task_count, dtype=torch.float64),
        "task_sum": torch.zeros(task_count, task_dim, dtype=torch.float64),
    }


def _validate_bank_batch(
    *,
    task_ids: torch.Tensor,
    progress: torch.Tensor,
    phase: torch.Tensor,
    initial_phase: torch.Tensor,
    causal: torch.Tensor,
    global_prompt_masked: torch.Tensor,
    valid_mask: torch.Tensor,
    state: dict[str, torch.Tensor],
) -> None:
    if progress.ndim != 2 or valid_mask.shape != progress.shape:
        raise ValueError("progress and valid_mask must both be [B,T].")
    if task_ids.ndim != 1 or len(task_ids) != progress.shape[0]:
        raise ValueError("task_ids must be [B] and agree with the batch.")
    if phase.shape[:2] != progress.shape or causal.shape[:2] != progress.shape:
        raise ValueError("Bank batch time dimensions disagree.")
    if initial_phase.ndim != 2 or initial_phase.shape[0] != len(task_ids):
        raise ValueError("initial_phase must be [B, phase_dim].")
    if global_prompt_masked.ndim != 2 or global_prompt_masked.shape[0] != len(task_ids):
        raise ValueError("Masked global prompt must be [B, task_dim].")
    if state["phase_sum"].shape[-1] != phase.shape[-1] or state["value_sum"].shape[-1] != causal.shape[-1]:
        raise ValueError("Bank representation dimensions disagree with accumulator.")
    if initial_phase.shape[-1] != phase.shape[-1]:
        raise ValueError("Initial and transition phase dimensions disagree.")
    if global_prompt_masked.shape[-1] != state["task_sum"].shape[-1]:
        raise ValueError("Masked global prompt dimension disagrees with accumulator.")


def update_bank_accumulator(
    state: dict[str, torch.Tensor],
    *,
    task_ids: torch.Tensor,
    progress: torch.Tensor,
    phase: torch.Tensor,
    initial_phase: torch.Tensor,
    causal: torch.Tensor,
    global_prompt_masked: torch.Tensor,
    valid_mask: torch.Tensor,
    samples_per_episode: int,
) -> dict[str, torch.Tensor]:
    """Add one variable-length batch to fixed CPU bank sums.

    The encoder output is copied to CPU before accumulation.  This keeps only
    the fixed bank-size tensors alive across batches and makes unequal padded
    sequence lengths safe for both batch and shard export.
    """

    _validate_bank_batch(
        task_ids=task_ids,
        progress=progress,
        phase=phase,
        initial_phase=initial_phase,
        causal=causal,
        global_prompt_masked=global_prompt_masked,
        valid_mask=valid_mask,
        state=state,
    )
    task_ids = task_ids.detach().to(device="cpu", dtype=torch.long)
    progress = progress.detach().to(device="cpu", dtype=torch.float32)
    phase = phase.detach().to(device="cpu", dtype=torch.float32)
    initial_phase = initial_phase.detach().to(device="cpu", dtype=torch.float32)
    causal = causal.detach().to(device="cpu", dtype=torch.float32)
    global_prompt_masked = global_prompt_masked.detach().to(device="cpu", dtype=torch.float32)
    valid_mask = valid_mask.detach().to(device="cpu", dtype=torch.bool)
    task_count, phase_bins = state["count"].shape
    collect = _collection_mask(
        progress,
        valid_mask,
        phase_bins=phase_bins,
        samples_per_episode=samples_per_episode,
    )
    for row, task_id in enumerate(task_ids.tolist()):
        if task_id < 0 or task_id >= task_count:
            raise ValueError(f"Task id {task_id} is outside the bank task table.")
        state["task_count_sum"][task_id] += 1
        state["task_sum"][task_id] += global_prompt_masked[row].to(torch.float64)
        # B0 has a real phase query but no causal signal.  Its count and phase
        # key occupy bin zero exactly as deployment starts from B0.
        state["count"][task_id, 0] += 1
        state["phase_sum"][task_id, 0] += initial_phase[row].to(torch.float64)
        for time in torch.nonzero(collect[row], as_tuple=False).flatten().tolist():
            bin_id = int(torch.round(progress[row, time] * (phase_bins - 1)).clamp(0, phase_bins - 1).item())
            state["count"][task_id, bin_id] += 1
            state["progress_sum"][task_id, bin_id] += progress[row, time].double()
            state["phase_sum"][task_id, bin_id] += phase[row, time].double()
            state["value_sum"][task_id, bin_id] += causal[row, time].double()
    return state


def finalize_bank_accumulator(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Normalize fixed CPU sums into the train-bank payload fields."""

    count = state["count"]
    divisor = count.clamp_min(1.0)
    task_prototype = F.normalize(
        (state["task_sum"] / state["task_count_sum"].clamp_min(1.0).unsqueeze(-1)).float(),
        dim=-1,
    )
    phase_key = F.normalize((state["phase_sum"] / divisor.unsqueeze(-1)).float(), dim=-1)
    causal_value = F.normalize((state["value_sum"] / divisor.unsqueeze(-1)).float(), dim=-1)
    return {
        "count": count.to(torch.long),
        "progress": (state["progress_sum"] / divisor).float(),
        "task_prototype": task_prototype,
        "phase_key": _fill_empty_bins(phase_key, count),
        "causal_value": _fill_empty_bins(causal_value, count),
    }


def merge_bank_accumulators(
    states: Sequence[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Sum shard accumulators after metadata validation."""

    if not states:
        raise ValueError("At least one bank accumulator is required.")
    keys = tuple(states[0])
    if any(tuple(state) != keys for state in states[1:]):
        raise ValueError("Bank accumulator keys differ across shards.")
    merged = {key: states[0][key].clone() for key in keys}
    for state in states[1:]:
        for key in keys:
            if state[key].shape != merged[key].shape:
                raise ValueError(f"Bank accumulator shape differs for {key}.")
            merged[key] += state[key]
    return merged


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
    """Aggregate one batch; :func:`update_bank_accumulator` handles streaming."""

    state = _new_bank_accumulator(
        task_count=task_count,
        phase_bins=phase_bins,
        task_dim=task_dim,
        phase_dim=phase_dim,
        signal_dim=signal_dim,
    )
    update_bank_accumulator(
        state,
        task_ids=task_ids,
        progress=progress,
        phase=phase,
        initial_phase=initial_phase,
        causal=causal,
        global_prompt_masked=global_prompt_masked,
        valid_mask=valid_mask,
        samples_per_episode=samples_per_episode,
    )
    return finalize_bank_accumulator(state)


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


def make_live_record(
    *,
    record_index: int,
    task_id: int,
    length: int,
    initial_phase: torch.Tensor,
    phase_token: torch.Tensor,
    causal_signal: torch.Tensor,
) -> dict[str, Any]:
    """Build one cache row while dropping right-padding from model outputs."""

    decision_frames = expected_decision_frames(length)
    transition_count = len(decision_frames) - 1
    if initial_phase.ndim != 1 or initial_phase.shape[-1] != 128:
        raise ValueError("Initial phase token must be [128].")
    if phase_token.ndim != 2 or phase_token.shape[-1] != 128:
        raise ValueError("Phase token must be [T,128].")
    if causal_signal.ndim != 2 or causal_signal.shape[-1] != 256:
        raise ValueError("Causal signal must be [T,256].")
    if phase_token.shape[0] < transition_count or causal_signal.shape[0] < transition_count:
        raise ValueError("Encoder outputs are shorter than the source H15 episode.")
    phases = torch.cat([initial_phase[None], phase_token[:transition_count]], dim=0)
    causal = causal_signal[:transition_count]
    record = {
        "record_index": int(record_index),
        "task_id": int(task_id),
        "decision_frames": torch.tensor(decision_frames, dtype=torch.int32),
        "phase_queries": phases.detach().cpu().to(torch.float16),
        "causal_signals": causal.detach().cpu().to(torch.float16),
    }
    validate_live_record(record, length=length, task_id=task_id)
    return record


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


def _shard_indices(indices: Sequence[int], *, rank: int, world_size: int) -> list[int]:
    """Deterministically stride a globally ordered source list across shards."""

    return [int(index) for position, index in enumerate(indices) if position % world_size == rank]


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
    selected_global: Sequence[int],
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    from torch.utils.data import DataLoader
    from scripts.train_robotwin_zte_v2 import PaddedEpisodeDataset, collate_robotwin_episodes

    original_record_indices = list(dataset._record_indices)  # noqa: SLF001
    if [int(value) for value in original_record_indices] != sorted(int(value) for value in original_record_indices):
        raise RuntimeError("Complete source record indices are not in source order.")
    selected_global = [int(index) for index in selected_global]
    if not set(selected_global).issubset(set(original_record_indices)):
        raise RuntimeError("Selected source records are not all complete H15 episodes.")
    selected = _shard_indices(
        selected_global,
        rank=args.rank,
        world_size=args.world_size,
    )
    selected_set = set(selected)
    # The v2 dataset indexes only records with at least one H15 transition.
    dataset._record_indices = [index for index in original_record_indices if index in selected_set]  # noqa: SLF001
    if len(dataset) != len(selected):
        raise RuntimeError("Selected record count disagrees with the v2 complete-episode dataset.")
    if [int(value) for value in dataset._record_indices] != sorted(selected):  # noqa: SLF001
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
    state = _new_bank_accumulator(
        task_count=len(task_names),
        phase_bins=args.phase_bins,
        task_dim=int(checkpoint["zte_config"]["task_dim"]),
        phase_dim=int(checkpoint["zte_config"]["phase_dim"]),
        signal_dim=int(checkpoint["zte_config"]["signal_dim"]),
    )
    source_position = 0
    with torch.inference_mode():
        for batch_index, raw_batch in enumerate(loader):
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
                time_count = int(valid_mask[local_row].sum().item())
                expected_count = len(expected_decision_frames(length)) - 1
                if time_count != expected_count:
                    raise RuntimeError("Dataset valid-mask length disagrees with source H15 frame sequence.")
                live_records.append(
                    make_live_record(
                        record_index=source_index,
                        task_id=task_id,
                        length=length,
                        initial_phase=outputs.initial_phase_token[local_row],
                        phase_token=outputs.phase_token[local_row, :time_count],
                        causal_signal=outputs.causal_signal[local_row, :time_count],
                    )
                )
            update_bank_accumulator(
                state,
                task_ids=batch["task_id"][real_rows],
                progress=batch["progress"][real_rows],
                phase=outputs.phase_token[real_rows],
                initial_phase=outputs.initial_phase_token[real_rows],
                causal=outputs.causal_signal[real_rows],
                global_prompt_masked=masked.global_prompt[real_rows],
                valid_mask=valid_mask[real_rows],
                samples_per_episode=args.samples_per_episode,
            )
            if (batch_index + 1) % args.log_every == 0 or source_position == count:
                print(
                    f"export {subset} rank {args.rank}/{args.world_size}: "
                    f"{source_position}/{count} episodes",
                    flush=True,
                )
            del outputs, masked, batch, raw_batch
    if source_position != count:
        raise RuntimeError(f"Exported {source_position} records, expected {count}.")
    live_records.sort(key=lambda row: int(row["record_index"]))
    if [int(row["record_index"]) for row in live_records] != sorted(selected):
        raise RuntimeError("Live records are missing or duplicated source record indices.")
    return {
        "schema": LIVE_QUERY_SCHEMA,
        "transition_horizon": HORIZON,
        "stage1_checkpoint_schema": V2_STAGE1_SCHEMA,
        "task_names": tuple(task_names),
        "records": live_records,
    }, state


def _assert_output_available(output_dir: Path, *, shard_path: Path | None = None) -> None:
    """Refuse accidental overwrite of a completed or prior export."""

    output_dir.mkdir(parents=True, exist_ok=True)
    final_names = {"train_causal_bank.pt", "live_queries_h15.pt", "export_summary.json"}
    if any((output_dir / name).exists() for name in final_names):
        raise FileExistsError(f"Refusing to overwrite existing artifact output: {output_dir}")
    if shard_path is not None and shard_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing shard: {shard_path}")
    if shard_path is not None:
        summary_path = shard_path.with_name(
            f"export_summary_rank{shard_path.stem.removeprefix('shard_rank')}.json"
        )
        if summary_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing shard summary: {summary_path}")


def _load_checkpoint_payload(args: Args, handoff):
    from openpi.zeva.stage1_checkpoint import V2_SCHEMA

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") != V2_SCHEMA:
        raise ValueError(f"v2 artifact export requires schema {V2_SCHEMA!r}.")
    if int(payload.get("zte_config", {}).get("executed_action_steps", -1)) != HORIZON:
        raise ValueError("v2 checkpoint is not an H15 encoder.")
    stats_sha = sha256_file(handoff.statistics)
    manifest = payload.get("manifest", {})
    if manifest.get("statistics_sha256") != stats_sha:
        raise ValueError("v2 checkpoint statistics do not match the handoff statistics.")
    return payload, sha256_file(args.checkpoint), stats_sha


def _checkpoint(args: Args, handoff, device: torch.device):
    from openpi.zeva.stage1_checkpoint import load_stage1_encoder

    payload, checkpoint_sha, stats_sha = _load_checkpoint_payload(args, handoff)
    encoder = load_stage1_encoder(payload, device=device).eval().requires_grad_(False)
    return payload, encoder, checkpoint_sha, stats_sha


def _source_metadata(source_by_split, args: Args) -> dict[str, Any]:
    metadata = {}
    for split, dataset in source_by_split.items():
        valid_indices = [int(index) for index in dataset._record_indices]  # noqa: SLF001
        if valid_indices != sorted(valid_indices) or len(set(valid_indices)) != len(valid_indices):
            raise RuntimeError(f"{split} complete source record indices are not unique and ordered.")
        expected_valid = _select_record_indices(dataset.dataset._records, None)  # noqa: SLF001
        if valid_indices != expected_valid:
            raise RuntimeError(f"{split} dataset omitted or reordered complete H15 source records.")
        metadata[split] = {
            "valid_indices": valid_indices,
            "episode_count": len(valid_indices),
            "adapter_record_count": len(dataset.dataset._records),  # noqa: SLF001
            "selected_indices": _select_record_indices(dataset.dataset._records, args.max_episodes),  # noqa: SLF001
        }
    return metadata


def _build_final_artifacts(
    *,
    args: Args,
    checkpoint: dict[str, Any],
    checkpoint_sha: str,
    stats_sha: str,
    goal_sha: str,
    adapter_sha: str,
    task_names: Sequence[str],
    split_payloads: dict[str, dict[str, Any]],
    train_state: dict[str, torch.Tensor],
    source_metadata: dict[str, Any],
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from openpi.zeva.transition_encoder_v2 import CausalTransitionEncoderV2

    expected = {
        split: source_metadata[split]["selected_indices"]
        for split in ("train", "validation")
    }
    observed = {
        split: [int(row["record_index"]) for row in split_payloads[split]["records"]]
        for split in ("train", "validation")
    }
    if observed != expected:
        raise RuntimeError(f"Final export source coverage differs: observed={observed}, expected={expected}.")
    bank_complete = observed["train"] == source_metadata["train"]["valid_indices"]
    complete = bank_complete and all(
        observed[split] == source_metadata[split]["valid_indices"] for split in observed
    )
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
        total_episodes=source_metadata["train"]["episode_count"],
    )
    bank_tensors = finalize_bank_accumulator(train_state)
    bank_payload = {
        "schema": "zeva-robotwin-train-causal-bank-v2",
        "split": "train95",
        "stage1_checkpoint_schema": V2_STAGE1_SCHEMA,
        "incomplete": not bank_complete,
        "usable_for_training": bank_complete,
        "task_names": tuple(task_names),
        "phase_bins": args.phase_bins,
        **bank_tensors,
        "manifest": bank_manifest,
    }
    live_payload = {
        "schema": LIVE_QUERY_SCHEMA,
        "transition_horizon": HORIZON,
        "stage1_checkpoint_schema": V2_STAGE1_SCHEMA,
        "zte_checkpoint_schema": V2_STAGE1_SCHEMA,
        "zte_checkpoint_sha256": checkpoint_sha,
        "statistics_sha256": stats_sha,
        "goal_embeddings_sha256": goal_sha,
        "task_names": tuple(task_names),
        "incomplete": not complete,
        "usable_for_training": complete,
        "source_record_counts": {
            split: source_metadata[split]["episode_count"] for split in source_metadata
        },
        "source_adapter_record_counts": {
            split: source_metadata[split]["adapter_record_count"] for split in source_metadata
        },
        "exported_record_counts": {
            split: len(payload["records"]) for split, payload in split_payloads.items()
        },
        "splits": {split: payload["records"] for split, payload in split_payloads.items()},
        "manifest": {
            "stage1_checkpoint_schema": V2_STAGE1_SCHEMA,
            "stage1_checkpoint_sha256": checkpoint_sha,
            "stage1_step": int(checkpoint["step"]),
            "statistics_sha256": stats_sha,
            "goal_embeddings_sha256": goal_sha,
            "dataset_adapter_sha256": adapter_sha,
            "incomplete": not complete,
            "usable_for_training": complete,
            "causal_transition_horizon": HORIZON,
            "recurrent_history": "complete-consecutive-H15-rollout-from-B0",
            "source_record_counts": {
                split: source_metadata[split]["episode_count"] for split in source_metadata
            },
            "source_adapter_record_counts": {
                split: source_metadata[split]["adapter_record_count"] for split in source_metadata
            },
            "source_files": {
                "exporter": sha256_file(Path(__file__).resolve()),
                "transition_encoder": sha256_file(inspect.getfile(CausalTransitionEncoderV2)),
            },
        },
    }
    validate_live_payload(live_payload, task_names=task_names, complete=complete)
    _assert_output_available(output_dir)
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
        "incomplete": not complete,
        "usable_for_training": complete,
        "train_records": len(split_payloads["train"]["records"]),
        "validation_records": len(split_payloads["validation"]["records"]),
    }
    (output_dir / "export_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return bank_payload, live_payload


def _save_shard(
    *,
    args: Args,
    checkpoint: dict[str, Any],
    checkpoint_sha: str,
    stats_sha: str,
    goal_sha: str,
    adapter_sha: str,
    task_names: Sequence[str],
    split_payloads: dict[str, dict[str, Any]],
    train_state: dict[str, torch.Tensor],
    source_metadata: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    shard_path = output_dir / f"shard_rank{args.rank:04d}.pt"
    _assert_output_available(output_dir, shard_path=shard_path)
    payload = {
        "schema": PARTIAL_SCHEMA,
        "stage1_checkpoint_schema": V2_STAGE1_SCHEMA,
        "rank": args.rank,
        "world_size": args.world_size,
        "checkpoint_sha256": checkpoint_sha,
        "statistics_sha256": stats_sha,
        "goal_embeddings_sha256": goal_sha,
        "dataset_adapter_sha256": adapter_sha,
        "stage1_step": int(checkpoint["step"]),
        "task_names": tuple(task_names),
        "phase_bins": args.phase_bins,
        "samples_per_episode": args.samples_per_episode,
        "representation_dims": {
            "task_dim": int(checkpoint["zte_config"]["task_dim"]),
            "phase_dim": int(checkpoint["zte_config"]["phase_dim"]),
            "signal_dim": int(checkpoint["zte_config"]["signal_dim"]),
        },
        "source_episode_counts": {
            split: source_metadata[split]["episode_count"] for split in source_metadata
        },
        "source_adapter_record_counts": {
            split: source_metadata[split]["adapter_record_count"] for split in source_metadata
        },
        "source_valid_record_indices": {
            split: source_metadata[split]["valid_indices"] for split in source_metadata
        },
        "global_selected_record_indices": {
            split: source_metadata[split]["selected_indices"] for split in source_metadata
        },
        "shard_record_indices": {
            split: [int(row["record_index"]) for row in split_payloads[split]["records"]]
            for split in split_payloads
        },
        "export_args": dataclasses.asdict(args),
        "bank_state": train_state,
        "splits": {split: payload["records"] for split, payload in split_payloads.items()},
        "incomplete": True,
        "usable_for_training": False,
    }
    torch.save(payload, shard_path)
    summary = {
        "schema": PARTIAL_SCHEMA,
        "stage1_checkpoint_schema": V2_STAGE1_SCHEMA,
        "rank": args.rank,
        "world_size": args.world_size,
        "shard": str(shard_path),
        "incomplete": True,
        "usable_for_training": False,
        "train_records": len(payload["splits"]["train"]),
        "validation_records": len(payload["splits"]["validation"]),
    }
    (output_dir / f"export_summary_rank{args.rank:04d}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return payload


def _normalized_export_args(values: dict[str, Any]) -> dict[str, Any]:
    result = dict(values)
    # Device may intentionally differ for the CPU-only merge; all export and
    # source-selection knobs remain part of the equality check.
    for key in ("rank", "world_size", "merge_shards", "device"):
        result.pop(key, None)
    return result


def _validate_shards(
    *,
    shards: Sequence[dict[str, Any]],
    args: Args,
    source_metadata: dict[str, Any],
    checkpoint_sha: str,
    stats_sha: str,
    goal_sha: str,
    adapter_sha: str,
) -> None:
    if len(shards) != args.world_size:
        raise ValueError("The shard set does not contain exactly world_size artifacts.")
    ranks = {int(shard.get("rank", -1)) for shard in shards}
    if ranks != set(range(args.world_size)):
        raise ValueError("Artifact shard ranks are missing or duplicated.")
    reference = shards[0]
    compare_keys = (
        "schema", "stage1_checkpoint_schema", "world_size", "checkpoint_sha256",
        "statistics_sha256", "goal_embeddings_sha256", "dataset_adapter_sha256",
        "stage1_step", "task_names", "phase_bins", "samples_per_episode",
        "representation_dims", "source_episode_counts", "source_adapter_record_counts",
        "source_valid_record_indices", "global_selected_record_indices",
    )
    for shard in shards:
        if shard.get("schema") != PARTIAL_SCHEMA:
            raise ValueError("Unsupported artifact shard schema.")
        if shard.get("stage1_checkpoint_schema") != V2_STAGE1_SCHEMA:
            raise ValueError("Artifact shard Stage1 schema differs from v2.")
        if any(shard.get(key) != reference.get(key) for key in compare_keys):
            raise ValueError("Artifact shard metadata/configuration differs across ranks.")
        if shard["checkpoint_sha256"] != checkpoint_sha or shard["statistics_sha256"] != stats_sha:
            raise ValueError("Artifact shard provenance does not match the selected checkpoint/statistics.")
        if shard["goal_embeddings_sha256"] != goal_sha or shard["dataset_adapter_sha256"] != adapter_sha:
            raise ValueError("Artifact shard goal/dataset provenance differs from the selected source.")
        if _normalized_export_args(shard["export_args"]) != _normalized_export_args(reference["export_args"]):
            raise ValueError("Artifact shard export arguments differ across ranks.")
    if _normalized_export_args(dataclasses.asdict(args)) != _normalized_export_args(reference["export_args"]):
        raise ValueError("Merge arguments differ from the export shard configuration.")
    for split, metadata in source_metadata.items():
        if reference["source_valid_record_indices"][split] != metadata["valid_indices"]:
            raise ValueError(f"{split} source record identity differs from the current adapter.")
        if reference["global_selected_record_indices"][split] != metadata["selected_indices"]:
            raise ValueError(f"{split} selected source coverage differs from the current adapter.")
        expected_global = metadata["selected_indices"]
        observed: list[int] = []
        for shard in shards:
            expected_shard = _shard_indices(
                expected_global,
                rank=int(shard["rank"]),
                world_size=args.world_size,
            )
            actual_shard = [int(value) for value in shard["shard_record_indices"][split]]
            if actual_shard != expected_shard:
                raise ValueError(f"{split} shard rank {shard['rank']} is not the deterministic source slice.")
            observed.extend(actual_shard)
            live_ids = [int(row["record_index"]) for row in shard["splits"][split]]
            if live_ids != actual_shard:
                raise ValueError(f"{split} shard live rows disagree with source identity.")
        if sorted(observed) != sorted(expected_global) or len(observed) != len(set(observed)):
            raise ValueError(f"{split} shard coverage has gaps or overlaps.")


def run_merge(args: Args) -> tuple[dict[str, Any], dict[str, Any]]:
    """Merge deterministic CPU shard files without re-running the encoder."""

    validate_export_args(args)
    if args.world_size <= 1:
        raise ValueError("Shard merge requires world_size > 1.")
    output_dir = Path(args.output_dir)
    shard_paths = [output_dir / f"shard_rank{rank:04d}.pt" for rank in range(args.world_size)]
    missing = [str(path) for path in shard_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing artifact shards: {missing}")
    _assert_output_available(output_dir)
    shards = [torch.load(path, map_location="cpu", weights_only=False) for path in shard_paths]
    from openpi.zeva.robotwin_contract import RobotWinHandoff

    handoff = RobotWinHandoff.from_root(args.handoff_root)
    checkpoint, checkpoint_sha, stats_sha = _load_checkpoint_payload(args, handoff)
    goal_sha = sha256_file(args.goal_embeddings)
    adapter_manifest = Path(args.dataset_root) / "adapter.json"
    adapter_sha = sha256_file(adapter_manifest)
    source_by_split = {split: _dataset(args, split) for split in ("train", "validation")}
    source_metadata = _source_metadata(source_by_split, args)
    task_names = source_by_split["train"].task_names
    if source_by_split["validation"].task_names != task_names:
        raise ValueError("Train and validation task ordering differ.")
    _validate_shards(
        shards=shards,
        args=args,
        source_metadata=source_metadata,
        checkpoint_sha=checkpoint_sha,
        stats_sha=stats_sha,
        goal_sha=goal_sha,
        adapter_sha=adapter_sha,
    )
    split_payloads = {
        split: {
            "schema": LIVE_QUERY_SCHEMA,
            "transition_horizon": HORIZON,
            "stage1_checkpoint_schema": V2_STAGE1_SCHEMA,
            "task_names": tuple(task_names),
            "records": sorted(
                [row for shard in shards for row in shard["splits"][split]],
                key=lambda row: int(row["record_index"]),
            ),
        }
        for split in ("train", "validation")
    }
    train_state = merge_bank_accumulators([shard["bank_state"] for shard in shards])
    return _build_final_artifacts(
        args=args,
        checkpoint=checkpoint,
        checkpoint_sha=checkpoint_sha,
        stats_sha=stats_sha,
        goal_sha=goal_sha,
        adapter_sha=adapter_sha,
        task_names=task_names,
        split_payloads=split_payloads,
        train_state=train_state,
        source_metadata=source_metadata,
        output_dir=output_dir,
    )


def run(args: Args) -> tuple[dict[str, Any], dict[str, Any]]:
    validate_export_args(args)
    if args.merge_shards:
        return run_merge(args)
    torch.manual_seed(args.seed)
    from openpi.zeva.robotwin_contract import MeanStdActionNormalizer, RobotWinHandoff

    output_dir = Path(args.output_dir)
    if args.world_size == 1:
        _assert_output_available(output_dir)
    else:
        _assert_output_available(output_dir, shard_path=output_dir / f"shard_rank{args.rank:04d}.pt")
    handoff = RobotWinHandoff.from_root(args.handoff_root)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    checkpoint, encoder, checkpoint_sha, stats_sha = _checkpoint(args, handoff, device)
    goal_sha = sha256_file(args.goal_embeddings)
    adapter_manifest = Path(args.dataset_root) / "adapter.json"
    adapter_sha = sha256_file(adapter_manifest)
    normalizer = MeanStdActionNormalizer.from_stats_file(handoff.statistics)
    source_by_split = {split: _dataset(args, split) for split in ("train", "validation")}
    source_metadata = _source_metadata(source_by_split, args)
    task_names = source_by_split["train"].task_names
    if source_by_split["validation"].task_names != task_names:
        raise ValueError("Train and validation task ordering differ.")
    split_payloads: dict[str, dict[str, Any]] = {}
    train_state: dict[str, torch.Tensor] | None = None
    for split, dataset in source_by_split.items():
        payload, state = _export_split(
            args=args,
            subset=split,
            dataset=dataset,
            encoder=encoder,
            normalizer=normalizer,
            device=device,
            task_names=task_names,
            checkpoint=checkpoint,
            selected_global=source_metadata[split]["selected_indices"],
        )
        split_payloads[split] = payload
        if split == "train":
            train_state = state
    assert train_state is not None
    if args.world_size > 1:
        partial = _save_shard(
            args=args,
            checkpoint=checkpoint,
            checkpoint_sha=checkpoint_sha,
            stats_sha=stats_sha,
            goal_sha=goal_sha,
            adapter_sha=adapter_sha,
            task_names=task_names,
            split_payloads=split_payloads,
            train_state=train_state,
            source_metadata=source_metadata,
            output_dir=output_dir,
        )
        return partial, {
            "schema": LIVE_QUERY_SCHEMA,
            "splits": partial["splits"],
            "incomplete": True,
            "usable_for_training": False,
        }
    return _build_final_artifacts(
        args=args,
        checkpoint=checkpoint,
        checkpoint_sha=checkpoint_sha,
        stats_sha=stats_sha,
        goal_sha=goal_sha,
        adapter_sha=adapter_sha,
        task_names=task_names,
        split_payloads=split_payloads,
        train_state=train_state,
        source_metadata=source_metadata,
        output_dir=output_dir,
    )


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
