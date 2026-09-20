"""Frozen validation5 audit for causal, within-episode ZeVA PIM.

For decision ``t``, aligned PIM contains only the same episode's cached BITs
from ``[0,t)``.  The shuffled control keeps the same phase keys but rotates the
corresponding BIT values, breaking phase/BIT alignment without introducing a
different episode.  Every condition uses identical explicit diffusion noise.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scripts.cte_eap_validation_contract import (
    check_cache_coverage,
    decision_noise_seed,
    enumerate_decisions,
)
from scripts.report_robotwin_pim_validation import sha, write_json


CONDITIONS = ("parent", "aligned", "shuffled", "pim_off")
SCHEMA = "zeva-episode-pim-validation5-v1"


def validate_checkpoint(checkpoint: Path) -> dict:
    required = ["COMPLETE", "model.safetensors", "zeva_adapter.pth", "training_state.pth"]
    required.extend(f"rng_rank{rank}.pth" for rank in range(8))
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing:
        raise ValueError(f"Incomplete Episode-PIM checkpoint: {missing}")
    manifest = json.loads((checkpoint.parent / "manifest.json").read_text())
    if (
        checkpoint.name != "002000"
        or manifest.get("schema") != "zeva-cte-bit-episode-pim-eap-v1-training-v1"
        or manifest.get("global_batch") != 256
        or manifest.get("history_contract") != "same episode, strictly earlier BIT boundaries only"
        or manifest.get("cross_episode")
        or manifest.get("cross_attempt")
        or not manifest.get("shared_parent_frozen")
    ):
        raise ValueError("Checkpoint differs from the fixed Episode-PIM protocol.")
    return manifest


def causal_history(row: dict, timestep: int, capacity: int, *, shuffled: bool):
    import torch

    if timestep <= 0:
        return None
    phase = torch.as_tensor(row["phase"][:timestep], dtype=torch.float32)
    bit = torch.as_tensor(row["effect"][:timestep], dtype=torch.float32)
    if len(phase) > capacity:
        index = torch.linspace(0, len(phase) - 1, capacity).round().long().unique()
        phase, bit = phase[index], bit[index]
    if shuffled and len(bit) > 1:
        bit = torch.roll(bit, shifts=1, dims=0)
    mask = torch.ones(1, len(phase), dtype=torch.bool, device="cuda")
    return phase.unsqueeze(0).cuda(), bit.unsqueeze(0).cuda(), mask


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "dataset-root", "cte-checkpoint", "cte-artifacts", "retrieval-checkpoint",
        "foundation-checkpoint", "parent-checkpoint", "checkpoint", "expected-decisions",
        "output-dir",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--tasks", type=Path, default=Path("configs/robotwin_zeva_advantage10.json"))
    parser.add_argument("--handoff", type=Path, default=Path(
        "/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1"))
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--merge", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard topology.")
    tasks = json.loads(args.tasks.read_text())["task_names"]
    if len(tasks) != 10 or len(tasks) != len(set(tasks)):
        raise ValueError("The frozen ten-task list is required.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.merge:
        merge(args, tasks)
        return

    import torch
    from openpi.zeva.cte_eap_policy import ZevaCTEEAPPolicy
    from openpi.zeva.pim_policy import ZevaEpisodePIMPolicy
    from openpi.zeva.robotwin_contract import ROBOTWIN_CAMERA_KEYS, prepare_robotwin_pi_image
    from scripts.train_robotwin_stage2 import _diagnostic_action_valid_mask
    from scripts.train_robotwin_zte import TorchCodecRoboTwinDataset

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = validate_checkpoint(args.checkpoint)
    source = TorchCodecRoboTwinDataset(args.dataset_root / "adapter.json", "validation")
    decisions = enumerate_decisions(source.dataset._records, tasks)
    expected = json.loads(args.expected_decisions.read_text())
    if [row["sample_id"] for row in decisions] != expected:
        raise ValueError("Frozen validation5 decision manifest changed.")
    cte = torch.load(args.cte_artifacts, map_location="cpu", weights_only=False)
    cache = cte["splits"]["validation"]
    check_cache_coverage(decisions, cache)
    capacity = int(manifest["args"]["max_entries"])

    parent = ZevaCTEEAPPolicy.from_handoff(
        args.handoff, args.foundation_checkpoint, args.cte_checkpoint, args.cte_artifacts,
        args.retrieval_checkpoint, device="cuda", stage2_checkpoint=args.parent_checkpoint,
        exploratory_epoch40=True,
    ).eval()
    policy = ZevaEpisodePIMPolicy.load_trained(
        args.handoff, args.foundation_checkpoint, args.cte_checkpoint, args.cte_artifacts,
        args.retrieval_checkpoint, args.checkpoint, device="cuda", exploratory_epoch40=True,
    ).eval()
    for model in (parent, policy):
        model.requires_grad_(False)
        config = model.foundation.config
        if (config.chunk_size, config.max_action_dim, config.num_inference_steps) != (50, 32, 10):
            raise ValueError("PI inference contract differs from H50x32/10-step protocol.")

    shard = [row for position, row in enumerate(decisions)
             if position % args.num_shards == args.shard_index]
    rows_path = args.output_dir / f"rows-shard-{args.shard_index:02d}.jsonl"
    meta_path = args.output_dir / f"meta-shard-{args.shard_index:02d}.json"
    with torch.inference_mode(), rows_path.open("x") as stream:
        for completed, decision in enumerate(shard, 1):
            index, frame = decision["record_index"], decision["frame"]
            record = source.dataset._records[index]
            raw = dict(source.dataset[int(source.dataset._cumulative[index]) + frame])
            images = source.read_images(record, [frame])
            for key in ROBOTWIN_CAMERA_KEYS:
                raw[key] = prepare_robotwin_pi_image(images[key][0], name=key)
            processed = policy.preprocessor(dict(raw))
            parent_processed = parent.preprocessor(dict(raw))
            for key in processed:
                if isinstance(processed[key], torch.Tensor):
                    torch.testing.assert_close(processed[key], parent_processed[key], rtol=0, atol=0)
            target = processed["action"].float().reshape(1, 50, 16)
            valid, mask_source = _diagnostic_action_valid_mask(
                processed, batch_size=1, horizon=50, device=target.device)
            valid = valid[:, :15]
            if not valid.any():
                raise ValueError("A frozen validation decision has no valid H15 actions.")
            row_cache = cache[decision["record_id"]]
            timestep = int(decision["cache_index"])
            phase = row_cache["phase"][timestep].unsqueeze(0).cuda().float()
            effect = row_cache["effect"][timestep].unsqueeze(0).cuda().float()
            aligned = causal_history(row_cache, timestep, capacity, shuffled=False)
            shuffled = causal_history(row_cache, timestep, capacity, shuffled=True)
            seed = decision_noise_seed(decision["sample_id"])
            noise = torch.randn((1, 50, 32), generator=torch.Generator().manual_seed(seed)).cuda()
            parent_global = parent.memory(parent.language(raw["task"]))
            policy_global = policy.memory(policy.language(raw["task"]))
            result = {
                "sample_id": decision["sample_id"], "task": decision["task"],
                "record_id": decision["record_id"], "cache_index": timestep,
                "history_entries": min(timestep, capacity), "noise_seed": seed,
                "valid_action_steps": int(valid.sum()), "mask_source": mask_source,
            }
            for condition in CONDITIONS:
                model = parent if condition == "parent" else policy
                model.foundation.reset()
                if condition == "parent":
                    parent.pbd.activate(parent_global, phase, effect)
                elif condition == "pim_off" or aligned is None:
                    policy.pbd.activate(policy_global, phase, effect, include_pim=False)
                else:
                    pim_phase, pim_bit, pim_mask = aligned if condition == "aligned" else shuffled
                    policy.pbd.activate(
                        policy_global, phase, effect, pim_phase=pim_phase,
                        pim_bit=pim_bit, pim_mask=pim_mask, include_pim=True,
                    )
                try:
                    prediction = model.foundation.predict_action_chunk(
                        {key: value.clone() if isinstance(value, torch.Tensor) else value
                         for key, value in processed.items()},
                        noise=noise.clone(), num_steps=10,
                    )
                finally:
                    model.pbd.clear()
                if prediction.shape != (1, 50, 16) or not torch.isfinite(prediction).all():
                    raise ValueError("Invalid sampled H50 EEF16 action.")
                error = (prediction[:, :15].float() - target[:, :15]).square().mean(-1)
                result[condition] = float(error[valid].mean())
            stream.write(json.dumps(result, allow_nan=False) + "\n")
            stream.flush()
            if completed % 100 == 0:
                print(json.dumps({"shard": args.shard_index, "completed": completed,
                                  "total": len(shard)}), flush=True)
    write_json(meta_path, {
        "schema": SCHEMA + "-shard", "shard_index": args.shard_index,
        "num_shards": args.num_shards, "rows": len(shard),
        "expected_decisions_sha256": sha(args.expected_decisions),
        "checkpoint_model_sha256": sha(args.checkpoint / "model.safetensors"),
        "checkpoint_adapter_sha256": sha(args.checkpoint / "zeva_adapter.pth"),
        "parent_model_sha256": sha(args.parent_checkpoint / "model.safetensors"),
        "rows_sha256": sha(rows_path), "source_sha256": sha(Path(__file__)),
    })


def merge(args, tasks: list[str]) -> None:
    expected = json.loads(args.expected_decisions.read_text())
    rows, identities = [], None
    for shard in range(args.num_shards):
        meta_path = args.output_dir / f"meta-shard-{shard:02d}.json"
        rows_path = args.output_dir / f"rows-shard-{shard:02d}.jsonl"
        if not meta_path.is_file() or not rows_path.is_file():
            raise ValueError(f"Missing completed shard {shard}.")
        meta = json.loads(meta_path.read_text())
        current = {key: meta[key] for key in (
            "num_shards", "expected_decisions_sha256", "checkpoint_model_sha256",
            "checkpoint_adapter_sha256", "parent_model_sha256", "source_sha256")}
        if identities is None:
            identities = current
        if current != identities or meta["shard_index"] != shard or meta["rows_sha256"] != sha(rows_path):
            raise ValueError("Shard topology, identity or content hash mismatch.")
        shard_rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
        if len(shard_rows) != meta["rows"]:
            raise ValueError("Shard row count differs from its manifest.")
        rows.extend(shard_rows)
    by_id = {row["sample_id"]: row for row in rows}
    if len(rows) != len(by_id) or set(by_id) != set(expected):
        raise ValueError("Merged report does not cover every frozen decision exactly once.")
    rows = [by_id[sample_id] for sample_id in expected]
    averages = {condition: sum(row[condition] for row in rows) / len(rows)
                for condition in CONDITIONS}
    if not all(math.isfinite(value) for value in averages.values()):
        raise ValueError("Non-finite aggregate metric.")
    starts = [row for row in rows if row["cache_index"] == 0]
    start_exact = all(
        row[condition] == row["parent"]
        for row in starts for condition in ("aligned", "shuffled", "pim_off")
    )
    per_task = {}
    for task in tasks:
        selected = [row for row in rows if row["task"] == task]
        per_task[task] = {condition: sum(row[condition] for row in selected) / len(selected)
                          for condition in CONDITIONS}
    checks = {
        "complete_frozen_coverage": len(rows) == len(expected),
        "episode_start_parent_exact": start_exact,
        "aligned_not_worse_than_parent": averages["aligned"] <= averages["parent"],
        "aligned_beats_within_episode_phase_bit_shuffle": averages["aligned"] < averages["shuffled"],
        "aligned_not_worse_than_pim_off": averages["aligned"] <= averages["pim_off"],
    }
    report = {
        "schema": SCHEMA, "split": "validation", "formal_labels_used": False,
        "checkpoint_step": 2000, "metric": "sample_mean_normalized_executed_h15_action_mse",
        "matched_noise": True, "num_shards": args.num_shards,
        "expected_decisions_sha256": sha(args.expected_decisions),
        "checkpoint_model_sha256": identities["checkpoint_model_sha256"],
        "checkpoint_adapter_sha256": identities["checkpoint_adapter_sha256"],
        "parent_model_sha256": identities["parent_model_sha256"],
        "episode_start_rows": len(starts), "averages": averages, "per_task": per_task,
        "gate": {"passed": all(checks.values()), "checks": checks,
                 "fixed_before_results": True, "offline_gate_not_success_rate": True},
        "rows": rows,
    }
    write_json(args.output_dir / "report.json", report)
    print(json.dumps({"report": str(args.output_dir / "report.json"),
                      "passed": report["gate"]["passed"], "averages": averages}), flush=True)


if __name__ == "__main__":
    main()
