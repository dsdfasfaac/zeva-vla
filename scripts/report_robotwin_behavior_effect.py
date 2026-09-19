"""Full validation5 sampled H15 errors, explicit common diffusion noise, no test labels.

Writes an independent raw-record expected-id manifest BEFORE loading CTE cache.
No checkpoint search, sample cap, partial promotion, or automatic closed-loop run.
All four conditions use the same processed inputs, H50x32 initial noise and ten
denoising steps. Cached phase/effect contain expert-history recurrence: this is
an offline gate, NOT a closed-loop success-rate measurement.
"""
import argparse
import json
from pathlib import Path

from scripts.behavior_effect_validation_contract import (
    check_cache_coverage, decision_noise_seed, enumerate_decisions, within_task_permutation,
)
from scripts.select_robotwin_behavior_effect import sha, select


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset-root", "cte-checkpoint", "artifacts", "retrieval-checkpoint",
                 "foundation-checkpoint", "checkpoint", "output-dir", "expected-decisions"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=Path("configs/robotwin_behavior_effect_20260918.json"))
    parser.add_argument("--tasks", type=Path, default=Path("configs/robotwin_zeva_advantage10.json"))
    parser.add_argument("--handoff", type=Path, default=Path(
        "/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1"))
    parser.add_argument("--exploratory-epoch40", action="store_true",
                        help="Evaluate the isolated epoch40 branch without making it epoch80-promotion eligible.")
    args = parser.parse_args()
    import torch
    from openpi.zeva.behavior_effect import SCHEMA
    from openpi.zeva.behavior_effect_policy import ZevaBehaviorEffectPolicy
    from openpi.zeva.robotwin_policy import RobotWinZevaPolicy
    from openpi.zeva.robotwin_contract import ROBOTWIN_CAMERA_KEYS, prepare_robotwin_pi_image
    from scripts.train_robotwin_zte import TorchCodecRoboTwinDataset
    from scripts.train_robotwin_stage2 import _diagnostic_action_valid_mask

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    plan = json.loads(args.plan.read_text())
    tasks = json.loads(args.tasks.read_text())["task_names"]
    if len(tasks) != 10 or len(set(tasks)) != 10:
        raise ValueError("Exactly the fixed ten tasks are required.")
    if json.loads(Path(plan["protocol"]["task_subset"]).read_text())["task_names"] != tasks:
        raise ValueError("Task list differs from preregistration.")
    baseline_path = Path(plan["comparison_base"])
    if sha(baseline_path / "model.safetensors") != plan["comparison_base_model_sha256"]:
        raise ValueError("Normally trained Base1000 SHA mismatch.")
    for name in ("COMPLETE", "training_state.pth", "zeva_adapter.pth", "model.safetensors"):
        if not (args.checkpoint / name).is_file():
            raise ValueError(f"Incomplete Stage2 checkpoint: {name}")
    state = torch.load(args.checkpoint / "training_state.pth", map_location="cpu", weights_only=False)
    if state["step"] != 5000 or state["manifest"]["global_batch"] != 256:
        raise ValueError("Only fixed step5000/global256 is eligible.")
    if bool(state["manifest"]["args"].get("exploratory_epoch40", False)) != args.exploratory_epoch40:
        raise ValueError("Requested evaluation mode differs from checkpoint exploratory lineage.")
    del state
    args.output_dir.mkdir(parents=True, exist_ok=False)
    dataset_path = args.dataset_root / "adapter.json"
    source = TorchCodecRoboTwinDataset(dataset_path, "validation")
    decisions = enumerate_decisions(source.dataset._records, tasks)
    expected = [row["sample_id"] for row in decisions]
    independent_expected = json.loads(args.expected_decisions.read_text())
    if independent_expected != expected:
        raise ValueError("Independent pre-Stage2 raw validation enumeration differs.")
    permutation = within_task_permutation(decisions)
    # These manifests are independent of cache availability and model errors.
    write_json(args.output_dir / "expected-decisions.json", expected)
    if sha(args.output_dir / "expected-decisions.json") != sha(args.expected_decisions):
        raise ValueError("Independent frozen expected-ID file changed when copied.")
    write_json(args.output_dir / "decision-plan.json", {"schema": "zeva-validation5-decisions-v1",
               "split": "validation", "adapter_sha256": sha(dataset_path),
               "tasks": tasks, "decisions": decisions, "within_task_permutation": permutation,
               "permutation_seed": 20260918, "enumerator_sha256": sha(Path(__file__).with_name(
                   "behavior_effect_validation_contract.py"))})
    artifact = torch.load(args.artifacts, map_location="cpu", weights_only=False)
    if (artifact.get("schema") != SCHEMA + "-artifacts" or artifact["tasks"] != tasks
            or artifact["adapter_sha256"] != sha(dataset_path)
            or artifact["execution_horizon"] != 15 or artifact["policy_horizon"] != 50):
        raise ValueError("CTE artifact schema/data/tasks/horizon mismatch.")
    cache = artifact["splits"]["validation"]
    check_cache_coverage(decisions, cache)
    train_source = TorchCodecRoboTwinDataset(dataset_path, "train")
    train_ids = {row["record_id"] for row in enumerate_decisions(train_source.dataset._records, tasks)}
    bank_ids = artifact["bank_record_ids"]
    if (artifact["bank_subset"] != "train" or len(bank_ids) != len(set(bank_ids))
            or set(bank_ids) != train_ids or train_ids & set(cache)):
        raise ValueError("Memory is not the complete disjoint train95 bank.")
    for entry in cache.values():
        if not torch.isfinite(entry["phase"]).all() or not torch.isfinite(entry["effect"]).all():
            raise ValueError("Non-finite recurrent features.")
    policy = ZevaBehaviorEffectPolicy.from_handoff(
        args.handoff, args.foundation_checkpoint, args.cte_checkpoint, args.artifacts,
        args.retrieval_checkpoint, stage2_checkpoint=args.checkpoint, device="cuda",
        exploratory_epoch40=args.exploratory_epoch40).eval()
    base = RobotWinZevaPolicy.from_handoff(
        args.handoff, foundation_checkpoint=args.foundation_checkpoint,
        stage2_checkpoint=baseline_path, install_injection_hooks=False, device="cuda").eval()
    for foundation in (base.foundation, policy.foundation):
        config = foundation.config
        if (config.chunk_size != 50 or config.max_action_dim != 32 or config.num_inference_steps != 10
                or config.use_visual_memory or config.use_proprioceptive_memory):
            raise ValueError("PI configuration differs from the frozen stateless H50 baseline.")
        foundation.requires_grad_(False)
    by_id = {row["sample_id"]: row for row in decisions}

    def features(row):
        entry = cache[row["record_id"]]
        return [entry[key][row["cache_index"]].unsqueeze(0).cuda().float() for key in ("phase", "effect")]

    rows = []
    with torch.inference_mode(), (args.output_dir / "rows.partial.jsonl").open("x") as partial:
        for decision in decisions:
            index, frame = decision["record_index"], decision["frame"]
            record = source.dataset._records[index]
            raw = dict(source.dataset[int(source.dataset._cumulative[index]) + frame])
            images = source.read_images(record, [frame])
            for key in ROBOTWIN_CAMERA_KEYS:
                raw[key] = prepare_robotwin_pi_image(images[key][0], name=key)
            processed = policy.preprocessor(dict(raw))
            base_processed = base.preprocessor(dict(raw))
            if set(processed) != set(base_processed):
                raise ValueError("Base/ZeVA preprocessing fields differ.")
            for key in processed:
                if isinstance(processed[key], torch.Tensor):
                    torch.testing.assert_close(processed[key], base_processed[key], rtol=0, atol=0)
                elif processed[key] != base_processed[key]:
                    raise ValueError(f"Base/ZeVA preprocessing differs: {key}")
            target = processed["action"].float().reshape(1, 50, 16)
            valid, mask_source = _diagnostic_action_valid_mask(
                processed, batch_size=1, horizon=50, device=target.device)
            valid = valid[:, :15]
            if not valid.any():
                raise ValueError("Expected validation decision has zero valid executed actions.")
            seed = decision_noise_seed(decision["sample_id"])
            noise = torch.randn((1, 50, 32), generator=torch.Generator().manual_seed(seed)).cuda()
            global_token = policy.memory(policy.language(raw["task"]))
            aligned = features(decision)
            shuffled = features(by_id[permutation[decision["sample_id"]]])
            row = {"sample_id": decision["sample_id"], "task": decision["task"],
                   "shuffled_sample_id": permutation[decision["sample_id"]], "noise_seed": seed,
                   "valid_action_steps": int(valid.sum()), "mask_source": mask_source}
            for condition in ("base", "aligned", "within_task_shuffled", "effect_off"):
                foundation = base.foundation if condition == "base" else policy.foundation
                foundation.reset()
                if condition != "base":
                    phase, effect = shuffled if condition == "within_task_shuffled" else aligned
                    policy.pbd.activate(global_token, phase, effect, include_effect=condition != "effect_off")
                try:
                    prediction = foundation.predict_action_chunk(
                        {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in processed.items()},
                        noise=noise.clone(), num_steps=10)
                finally:
                    policy.pbd.clear()
                if prediction.shape != (1, 50, 16) or not torch.isfinite(prediction).all():
                    raise ValueError("Invalid H50 EEF16 sample.")
                error = (prediction[:, :15].float() - target[:, :15]).square().mean(-1)
                row[condition] = float(error[valid].mean())
            rows.append(row)
            partial.write(json.dumps(row, allow_nan=False) + "\n")
            partial.flush()
            if len(rows) % 100 == 0:
                print(json.dumps({"completed": len(rows), "total": len(decisions)}), flush=True)
    report = {"schema": "zeva-behavior-effect-validation5-v1", "split": "validation",
              "formal_labels_used": False, "checkpoint_step": 5000, "checkpoint": str(args.checkpoint.resolve()),
              "exploratory_epoch40": args.exploratory_epoch40,
              "formal_promotion_eligible": not args.exploratory_epoch40,
              "output_horizon": 50, "execution_horizon": 15,
              "metric": "sample_mean_normalized_executed_h15_action_mse", "matched_noise": True,
              "noise_contract": "explicit identical FP32 H50x32 noise, 10 denoising steps, TF32 disabled",
              "history_contract": "offline expert H15 recurrent cache; not closed-loop success",
              "baseline_model_sha256": plan["comparison_base_model_sha256"],
              "model_sha256": sha(args.checkpoint / "model.safetensors"),
              "adapter_sha256": sha(args.checkpoint / "zeva_adapter.pth"),
              "identity": policy.identity, "source_sha256": sha(Path(__file__)),
              "expected_decisions_sha256": sha(args.output_dir / "expected-decisions.json"),
              "decision_plan_sha256": sha(args.output_dir / "decision-plan.json"), "rows": rows}
    # Fail closed before producing a complete report; gate failure is a valid report.
    gate = select(report, expected, plan, tasks)
    write_json(args.output_dir / "report.json", report)
    print(json.dumps({"report": str(args.output_dir / "report.json"), "passed": gate["passed"],
                      "next": ("exploratory diagnosis only; epoch80 selector rejects this branch"
                               if args.exploratory_epoch40 else
                               "run independent selector; no automatic promotion")}), flush=True)


if __name__ == "__main__":
    main()
