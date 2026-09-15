"""Read-only, fixed training-sample flow/NLL gradient audit; no optimizer."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_model
from torch.utils.data import DataLoader, Subset

from scripts import eval_robotwin_stage2_diagnostics as diag
from scripts import train_robotwin_stage2 as train


GROUPS = ("task_token_projector", "memory_context_encoder", "action_prior",
          "causal_action_projector", "prior_action_projector", "residual_gate_router")


def gradient_stats(left, right) -> dict:
    """Undefined cosine/ratio stay unavailable, never invent an aligned zero."""
    if len(left) != len(right):
        raise ValueError("Gradient lists must have equal length")
    aa = bb = ab = 0.0
    for x, y in zip(left, right):
        if x is not None:
            aa += float(x.detach().double().square().sum())
        if y is not None:
            bb += float(y.detach().double().square().sum())
        if x is not None and y is not None:
            ab += float((x.detach().double() * y.detach().double()).sum())
    return {"flow_gradient_norm": aa ** 0.5,
            "weighted_nll_gradient_norm": bb ** 0.5,
            "nll_to_flow_norm_ratio": (bb / aa) ** 0.5 if aa > 0 else None,
            "cosine": ab / (aa * bb) ** 0.5 if aa > 0 and bb > 0 else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    cli = parser.parse_args()
    if cli.output.exists():
        raise FileExistsError(cli.output)
    source = diag._read_json(cli.manifest)
    args = diag._stage2_args_from_manifest(source, diag.EvalArgs(
        checkpoint=str(cli.checkpoint), manifest=str(cli.manifest), output=str(cli.output)))
    if args.training_variant != "zeva":
        raise ValueError("This bounded audit only supports the current dual-residual zeva variant")
    seed, batch_size, batches = 20260915, 8, 4
    device = torch.device("cuda:0")
    torch.manual_seed(seed)
    handoff = diag.RobotWinHandoff.from_root(args.handoff_root)
    handoff = dataclasses.replace(handoff, checkpoint=Path(args.foundation_checkpoint).resolve())
    handoff.validate()
    bank = diag.RobotWinCausalBank.load(args.causal_bank, device=device)
    lineage = diag._validate_lineage(args, handoff, bank, train._validated_runtime_versions())
    policy = diag.RobotWinZevaPolicy.from_handoff(
        args.handoff_root, device=str(device), foundation_checkpoint=args.foundation_checkpoint,
        goal_embedding_checkpoint=args.goal_embedding_checkpoint, zte_checkpoint=args.zte_checkpoint,
        retrieval_checkpoint=args.task_retrieval, causal_bank=args.causal_bank,
        stage2_checkpoint=args.initial_stage2_checkpoint)
    diag._configure_policy(policy, "zeva", args.prior_injection_horizon)
    load_model(policy.foundation, cli.checkpoint / "model.safetensors", strict=True)
    policy.load_adapter(cli.checkpoint / "zeva_adapter.pth")
    policy.train()
    policy.enforce_action_expert_stage2_mode()
    # autograd.grad is incompatible with reentrant checkpointing. This changes
    # memory/recomputation only; training flags and stochastic masks stay intact.
    policy.foundation.model.gradient_checkpointing_disable()
    if any(p.requires_grad for p in policy.causal_transition_encoder.parameters()):
        raise AssertionError("Stage1 unexpectedly trainable")
    if any(p.requires_grad for p in policy.foundation.model.paligemma_with_expert.paligemma.parameters()):
        raise AssertionError("Vision-language backbone unexpectedly trainable")
    dataset = train.RobotWinStage2Dataset(
        Path(args.dataset_root) / "adapter.json", args.live_queries, subset="train",
        config=policy.zeva_config, selected_tasks=diag._selected_tasks(source, args),
        video_backend=args.video_backend, decoder_threads=args.decoder_threads)
    if dataset.task_names != bank.task_names:
        raise AssertionError("Dataset/bank task order mismatch")
    indexes = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[:batch_size*batches].tolist()
    loader = DataLoader(Subset(dataset, indexes), batch_size=batch_size, shuffle=False, num_workers=0)
    named = [(group, name, p) for group in GROUPS
             for name, p in getattr(policy, group).named_parameters() if p.requires_grad]
    parameters = [p for _, _, p in named]
    original = [p.detach().cpu().clone() for p in parameters]
    observations = []
    for index, raw in enumerate(loader):
        torch.manual_seed(seed + index)
        torch.cuda.manual_seed_all(seed + index)
        sample_ids = raw.pop("zeva.sample_index").tolist()
        task_ids = raw.pop("zeva.task_id")
        values = [raw.pop("zeva." + name) for name in
                  ("phase_query", "live_brief", "live_brief_mask", "live_retrieved", "live_retrieved_mask")]
        tasks = list(raw["task"])
        processed = train._preprocess_with_task_only_goal(policy, policy.preprocessor, raw)
        phase, brief, brief_mask, retrieved, retrieved_mask = values
        bank_batch, confidence, _ = train._retrieve(
            policy, processed, phase, task_ids, brief, brief_mask, retrieved, retrieved_mask,
            bank, policy.zeva_config, args, training=True)
        losses = train._losses(
            policy, processed, bank_batch, confidence, None, None,
            args.prior_loss_weight, args.preserve_loss_weight, args.gate_regularization_weight,
            args.prior_residual_dropout_probability,
            prior_supervision_horizon=args.prior_injection_horizon, training=True)
        flow = torch.autograd.grad(losses["flow"], parameters, retain_graph=True, allow_unused=True)
        prior = torch.autograd.grad(args.prior_loss_weight * losses["prior"], parameters, allow_unused=True)
        groups = {}
        for group in GROUPS:
            positions = [i for i, (g, _, _) in enumerate(named) if g == group]
            groups[group] = gradient_stats([flow[i] for i in positions], [prior[i] for i in positions])
        row = {"batch": index, "sample_ids": sample_ids, "tasks": tasks,
               "flow": float(losses["flow"].detach()),
               "weighted_nll": float((args.prior_loss_weight * losses["prior"]).detach()),
               "groups": groups}
        observations.append(row)
        print(json.dumps(row, allow_nan=False), flush=True)
        del flow, prior, losses, processed, bank_batch
    if not all(torch.equal(before, p.detach().cpu()) for before, p in zip(original, parameters, strict=True)):
        raise AssertionError("Audited parameters changed without an optimizer")
    result = {"schema": "zeva-objective-gradient-audit-v1", "checkpoint": str(cli.checkpoint),
              "model_sha256": train._sha256(cli.checkpoint / "model.safetensors"),
              "adapter_sha256": train._sha256(cli.checkpoint / "zeva_adapter.pth"),
              "lineage": lineage, "subset": "train", "seed": seed, "batch_size": batch_size,
              "batches": batches, "dataset_indexes": indexes,
              "sample_indexes_sha256": hashlib.sha256(json.dumps(indexes).encode()).hexdigest(),
              "optimizer_created": False, "checkpoint_written": False,
              "audited_parameters_unchanged": True, "torch_compile": False,
              "gradient_checkpointing": False, "training_masks_enabled": True,
              "objective_horizon": 50, "prior_weight": args.prior_loss_weight,
              "excluded_from_attribution": ["preserve hinge", "gate regularizer", "clipping", "Adam state"],
              "caveat": "32 fixed training decisions; local raw-gradient attribution, not optimizer-update or generalization proof",
              "observations": observations}
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    with cli.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
