"""LIBERO Stage 3: selectively adapt final PI0.5 action blocks after Stage 2A."""

from __future__ import annotations

import dataclasses
from contextlib import nullcontext
import inspect
import json
import math
from pathlib import Path

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
import torch
import torch.nn.functional as F  # noqa: N812
from torch.func import functional_call
from torch.utils.data import DataLoader
import tqdm
import tyro

from openpi.zeva.libero_bank import LiberoCausalBank
from openpi.zeva.libero_contract import LIBERO_EXECUTION_HORIZON
from openpi.zeva.libero_contract import LIBERO_POLICY_HORIZON
from openpi.zeva.libero_contract import LiberoHandoff
from openpi.zeva.libero_contract import sha256
from openpi.zeva.libero_policy import LiberoZevaPolicy
from scripts.train_libero_stage2 import LiberoStage2ADataset
from scripts.train_libero_stage2 import _foundation_loss
from scripts.train_libero_stage2 import _retrieve


@dataclasses.dataclass
class Args:
    handoff_root: str = "/data1/dingxin/libero-memory-baseline-v1"
    dataset_root: str = "/data1/dingxin/libero-memory-baseline-v1/data"
    tokenizer_path: str = (
        "/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/"
        "robotwin-memory-baseline-v1/checkpoint/pretrained_model/tokenizer"
    )
    stage2_checkpoint: str = "/data1/dingxin/zeva-runs/libero-v3-h5-lang/stage2a-adapter/005000"
    zte_checkpoint: str = "/data1/dingxin/zeva-runs/libero-v3-h5-lang/stage1-zte/zte_best.pth"
    causal_bank: str = "/data1/dingxin/zeva-runs/libero-v3-h5-lang/stage1-zte/train_causal_bank.pt"
    live_queries: str = "/data1/dingxin/zeva-runs/libero-v3-h5-lang/stage1-zte/live_queries_h5.pt"
    task_retrieval: str = (
        "/data1/dingxin/zeva-runs/libero-v3-h5-lang/"
        "stage1.5-task-retrieval/task_retrieval.pth"
    )
    save_dir: str = "/data1/dingxin/zeva-runs/libero-v3-h5-lang/stage3-selective-action"
    resume_checkpoint: str | None = None
    steps: int = 2_000
    final_action_layers: int = 2
    batch_size: int = 16
    gradient_accumulation_steps: int = 2
    num_workers: int = 4
    learning_rate: float = 5e-6
    weight_decay: float = 1e-10
    adam_beta2: float = 0.95
    warmup_steps: int = 200
    preserve_loss_weight: float = 1.0
    anchor_loss_weight: float = 1e-4
    phase_noise_std: float = 0.01
    memory_dropout: float = 0.05
    retrieval_confidence_floor: float = 0.2
    save_freq: int = 250
    save_checkpoints: bool = True
    eval_batches: int = 32
    log_freq: int = 10
    seed: int = 1000


def _stage2_manifest(args: Args, handoff: LiberoHandoff, bank: LiberoCausalBank):
    checkpoint = Path(args.stage2_checkpoint).resolve()
    for name in ("zeva_adapter.pth", "training_state.pt"):
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(checkpoint / name)
    manifest_path = checkpoint.parent / "manifest.json"
    gate_path = checkpoint.parent / "stage2_gate.json"
    stage2 = json.loads(manifest_path.read_text())
    gate = json.loads(gate_path.read_text())
    if stage2.get("schema") != "zeva-libero-stage2a-manifest-v2":
        raise ValueError("Stage 3 requires a protected LIBERO Stage 2A checkpoint.")
    if gate.get("schema") != "zeva-libero-stage2a-gate-v2" or not gate.get("passed"):
        raise ValueError("Stage 3 requires a passed LIBERO Stage 2A gate.")
    if Path(gate["best"]["checkpoint"]).resolve() != checkpoint:
        raise ValueError("Stage 3 must start from the validation-selected Stage 2A checkpoint.")
    if stage2["causal_bank_sha256"] != sha256(args.causal_bank):
        raise ValueError("Stage 2A and Stage 3 causal banks differ.")
    if stage2["zte_checkpoint_sha256"] != sha256(args.zte_checkpoint):
        raise ValueError("Stage 2A and Stage 3 ZTE checkpoints differ.")
    if stage2["task_retrieval_sha256"] != sha256(args.task_retrieval):
        raise ValueError("Stage 2A and Stage 3 task retrieval heads differ.")
    if stage2["live_queries_sha256"] != sha256(args.live_queries):
        raise ValueError("Stage 2A and Stage 3 live-query caches differ.")
    if stage2["statistics_sha256"] != sha256(handoff.statistics):
        raise ValueError("Stage 2A and Stage 3 normalization differs.")
    if bank.manifest["stage1_checkpoint_sha256"] != stage2["zte_checkpoint_sha256"]:
        raise ValueError("The causal bank was exported from another ZTE.")
    return stage2, gate


def _manifest(args, handoff, bank, stage2, gate, trainable_names):
    checkpoint = Path(args.stage2_checkpoint).resolve()
    return {
        "schema": "zeva-libero-stage3-selective-action-manifest-v1",
        "handoff_root": str(handoff.root),
        "statistics_sha256": sha256(handoff.statistics),
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "train_split": "official-train-1614",
        "validation_split": "official-validation-79",
        "stage2_checkpoint": str(checkpoint),
        "stage2_step": int(checkpoint.name),
        "stage2_adapter_sha256": sha256(checkpoint / "zeva_adapter.pth"),
        "stage2_training_state_sha256": sha256(checkpoint / "training_state.pt"),
        "stage2_manifest": stage2,
        "stage2_gate": gate,
        "zte_checkpoint": str(Path(args.zte_checkpoint).resolve()),
        "zte_checkpoint_sha256": sha256(args.zte_checkpoint),
        "causal_bank": str(Path(args.causal_bank).resolve()),
        "causal_bank_sha256": sha256(args.causal_bank),
        "live_queries": str(Path(args.live_queries).resolve()),
        "live_queries_sha256": sha256(args.live_queries),
        "task_retrieval": str(Path(args.task_retrieval).resolve()),
        "task_retrieval_sha256": sha256(args.task_retrieval),
        "policy_horizon": LIBERO_POLICY_HORIZON,
        "execution_horizon": LIBERO_EXECUTION_HORIZON,
        "baseline_protocol": "matched frozen Stage 2A action parameters with identical RNG",
        "frozen": ["pi05_vlm", "pi05_action_layers_0_to_15", "zte", "causal_bank",
                   "task_retrieval", "stage2a_adapter"],
        "trainable": list(trainable_names),
        "train_args": dataclasses.asdict(args),
        "source_sha256": {
            "trainer": sha256(Path(__file__).resolve()),
            "libero_policy": sha256(Path(inspect.getfile(LiberoZevaPolicy)).resolve()),
        },
    }


def _split_batch(batch, processor, accelerator):
    task_ids = batch.pop("zeva.task_id")
    batch.pop("zeva.progress")
    phase_queries = batch.pop("zeva.phase_query")
    live_brief = batch.pop("zeva.live_brief")
    live_brief_mask = batch.pop("zeva.live_brief_mask")
    live_retrieved = batch.pop("zeva.live_retrieved")
    live_retrieved_mask = batch.pop("zeva.live_retrieved_mask")
    observation, actions = processor(batch, device=accelerator.device)
    return (observation, actions, task_ids, phase_queries, live_brief, live_brief_mask,
            live_retrieved, live_retrieved_mask)


def _matched_stage2_flow(policy, anchor, observation, actions, bank_batch, confidence):
    unwrapped = policy.module if hasattr(policy, "module") else policy
    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(actions.device)
    with torch.no_grad():
        output, _ = functional_call(
            unwrapped,
            anchor,
            (observation, actions, bank_batch, confidence),
            strict=False,
        )
        baseline = _foundation_loss(output)
    torch.random.set_rng_state(cpu_state)
    torch.cuda.set_rng_state(cuda_state, actions.device)
    unwrapped.set_foundation_rng_state(cpu_state, cuda_state)
    return baseline


def _anchor_regularizer(policy, anchor):
    unwrapped = policy.module if hasattr(policy, "module") else policy
    parameters = dict(unwrapped.named_parameters())
    terms = [F.mse_loss(parameters[name].float(), value.float()) for name, value in anchor.items()]
    return torch.stack(terms).mean()


def _losses(policy, anchor, observation, actions, bank_batch, confidence, preserve_weight, anchor_weight):
    baseline = _matched_stage2_flow(policy, anchor, observation, actions, bank_batch, confidence)
    output, _ = policy(observation, actions, bank_batch, confidence)
    flow = _foundation_loss(output)
    preserve = F.relu(flow - baseline.detach())
    anchor_loss = _anchor_regularizer(policy, anchor)
    total = flow + preserve_weight * preserve + anchor_weight * anchor_loss
    return {"total": total, "flow": flow, "baseline": baseline.detach(),
            "preserve": preserve, "anchor": anchor_loss}


def _assert_gradients(policy: LiberoZevaPolicy, trainable_names):
    selected = set(trainable_names)
    for name, parameter in policy.named_parameters():
        if name in selected:
            if parameter.grad is None:
                raise RuntimeError(f"Stage 3 trainable action parameter received no gradient: {name}")
        elif parameter.grad is not None:
            raise RuntimeError(f"Stage 3 frozen parameter received a gradient: {name}")


@torch.no_grad()
def evaluate(policy, anchor, loader, bank, processor, config, args, accelerator):
    policy.eval()
    totals: dict[str, list[torch.Tensor]] = {
        "total": [], "flow": [], "baseline": [], "retrieval_accuracy": []
    }
    for batch_index, batch in enumerate(loader):
        if batch_index >= args.eval_batches:
            break
        unpacked = _split_batch(batch, processor, accelerator)
        observation, actions, task_ids, phase_queries = unpacked[:4]
        bank_batch, confidence, accuracy = _retrieve(
            policy, observation, phase_queries, task_ids, *unpacked[4:],
            bank, config, args, training=False
        )
        losses = _losses(
            policy, anchor, observation, actions, bank_batch, confidence,
            args.preserve_loss_weight, args.anchor_loss_weight
        )
        for name in ("total", "flow", "baseline"):
            totals[name].append(accelerator.gather_for_metrics(losses[name].reshape(1)))
        totals["retrieval_accuracy"].append(accelerator.gather_for_metrics(accuracy.reshape(1)))
    return {name: float(torch.cat(values).mean()) if values else math.inf
            for name, values in totals.items()}


def main(args: Args):
    accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(
        find_unused_parameters=False, gradient_as_bucket_view=True
    )])
    torch.manual_seed(args.seed + accelerator.process_index)
    handoff = LiberoHandoff.from_root(args.handoff_root)
    bank = LiberoCausalBank.load(args.causal_bank, device=accelerator.device)
    stage2, stage2_gate = _stage2_manifest(args, handoff, bank)
    policy = LiberoZevaPolicy.from_handoff(
        args.handoff_root, tokenizer_path=args.tokenizer_path, device=str(accelerator.device),
        zte_checkpoint=args.zte_checkpoint,
        adapter_checkpoint=Path(args.stage2_checkpoint) / "zeva_adapter.pth",
        retrieval_checkpoint=args.task_retrieval, causal_bank=args.causal_bank
    )
    trainable = policy.configure_selective_action_stage3(args.final_action_layers)
    trainable_names = tuple(f"foundation.{name}" for name in policy._stage3_trainable_names)  # noqa: SLF001
    named_parameters = dict(policy.named_parameters())
    anchor = {name: named_parameters[name].detach().clone() for name in trainable_names}
    manifest = _manifest(args, handoff, bank, stage2, stage2_gate, trainable_names)
    manifest["world_size"] = accelerator.num_processes
    manifest["effective_global_batch_size"] = (
        args.batch_size * args.gradient_accumulation_steps * accelerator.num_processes
    )
    processor, config = policy.processor, policy.zeva_config
    train_dataset = LiberoStage2ADataset(args.dataset_root, args.live_queries, "train", config)
    validation_dataset = LiberoStage2ADataset(args.dataset_root, args.live_queries, "validation", config)
    if train_dataset.task_names != bank.task_names or validation_dataset.task_names != bank.task_names:
        raise ValueError("LIBERO task ordering differs from the causal bank.")
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=args.num_workers > 0
    )
    validation_workers = min(2, args.num_workers)
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=validation_workers, pin_memory=True,
        persistent_workers=validation_workers > 0
    )
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, betas=(0.9, args.adam_beta2), eps=1e-8,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / max(1, args.warmup_steps))
    )
    start_step, best_validation = 0, math.inf
    if args.resume_checkpoint is not None:
        resume = Path(args.resume_checkpoint)
        state = torch.load(resume / "training_state.pt", map_location="cpu", weights_only=False)
        if state.get("schema") != "zeva-libero-stage3-selective-action-training-v1":
            raise ValueError("Unsupported LIBERO Stage 3 resume checkpoint.")
        if state["manifest"]["stage2_adapter_sha256"] != manifest["stage2_adapter_sha256"]:
            raise ValueError("Stage 3 resume checkpoint uses another Stage 2A adapter.")
        policy.load_stage3_action(resume / "stage3_action.pth")
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        start_step = int(state["step"])
        best_validation = float(state.get("best_validation", math.inf))
    policy, optimizer, scheduler, train_loader, validation_loader = accelerator.prepare(
        policy, optimizer, scheduler, train_loader, validation_loader
    )
    save_dir = Path(args.save_dir)
    if accelerator.is_main_process:
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    accelerator.wait_for_everyone()
    iterator = iter(train_loader)
    progress_bar = tqdm.trange(
        start_step, args.steps, initial=start_step, total=args.steps,
        disable=not accelerator.is_local_main_process
    )
    for step in progress_bar:
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated: dict[str, list[torch.Tensor]] = {}
        accuracies = []
        for micro_step in range(args.gradient_accumulation_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            unpacked = _split_batch(batch, processor, accelerator)
            observation, actions, task_ids, phase_queries = unpacked[:4]
            bank_batch, confidence, accuracy = _retrieve(
                policy, observation, phase_queries, task_ids, *unpacked[4:],
                bank, config, args, training=True
            )
            sync_context = (
                policy.no_sync()
                if micro_step + 1 < args.gradient_accumulation_steps and hasattr(policy, "no_sync")
                else nullcontext()
            )
            with sync_context:
                micro_losses = _losses(
                    policy, anchor, observation, actions, bank_batch, confidence,
                    args.preserve_loss_weight, args.anchor_loss_weight
                )
                accelerator.backward(micro_losses["total"] / args.gradient_accumulation_steps)
            if step == start_step and micro_step == 0:
                _assert_gradients(accelerator.unwrap_model(policy), trainable_names)
            for name, value in micro_losses.items():
                accumulated.setdefault(name, []).append(value.detach())
            accuracies.append(accuracy.detach())
        losses = {name: torch.stack(values).mean() for name, values in accumulated.items()}
        accuracy = torch.stack(accuracies).mean()
        accelerator.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()
        completed = step + 1
        if completed % args.log_freq == 0:
            progress_bar.set_postfix(
                loss=f"{float(losses['total']):.4f}", flow=f"{float(losses['flow']):.4f}",
                base=f"{float(losses['baseline']):.4f}", task=f"{float(accuracy):.3f}"
            )
        if args.save_checkpoints and (completed % args.save_freq == 0 or completed == args.steps):
            validation = evaluate(policy, anchor, validation_loader, bank, processor, config, args, accelerator)
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped = accelerator.unwrap_model(policy)
                step_dir = save_dir / f"{completed:06d}"
                step_dir.mkdir(parents=True, exist_ok=True)
                torch.save(unwrapped.stage3_action_state_dict(manifest), step_dir / "stage3_action.pth")
                torch.save({
                    "schema": "zeva-libero-stage3-selective-action-training-v1",
                    "step": completed, "validation": validation,
                    "validation_loss": validation["total"],
                    "best_validation": min(best_validation, validation["total"]),
                    "gradient_invariants_passed": True, "manifest": manifest,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                }, step_dir / "training_state.pt")
                latest = {"step": completed, "checkpoint": str(step_dir)}
                (save_dir / "latest.json").write_text(json.dumps(latest, indent=2) + "\n")
                if validation["total"] <= best_validation:
                    best_validation = validation["total"]
                    (save_dir / "best.json").write_text(json.dumps({
                        "step": completed, "validation": validation,
                        "checkpoint": str(step_dir)
                    }, indent=2) + "\n")
                (save_dir / "last_metrics.json").write_text(json.dumps({
                    "step": completed, "train": {key: float(value) for key, value in losses.items()},
                    "train_retrieval_accuracy": float(accuracy), "validation": validation
                }, indent=2) + "\n")
            accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main(tyro.cli(Args))
