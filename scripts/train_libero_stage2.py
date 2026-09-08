"""Formal LIBERO Stage 2A: protected ZeVA residuals on frozen PI0.5 and ZTE."""

from __future__ import annotations

import dataclasses
from contextlib import nullcontext
import inspect
import json
import math
from pathlib import Path
from typing import Any

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader, Dataset
import tqdm
import tyro

from openpi.zeva.libero_bank import LiberoCausalBank
from openpi.zeva.libero_contract import LIBERO_ACTION_DIM
from openpi.zeva.libero_contract import LIBERO_CHECKPOINT_SHA256
from openpi.zeva.libero_contract import LIBERO_EXECUTION_HORIZON
from openpi.zeva.libero_contract import LIBERO_POLICY_HORIZON
from openpi.zeva.libero_contract import LiberoHandoff
from openpi.zeva.libero_contract import sha256
from openpi.zeva.libero_data import LiberoStage2Dataset
from openpi.zeva.libero_policy import LiberoZevaPolicy
from openpi.zeva.memory import CausalMemoryManager


@dataclasses.dataclass
class Args:
    handoff_root: str = "/data1/dingxin/libero-memory-baseline-v1"
    dataset_root: str = "/data1/dingxin/libero-memory-baseline-v1/data"
    tokenizer_path: str = (
        "/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/"
        "robotwin-memory-baseline-v1/checkpoint/pretrained_model/tokenizer"
    )
    zte_checkpoint: str = "/data1/dingxin/zeva-runs/libero-v3-h5-lang/stage1-zte/zte_best.pth"
    causal_bank: str = "/data1/dingxin/zeva-runs/libero-v3-h5-lang/stage1-zte/train_causal_bank.pt"
    live_queries: str = "/data1/dingxin/zeva-runs/libero-v3-h5-lang/stage1-zte/live_queries_h5.pt"
    task_retrieval: str = (
        "/data1/dingxin/zeva-runs/libero-v3-h5-lang/"
        "stage1.5-task-retrieval/task_retrieval.pth"
    )
    save_dir: str = "/data1/dingxin/zeva-runs/libero-v3-h5-lang/stage2a-adapter"
    resume_checkpoint: str | None = None
    steps: int = 5_000
    # Per-rank micro-batch. 16 x 2 accumulation x 8 GPUs = effective global 256.
    batch_size: int = 16
    gradient_accumulation_steps: int = 2
    num_workers: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 1e-10
    adam_beta2: float = 0.95
    warmup_steps: int = 500
    prior_loss_weight: float = 0.5
    preserve_loss_weight: float = 1.0
    gate_regularization_weight: float = 1e-3
    phase_noise_std: float = 0.02
    memory_dropout: float = 0.1
    retrieval_confidence_floor: float = 0.2
    save_freq: int = 500
    save_checkpoints: bool = True
    eval_batches: int = 32
    log_freq: int = 10
    seed: int = 1000


class LiberoStage2ADataset(Dataset):
    """H5 PI decisions paired with cached deployment-recurrent ZTE queries."""

    def __init__(self, dataset_root: str | Path, live_queries: str | Path, subset: str, config):
        self.source = LiberoStage2Dataset(dataset_root, subset=subset)
        self.task_names = self.source.task_names
        self.config = config
        cache = torch.load(live_queries, map_location="cpu", weights_only=False)
        if cache.get("schema") != "zeva-libero-live-queries-h5-v1":
            raise ValueError("Stage 2A requires deployment-recurrent LIBERO H5 live queries.")
        if tuple(cache["task_names"]) != self.task_names:
            raise ValueError("Live-query task ordering differs from the LIBERO dataset.")
        self.live_records = cache["splits"][subset]
        if len(self.live_records) != len(self.source.table.episodes):
            raise ValueError("Live-query cache record count differs from the LIBERO dataset.")
        self._samples = []
        for record_index, live in enumerate(self.live_records):
            valid_count = int(self.source.cumulative[record_index + 1] - self.source.cumulative[record_index])
            if int(live["task_id"]) != self.source.task_ids[self.source.table.task(record_index)]:
                raise ValueError("Live-query task label differs from the LIBERO episode metadata.")
            if len(live["decision_frames"]) != len(live["phase_queries"]):
                raise ValueError("Live-query decisions and phase queries have different lengths.")
            if len(live["causal_signals"]) + 1 != len(live["phase_queries"]):
                raise ValueError("Live-query recurrence is missing a transition signal.")
            self._samples.extend(
                (record_index, decision_index, int(frame))
                for decision_index, frame in enumerate(live["decision_frames"])
                if int(frame) < valid_count
            )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record_index, decision_index, frame = self._samples[index]
        source_index = int(self.source.cumulative[record_index]) + frame
        result = dict(self.source[source_index])
        live = self.live_records[record_index]
        result["zeva.phase_query"] = live["phase_queries"][decision_index].float()
        memory = CausalMemoryManager(
            brief_size=self.config.brief_memory_size,
            persistent_size=self.config.persistent_memory_size,
            retrieval_top_k=self.config.retrieval_top_k,
            merge_phase_weight=self.config.merge_phase_weight,
            merge_signal_weight=self.config.merge_signal_weight,
            merge_threshold=self.config.merge_threshold,
            use_brief_memory=self.config.use_brief_memory,
            use_persistent_memory=self.config.use_persistent_memory,
        )
        for transition_index in range(decision_index):
            memory.update(
                live["phase_queries"][transition_index + 1],
                live["causal_signals"][transition_index],
            )
        brief = memory.brief_tensor(device="cpu")
        retrieved = memory.retrieve(result["zeva.phase_query"], device="cpu")
        result["zeva.live_brief"], result["zeva.live_brief_mask"] = self._pad_memory(
            brief, self.config.brief_memory_size, self.config.signal_dim
        )
        result["zeva.live_retrieved"], result["zeva.live_retrieved_mask"] = self._pad_memory(
            retrieved, self.config.retrieval_top_k, self.config.signal_dim
        )
        return result

    @staticmethod
    def _pad_memory(
        values: torch.Tensor | None, size: int, dimension: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        padded = torch.zeros((size, dimension), dtype=torch.float32)
        mask = torch.zeros(size, dtype=torch.bool)
        if values is not None:
            values = values[0].float()
            count = min(size, len(values))
            padded[:count] = values[:count]
            mask[:count] = True
        return padded, mask


def _foundation_loss(output: Any) -> torch.Tensor:
    value = output[0] if isinstance(output, tuple) else output
    if not torch.is_tensor(value):
        raise TypeError(f"Unexpected PI0.5 output {type(value)!r}.")
    return value.mean()


def _losses(
    policy: nn.Module,
    observation,
    actions: torch.Tensor,
    bank_batch,
    injection_confidence: torch.Tensor,
    baseline_flow: torch.Tensor,
    foundation_rng_state: tuple[torch.Tensor, torch.Tensor],
    prior_weight: float,
    preserve_weight: float,
    gate_regularization_weight: float,
) -> dict[str, torch.Tensor]:
    unwrapped = policy.module if hasattr(policy, "module") else policy
    unwrapped.set_foundation_rng_state(*foundation_rng_state)
    foundation_output, action_prior = policy(observation, actions, bank_batch, injection_confidence)
    flow = _foundation_loss(foundation_output)
    prior = F.mse_loss(action_prior, actions[..., :LIBERO_ACTION_DIM].to(action_prior.dtype))
    preserve = F.relu(flow - baseline_flow.detach())
    gate = unwrapped.injection_gate_regularizer()
    total = flow + prior_weight * prior + preserve_weight * preserve + gate_regularization_weight * gate
    return {
        "total": total,
        "flow": flow,
        "prior": prior,
        "baseline": baseline_flow.detach(),
        "preserve": preserve,
        "gate": gate,
    }


def _assert_adapter_gradients(policy: LiberoZevaPolicy) -> None:
    if any(parameter.grad is not None for parameter in policy.foundation.parameters()):
        raise RuntimeError("Stage 2A invariant failed: frozen PI0.5 received gradients.")
    if any(parameter.grad is not None for parameter in policy.causal_transition_encoder.parameters()):
        raise RuntimeError("Stage 2A invariant failed: frozen ZTE received gradients.")
    modules = (
        policy.task_token_projector,
        policy.memory_context_encoder,
        policy.action_prior,
        policy.causal_prefix_projector,
        policy.prior_action_projector,
    )
    if any(not any(parameter.grad is not None for parameter in module.parameters()) for module in modules):
        raise RuntimeError("Stage 2A invariant failed: a trainable ZeVA module received no gradients.")
    if policy.prefix_gate_logit.grad is None or policy.action_gate_logit.grad is None:
        raise RuntimeError("Stage 2A invariant failed: residual gates received no gradients.")


def _retrieve(
    policy: nn.Module,
    observation,
    phase_queries: torch.Tensor,
    task_ids: torch.Tensor,
    live_brief: torch.Tensor,
    live_brief_mask: torch.Tensor,
    live_retrieved: torch.Tensor,
    live_retrieved_mask: torch.Tensor,
    bank: LiberoCausalBank,
    config,
    args: Args,
    *,
    training: bool,
):
    unwrapped = policy.module if hasattr(policy, "module") else policy
    with torch.no_grad():
        predicted_ids, scores = unwrapped.retrieve_task_ids_from_language(observation)
        phase_queries = phase_queries.to(bank.phase_key.device, dtype=bank.phase_key.dtype)
        if training and args.phase_noise_std > 0:
            phase_queries = F.normalize(
                phase_queries + torch.randn_like(phase_queries) * args.phase_noise_std, dim=-1
            )
        bank_batch = bank.retrieve(
            predicted_ids,
            phase_queries,
            brief_size=config.brief_memory_size,
            retrieval_top_k=config.retrieval_top_k,
        )
        offline_brief_size = bank_batch.brief_signals.shape[1]
        offline_retrieved_size = bank_batch.retrieved_signals.shape[1]
        bank_batch.brief_signals = torch.cat(
            (bank_batch.brief_signals, live_brief.to(bank.phase_key.device)), dim=1
        )
        bank_batch.retrieved_signals = torch.cat(
            (bank_batch.retrieved_signals, live_retrieved.to(bank.phase_key.device)), dim=1
        )
        bank_batch.brief_mask = torch.cat(
            (
                torch.ones(len(task_ids), offline_brief_size, dtype=torch.bool, device=bank.phase_key.device),
                live_brief_mask.to(bank.phase_key.device),
            ),
            dim=1,
        )
        bank_batch.retrieved_mask = torch.cat(
            (
                torch.ones(len(task_ids), offline_retrieved_size, dtype=torch.bool, device=bank.phase_key.device),
                live_retrieved_mask.to(bank.phase_key.device),
            ),
            dim=1,
        )
        confidence = unwrapped.calibrate_retrieval_confidence(
            scores, floor=args.retrieval_confidence_floor
        )
        if training and args.memory_dropout > 0:
            keep = torch.rand_like(confidence) >= args.memory_dropout
            confidence = confidence * keep.to(confidence.dtype)
        accuracy = (predicted_ids == task_ids.to(predicted_ids.device)).float().mean()
    return bank_batch, confidence, accuracy


def _matched_baseline_flow(policy: nn.Module, observation, actions: torch.Tensor):
    unwrapped = policy.module if hasattr(policy, "module") else policy
    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(actions.device)
    with torch.no_grad():
        baseline = _foundation_loss(unwrapped.foundation(observation, actions))
    torch.random.set_rng_state(cpu_state)
    torch.cuda.set_rng_state(cuda_state, actions.device)
    return baseline, (cpu_state, cuda_state)


def _manifest(args: Args, handoff: LiberoHandoff, bank: LiberoCausalBank) -> dict[str, Any]:
    zte = torch.load(args.zte_checkpoint, map_location="cpu", weights_only=False)
    if zte.get("schema") != "zeva-libero-zte-stage1-checkpoint-v3":
        raise ValueError("Stage 2A requires a formal LIBERO Stage 1 v3 checkpoint.")
    if zte["manifest"].get("causal_transition_horizon") != LIBERO_EXECUTION_HORIZON:
        raise ValueError("LIBERO Stage 2A requires an H5 ZTE.")
    if bank.manifest["stage1_checkpoint_sha256"] != sha256(args.zte_checkpoint):
        raise ValueError("Causal bank was not exported from the selected ZTE.")
    if bank.manifest["statistics_sha256"] != sha256(handoff.statistics):
        raise ValueError("Causal bank normalization differs from PI0.5.")
    return {
        "schema": "zeva-libero-stage2a-manifest-v2",
        "handoff_root": str(handoff.root),
        "foundation_checkpoint": str(handoff.checkpoint),
        "foundation_checkpoint_sha256": LIBERO_CHECKPOINT_SHA256,
        "statistics_sha256": sha256(handoff.statistics),
        "tokenizer_path": str(Path(args.tokenizer_path).resolve()),
        "zte_checkpoint": str(Path(args.zte_checkpoint).resolve()),
        "zte_checkpoint_sha256": sha256(args.zte_checkpoint),
        "zte_step": int(zte["step"]),
        "causal_bank": str(Path(args.causal_bank).resolve()),
        "causal_bank_sha256": sha256(args.causal_bank),
        "causal_bank_manifest": bank.manifest,
        "live_queries": str(Path(args.live_queries).resolve()),
        "live_queries_sha256": sha256(args.live_queries),
        "task_retrieval": str(Path(args.task_retrieval).resolve()),
        "task_retrieval_sha256": sha256(args.task_retrieval),
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "train_split": "official-train-1614",
        "validation_split": "official-validation-79",
        "policy_horizon": LIBERO_POLICY_HORIZON,
        "execution_horizon": LIBERO_EXECUTION_HORIZON,
        "retrieval_protocol": "task-language inferred task + recurrent H5 ZTE phase; no oracle progress",
        "frozen": ["pi05_libero_baseline", "zte", "causal_bank", "task_retrieval"],
        "trainable": [
            "task_token_projector",
            "memory_context_encoder",
            "action_prior",
            "causal_prefix_projector",
            "prior_action_projector",
            "prefix_gate_logit",
            "action_gate_logit",
        ],
        "train_args": dataclasses.asdict(args),
        "source_sha256": {
            "trainer": sha256(Path(__file__).resolve()),
            "libero_policy": sha256(Path(inspect.getfile(LiberoZevaPolicy)).resolve()),
        },
    }


@torch.no_grad()
def evaluate(policy, loader, bank, processor, config, args, accelerator):
    policy.eval()
    totals: dict[str, list[torch.Tensor]] = {
        "total": [], "flow": [], "prior": [], "baseline": [], "retrieval_accuracy": []
    }
    for batch_index, batch in enumerate(loader):
        if batch_index >= args.eval_batches:
            break
        task_ids = batch.pop("zeva.task_id")
        batch.pop("zeva.progress")
        phase_queries = batch.pop("zeva.phase_query")
        live_brief = batch.pop("zeva.live_brief")
        live_brief_mask = batch.pop("zeva.live_brief_mask")
        live_retrieved = batch.pop("zeva.live_retrieved")
        live_retrieved_mask = batch.pop("zeva.live_retrieved_mask")
        observation, actions = processor(batch, device=accelerator.device)
        bank_batch, confidence, retrieval_accuracy = _retrieve(
            policy, observation, phase_queries, task_ids, live_brief, live_brief_mask,
            live_retrieved, live_retrieved_mask, bank, config, args, training=False
        )
        baseline_flow, foundation_rng_state = _matched_baseline_flow(policy, observation, actions)
        losses = _losses(
            policy, observation, actions, bank_batch, confidence, baseline_flow,
            foundation_rng_state, args.prior_loss_weight, args.preserve_loss_weight,
            args.gate_regularization_weight
        )
        for name in ("total", "flow", "prior", "baseline"):
            totals[name].append(accelerator.gather_for_metrics(losses[name].detach().reshape(1)))
        totals["retrieval_accuracy"].append(
            accelerator.gather_for_metrics(retrieval_accuracy.detach().reshape(1))
        )
    return {
        name: float(torch.cat(values).mean()) if values else math.inf
        for name, values in totals.items()
    }


def main(args: Args) -> None:
    accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(
        find_unused_parameters=False, gradient_as_bucket_view=True
    )])
    torch.manual_seed(args.seed + accelerator.process_index)
    handoff = LiberoHandoff.from_root(args.handoff_root)
    policy = LiberoZevaPolicy.from_handoff(
        args.handoff_root,
        tokenizer_path=args.tokenizer_path,
        device=str(accelerator.device),
        zte_checkpoint=args.zte_checkpoint,
        retrieval_checkpoint=args.task_retrieval,
        causal_bank=args.causal_bank,
    )
    trainable = policy.configure_adapter_stage2()
    if hasattr(policy.foundation, "gradient_checkpointing_enable"):
        policy.foundation.gradient_checkpointing_enable()
    bank = LiberoCausalBank.load(args.causal_bank, device=accelerator.device)
    manifest = _manifest(args, handoff, bank)
    manifest["world_size"] = accelerator.num_processes
    manifest["effective_global_batch_size"] = (
        args.batch_size * args.gradient_accumulation_steps * accelerator.num_processes
    )
    live_cache = torch.load(args.live_queries, map_location="cpu", weights_only=False)
    if live_cache.get("zte_checkpoint_sha256") != sha256(args.zte_checkpoint):
        raise ValueError("Live queries were not exported from the selected ZTE checkpoint.")
    if live_cache.get("statistics_sha256") != sha256(handoff.statistics):
        raise ValueError("Live queries use different PI0.5 normalization statistics.")
    retrieval_checkpoint = torch.load(args.task_retrieval, map_location="cpu", weights_only=False)
    if retrieval_checkpoint.get("causal_bank_sha256") != sha256(args.causal_bank):
        raise ValueError("Task retrieval was not trained against the selected causal bank.")
    if float(retrieval_checkpoint.get("validation_accuracy", 0.0)) < 0.95:
        raise ValueError("Task-language retrieval validation accuracy is below the 95% Stage 1.5 gate.")
    processor, config = policy.processor, policy.zeva_config
    train_dataset = LiberoStage2ADataset(
        args.dataset_root, args.live_queries, subset="train", config=config
    )
    validation_dataset = LiberoStage2ADataset(
        args.dataset_root, args.live_queries, subset="validation", config=config
    )
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
        if state.get("schema") != "zeva-libero-stage2a-training-state-v2":
            raise ValueError("Unsupported LIBERO Stage 2A resume checkpoint.")
        if state["manifest"]["causal_bank_sha256"] != manifest["causal_bank_sha256"]:
            raise ValueError("Stage 2A resume checkpoint uses a different causal bank.")
        policy.load_adapter(resume / "zeva_adapter.pth")
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
        start_step, args.steps, disable=not accelerator.is_local_main_process,
        initial=start_step, total=args.steps
    )
    for step in progress_bar:
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated_losses: dict[str, list[torch.Tensor]] = {}
        retrieval_accuracies = []
        for micro_step in range(args.gradient_accumulation_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            task_ids = batch.pop("zeva.task_id")
            batch.pop("zeva.progress")
            phase_queries = batch.pop("zeva.phase_query")
            live_brief = batch.pop("zeva.live_brief")
            live_brief_mask = batch.pop("zeva.live_brief_mask")
            live_retrieved = batch.pop("zeva.live_retrieved")
            live_retrieved_mask = batch.pop("zeva.live_retrieved_mask")
            observation, actions = processor(batch, device=accelerator.device)
            bank_batch, confidence, retrieval_accuracy = _retrieve(
                policy, observation, phase_queries, task_ids, live_brief, live_brief_mask,
                live_retrieved, live_retrieved_mask, bank, config, args, training=True
            )
            sync_context = (
                policy.no_sync()
                if micro_step + 1 < args.gradient_accumulation_steps and hasattr(policy, "no_sync")
                else nullcontext()
            )
            with sync_context:
                baseline_flow, foundation_rng_state = _matched_baseline_flow(policy, observation, actions)
                micro_losses = _losses(
                    policy, observation, actions, bank_batch, confidence, baseline_flow,
                    foundation_rng_state, args.prior_loss_weight, args.preserve_loss_weight,
                    args.gate_regularization_weight
                )
                accelerator.backward(micro_losses["total"] / args.gradient_accumulation_steps)
            if step == start_step and micro_step == 0:
                _assert_adapter_gradients(accelerator.unwrap_model(policy))
            for name, value in micro_losses.items():
                accumulated_losses.setdefault(name, []).append(value.detach())
            retrieval_accuracies.append(retrieval_accuracy.detach())
        losses = {name: torch.stack(values).mean() for name, values in accumulated_losses.items()}
        retrieval_accuracy = torch.stack(retrieval_accuracies).mean()
        accelerator.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()
        completed = step + 1
        if completed % args.log_freq == 0:
            progress_bar.set_postfix(
                loss=f"{float(losses['total']):.4f}", flow=f"{float(losses['flow']):.4f}",
                base=f"{float(losses['baseline']):.4f}", prior=f"{float(losses['prior']):.4f}",
                task=f"{float(retrieval_accuracy):.3f}"
            )
        if args.save_checkpoints and (completed % args.save_freq == 0 or completed == args.steps):
            validation = evaluate(policy, validation_loader, bank, processor, config, args, accelerator)
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped = accelerator.unwrap_model(policy)
                step_dir = save_dir / f"{completed:06d}"
                step_dir.mkdir(parents=True, exist_ok=True)
                torch.save(unwrapped.adapter_state_dict(manifest), step_dir / "zeva_adapter.pth")
                torch.save({
                    "schema": "zeva-libero-stage2a-training-state-v2",
                    "step": completed,
                    "validation": validation,
                    "validation_loss": validation["total"],
                    "best_validation": min(best_validation, validation["total"]),
                    "gradient_invariants_passed": True,
                    "manifest": manifest,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                }, step_dir / "training_state.pt")
                latest = {"step": completed, "checkpoint": str(step_dir)}
                (save_dir / "latest.json").write_text(json.dumps(latest, indent=2) + "\n")
                if validation["total"] <= best_validation:
                    best_validation = validation["total"]
                    (save_dir / "best.json").write_text(json.dumps({
                        "step": completed, "validation": validation, "checkpoint": str(step_dir)
                    }, indent=2) + "\n")
                (save_dir / "last_metrics.json").write_text(json.dumps({
                    "step": completed,
                    "train_loss": float(losses["total"]),
                    "train_flow": float(losses["flow"]),
                    "train_baseline_flow": float(losses["baseline"]),
                    "validation": validation,
                }, indent=2) + "\n")
            accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main(tyro.cli(Args))
