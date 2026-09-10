"""Formal Stage 2: tune the PI0.5 action expert and Zeva with PaliGemma frozen."""

from __future__ import annotations

from contextlib import nullcontext
import dataclasses
import hashlib
import importlib.metadata
import inspect
import json
import math
from pathlib import Path
from typing import Any

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from safetensors.torch import load_model
from safetensors.torch import save_model
import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import tqdm
import tyro

from openpi.zeva.causal_bank import RobotWinCausalBank
from openpi.zeva.memory import CausalMemoryManager
from openpi.zeva.robotwin_contract import ROBOTWIN_ACTION_DIM
from openpi.zeva.robotwin_contract import ROBOTWIN_ACTION_HORIZON
from openpi.zeva.robotwin_contract import ROBOTWIN_CAMERA_KEYS
from openpi.zeva.robotwin_contract import RobotWinHandoff
from openpi.zeva.robotwin_contract import prepare_robotwin_pi_image
from openpi.zeva.robotwin_policy import RobotWinZevaPolicy
from openpi.zeva.robotwin_policy import gaussian_action_prior_nll

try:
    from scripts.train_robotwin_zte import FFmpegRoboTwinDataset
    from scripts.train_robotwin_zte import TorchCodecRoboTwinDataset
except ModuleNotFoundError:  # Direct `python scripts/...py` execution.
    from train_robotwin_zte import FFmpegRoboTwinDataset
    from train_robotwin_zte import TorchCodecRoboTwinDataset


@dataclasses.dataclass
class Args:
    handoff_root: str = (
        "/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1"
    )
    dataset_root: str = "/data1/dingxin/robotwin-lerobot-sidney-eef16-v1/data"
    foundation_checkpoint: str | None = None
    # Optional weights-only Stage 2 checkpoint used to initialize the PI0.5
    # foundation after loading the released config/processors.  This lets a
    # ZeVA residual run start from the already specialized matched Base rather
    # than following a second, independently drifting action-expert trajectory.
    initial_stage2_checkpoint: str | None = None
    # Optional immutable, independently trained Base action path used as the
    # paired teacher while the ZeVA action expert continues to update.
    anchor_stage2_checkpoint: str | None = None
    # Optional immutable foundation path used as the paired teacher when the
    # current run tunes the action expert directly from untouched best-v1.
    # The v11 prior_zeva variant requires this to be the same checkpoint as
    # foundation_checkpoint, so the teacher is the independently defined
    # untouched PI rather than a moving residual-off student.
    anchor_foundation_checkpoint: str | None = None
    # Optional frozen language-coordinate source for Zeva.  This allows a new
    # PI policy checkpoint to reuse an existing Stage 1 lineage explicitly,
    # rather than silently recomputing B0 from the new PI embedding table.
    goal_embedding_checkpoint: str | None = None
    zte_checkpoint: str = "/data1/dingxin/zeva-runs/robotwin-v5-h15-tasklang/stage1-zte/zte_best.pth"
    causal_bank: str = "/data1/dingxin/zeva-runs/robotwin-v5-h15-tasklang/stage1-zte/train_causal_bank.pt"
    live_queries: str = "/data1/dingxin/zeva-runs/robotwin-v5-h15-tasklang/stage1-zte/live_queries_h15.pt"
    task_retrieval: str = (
        "/data1/dingxin/zeva-runs/robotwin-v5-h15-tasklang/"
        "stage1.5-task-retrieval/task_retrieval.pth"
    )
    save_dir: str = (
        "/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/"
        "stage2-action-expert-v8-accelerated"
    )
    # Optional task-specialization manifest.  Filtering happens only at the
    # Stage 2 sample layer: the full Stage 1 task vocabulary, ZTE, causal bank,
    # and task-language retrieval IDs stay unchanged.
    task_subset: str | None = None
    training_variant: str = "zeva"
    resume_checkpoint: str | None = None
    steps: int = 5_000
    # Per-rank micro-batch. 16 x 2 accumulation x 8 GPUs = effective global 256.
    # Freezing PaliGemma removes its gradients and optimizer state, allowing a
    # larger micro-batch than full-PI0.5 Stage 2 while preserving global batch.
    batch_size: int = 16
    gradient_accumulation_steps: int = 2
    num_workers: int = 4
    video_backend: str = "torchcodec"
    # Limit each AV1 decoder; filter_threads are serialized in the shared
    # FFmpeg adapter to prevent host-wide thread oversubscription.
    decoder_threads: int = 1
    # The already RoboTwin-trained PI0.5 action expert uses a 10x smaller
    # learning rate than the new Zeva modules. PaliGemma stays frozen.
    action_expert_learning_rate: float = 5e-6
    learning_rate: float = 5e-5
    weight_decay: float = 1e-10
    adam_beta2: float = 0.95
    warmup_steps: int = 500
    # BehaviorVLA sums Gaussian NLL over action dimensions and scales it by 0.01.
    prior_loss_weight: float = 0.01
    preserve_loss_weight: float = 1.0
    # Positive margin required of residual-on relative to the matched
    # residual-off forward.  Zero preserves the historical non-regression
    # hinge; a small positive value gives a fresh zero residual a useful
    # optimization signal toward measurable improvement.
    paired_improvement_margin: float = 0.0
    # Run the matched frozen teacher on one optimizer step out of every four.
    # Scaling a sampled preserve loss by the interval keeps its expected weight.
    baseline_preserve_interval: int = 4
    gate_regularization_weight: float = 1e-3
    phase_noise_std: float = 0.02
    memory_dropout: float = 0.1
    # Drop the complete action-prior residual for 40% of training examples.
    # This matches BehaviorVLA's Bernoulli keep probability of 0.6 and is
    # intentionally stronger than the separate 10% whole-memory dropout.
    prior_residual_dropout_probability: float = 0.4
    # A v11 prior is applied only to the executed prefix.  Keep 50 for legacy
    # variants; the v11 launcher sets 15 to match RoboTwin's H15 replan loop.
    prior_injection_horizon: int = 50
    # Fresh residual branches may use a stronger gate while retaining exact
    # step-zero Base equivalence because both injection projectors start at 0.
    initial_residual_gate_probability: float = 0.01
    retrieval_confidence_floor: float = 0.2
    save_freq: int = 500
    save_checkpoints: bool = True
    eval_batches: int = 32
    log_freq: int = 10
    compile_model: bool = True
    compile_mode: str = "default"
    seed: int = 1000


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_identity(path: str | Path) -> dict[str, Any]:
    checkpoint = Path(path).resolve()
    identity: dict[str, Any] = {
        "path": str(checkpoint),
        "config_sha256": _sha256(checkpoint / "config.json"),
        "tokenizer_sha256": _sha256(checkpoint / "tokenizer" / "tokenizer.json"),
    }
    declared_hash = checkpoint / "TRANSFER_VALID.sha256"
    if declared_hash.is_file():
        fields = declared_hash.read_text().strip().split()
        if len(fields) == 2 and fields[1] == "model.safetensors" and len(fields[0]) == 64:
            identity["model_sha256"] = fields[0]
            identity["model_sha256_source"] = str(declared_hash)
    return identity


def _initial_stage2_identity(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    checkpoint = Path(path).resolve()
    model = checkpoint / "model.safetensors"
    training_state = checkpoint / "training_state.pt"
    run_manifest = checkpoint.parent / "manifest.json"
    for required in (model, training_state, run_manifest):
        if not required.is_file() or required.stat().st_size <= 0:
            raise FileNotFoundError(f"Missing initial Stage 2 artifact: {required}")
    return {
        "path": str(checkpoint),
        "step": int(checkpoint.name),
        "model_size": model.stat().st_size,
        "training_state_size": training_state.stat().st_size,
        "source_manifest": str(run_manifest),
        "source_manifest_sha256": _sha256(run_manifest),
    }


def _validated_runtime_versions() -> dict[str, str]:
    import tokenizers  # noqa: PLC0415

    versions = {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "tokenizers", "accelerate")
    }
    versions["tokenizers_module"] = tokenizers.__version__
    if (
        versions["transformers"] != "5.5.4"
        or versions["tokenizers"] != "0.22.2"
        or not versions["tokenizers_module"].startswith("0.21.")
    ):
        raise RuntimeError(
            "RoboTwin Stage 2 requires the anchor-validated native Transformers 5.5.4 "
            "runtime with tokenizers 0.22.2 distribution metadata and the host-compatible "
            "0.21.x extension module; got "
            f"{versions['transformers']} / {versions['tokenizers']} / "
            f"{versions['tokenizers_module']}."
        )
    return versions


def _load_task_subset(path: str | Path | None) -> tuple[str, ...] | None:
    if path is None:
        return None
    payload = json.loads(Path(path).read_text())
    if payload.get("schema") != "zeva-robotwin-task-subset-v1":
        raise ValueError("Stage 2 task subset must use zeva-robotwin-task-subset-v1.")
    task_names = payload.get("task_names")
    if not isinstance(task_names, list) or not task_names:
        raise ValueError("Stage 2 task subset must contain a non-empty task_names list.")
    if any(not isinstance(task, str) or not task for task in task_names):
        raise ValueError("Every task subset entry must be a non-empty string.")
    if len(task_names) != len(set(task_names)):
        raise ValueError("Stage 2 task subset contains duplicate task names.")
    return tuple(task_names)


class RobotWinStage2Dataset(Dataset):
    """H15 PI decisions paired with cached deployment-recurrent ZTE queries."""

    def __init__(
        self,
        adapter_manifest: str | Path,
        live_queries: str | Path,
        subset: str,
        config,
        selected_tasks: tuple[str, ...] | None = None,
        video_backend: str = "torchcodec",
        decoder_threads: int = 1,
    ):
        if video_backend == "torchcodec":
            self.source = TorchCodecRoboTwinDataset(adapter_manifest, subset=subset)
        elif video_backend == "ffmpeg":
            self.source = FFmpegRoboTwinDataset(
                adapter_manifest, subset=subset, decoder_threads=decoder_threads
            )
        else:
            raise ValueError(f"Unsupported Stage 2 video backend: {video_backend!r}.")
        self.dataset = self.source.dataset
        task_names = sorted({record["key"][1] for record in self.dataset._records})  # noqa: SLF001
        self.task_names = tuple(task_names)
        self._task_ids = {name: index for index, name in enumerate(task_names)}
        if selected_tasks is None:
            selected_tasks = self.task_names
        unknown_tasks = sorted(set(selected_tasks) - set(self.task_names))
        if unknown_tasks:
            raise ValueError(f"Unknown Stage 2 task subset entries: {unknown_tasks}.")
        self.selected_task_names = tuple(selected_tasks)
        selected_task_set = set(self.selected_task_names)
        self.config = config
        cache = torch.load(live_queries, map_location="cpu")
        if cache.get("schema") != "zeva-robotwin-live-queries-h15-v1":
            raise ValueError("Stage 2A requires deployment-recurrent H15 live queries.")
        if tuple(cache["task_names"]) != self.task_names:
            raise ValueError("Live-query task ordering differs from the Stage 2 dataset.")
        self.live_records = cache["splits"][subset]
        if len(self.live_records) != len(self.dataset._records):  # noqa: SLF001
            raise ValueError("Live-query cache record count differs from the Stage 2 dataset.")
        self._samples = [
            (record_index, decision_index, int(frame))
            for record_index, live in enumerate(self.live_records)
            if self.dataset._records[record_index]["key"][1] in selected_task_set  # noqa: SLF001
            for decision_index, frame in enumerate(live["decision_frames"])
        ]
        if not self._samples:
            raise ValueError(f"Task subset selected no samples from split {subset!r}.")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record_index, decision_index, frame = self._samples[index]
        record = self.dataset._records[record_index]  # noqa: SLF001
        episode_start = int(self.dataset._cumulative[record_index])  # noqa: SLF001
        result = dict(self.dataset[episode_start + frame])
        images = self.source.read_images(record, [frame])
        for key in ROBOTWIN_CAMERA_KEYS:
            # FFmpeg returns CHW uint8 [0,255], while the frozen PI0.5
            # preprocessor declares VISUAL=IDENTITY.  Normalize explicitly at
            # the dataset boundary so training exactly matches deployment.
            result[key] = prepare_robotwin_pi_image(images[key][0], name=key)
        result["zeva.task_id"] = torch.tensor(self._task_ids[record["key"][1]], dtype=torch.long)
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


def _foundation_loss(output: Any, *, reduction: str = "mean") -> torch.Tensor:
    value = output[0] if isinstance(output, tuple) else output
    if not torch.is_tensor(value):
        raise TypeError(f"Unexpected PI0.5 training output: {type(value)!r}.")
    if reduction == "mean":
        return value.mean()
    if reduction == "none":
        if value.ndim != 1:
            raise ValueError(
                "PI0.5 reduction='none' must return one matched loss per example; "
                f"got {tuple(value.shape)}."
            )
        return value
    raise ValueError(f"Unsupported foundation loss reduction: {reduction!r}.")


def _preprocess_with_task_only_goal(
    policy: nn.Module,
    preprocessor,
    raw_batch: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Match deployment's explicit task-only goal embedding exactly."""
    unwrapped = policy.module if hasattr(policy, "module") else policy
    tasks = raw_batch["task"]
    processed = preprocessor(raw_batch)
    processed["zeva.goal_embedding"] = unwrapped._task_only_goal_embedding(  # noqa: SLF001
        tasks, processed["action"].device
    )
    return processed


def _losses(
    policy: nn.Module,
    processed: dict[str, torch.Tensor],
    bank_batch,
    injection_confidence: torch.Tensor,
    baseline_flow: torch.Tensor | None,
    foundation_rng_state: tuple[torch.Tensor, torch.Tensor] | None,
    prior_weight: float,
    preserve_weight: float,
    gate_regularization_weight: float,
    prior_residual_dropout_probability: float,
    paired_improvement_margin: float = 0.0,
    preserve_scale: float = 1.0,
    prior_supervision_horizon: int = ROBOTWIN_ACTION_HORIZON,
    *,
    training: bool,
) -> dict[str, torch.Tensor]:
    unwrapped = policy.module if hasattr(policy, "module") else policy
    if foundation_rng_state is not None:
        unwrapped.set_foundation_rng_state(*foundation_rng_state)
    prior_residual_mask = None
    if training and prior_residual_dropout_probability > 0:
        batch_size = processed["action"].shape[0]
        prior_residual_mask = (
            torch.rand(batch_size, device=processed["action"].device)
            >= prior_residual_dropout_probability
        ).to(processed["action"].dtype)
    foundation_output, action_prior = policy(
        processed,
        bank_phase_token=bank_batch.phase_token,
        bank_brief_signals=bank_batch.brief_signals,
        bank_retrieved_signals=bank_batch.retrieved_signals,
        bank_brief_mask=bank_batch.brief_mask,
        bank_retrieved_mask=bank_batch.retrieved_mask,
        injection_confidence=injection_confidence,
        prior_residual_mask=prior_residual_mask,
        foundation_reduction="none",
    )
    flow_per_sample = _foundation_loss(foundation_output, reduction="none")
    flow = flow_per_sample.mean()
    target = processed["action"][..., :ROBOTWIN_ACTION_DIM].to(action_prior.mean.dtype)
    prior_horizon = min(int(prior_supervision_horizon), target.shape[1])
    if prior_horizon <= 0:
        raise ValueError("prior_supervision_horizon must be positive.")
    supervised_prior = action_prior._replace(
        mean=action_prior.mean[:, :prior_horizon],
        log_std=action_prior.log_std[:, :prior_horizon],
    )
    prior = gaussian_action_prior_nll(supervised_prior, target[:, :prior_horizon])
    if baseline_flow is None:
        baseline = flow.detach().new_full((), float("nan"))
        baseline_per_sample = flow_per_sample.detach().new_full(
            flow_per_sample.shape, float("nan")
        )
        preserve = flow.new_zeros(())
        baseline_sampled = flow.new_zeros(())
    else:
        baseline_per_sample = baseline_flow.detach()
        if baseline_per_sample.shape != flow_per_sample.shape:
            raise ValueError(
                "Matched baseline and ZeVA flow losses must have identical per-example shapes."
            )
        baseline = baseline_per_sample.mean()
        # Preserve every example independently.  The previous aggregate hinge
        # allowed degradation on one task to cancel improvement on another.
        preserve = (
            F.relu(
                flow_per_sample
                - baseline_per_sample
                + float(paired_improvement_margin)
            ).mean()
            * preserve_scale
        )
        baseline_sampled = flow.new_ones(())
    task_schema = unwrapped._task_schema(processed)  # noqa: SLF001
    gate = unwrapped.injection_gate_regularizer(task_schema, bank_batch.phase_token)
    paired_improvement = (
        flow.new_full((), float("nan")) if baseline_flow is None else baseline - flow
    )
    paired_win_fraction = (
        flow.new_full((), float("nan"))
        if baseline_flow is None
        else (flow_per_sample < baseline_per_sample).to(flow.dtype).mean()
    )
    paired_degradation = (
        flow.new_full((), float("nan"))
        if baseline_flow is None
        else F.relu(flow_per_sample - baseline_per_sample).mean()
    )
    total = flow + prior_weight * prior + preserve_weight * preserve + gate_regularization_weight * gate
    return {
        "total": total,
        "flow": flow,
        "prior": prior,
        "baseline": baseline,
        "baseline_sampled": baseline_sampled,
        "preserve": preserve,
        "gate": gate,
        "paired_improvement": paired_improvement,
        "paired_win_fraction": paired_win_fraction,
        "paired_degradation": paired_degradation,
        "flow_per_sample": flow_per_sample,
        "baseline_per_sample": baseline_per_sample,
        "prior_std": supervised_prior.log_std.detach().exp().mean(),
        "prior_residual_keep": (
            prior.new_tensor(1.0)
            if prior_residual_mask is None
            else prior_residual_mask.detach().mean()
        ),
    }


def _baseline_losses(policy: nn.Module, processed: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Flow-only loss for the matched PI0.5 action-expert baseline."""
    output = policy(processed, foundation_only=True)
    flow = _foundation_loss(output)
    nan = flow.detach().new_full((), float("nan"))
    zero = flow.detach().new_zeros(())
    return {
        "total": flow,
        "flow": flow,
        "prior": nan,
        "baseline": flow.detach(),
        "baseline_sampled": zero,
        "preserve": zero,
        "gate": zero,
        "paired_improvement": nan,
        "paired_win_fraction": nan,
        "paired_degradation": nan,
        "flow_per_sample": flow.reshape(1),
        "baseline_per_sample": flow.detach().reshape(1),
        "prior_std": nan,
        "prior_residual_keep": nan,
    }


def _assert_action_expert_finetune_gradients(
    policy: RobotWinZevaPolicy,
    *,
    include_zeva: bool,
    action_expert_trainable: bool = True,
    prior_only: bool = False,
) -> None:
    core = policy.foundation.model
    backbone = core.paligemma_with_expert.paligemma
    if any(parameter.requires_grad for parameter in backbone.parameters()):
        raise RuntimeError("Stage 2 invariant failed: frozen PaliGemma is trainable.")
    if any(parameter.grad is not None for parameter in backbone.parameters()):
        raise RuntimeError("Stage 2 invariant failed: frozen PaliGemma received gradients.")
    action_modules = (
        core.paligemma_with_expert.gemma_expert.model,
        core.action_in_proj,
        core.action_out_proj,
        core.time_mlp_in,
        core.time_mlp_out,
    )
    if action_expert_trainable:
        if any(
            not any(parameter.grad is not None for parameter in module.parameters())
            for module in action_modules
        ):
            raise RuntimeError("Stage 2 invariant failed: an action-expert module received no gradients.")
    elif any(parameter.grad is not None for module in action_modules for parameter in module.parameters()):
        raise RuntimeError("Stage 2 invariant failed: frozen action expert received gradients.")
    if any(parameter.grad is not None for parameter in policy.causal_transition_encoder.parameters()):
        raise RuntimeError("Stage 2 invariant failed: frozen ZTE received gradients.")
    if policy.retrieval_head is not None and any(
        parameter.grad is not None for parameter in policy.retrieval_head.parameters()
    ):
        raise RuntimeError("Stage 2 invariant failed: frozen task retrieval received gradients.")
    full_zeva_modules = (
        policy.task_token_projector,
        policy.memory_context_encoder,
        policy.action_prior,
        policy.causal_action_projector,
        policy.prior_action_projector,
        policy.residual_gate_router,
    )
    prior_zeva_modules = (
        policy.task_token_projector,
        policy.memory_context_encoder,
        policy.action_prior,
        policy.prior_action_projector,
    )
    if include_zeva:
        required_modules = (
            prior_zeva_modules
            if prior_only and action_expert_trainable
            else prior_zeva_modules + (policy.residual_gate_router,)
            if prior_only
            else full_zeva_modules
        )
        if any(
            not any(parameter.grad is not None for parameter in module.parameters())
            for module in required_modules
        ):
            raise RuntimeError("Stage 2 invariant failed: a trainable Zeva module received no gradients.")
        if prior_only:
            if any(parameter.grad is not None for parameter in policy.causal_action_projector.parameters()):
                raise RuntimeError("Prior-only Stage 2 gave gradients to the disabled context branch.")
            if policy.context_gate_logit.grad is not None or policy.prior_gate_logit.grad is not None:
                raise RuntimeError("Prior-only Stage 2 unexpectedly optimized a fixed scalar gate.")
            if action_expert_trainable and any(
                parameter.grad is not None
                for parameter in policy.residual_gate_router.parameters()
            ):
                raise RuntimeError("v11 unexpectedly optimized the fixed prior gate router.")
        elif policy.context_gate_logit.grad is None or policy.prior_gate_logit.grad is None:
            raise RuntimeError("Stage 2 invariant failed: residual gates received no gradients.")
    elif any(
        parameter.grad is not None
        for module in full_zeva_modules
        for parameter in module.parameters()
    ) or policy.context_gate_logit.grad is not None or policy.prior_gate_logit.grad is not None:
        raise RuntimeError("Matched PI baseline unexpectedly received Zeva gradients.")


def _retrieve(
    policy: nn.Module,
    processed: dict[str, torch.Tensor],
    phase_queries: torch.Tensor,
    task_ids: torch.Tensor,
    live_brief: torch.Tensor,
    live_brief_mask: torch.Tensor,
    live_retrieved: torch.Tensor,
    live_retrieved_mask: torch.Tensor,
    bank: RobotWinCausalBank,
    config,
    args: Args,
    *,
    training: bool,
) -> tuple[Any, torch.Tensor, torch.Tensor]:
    unwrapped = policy.module if hasattr(policy, "module") else policy
    with torch.no_grad():
        predicted_ids, scores = unwrapped.retrieve_task_ids_from_language(processed)
        phase_queries = phase_queries.to(bank.phase_key.device, dtype=bank.phase_key.dtype)
        if training and args.phase_noise_std > 0:
            phase_queries = F.normalize(phase_queries + torch.randn_like(phase_queries) * args.phase_noise_std, dim=-1)
        bank_batch = bank.retrieve(
            predicted_ids,
            phase_queries,
            brief_size=config.brief_memory_size,
            retrieval_top_k=config.retrieval_top_k,
        )
        bank_batch.brief_signals = torch.cat(
            [bank_batch.brief_signals, live_brief.to(bank.phase_key.device)], dim=1
        )
        bank_batch.retrieved_signals = torch.cat(
            [bank_batch.retrieved_signals, live_retrieved.to(bank.phase_key.device)], dim=1
        )
        bank_batch.brief_mask = torch.cat(
            [
                torch.ones(
                    bank_batch.brief_signals.shape[0],
                    bank_batch.brief_signals.shape[1] - live_brief_mask.shape[1],
                    dtype=torch.bool,
                    device=bank.phase_key.device,
                ),
                live_brief_mask.to(bank.phase_key.device),
            ],
            dim=1,
        )
        bank_batch.retrieved_mask = torch.cat(
            [
                torch.ones(
                    bank_batch.retrieved_signals.shape[0],
                    bank_batch.retrieved_signals.shape[1] - live_retrieved_mask.shape[1],
                    dtype=torch.bool,
                    device=bank.phase_key.device,
                ),
                live_retrieved_mask.to(bank.phase_key.device),
            ],
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


def _matched_baseline_flow(
    policy: nn.Module, processed: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    unwrapped = policy.module if hasattr(policy, "module") else policy
    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(processed["action"].device)
    with torch.no_grad():
        anchor_parameters = getattr(unwrapped, "_foundation_anchor_parameters", {})
        output = (
            unwrapped.foundation_anchor_forward(processed, reduction="none")
            if anchor_parameters
            else unwrapped.foundation(processed, reduction="none")
        )
        baseline = _foundation_loss(
            output, reduction="none"
        )
    torch.random.set_rng_state(cpu_state)
    torch.cuda.set_rng_state(cuda_state, processed["action"].device)
    return baseline, (cpu_state, cuda_state)


def _manifest(
    args: Args,
    handoff: RobotWinHandoff,
    bank: RobotWinCausalBank,
    runtime_versions: dict[str, str],
) -> dict[str, Any]:
    zeva_enabled = args.training_variant in {
        "zeva",
        "adapter",
        "prior_adapter",
        "prior_zeva",
    }
    prior_only = args.training_variant in {"prior_adapter", "prior_zeva"}
    frozen_foundation = args.training_variant in {"adapter", "prior_adapter"}
    zte_checkpoint = torch.load(args.zte_checkpoint, map_location="cpu")
    if zte_checkpoint.get("schema") != "zeva-robotwin-zte-stage1-checkpoint-v5":
        raise ValueError("Aligned Stage 2 requires a Stage 1 v5 checkpoint.")
    if zte_checkpoint["manifest"].get("causal_transition_horizon") != 15:
        raise ValueError("Aligned Stage 2 requires a causal transition horizon of 15.")
    if bank.manifest["stage1_checkpoint_sha256"] != _sha256(args.zte_checkpoint):
        raise ValueError("The causal bank was not exported from the selected Stage 1 checkpoint.")
    if bank.manifest["statistics_sha256"] != _sha256(handoff.statistics):
        raise ValueError("The causal bank does not match the selected PI0.5 normalization.")
    return {
        "schema": "zeva-robotwin-stage2-action-expert-manifest-v11",
        "handoff_root": str(handoff.root),
        "foundation_checkpoint": str(handoff.checkpoint),
        "foundation_identity": _checkpoint_identity(handoff.checkpoint),
        "initial_stage2_identity": _initial_stage2_identity(
            args.initial_stage2_checkpoint
        ),
        "anchor_stage2_identity": _initial_stage2_identity(
            args.anchor_stage2_checkpoint
        ),
        "anchor_foundation_identity": (
            _checkpoint_identity(args.anchor_foundation_checkpoint)
            if args.anchor_foundation_checkpoint is not None
            else None
        ),
        "goal_embedding_identity": _checkpoint_identity(
            args.goal_embedding_checkpoint or handoff.checkpoint
        ),
        "statistics_sha256": _sha256(handoff.statistics),
        "zte_checkpoint": str(Path(args.zte_checkpoint).resolve()),
        "zte_checkpoint_sha256": _sha256(args.zte_checkpoint),
        "zte_step": int(zte_checkpoint["step"]),
        "causal_bank": str(Path(args.causal_bank).resolve()),
        "causal_bank_sha256": _sha256(args.causal_bank),
        "causal_bank_manifest": bank.manifest,
        "live_queries": str(Path(args.live_queries).resolve()),
        "live_queries_sha256": _sha256(args.live_queries),
        "task_retrieval": str(Path(args.task_retrieval).resolve()),
        "task_retrieval_sha256": _sha256(args.task_retrieval),
        "dataset_adapter": str((Path(args.dataset_root) / "adapter.json").resolve()),
        "train_split": "train95",
        "validation_split": "validation5",
        "retrieval_protocol": "task-language inferred task + recurrent H15 ZTE phase; no oracle progress",
        "pi_image_contract": {
            "layout": "CHW",
            "dtype": "float32",
            "range": [0.0, 1.0],
            "source_uint8_transform": "x / 255",
            "foundation_visual_processor": "IDENTITY",
        },
        "runtime_versions": runtime_versions,
        "training_variant": args.training_variant,
        "training_mode": (
            "frozen_paligemma_action_expert_prior_residual_pbd_v11"
            if args.training_variant == "prior_zeva"
            else "frozen_pi05_zeva_prior_only_fixed_gate_v11"
            if args.training_variant == "prior_adapter"
            else "frozen_pi05_zeva_task_phase_routed_dual_residual_v10"
            if frozen_foundation
            else "frozen_paligemma_action_expert_dual_residual_accelerated_v8"
            if zeva_enabled
            else "frozen_paligemma_action_expert_only_matched_baseline_v1"
        ),
        "foundation_forward": {
            "attention": "released_joint_paligemma_action_expert_forward",
            "paligemma_requires_grad": False,
            "injection_before_vlm": False,
            "torch_compile": args.compile_model,
            "torch_compile_mode": args.compile_mode,
        },
        "video_decode": {
            "backend": args.video_backend,
            "persistent_decoder_cache": args.video_backend == "torchcodec",
        },
        "paired_baseline_preservation": {
            "enabled": zeva_enabled,
            "teacher": (
                "independent_untouched_foundation_path"
                if args.anchor_foundation_checkpoint is not None
                else (
                    "independent_frozen_base_action_path"
                    if args.anchor_stage2_checkpoint is not None
                    else ("current_student_residual_off" if zeva_enabled else "none")
                )
            ),
            "sampling": (
                "fixed_optimizer_step_interval" if zeva_enabled else "none"
            ),
            "interval": args.baseline_preserve_interval if zeva_enabled else None,
            "sampled_loss_scale": (
                args.baseline_preserve_interval if zeva_enabled else None
            ),
            "validation": (
                "every_batch_matched_teacher" if zeva_enabled else "self_flow"
            ),
            "hinge_scope": "per_example_no_cross_task_cancellation" if zeva_enabled else None,
            "positive_improvement_margin": (
                args.paired_improvement_margin if zeva_enabled else None
            ),
        },
        "causal_context_residual": {
            "source": "memory_context_encoder",
            "target": (
                "prior_conditioning_only"
                if prior_only
                else "noisy_action_embedding"
            ),
            "direct_injection_enabled": not prior_only,
            "broadcast_horizon": 50,
            "gate": (
                "disabled"
                if prior_only
                else "bounded_task_language_plus_recurrent_h15_phase_router"
            ),
        },
        "action_prior": {
            "distribution": "diagonal_gaussian",
            "parameterization": "mean_and_log_std",
            "log_std_range": [-5.0, 2.0],
            "loss": "negative_log_likelihood_sum_action_mean_executed_horizon",
            "supervision_horizon": args.prior_injection_horizon,
            "residual_source": "mean",
            "residual_target": "noisy_action_embedding",
            "residual_dropout_probability": args.prior_residual_dropout_probability,
            "injection_horizon": args.prior_injection_horizon,
            "scalar_gate": "fixed_0.5" if prior_only else "learned_sigmoid",
        },
        "frozen": [
            "pi05_paligemma_vision_tower",
            "pi05_paligemma_language_backbone",
            "zte",
            "causal_bank",
            "task_retrieval",
        ] + ([
            "task_token_projector",
            "memory_context_encoder",
            "action_prior",
            "causal_action_projector",
            "prior_action_projector",
            "context_gate_logit",
            "prior_gate_logit",
            "residual_gate_router",
        ] if args.training_variant == "baseline" else []) + ([
            "pi05_gemma_action_expert",
            "pi05_action_input_output_and_time_projections",
        ] if frozen_foundation else []) + (([
            "causal_action_projector",
            "context_gate_logit",
            "prior_gate_logit",
        ] + (["residual_gate_router"] if args.training_variant == "prior_zeva" else []))
        if prior_only else []),
        "trainable": ([] if frozen_foundation else [
            "pi05_gemma_action_expert",
            "pi05_action_input_output_and_time_projections",
        ]) + (([
            "task_token_projector",
            "memory_context_encoder",
            "action_prior",
            "prior_action_projector",
        ] + ([
            "residual_gate_router",
        ] if args.training_variant == "prior_adapter" else [])) if prior_only else ([
            "task_token_projector",
            "memory_context_encoder",
            "action_prior",
            "causal_action_projector",
            "prior_action_projector",
            "context_gate_logit",
            "prior_gate_logit",
            "residual_gate_router",
        ] if zeva_enabled else [])),
        "train_args": dataclasses.asdict(args),
        "source_sha256": {
            "trainer": _sha256(Path(__file__).resolve()),
            "robotwin_policy": _sha256(Path(inspect.getfile(RobotWinZevaPolicy)).resolve()),
        },
    }


@torch.no_grad()
def evaluate(
    policy: nn.Module,
    loader: DataLoader,
    bank: RobotWinCausalBank,
    preprocessor,
    config,
    args: Args,
    accelerator: Accelerator,
) -> dict[str, Any]:
    policy.eval()
    totals: dict[str, list[torch.Tensor]] = {
        "total": [],
        "flow": [],
        "prior": [],
        "prior_std": [],
        "baseline": [],
        "retrieval_accuracy": [],
        "paired_improvement": [],
        "paired_win_fraction": [],
        "paired_degradation": [],
    }
    paired_by_task: dict[int, dict[str, list[torch.Tensor]]] = {}
    for batch_index, raw_batch in enumerate(loader):
        if batch_index >= args.eval_batches:
            break
        task_ids = raw_batch.pop("zeva.task_id")
        phase_queries = raw_batch.pop("zeva.phase_query")
        live_brief = raw_batch.pop("zeva.live_brief")
        live_brief_mask = raw_batch.pop("zeva.live_brief_mask")
        live_retrieved = raw_batch.pop("zeva.live_retrieved")
        live_retrieved_mask = raw_batch.pop("zeva.live_retrieved_mask")
        processed = _preprocess_with_task_only_goal(policy, preprocessor, raw_batch)
        if args.training_variant in {"zeva", "adapter", "prior_adapter", "prior_zeva"}:
            bank_batch, confidence, retrieval_accuracy = _retrieve(
                policy,
                processed,
                phase_queries,
                task_ids,
                live_brief,
                live_brief_mask,
                live_retrieved,
                live_retrieved_mask,
                bank,
                config,
                args,
                training=False,
            )
            baseline_flow, foundation_rng_state = _matched_baseline_flow(policy, processed)
            losses = _losses(
                policy,
                processed,
                bank_batch,
                confidence,
                baseline_flow,
                foundation_rng_state,
                args.prior_loss_weight,
                args.preserve_loss_weight,
                args.gate_regularization_weight,
                args.prior_residual_dropout_probability,
                paired_improvement_margin=0.0,
                prior_supervision_horizon=args.prior_injection_horizon,
                training=False,
            )
        else:
            losses = _baseline_losses(policy, processed)
            retrieval_accuracy = losses["flow"].detach().new_full((), float("nan"))
        if args.training_variant in {"zeva", "adapter", "prior_adapter", "prior_zeva"}:
            gathered_task_ids = accelerator.gather_for_metrics(task_ids.detach()).cpu()
            gathered_flow = accelerator.gather_for_metrics(
                losses["flow_per_sample"].detach()
            ).cpu()
            gathered_baseline = accelerator.gather_for_metrics(
                losses["baseline_per_sample"].detach()
            ).cpu()
            for task_id in gathered_task_ids.unique().tolist():
                mask = gathered_task_ids == task_id
                slot = paired_by_task.setdefault(
                    int(task_id), {"flow": [], "baseline": []}
                )
                slot["flow"].append(gathered_flow[mask])
                slot["baseline"].append(gathered_baseline[mask])
        for name in (
            "total",
            "flow",
            "prior",
            "prior_std",
            "baseline",
            "paired_improvement",
            "paired_win_fraction",
            "paired_degradation",
        ):
            totals[name].append(accelerator.gather_for_metrics(losses[name].detach().reshape(1)))
        totals["retrieval_accuracy"].append(
            accelerator.gather_for_metrics(retrieval_accuracy.detach().reshape(1))
        )
    result = {
        name: float(torch.cat(values).mean()) if values else math.inf
        for name, values in totals.items()
    }
    per_task = {}
    for task_id, values in sorted(paired_by_task.items()):
        flow = torch.cat(values["flow"])
        baseline = torch.cat(values["baseline"])
        delta = baseline - flow
        per_task[bank.task_names[task_id]] = {
            "count": int(delta.numel()),
            "paired_improvement": float(delta.mean()),
            "paired_win_fraction": float((delta > 0).float().mean()),
            "paired_degradation": float(F.relu(-delta).mean()),
        }
    result["per_task_paired"] = per_task
    result["minimum_task_paired_improvement"] = (
        min(row["paired_improvement"] for row in per_task.values())
        if per_task
        else float("nan")
    )
    return result


def main(args: Args) -> None:
    runtime_versions = _validated_runtime_versions()
    if args.training_variant not in {
        "zeva",
        "adapter",
        "prior_adapter",
        "prior_zeva",
        "baseline",
    }:
        raise ValueError(
            "training_variant must be 'zeva', 'adapter', 'prior_adapter', 'prior_zeva', or 'baseline'."
        )
    zeva_enabled = args.training_variant in {
        "zeva",
        "adapter",
        "prior_adapter",
        "prior_zeva",
    }
    prior_only = args.training_variant in {"prior_adapter", "prior_zeva"}
    if zeva_enabled and args.foundation_checkpoint is not None:
        default_checkpoint = Path(args.handoff_root).resolve() / "checkpoint" / "pretrained_model"
        if Path(args.foundation_checkpoint).resolve() != default_checkpoint and args.goal_embedding_checkpoint is None:
            raise ValueError(
                "A changed PI foundation requires an explicit goal_embedding_checkpoint so the "
                "Stage 1 language coordinate cannot drift silently."
            )
    accelerator = Accelerator(
        # ``AcceleratedScheduler`` otherwise advances once per process when
        # ``split_batches=False``.  Our CLI ``warmup_steps`` is expressed in
        # global optimizer steps, so the scheduler must advance exactly once
        # after each synchronized optimizer update, not world_size times.
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[
            DistributedDataParallelKwargs(
                # The v8 trainable set contains only action-path and Zeva
                # parameters, all of which participate in every train step.
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
            )
        ]
    )
    torch.manual_seed(args.seed + accelerator.process_index)
    handoff = RobotWinHandoff.from_root(args.handoff_root)
    if args.foundation_checkpoint is not None:
        handoff = dataclasses.replace(
            handoff, checkpoint=Path(args.foundation_checkpoint).resolve()
        )
        handoff.validate()
    policy = RobotWinZevaPolicy.from_handoff(
        args.handoff_root,
        device=str(accelerator.device),
        foundation_checkpoint=args.foundation_checkpoint,
        goal_embedding_checkpoint=(
            args.goal_embedding_checkpoint if zeva_enabled else None
        ),
        stage2_checkpoint=args.initial_stage2_checkpoint,
        zte_checkpoint=args.zte_checkpoint,
        retrieval_checkpoint=args.task_retrieval,
        causal_bank=args.causal_bank,
    )
    if args.training_variant == "zeva":
        trainable = policy.configure_action_expert_finetune_stage2()
    elif args.training_variant == "adapter":
        trainable = policy.configure_adapter_stage2()
    elif args.training_variant == "prior_adapter":
        trainable = policy.configure_prior_only_adapter_stage2(prior_gate_probability=0.5)
    elif args.training_variant == "prior_zeva":
        trainable = policy.configure_action_expert_prior_finetune_stage2(
            prior_gate_probability=0.5,
            prior_injection_horizon=args.prior_injection_horizon,
        )
        if args.prior_injection_horizon != 15:
            raise ValueError("Formal v11 requires H15 prior injection and supervision.")
        if policy._direct_context_injection_enabled:  # noqa: SLF001
            raise RuntimeError("v11 invariant failed: direct context injection is enabled.")
        if any(parameter.requires_grad for parameter in policy.residual_gate_router.parameters()):
            raise RuntimeError("v11 invariant failed: prior gate router is trainable.")
        if any(torch.count_nonzero(parameter).item() for parameter in policy.residual_gate_router.parameters()):
            raise RuntimeError("v11 invariant failed: fixed prior gate router is not identity-scaled.")
        if torch.count_nonzero(policy.causal_action_projector.weight).item() or torch.count_nonzero(
            policy.causal_action_projector.bias
        ).item():
            raise RuntimeError("v11 invariant failed: direct context projector is nonzero.")
    else:
        trainable = policy.configure_action_expert_only_finetune()
    # Prior-only variants set the deployed 0.5 guidance inside their
    # configure_* method.  Do not overwrite that fixed value with the legacy
    # fresh-dual-residual initialization probability.
    if zeva_enabled and not prior_only and args.resume_checkpoint is None:
        policy.initialize_residual_gate_probability(
            args.initial_residual_gate_probability
        )
    if args.anchor_stage2_checkpoint is not None:
        if args.training_variant != "zeva":
            raise ValueError("An independent Stage 2 Base anchor is only valid for joint ZeVA training.")
        policy.load_foundation_anchor(args.anchor_stage2_checkpoint)
    if args.anchor_foundation_checkpoint is not None:
        if args.training_variant != "prior_zeva":
            raise ValueError(
                "An untouched foundation anchor is only valid for the prior_zeva variant."
            )
        if Path(args.anchor_foundation_checkpoint).resolve() != handoff.checkpoint.resolve():
            raise ValueError(
                "prior_zeva requires the immutable anchor to be the same untouched foundation "
                "checkpoint used to initialize the action expert."
            )
        policy.load_foundation_anchor(args.anchor_foundation_checkpoint)
    elif args.training_variant == "prior_zeva":
        raise ValueError("prior_zeva requires --anchor-foundation-checkpoint.")
    core = policy.foundation.model
    if hasattr(core, "gradient_checkpointing_disable"):
        core.gradient_checkpointing_disable()
    # Preserve the released joint PaliGemma/action-expert attention math. The
    # backbone weights are frozen, while joint-layer checkpointing bounds the
    # activation memory of the H50 action-expert fine-tune.
    core.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True
    bank = RobotWinCausalBank.load(args.causal_bank, device=accelerator.device)
    manifest = _manifest(args, handoff, bank, runtime_versions)
    selected_tasks = _load_task_subset(args.task_subset)
    if args.task_subset is None:
        manifest["task_scope"] = {
            "mode": "all_tasks",
            "task_names": list(bank.task_names),
        }
    else:
        manifest["task_scope"] = {
            "mode": "specialization_subset",
            "task_names": list(selected_tasks or ()),
            "manifest": str(Path(args.task_subset).resolve()),
            "manifest_sha256": _sha256(args.task_subset),
            "stage1_artifacts_remain_full_vocabulary": True,
        }
    manifest["world_size"] = accelerator.num_processes
    manifest["effective_global_batch_size"] = (
        args.batch_size * args.gradient_accumulation_steps * accelerator.num_processes
    )
    manifest["lr_scheduler"] = {
        "name": "linear_warmup_then_constant",
        "warmup_global_optimizer_steps": args.warmup_steps,
        "accelerate_step_scheduler_with_optimizer": False,
        "expected_scheduler_steps_per_global_optimizer_step": 1,
    }
    if manifest["effective_global_batch_size"] != 256:
        raise ValueError(
            "Formal Stage 2 requires global batch 256, got "
            f"{manifest['effective_global_batch_size']}."
        )
    if not (0 < args.action_expert_learning_rate < args.learning_rate):
        raise ValueError("Action-expert learning rate must be positive and smaller than the Zeva rate.")
    if args.prior_loss_weight <= 0:
        raise ValueError("Gaussian action-prior NLL weight must be positive.")
    if not math.isfinite(args.paired_improvement_margin) or args.paired_improvement_margin < 0:
        raise ValueError("Paired improvement margin must be finite and non-negative.")
    if (
        not math.isfinite(args.initial_residual_gate_probability)
        or not 0.0 < args.initial_residual_gate_probability < 1.0
    ):
        raise ValueError("Initial residual gate probability must be finite and in (0, 1).")
    if not 0.0 <= args.prior_residual_dropout_probability < 1.0:
        raise ValueError("Prior residual dropout probability must be in [0, 1).")
    if not 0 < args.prior_injection_horizon <= 50:
        raise ValueError("prior_injection_horizon must be in [1, 50].")
    if args.baseline_preserve_interval <= 0:
        raise ValueError("baseline_preserve_interval must be positive.")
    if args.compile_mode not in {"default", "reduce-overhead", "max-autotune"}:
        raise ValueError(f"Unsupported torch.compile mode: {args.compile_mode!r}.")
    preprocessor = policy.preprocessor
    config = policy.zeva_config
    adapter_manifest = Path(args.dataset_root) / "adapter.json"
    live_cache = torch.load(args.live_queries, map_location="cpu")
    if live_cache.get("zte_checkpoint_sha256") != _sha256(args.zte_checkpoint):
        raise ValueError("Live queries were not exported from the selected ZTE checkpoint.")
    if live_cache.get("statistics_sha256") != _sha256(handoff.statistics):
        raise ValueError("Live queries use different PI0.5 normalization statistics.")
    retrieval_checkpoint = torch.load(args.task_retrieval, map_location="cpu")
    if retrieval_checkpoint.get("causal_bank_sha256") != _sha256(args.causal_bank):
        raise ValueError("Task retrieval was not trained against the selected causal bank.")
    if float(retrieval_checkpoint.get("validation_accuracy", 0.0)) < 0.95:
        raise ValueError("Task-language retrieval validation accuracy is below the 95% Stage 1.5 gate.")
    train_dataset = RobotWinStage2Dataset(
        adapter_manifest,
        args.live_queries,
        subset="train",
        config=config,
        selected_tasks=selected_tasks,
        video_backend=args.video_backend,
        decoder_threads=args.decoder_threads,
    )
    validation_dataset = RobotWinStage2Dataset(
        adapter_manifest,
        args.live_queries,
        subset="validation",
        config=config,
        selected_tasks=selected_tasks,
        video_backend=args.video_backend,
        decoder_threads=args.decoder_threads,
    )
    if train_dataset.task_names != bank.task_names or validation_dataset.task_names != bank.task_names:
        raise ValueError("Dataset task ordering differs from the Stage 1 causal bank.")
    if selected_tasks is not None and (
        train_dataset.selected_task_names != selected_tasks
        or validation_dataset.selected_task_names != selected_tasks
    ):
        raise ValueError("The requested Stage 2 task subset was not applied exactly.")
    manifest["task_scope"]["train_decision_samples"] = len(train_dataset)
    manifest["task_scope"]["validation_decision_samples"] = len(validation_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    # Creating validation workers after CUDA/compile has initialized makes
    # forked workers inherit a large CUDA context.  This first appeared in the
    # frozen-foundation adapter run, but the matched action-expert v8 pair
    # showed the same 60--79 GB retention after its first validation.  Keep
    # validation in each rank process for every formal Stage 2 variant.  This
    # changes only input loading, not sample order or model/optimizer state.
    validation_workers = 0
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=validation_workers,
        pin_memory=True,
        persistent_workers=validation_workers > 0,
    )
    action_expert_parameters = [
        parameter for parameter in policy.foundation.parameters() if parameter.requires_grad
    ]
    action_expert_parameter_ids = {id(parameter) for parameter in action_expert_parameters}
    zeva_parameters = [
        parameter for parameter in trainable if id(parameter) not in action_expert_parameter_ids
    ]
    if args.training_variant not in {"adapter", "prior_adapter"} and not action_expert_parameters:
        raise RuntimeError("Stage 2 requires a non-empty action-expert optimizer group.")
    if zeva_enabled and not zeva_parameters:
        raise RuntimeError("Zeva Stage 2 requires a non-empty Zeva optimizer group.")
    if args.training_variant == "baseline" and zeva_parameters:
        raise RuntimeError("Matched PI baseline must not contain trainable Zeva parameters.")
    manifest["optimizer_groups"] = {}
    if action_expert_parameters:
        manifest["optimizer_groups"]["pi05_action_expert"] = {
            "learning_rate": args.action_expert_learning_rate,
            "parameter_count": sum(parameter.numel() for parameter in action_expert_parameters),
        }
    if zeva_enabled:
        manifest["optimizer_groups"]["zeva"] = {
            "learning_rate": args.learning_rate,
            "parameter_count": sum(parameter.numel() for parameter in zeva_parameters),
        }
    optimizer_groups = []
    if action_expert_parameters:
        optimizer_groups.append({
            "params": action_expert_parameters,
            "lr": args.action_expert_learning_rate,
            "name": "pi05_action_expert",
        })
    if zeva_enabled:
        optimizer_groups.append(
            {"params": zeva_parameters, "lr": args.learning_rate, "name": "zeva"}
        )
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        lr=args.learning_rate,
        betas=(0.9, args.adam_beta2),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, (step + 1) / max(1, args.warmup_steps)),
    )
    start_step = 0
    best_validation = math.inf
    if args.resume_checkpoint is not None:
        resume_dir = Path(args.resume_checkpoint)
        checkpoint = torch.load(resume_dir / "training_state.pt", map_location="cpu")
        expected_schema = {
            "zeva": "zeva-robotwin-stage2-action-expert-training-state-v8",
            "adapter": "zeva-robotwin-stage2-frozen-foundation-training-state-v2",
            "prior_adapter": "zeva-robotwin-stage2-prior-only-training-state-v1",
            "prior_zeva": "zeva-robotwin-stage2-action-expert-prior-training-state-v1",
            "baseline": "robotwin-pi05-action-expert-baseline-training-state-v1",
        }[args.training_variant]
        if checkpoint.get("schema") != expected_schema:
            raise ValueError(
                f"{args.training_variant} training cannot resume checkpoint schema "
                f"{checkpoint.get('schema')!r}; expected {expected_schema!r}."
            )
        if checkpoint["manifest"]["causal_bank_sha256"] != manifest["causal_bank_sha256"]:
            raise ValueError("Stage 2 resume checkpoint uses a different causal bank.")
        if checkpoint["manifest"].get("task_scope") != manifest.get("task_scope"):
            raise ValueError("Stage 2 resume checkpoint uses a different task scope.")
        if checkpoint["manifest"].get("foundation_identity") != manifest.get("foundation_identity"):
            raise ValueError("Stage 2 resume checkpoint uses a different PI foundation.")
        if checkpoint["manifest"].get("initial_stage2_identity") != manifest.get(
            "initial_stage2_identity"
        ):
            raise ValueError("Stage 2 resume checkpoint uses a different initial Stage 2 Base.")
        if checkpoint["manifest"].get("anchor_stage2_identity") != manifest.get(
            "anchor_stage2_identity"
        ):
            raise ValueError("Stage 2 resume checkpoint uses a different independent Base anchor.")
        if checkpoint["manifest"].get("anchor_foundation_identity") != manifest.get(
            "anchor_foundation_identity"
        ):
            raise ValueError(
                "Stage 2 resume checkpoint uses a different immutable foundation anchor."
            )
        load_model(policy.foundation, resume_dir / "model.safetensors", strict=True)
        if zeva_enabled:
            policy.load_adapter(resume_dir / "zeva_adapter.pth")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_step = int(checkpoint["step"])
        best_validation = float(checkpoint.get("best_validation", math.inf))

    if args.compile_model:
        # ZeVA's suffix-injection hook is already installed. Dynamo therefore
        # captures the exact released joint attention with both action-space
        # residuals, rather than compiling the unmodified foundation first.
        torch.set_float32_matmul_precision("high")
        core.forward = torch.compile(core.forward, mode=args.compile_mode)

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
        start_step,
        args.steps,
        disable=not accelerator.is_local_main_process,
        initial=start_step,
        total=args.steps,
    )
    for step in progress_bar:
        policy.train()
        unwrapped_policy = accelerator.unwrap_model(policy)
        unwrapped_policy.enforce_action_expert_stage2_mode()
        if args.training_variant in {"adapter", "prior_adapter"}:
            # ``policy.train()`` recursively flips the frozen foundation back
            # to train mode.  Restore deterministic deployment behavior for
            # every real layer while retaining gradient checkpointing through
            # the frozen action expert into the residual inputs.
            unwrapped_policy.enforce_frozen_foundation_checkpointing_mode()
        optimizer.zero_grad(set_to_none=True)
        accumulated_losses: dict[str, list[torch.Tensor]] = {}
        retrieval_accuracies: list[torch.Tensor] = []
        sample_baseline = zeva_enabled and step % args.baseline_preserve_interval == 0
        for micro_step in range(args.gradient_accumulation_steps):
            try:
                raw_batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                raw_batch = next(iterator)
            task_ids = raw_batch.pop("zeva.task_id")
            phase_queries = raw_batch.pop("zeva.phase_query")
            live_brief = raw_batch.pop("zeva.live_brief")
            live_brief_mask = raw_batch.pop("zeva.live_brief_mask")
            live_retrieved = raw_batch.pop("zeva.live_retrieved")
            live_retrieved_mask = raw_batch.pop("zeva.live_retrieved_mask")
            processed = _preprocess_with_task_only_goal(policy, preprocessor, raw_batch)
            if zeva_enabled:
                bank_batch, confidence, retrieval_accuracy = _retrieve(
                    policy,
                    processed,
                    phase_queries,
                    task_ids,
                    live_brief,
                    live_brief_mask,
                    live_retrieved,
                    live_retrieved_mask,
                    bank,
                    config,
                    args,
                    training=True,
                )
            else:
                retrieval_accuracy = processed["action"].detach().new_full((), float("nan"))
            # Skip the redundant DDP all-reduce for all but the final micro-batch.
            sync_context = (
                policy.no_sync()
                if micro_step + 1 < args.gradient_accumulation_steps and hasattr(policy, "no_sync")
                else nullcontext()
            )
            with sync_context:
                if args.training_variant == "baseline":
                    micro_losses = _baseline_losses(policy, processed)
                else:
                    if sample_baseline:
                        baseline_flow, foundation_rng_state = _matched_baseline_flow(policy, processed)
                    else:
                        baseline_flow, foundation_rng_state = None, None
                    micro_losses = _losses(
                        policy,
                        processed,
                        bank_batch,
                        confidence,
                        baseline_flow,
                        foundation_rng_state,
                        args.prior_loss_weight,
                        args.preserve_loss_weight,
                        args.gate_regularization_weight,
                        args.prior_residual_dropout_probability,
                        paired_improvement_margin=args.paired_improvement_margin,
                        preserve_scale=(
                            float(args.baseline_preserve_interval) if sample_baseline else 1.0
                        ),
                        prior_supervision_horizon=args.prior_injection_horizon,
                        training=True,
                    )
                accelerator.backward(micro_losses["total"] / args.gradient_accumulation_steps)
            if step == start_step and micro_step == 0:
                _assert_action_expert_finetune_gradients(
                    accelerator.unwrap_model(policy),
                    include_zeva=zeva_enabled,
                    action_expert_trainable=args.training_variant
                    not in {"adapter", "prior_adapter"},
                    prior_only=args.training_variant in {"prior_adapter", "prior_zeva"},
                )
            for name, value in micro_losses.items():
                accumulated_losses.setdefault(name, []).append(value.detach())
            retrieval_accuracies.append(retrieval_accuracy.detach())
        losses = {name: torch.stack(values).mean() for name, values in accumulated_losses.items()}
        retrieval_accuracy = torch.stack(retrieval_accuracies).mean()
        accelerator.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()
        completed = step + 1
        scheduler_last_epoch = int(scheduler.state_dict()["last_epoch"])
        if scheduler_last_epoch != completed:
            raise RuntimeError(
                "LR scheduler/global-step drift: "
                f"last_epoch={scheduler_last_epoch}, completed={completed}. "
                "The scheduler must advance exactly once per global optimizer step."
            )
        if completed % args.log_freq == 0:
            baseline_sampled = bool(float(losses["baseline_sampled"]))
            progress_bar.set_postfix(
                loss=f"{float(losses['total'].detach()):.4f}",
                flow=f"{float(losses['flow'].detach()):.4f}",
                base=(f"{float(losses['baseline']):.4f}" if baseline_sampled else "skip"),
                prior=f"{float(losses['prior'].detach()):.4f}",
                pstd=f"{float(losses['prior_std']):.3f}",
                pkeep=f"{float(losses['prior_residual_keep']):.2f}",
                task=f"{float(retrieval_accuracy):.3f}",
                bsample=f"{int(baseline_sampled)}",
            )
        if args.save_checkpoints and (completed % args.save_freq == 0 or completed == args.steps):
            validation = evaluate(policy, validation_loader, bank, preprocessor, config, args, accelerator)
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped = accelerator.unwrap_model(policy)
                step_dir = save_dir / f"{completed:06d}"
                step_dir.mkdir(parents=True, exist_ok=True)
                save_model(unwrapped.foundation, step_dir / "model.safetensors")
                if zeva_enabled:
                    torch.save(unwrapped.stage2_state_dict(manifest), step_dir / "zeva_adapter.pth")
                torch.save(
                    {
                        "schema": {
                        "zeva": "zeva-robotwin-stage2-action-expert-training-state-v8",
                        "adapter": "zeva-robotwin-stage2-frozen-foundation-training-state-v2",
                        "prior_adapter": "zeva-robotwin-stage2-prior-only-training-state-v1",
                        "prior_zeva": "zeva-robotwin-stage2-action-expert-prior-training-state-v1",
                        "baseline": "robotwin-pi05-action-expert-baseline-training-state-v1",
                        }[args.training_variant],
                        "step": completed,
                        "validation": validation,
                        "validation_loss": validation["total"],
                        "best_validation": min(best_validation, validation["total"]),
                        "manifest": manifest,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                    },
                    step_dir / "training_state.pt",
                )
                (save_dir / "latest.json").write_text(
                    json.dumps({"step": completed, "checkpoint": str(step_dir)}, indent=2) + "\n"
                )
                if validation["total"] <= best_validation:
                    best_validation = validation["total"]
                    (save_dir / "best.json").write_text(
                        json.dumps(
                            {"step": completed, "validation": validation, "checkpoint": str(step_dir)},
                            indent=2,
                        )
                        + "\n"
                    )
                (save_dir / "last_metrics.json").write_text(
                    json.dumps(
                        {
                            "step": completed,
                            "train_loss": float(losses["total"]),
                            "train_flow": float(losses["flow"]),
                            "train_baseline_flow": (
                                float(losses["baseline"])
                                if bool(float(losses["baseline_sampled"]))
                                else None
                            ),
                            "train_baseline_sampled": bool(float(losses["baseline_sampled"])),
                            "train_prior_nll": float(losses["prior"]),
                            "train_prior_std": float(losses["prior_std"]),
                            "train_prior_residual_keep": float(losses["prior_residual_keep"]),
                            "validation": validation,
                        },
                        indent=2,
                    )
                    + "\n"
                )
            accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main(tyro.cli(Args))
