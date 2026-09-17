"""Formal Stage 2: tune the PI0.5 action expert and Zeva with PaliGemma frozen."""

from __future__ import annotations

from contextlib import contextmanager
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
from openpi.zeva.stage1_checkpoint import LEGACY_SCHEMAS
from openpi.zeva.stage1_checkpoint import V2_SCHEMA
from openpi.zeva.stage1_checkpoint import stage1_transition_horizon
from openpi.zeva.robotwin_policy import RobotWinZevaPolicy
from openpi.zeva.robotwin_policy import gaussian_action_prior_nll
from openpi.zeva.robotwin_policy import stage1_artifact_schema
from openpi.zeva.robotwin_policy import validate_stage1_v2_artifact_status

try:
    from scripts.train_robotwin_zte import FFmpegRoboTwinDataset
    from scripts.train_robotwin_zte import TorchCodecRoboTwinDataset
except ModuleNotFoundError:  # Direct `python scripts/...py` execution.
    from train_robotwin_zte import FFmpegRoboTwinDataset
    from train_robotwin_zte import TorchCodecRoboTwinDataset


UNTOUCHED_BEST_V1_MODEL_SHA256 = (
    "7d3e945c1d17eae24b9f374d818ee43415e6a789da5587397403ea26a91e0abe"
)

# RoboTwin emits PI0.5 chunks of H50 and executes only the first H15 before
# the recurrent replan.  The opt-in routed ZeVA objective below may use this
# executed prefix; the PI0.5 output and ordinary Base loss remain H50.
ROBOTWIN_EXECUTED_HORIZON = 15

LIVE_QUERY_SCHEMAS = frozenset({
    "zeva-robotwin-live-queries-h15-v1",
    "zeva-robotwin-live-queries-h15-v2",
})


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
    # Optional untouched-PI action cache.  Required by the formal v13 path so
    # the frozen diffusion model is evaluated once per observation, not once
    # per training epoch.
    base_action_cache: str | None = None
    # Only for a content-verified host replica of the cache's source dataset.
    # The reports must agree on every content hash and normalized adapter key.
    dataset_identity_source_report: str | None = None
    dataset_identity_replica_report: str | None = None
    # A cached-output run may train on a host with only the action/Joint/stats
    # indices, provided they are rehashed against a full source replica proof.
    cache_only_dataset_replica: bool = False
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
    # Experimental auxiliary-gradient routing. Flow conditioning stays attached;
    # Gaussian NLL updates its head but not shared task/context features.
    prior_nll_detach_context: bool = False
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
    # Optional gradient routing experiment for the standard joint ZeVA path.
    # The residual-on forward updates only ZeVA; a matched residual-off forward
    # updates only the current PI0.5 action expert.  It is deliberately opt-in
    # because it changes the objective, while preserving the normal path when
    # false.
    decouple_action_expert_gradient: bool = False
    # Only with gradient routing: optimize the ZeVA branch's flow on the
    # deployed H15 prefix.  The current-student Base branch still trains on
    # ordinary H50 flow, and inference still emits H50 chunks.
    zeva_h15_flow_objective: bool = False
    # A v11 prior is applied only to the executed prefix.  Keep 50 for legacy
    # variants; the v11 launcher sets 15 to match RoboTwin's H15 replan loop.
    prior_injection_horizon: int = 50
    # Fresh residual branches may use a stronger gate while retaining exact
    # step-zero Base equivalence because both injection projectors start at 0.
    initial_residual_gate_probability: float = 0.01
    # Direct post-diffusion residual corrector (v14).  The residual is
    # bounded in normalized EEF16 coordinates and only applied to H15.
    residual_bound: float = 0.25
    residual_regression_weight: float = 0.25
    residual_trust_region_weight: float = 0.01
    residual_trust_region_radius: float = 0.10
    retrieval_confidence_floor: float = 0.2
    save_freq: int = 500
    save_checkpoints: bool = True
    eval_batches: int = 32
    # Optional read-only validation telemetry.  It is intentionally excluded
    # from the normal path so existing validation numbers and checkpoints are
    # bit-for-bit unaffected unless a caller explicitly opts in.
    validation_diagnostics: bool = False
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
        "model_sha256": _sha256(model),
        "model_size": model.stat().st_size,
        "training_state_size": training_state.stat().st_size,
        "source_manifest": str(run_manifest),
        "source_manifest_sha256": _sha256(run_manifest),
    }


def _verify_cached_dataset_replica(
    *,
    cache_adapter_sha256: str,
    current_adapter: Path,
    source_report: str | None,
    replica_report: str | None,
    cache_only: bool = False,
) -> dict[str, str] | None:
    """Accept only a previously fully hashed, semantic-equivalent copy."""
    current_sha = _sha256(current_adapter)
    if current_sha == cache_adapter_sha256:
        if source_report is not None or replica_report is not None:
            raise ValueError("Replica reports are unnecessary for an exact adapter match.")
        return None
    if not source_report or not replica_report:
        raise ValueError("Base cache adapter differs; both dataset identity reports are required.")
    source_path = Path(source_report).resolve()
    replica_path = Path(replica_report).resolve()
    source = json.loads(source_path.read_text())
    replica = json.loads(replica_path.read_text())
    if source.get("schema") != "robotwin-dataset-content-identity-v1" or replica.get(
        "schema"
    ) != "robotwin-dataset-content-identity-v1":
        raise ValueError("Unsupported dataset identity report schema.")
    if source.get("adapter_sha256") != cache_adapter_sha256:
        raise ValueError("Source identity report does not match the Base action cache.")
    if replica.get("adapter_sha256") != current_sha:
        raise ValueError("Replica identity report does not match the training adapter.")
    if Path(replica.get("dataset_root", "")).resolve() != current_adapter.parent.resolve():
        raise ValueError("Replica identity report names another dataset root.")
    if source.get("components") != replica.get("components") or source.get(
        "semantic_adapter"
    ) != replica.get("semantic_adapter"):
        raise ValueError("Dataset replicas differ in content or normalized adapter semantics.")
    if set(source["components"]) != {"source", "eef-index", "joint14-index", "stats"}:
        raise ValueError("Dataset identity report lacks a required content component.")
    proof = {
        "source_report": str(source_path),
        "source_report_sha256": _sha256(source_path),
        "replica_report": str(replica_path),
        "replica_report_sha256": _sha256(replica_path),
        "replica_adapter_sha256": current_sha,
    }
    if cache_only:
        try:
            from scripts.audit_robotwin_dataset_identity import component  # noqa: PLC0415
        except ModuleNotFoundError:
            from audit_robotwin_dataset_identity import component  # noqa: PLC0415

        adapter = json.loads(current_adapter.read_text())
        for field, name in (
            ("eef_cache_root", "eef-index"),
            ("joint_cache_root", "joint14-index"),
            ("stats_path", "stats"),
        ):
            actual = component(Path(adapter[field]))
            if actual != replica["components"][name]:
                raise ValueError(f"Runtime cache-only dataset component {name} differs from proof.")
        proof["runtime_cache_only_components_verified"] = "eef-index,joint14-index,stats"
        proof["runtime_source_video_used"] = "false"
    return proof


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
        base_action_cache: str | Path | None = None,
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
        if cache.get("schema") not in LIVE_QUERY_SCHEMAS:
            raise ValueError("Stage 2A requires a supported deployment-recurrent H15 live-query cache.")
        if stage1_artifact_schema(cache) == V2_SCHEMA:
            validate_stage1_v2_artifact_status(cache, artifact_name="live-query cache")
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
        self._original_sample_indices = list(range(len(self._samples)))
        self.cached_base_actions: torch.Tensor | None = None
        if base_action_cache is not None:
            payload = torch.load(base_action_cache, map_location="cpu", weights_only=False)
            if payload.get("schema") not in {
                "zeva-robotwin-untouched-base-action-cache-v1",
                "zeva-robotwin-stage2-base-action-cache-v2",
            }:
                raise ValueError("Output correction requires a supported frozen Base action cache.")
            cached = payload["splits"][subset]
            indices = torch.as_tensor(cached["sample_indices"], dtype=torch.long)
            actions = torch.as_tensor(cached["base_actions"], dtype=torch.float32)
            if actions.shape != (len(indices), ROBOTWIN_ACTION_HORIZON, ROBOTWIN_ACTION_DIM):
                raise ValueError(f"Invalid {subset} Base cache shape: {tuple(actions.shape)}.")
            if len(indices) != len(indices.unique()) or bool((indices < 0).any()) or bool(
                (indices >= len(self._samples)).any()
            ):
                raise ValueError(f"Invalid or duplicate {subset} Base cache sample indices.")
            self._samples = [self._samples[int(index)] for index in indices]
            self._original_sample_indices = indices.tolist()
            self.cached_base_actions = actions

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record_index, decision_index, frame = self._samples[index]
        record = self.dataset._records[record_index]  # noqa: SLF001
        episode_start = int(self.dataset._cumulative[record_index])  # noqa: SLF001
        result = dict(self.dataset[episode_start + frame])
        if self.cached_base_actions is None:
            images = self.source.read_images(record, [frame])
            for key in ROBOTWIN_CAMERA_KEYS:
                # FFmpeg returns CHW uint8 [0,255], while the frozen PI0.5
                # preprocessor declares VISUAL=IDENTITY.  Normalize explicitly at
                # the dataset boundary so training exactly matches deployment.
                result[key] = prepare_robotwin_pi_image(images[key][0], name=key)
        else:
            result["zeva.base_actions"] = self.cached_base_actions[index]
        result["zeva.task_id"] = torch.tensor(self._task_ids[record["key"][1]], dtype=torch.long)
        result["zeva.sample_index"] = torch.tensor(
            self._original_sample_indices[index], dtype=torch.long
        )
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


def _diagnostic_rng_state(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Capture the rank-local RNG streams without requiring CUDA in tests."""
    cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    return torch.random.get_rng_state(), cuda_state


def _restore_diagnostic_rng_state(
    state: tuple[torch.Tensor, torch.Tensor | None], device: torch.device
) -> None:
    torch.random.set_rng_state(state[0])
    if state[1] is not None:
        torch.cuda.set_rng_state(state[1], device)


class _ConnectedRawFlowCapture:
    """Capture the one connected PI05 core loss already used by a policy call.

    The released PI05Policy invokes ``model.forward`` directly, so a PyTorch
    module forward hook does not fire.  Wrap the callable *after* optional
    torch.compile and before DDP instead.  The wrapper never replays the model,
    never changes its return value, and holds the graph only until ``take``.
    """

    def __init__(self, core: nn.Module):
        original_forward = core.forward
        self._enabled = False
        self._raw: torch.Tensor | None = None
        self.h50_equivalence_verified = False

        def capture_forward(*args, **kwargs):
            output = original_forward(*args, **kwargs)
            if self._enabled:
                if self._raw is not None:
                    raise RuntimeError("More than one PI05 core forward occurred during H15 capture.")
                raw = output[0] if isinstance(output, tuple) else output
                if not torch.is_tensor(raw) or raw.ndim != 3:
                    raise RuntimeError("PI05 core did not return connected [B,H,D] flow errors.")
                self._raw = raw
            return output

        core.forward = capture_forward

    def begin(self) -> None:
        if self._enabled or self._raw is not None:
            raise RuntimeError("A previous PI05 raw-flow capture was not consumed.")
        self._enabled = True

    def take(self) -> torch.Tensor:
        raw = self._raw
        self._raw = None
        self._enabled = False
        if raw is None:
            raise RuntimeError("PI05Policy did not invoke its core during H15 capture.")
        return raw

    def abort(self) -> None:
        self._raw = None
        self._enabled = False


def _connected_executed_flow_per_sample(
    raw: torch.Tensor,
    *,
    expected_batch_size: int,
    horizon: int = ROBOTWIN_EXECUTED_HORIZON,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce the deployed action prefix while preserving autograd edges."""
    if raw.ndim != 3 or raw.shape[0] != expected_batch_size:
        raise ValueError(f"Expected PI05 raw [B,H,D] with B={expected_batch_size}, got {tuple(raw.shape)}.")
    if raw.shape[1] < horizon or raw.shape[2] < ROBOTWIN_ACTION_DIM:
        raise ValueError(
            "PI05 raw flow is smaller than RoboTwin's executed H15/EEF16 contract: "
            f"{tuple(raw.shape)}."
        )
    action_flow = raw[:, :, :ROBOTWIN_ACTION_DIM]
    return (
        action_flow[:, :horizon].mean(dim=(1, 2)),
        action_flow.mean(dim=(1, 2)),
    )


class _ActionExpertGradientRouter:
    """Mask action-expert gradients on the residual-on branch.

    Hooks are installed on the unwrapped parameters before Accelerate wraps
    the policy.  This ordering makes the zero returned by the hook visible to
    DDP's reducer while retaining the parameter in the residual-on graph.  A
    residual-off backward switches the hook to pass-through, so accumulated
    action-expert gradients are exactly those from that current-student flow
    forward.  The manager is intentionally tiny and DDP-free; the training
    loop owns synchronization via its existing ``no_sync`` context.
    """

    _VALID_PHASES = frozenset({"residual_on", "residual_off"})

    def __init__(self, parameters: list[nn.Parameter] | tuple[nn.Parameter, ...]):
        self.parameters = tuple(parameter for parameter in parameters if parameter.requires_grad)
        if not self.parameters:
            raise ValueError("Action-expert gradient routing requires trainable parameters.")
        self._phase = "residual_off"
        self._audit_enabled = False
        self._audit: dict[str, Any] = {}
        self._handles = [
            parameter.register_hook(self._make_hook(parameter))
            for parameter in self.parameters
        ]

    @property
    def phase(self) -> str:
        return self._phase

    def set_phase(self, phase: str) -> None:
        phase = str(phase)
        if phase not in self._VALID_PHASES:
            raise ValueError(f"Unknown action-expert gradient-routing phase: {phase!r}.")
        self._phase = phase

    def begin_first_step_audit(self) -> None:
        self._audit_enabled = True
        self._audit = {
            "expected_parameters": len(self.parameters),
            "on_hook_parameters": set(),
            "off_hook_parameters": set(),
            "on_input_nonzero": False,
            "on_output_nonzero": False,
            "off_input_nonzero": False,
        }

    def first_step_audit(self) -> dict[str, Any]:
        audit = dict(self._audit)
        for name in ("on_hook_parameters", "off_hook_parameters"):
            audit[name] = len(audit.get(name, ()))
        return audit

    def end_first_step_audit(self) -> None:
        self._audit_enabled = False

    def assert_first_step_audit(self) -> None:
        audit = self.first_step_audit()
        expected = audit.get("expected_parameters", 0)
        if audit.get("on_hook_parameters") != expected:
            raise RuntimeError(
                "Gradient-routing invariant failed: residual-on did not mark every "
                f"action-expert parameter used ({audit.get('on_hook_parameters')}/{expected})."
            )
        if audit.get("off_hook_parameters") != expected:
            raise RuntimeError(
                "Gradient-routing invariant failed: residual-off did not produce a "
                f"gradient for every action-expert parameter ({audit.get('off_hook_parameters')}/{expected})."
            )
        if audit.get("on_output_nonzero"):
            raise RuntimeError(
                "Gradient-routing invariant failed: residual-on action-expert gradient "
                "was not masked to zero."
            )
        if not audit.get("on_input_nonzero"):
            raise RuntimeError(
                "Gradient-routing invariant failed: residual-on did not produce an "
                "action-expert gradient to mask."
            )
        if not audit.get("off_input_nonzero"):
            raise RuntimeError(
                "Gradient-routing invariant failed: residual-off action-expert gradient "
                "was entirely zero."
            )

    def _make_hook(self, parameter: nn.Parameter):
        parameter_id = id(parameter)

        def route(gradient: torch.Tensor) -> torch.Tensor:
            if self._audit_enabled:
                if self._phase == "residual_on":
                    self._audit["on_hook_parameters"].add(parameter_id)
                    self._audit["on_input_nonzero"] |= bool(torch.any(gradient.detach() != 0).item())
                else:
                    self._audit["off_hook_parameters"].add(parameter_id)
                    self._audit["off_input_nonzero"] |= bool(torch.any(gradient.detach() != 0).item())
            if self._phase == "residual_on":
                # Returning a fresh zero preserves DDP's used-parameter mark
                # and prevents any residual-on contribution from accumulating.
                result = torch.zeros_like(gradient)
                if self._audit_enabled:
                    self._audit["on_output_nonzero"] |= bool(torch.any(result != 0).item())
                return result
            return gradient

        return route

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._audit_enabled = False


def _validate_action_expert_gradient_routing_contract(
    training_variant: str,
    enabled: bool,
    prior_nll_detach_context: bool,
) -> None:
    """Reject routing combinations whose gradients have ambiguous ownership."""
    if not enabled:
        return
    if training_variant != "zeva":
        raise ValueError(
            "Action-expert gradient routing is restricted to the standard zeva variant."
        )
    if not prior_nll_detach_context:
        raise ValueError(
            "Action-expert gradient routing requires --prior-nll-detach-context so "
            "shared task/context features are not updated by the auxiliary NLL path."
        )


def _zero_parameter_gradient_link(
    parameters: list[nn.Parameter] | tuple[nn.Parameter, ...],
    reference: torch.Tensor,
) -> torch.Tensor:
    """Attach a zero-valued graph edge to every parameter in an off forward.

    DDP is configured with ``find_unused_parameters=False``.  The current
    foundation-only forward does not reference ZeVA parameters, so this
    explicit zero edge keeps every ZeVA reducer bucket marked as used without
    changing its gradient or the off-flow value.  One scalar per parameter is
    enough; reducing an entire large tensor would add needless work.
    """
    links = [
        parameter.reshape(-1)[0] * reference.new_zeros(())
        for parameter in parameters
        if parameter.requires_grad and parameter.numel()
    ]
    if not links:
        return reference.new_zeros(())
    return torch.stack(links).sum()


def _clip_stage2_gradients(
    accelerator: Accelerator,
    trainable: list[nn.Parameter] | tuple[nn.Parameter, ...],
    *,
    action_expert_parameters: list[nn.Parameter] | tuple[nn.Parameter, ...],
    zeva_parameters: list[nn.Parameter] | tuple[nn.Parameter, ...],
    decouple_action_expert_gradient: bool,
    max_norm: float = 1.0,
) -> None:
    """Clip the Stage 2 optimizer gradients using the configured contract.

    The ordinary Base action-expert path clips only its action-expert
    parameters because that is the complete ``trainable`` set for the Base
    optimizer.  When gradient routing is enabled, preserve that same action-
    expert-only norm while clipping the ZeVA adapter group independently.  A
    disabled route deliberately retains the historical one-call clip over
    the complete trainable set.
    """
    if decouple_action_expert_gradient:
        accelerator.clip_grad_norm_(action_expert_parameters, max_norm)
        accelerator.clip_grad_norm_(zeva_parameters, max_norm)
    else:
        accelerator.clip_grad_norm_(trainable, max_norm)


def _diagnostic_action_valid_mask(
    processed: dict[str, torch.Tensor],
    *,
    batch_size: int,
    horizon: int,
    device: torch.device,
) -> tuple[torch.Tensor, str]:
    """Return the action-step validity mask used by an opt-in H15 report.

    The normal PI0.5 forward path owns image/action preprocessing.  This
    helper only interprets an explicitly supplied action padding/mask field;
    when no such field exists, all fixed H50 action slots are valid exactly as
    in the existing flow loss.  It never treats zero-valued actions as
    padding, which would silently change the training contract.
    """
    padding_keys = (
        "action_is_pad",
        "action.is_pad",
        "observation.action_is_pad",
    )
    valid_keys = (
        "action_mask",
        "action.mask",
        "observation.action_mask",
    )
    source = "implicit_all_action_steps_valid"
    value: torch.Tensor | None = None
    is_padding = False
    for key in padding_keys:
        if key in processed:
            value = processed[key]
            is_padding = True
            source = key
            break
    if value is None:
        for key in valid_keys:
            if key in processed:
                value = processed[key]
                source = key
                break

    if value is None:
        return torch.ones((batch_size, horizon), dtype=torch.bool, device=device), source

    mask = torch.as_tensor(value, device=device).bool()
    if mask.ndim == 1:
        if batch_size != 1:
            raise ValueError(
                f"Action validity mask {source!r} must have a batch dimension, got {tuple(mask.shape)}."
            )
        mask = mask.unsqueeze(0)
    if mask.ndim == 3:
        # LeRobot's action padding is normally repeated over dimensions.  A
        # step is valid if at least one action dimension is not padded; for an
        # ordinary validity mask, any valid dimension keeps the step valid.
        mask = mask.all(dim=-1) if is_padding else mask.any(dim=-1)
    if mask.ndim != 2 or mask.shape[0] != batch_size:
        raise ValueError(
            f"Action validity mask {source!r} must be [B,H] or [B,H,D], got {tuple(mask.shape)}."
        )
    result = torch.zeros((batch_size, horizon), dtype=torch.bool, device=device)
    width = min(horizon, mask.shape[1])
    if width:
        result[:, :width] = ~mask[:, :width] if is_padding else mask[:, :width]
    return result, source


def _masked_action_flow_per_sample(
    action_losses: torch.Tensor,
    *,
    horizon: int,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce raw PI0.5 per-action flow errors over a valid H-prefix.

    ``action_losses`` is expected to be the unreduced ``[B,H,D]`` tensor from
    ``PI05Pytorch.forward``.  A ``[B,H]`` tensor is accepted for small test
    doubles.  The returned pair is ``(per_sample_mean, valid_step_count)``;
    samples with no valid action slots receive NaN and are excluded by report
    aggregation.  No action values are inspected to infer padding.
    """
    if action_losses.ndim == 2:
        action_losses = action_losses.unsqueeze(-1)
    if action_losses.ndim != 3:
        raise ValueError(
            "H-prefix flow diagnostics require unreduced [B,H,D] losses, got "
            f"{tuple(action_losses.shape)}."
        )
    if horizon <= 0:
        raise ValueError(f"Diagnostic horizon must be positive, got {horizon}.")
    batch_size, available_horizon, action_dim = action_losses.shape
    width = min(horizon, available_horizon)
    values = action_losses[:, :width].float()
    if valid_mask is None:
        valid = torch.ones(
            (batch_size, width), dtype=torch.bool, device=action_losses.device
        )
    else:
        valid = torch.as_tensor(valid_mask, device=action_losses.device).bool()
        if valid.shape != (batch_size, horizon):
            raise ValueError(
                "Diagnostic validity mask must match [B, requested_horizon], got "
                f"{tuple(valid.shape)} versus {(batch_size, horizon)}."
            )
        valid = valid[:, :width]
    valid_steps = valid.sum(dim=1)
    denominator = valid_steps * action_dim
    weighted = values * valid.unsqueeze(-1).to(values.dtype)
    per_sample = weighted.sum(dim=(1, 2)) / denominator.clamp_min(1).to(values.dtype)
    per_sample = per_sample.masked_fill(valid_steps == 0, float("nan"))
    return per_sample, valid_steps


def _diagnostic_value_summary(
    values: torch.Tensor,
    *,
    valid_steps: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Summarize finite per-sample diagnostics without weighting padding."""
    values = torch.as_tensor(values, dtype=torch.float32).reshape(-1).cpu()
    finite = torch.isfinite(values)
    result: dict[str, Any] = {
        "available": bool(finite.any()),
        "valid_examples": int(finite.sum()),
    }
    if finite.any():
        selected = values[finite]
        result.update(
            {
                "mean": float(selected.mean()),
                "min": float(selected.min()),
                "max": float(selected.max()),
            }
        )
    else:
        result["reason"] = "no finite examples"
    if valid_steps is not None:
        steps = torch.as_tensor(valid_steps, dtype=torch.float32).reshape(-1).cpu()
        if steps.numel() == values.numel():
            result["valid_steps"] = int(steps[finite].sum()) if finite.any() else 0
    return result


def _diagnostic_pair_summary(
    student: torch.Tensor,
    teacher: torch.Tensor,
    *,
    valid_steps: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Summarize a paired H15 comparison with an explicit teacher label."""
    student = torch.as_tensor(student, dtype=torch.float32).reshape(-1).cpu()
    teacher = torch.as_tensor(teacher, dtype=torch.float32).reshape(-1).cpu()
    finite = torch.isfinite(student) & torch.isfinite(teacher)
    if not bool(finite.any()):
        result = {"available": False, "valid_examples": 0, "reason": "no finite pairs"}
    else:
        delta = teacher[finite] - student[finite]
        result = {
            "available": True,
            "valid_examples": int(finite.sum()),
            "student_flow": float(student[finite].mean()),
            "teacher_flow": float(teacher[finite].mean()),
            "improvement": float(delta.mean()),
            "degradation": float(torch.relu(-delta).mean()),
            "win_fraction": float((delta > 0).float().mean()),
        }
    if valid_steps is not None:
        steps = torch.as_tensor(valid_steps, dtype=torch.float32).reshape(-1).cpu()
        if steps.numel() == student.numel():
            result["valid_steps"] = int(steps[finite].sum()) if finite.any() else 0
    return result


_DIAGNOSTIC_ACTIVE_FIELDS = (
    "_active_causal_context",
    "_active_action_prior",
    "_active_injection_confidence",
    "_active_prior_residual_mask",
    "_active_context_gate",
    "_active_prior_gate",
)


def _snapshot_active_injection(policy: nn.Module) -> dict[str, Any]:
    return {name: getattr(policy, name, None) for name in _DIAGNOSTIC_ACTIVE_FIELDS}


def _restore_active_injection(policy: nn.Module, snapshot: dict[str, Any]) -> None:
    for name, value in snapshot.items():
        setattr(policy, name, value)


@contextmanager
def _diagnostic_active_injection(
    policy: nn.Module,
    *,
    context: torch.Tensor | None,
    prior_mean: torch.Tensor | None,
    confidence: torch.Tensor | None,
    task_schema: torch.Tensor | None,
    phase_token: torch.Tensor | None,
) -> Any:
    """Temporarily expose the same residual inputs used by policy.forward."""
    snapshot = _snapshot_active_injection(policy)
    try:
        policy._clear_active_residuals()  # noqa: SLF001 - existing policy API
        if context is not None and prior_mean is not None:
            if task_schema is None or phase_token is None:
                raise ValueError("Residual diagnostics require task schema and phase token.")
            policy._active_causal_context = context  # noqa: SLF001
            policy._active_action_prior = prior_mean  # noqa: SLF001
            policy._activate_residual_gates(task_schema, phase_token)  # noqa: SLF001
            policy._active_injection_confidence = confidence  # noqa: SLF001
        yield
    finally:
        _restore_active_injection(policy, snapshot)


@contextmanager
def _diagnostic_foundation_anchor(policy: nn.Module) -> Any:
    """Temporarily swap the existing immutable action-path anchor, if loaded."""
    anchors = getattr(policy, "_foundation_anchor_parameters", {})
    if not anchors:
        yield False
        return
    named = dict(policy.foundation.named_parameters())
    original: dict[str, torch.Tensor] = {}
    try:
        with torch.no_grad():
            for name, anchor in anchors.items():
                parameter = named[name]
                original[name] = parameter.data
                parameter.data = anchor
        yield True
    finally:
        with torch.no_grad():
            for name, value in original.items():
                named[name].data = value


def _foundation_raw_flow_and_embedding(
    policy: nn.Module,
    processed: dict[str, torch.Tensor],
    *,
    rng_state: tuple[torch.Tensor, torch.Tensor | None],
    capture_embedding: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Read raw PI0.5 flow errors and pre-injection action embeddings.

    ``PI05Policy.forward(reduction='none')`` intentionally returns only a
    per-example reduction.  The opt-in diagnostic therefore calls its already
    public core ``PI05Pytorch.forward`` with the exact same preprocessor,
    noise, and time samplers.  It is under ``no_grad`` and restores RNG state;
    normal validation and training never enter this function.
    """
    foundation = policy.foundation
    model = foundation.model
    required = (
        getattr(foundation, "_preprocess_images", None),
        getattr(foundation, "_prepare_memory_states", None),
        getattr(foundation, "prepare_action", None),
        getattr(model, "sample_noise", None),
        getattr(model, "sample_time", None),
        getattr(model, "embed_suffix", None),
        getattr(model, "forward", None),
    )
    if any(value is None for value in required):
        raise RuntimeError("PI0.5 core does not expose the raw diagnostic forward API.")

    device = processed["action"].device
    saved_rng = _diagnostic_rng_state(device)
    try:
        _restore_diagnostic_rng_state(rng_state, device)
        images, image_masks = foundation._preprocess_images(processed)  # noqa: SLF001
        states, state_masks = foundation._prepare_memory_states(processed)  # noqa: SLF001
        tokens = processed["observation.language.tokens"]
        masks = processed["observation.language.attention_mask"]
        actions = foundation.prepare_action(processed)
        noise = model.sample_noise(actions.shape, actions.device)
        time = model.sample_time(actions.shape[0], actions.device)
        time_expanded = time[:, None, None]
        noisy_actions = time_expanded * noise + (1 - time_expanded) * actions

        noisy_embedding = None
        if capture_embedding:
            active_snapshot = _snapshot_active_injection(policy)
            try:
                policy._clear_active_residuals()  # noqa: SLF001
                # The pinned RoboTwin handoff exposes ``embed_suffix`` as
                # ``(noisy_actions, timestep)``.  A few older PI0.5 runtime
                # builds kept the non-PI05 state argument; retain a narrow
                # compatibility fallback for diagnostics only.  The normal
                # policy path is never routed through this branch.
                try:
                    suffix = model.embed_suffix(noisy_actions, time)
                except TypeError as first_error:
                    try:
                        suffix = model.embed_suffix(states, noisy_actions, time)
                    except TypeError:
                        raise first_error from None
                if not isinstance(suffix, tuple) or not suffix:
                    raise RuntimeError("PI0.5 embed_suffix returned no action embeddings.")
                noisy_embedding = suffix[0].detach()
            finally:
                _restore_active_injection(policy, active_snapshot)

        raw = model.forward(
            images,
            image_masks,
            tokens,
            masks,
            actions,
            noise,
            time,
            states=states,
            state_masks=state_masks,
        )
        raw = raw[0] if isinstance(raw, tuple) else raw
        if not torch.is_tensor(raw) or raw.ndim != 3:
            raise RuntimeError(
                "PI0.5 core did not return unreduced [B,H,D] flow errors; "
                f"got {type(raw)!r} with shape {getattr(raw, 'shape', None)}."
            )
        original_dim = int(foundation.config.output_features["action"].shape[0])
        return raw[:, :, :original_dim].detach(), noisy_embedding
    finally:
        _restore_diagnostic_rng_state(saved_rng, device)


def _diagnostic_residual_tensors(
    policy: nn.Module,
    *,
    task_schema: torch.Tensor,
    phase_token: torch.Tensor,
    context: torch.Tensor,
    prior_mean: torch.Tensor,
    confidence: torch.Tensor | None,
    action_embedding_dtype: torch.dtype | None = None,
    effective_gates: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the exact gated residual tensors used by the installed hook."""
    projector_dtype = next(policy.causal_action_projector.parameters()).dtype
    target_dtype = action_embedding_dtype or projector_dtype
    confidence_value = (
        context.new_ones(context.shape[0]) if confidence is None else confidence
    )
    if effective_gates is None:
        context_gate, prior_gate = policy.residual_injection_gates(task_schema, phase_token)
    else:
        context_gate, prior_gate = effective_gates
    context_gate = context_gate.to(context.dtype) * confidence_value.to(context.dtype)
    prior_gate = prior_gate.to(prior_mean.dtype) * confidence_value.to(prior_mean.dtype)

    if policy._direct_context_injection_enabled:  # noqa: SLF001
        # The installed hook broadcasts one projected context vector over
        # every action token.  Keep the singleton horizon here so the
        # measurement can expand it explicitly, rather than accidentally
        # relying on PyTorch's batch broadcasting rules.
        context_delta = policy.causal_action_projector(context).to(target_dtype)
        if context_delta.ndim != 2:
            raise ValueError(
                "The installed context hook expects a single [B,E] context projection, "
                f"got {tuple(context_delta.shape)}."
            )
        context_delta = context_delta[:, None, :]
        context_delta = context_delta * context_gate[:, None, None]
    else:
        context_delta = context.new_zeros(
            (context.shape[0], 1, policy.causal_action_projector.out_features),
            dtype=target_dtype,
        )
    prior_delta = policy.prior_action_projector(prior_mean).to(target_dtype)
    if prior_delta.ndim == 2:
        prior_delta = prior_delta[:, None, :]
    prior_delta = prior_delta * prior_gate[:, None, None]
    prior_horizon = min(
        int(getattr(policy, "_prior_injection_horizon", ROBOTWIN_ACTION_HORIZON)),
        prior_delta.shape[1],
    )
    if prior_horizon < prior_delta.shape[1]:
        prior_delta = prior_delta.clone()
        prior_delta[:, prior_horizon:] = 0
    return context_delta, prior_delta


def _expand_diagnostic_residual(
    value: torch.Tensor,
    *,
    batch_size: int,
    width: int,
    name: str,
) -> torch.Tensor:
    """Normalize a residual tensor to ``[B, H, E]`` for norm accounting."""
    value = torch.as_tensor(value)
    if value.ndim == 2:
        value = value[:, None, :]
    if value.ndim != 3 or value.shape[0] != batch_size:
        raise ValueError(
            f"{name} must be [B,H,E] (or [B,E]), got {tuple(value.shape)}."
        )
    if value.shape[1] == 1 and width > 1:
        value = value.expand(-1, width, -1)
    if value.shape[1] < width:
        raise ValueError(
            f"{name} has only {value.shape[1]} steps but {width} are required."
        )
    return value[:, :width]


def _relative_residual_values(
    reference: torch.Tensor,
    context_delta: torch.Tensor,
    prior_delta: torch.Tensor,
    *,
    valid_mask: torch.Tensor,
    horizon: int,
) -> dict[str, torch.Tensor]:
    """Return measured residual/reference norms, never gate-only estimates.

    This function deliberately returns per-example tensors.  The evaluator
    gathers those tensors across ranks before reducing them, so a large batch
    or a padded batch cannot change the metric through rank-local averaging.
    """
    if reference.ndim != 3:
        raise ValueError(
            "The pre-injection action embedding must be [B,H,E], got "
            f"{tuple(reference.shape)}."
        )
    width = min(horizon, reference.shape[1])
    if width <= 0:
        raise ValueError("The pre-injection action embedding has an empty horizon.")
    batch_size = reference.shape[0]
    context = _expand_diagnostic_residual(
        context_delta,
        batch_size=batch_size,
        width=width,
        name="context_delta",
    )
    prior = _expand_diagnostic_residual(
        prior_delta,
        batch_size=batch_size,
        width=width,
        name="prior_delta",
    )
    valid_mask = torch.as_tensor(valid_mask, device=reference.device).bool()
    if valid_mask.ndim != 2 or valid_mask.shape[0] != batch_size:
        raise ValueError(
            "Residual diagnostic validity mask must be [B,H], got "
            f"{tuple(valid_mask.shape)}."
        )
    valid = valid_mask[:, :width]
    ref = reference[:, :width].float()
    context = context.float()
    prior = prior.float()
    mask = valid.unsqueeze(-1).to(ref.dtype)
    ref_norm = torch.linalg.vector_norm((ref * mask).reshape(ref.shape[0], -1), dim=1)
    context_norm = torch.linalg.vector_norm(
        (context * mask).reshape(context.shape[0], -1), dim=1
    )
    prior_norm = torch.linalg.vector_norm(
        (prior * mask).reshape(prior.shape[0], -1), dim=1
    )
    total_norm = torch.linalg.vector_norm(
        ((context + prior) * mask).reshape(ref.shape[0], -1), dim=1
    )
    valid_steps = valid.sum(dim=1)
    denominator = ref_norm.clamp_min(torch.finfo(ref_norm.dtype).eps)
    finite = valid_steps > 0
    finite &= torch.isfinite(ref_norm)
    finite &= torch.isfinite(context_norm)
    finite &= torch.isfinite(prior_norm)
    finite &= torch.isfinite(total_norm)
    values: dict[str, torch.Tensor] = {
        "reference_norm": ref_norm,
        "context_delta_norm": context_norm,
        "prior_delta_norm": prior_norm,
        "total_delta_norm": total_norm,
        "context_relative_norm": context_norm / denominator,
        "prior_relative_norm": prior_norm / denominator,
        "total_relative_norm": total_norm / denominator,
        "valid_steps": valid_steps,
        "finite": finite,
    }
    for name, value in tuple(values.items()):
        if name not in {"valid_steps", "finite"}:
            values[name] = value.masked_fill(~finite, float("nan"))
    return values


def _collect_validation_diagnostics(
    policy: nn.Module,
    processed: dict[str, torch.Tensor],
    losses: dict[str, torch.Tensor],
    *,
    foundation_rng_state: tuple[torch.Tensor, torch.Tensor | None] | None,
    injection_confidence: torch.Tensor | None,
    phase_token: torch.Tensor | None,
    training_variant: str,
) -> dict[str, Any]:
    """Collect one opt-in, read-only PI0.5/ZeVA validation observation.

    The normal policy forward has already run before this helper is called.
    Every additional forward below runs under ``no_grad`` with RNG replay and
    restores the policy's transient injection fields, so enabling the report
    cannot mutate model state or alter the ordinary validation numbers.  The
    raw PI0.5 core is intentionally optional: unsupported runtime versions
    produce an explicit unavailable report instead of approximating H15 from a
    scalar H50 loss.
    """
    unavailable: dict[str, Any] = {
        "available": False,
        "mask_source": None,
        "reason": None,
        "_student_h50": None,
        "_student_h15": None,
        "_current_h50": None,
        "_current_h15": None,
        "_fixed_h50": None,
        "_fixed_h15": None,
        "_valid_steps_h50": None,
        "_valid_steps_h15": None,
        "_residual_h50": None,
        "_residual_h15": None,
    }
    if training_variant not in {
        "zeva",
        "adapter",
        "prior_adapter",
        "prior_zeva",
        "action_expert_control",
    }:
        unavailable["reason"] = (
            "raw residual diagnostics are defined for the PI0.5 noisy-action "
            f"path, not training_variant={training_variant!r}"
        )
        return unavailable
    if foundation_rng_state is None:
        unavailable["reason"] = "matched foundation RNG state was not captured"
        return unavailable

    unwrapped = policy.module if hasattr(policy, "module") else policy
    context = losses.get("_diagnostic_context")
    prior_mean = losses.get("_diagnostic_action_prior_mean")
    task_schema = losses.get("_diagnostic_task_schema")

    try:
        # First replay the exact current student weights with both residual
        # inputs disabled.  This is the separately labeled *current* teacher;
        # it is not silently called a fixed teacher because its action expert
        # may continue to drift during a run.
        with _diagnostic_active_injection(
            unwrapped,
            context=None,
            prior_mean=None,
            confidence=None,
            task_schema=None,
            phase_token=None,
        ):
            current_h50_raw, _ = _foundation_raw_flow_and_embedding(
                unwrapped,
                processed,
                rng_state=foundation_rng_state,
                capture_embedding=False,
            )

        student_h50_raw = None
        noisy_embedding = None
        residual_tensors = None
        if all(
            value is not None
            for value in (context, prior_mean, task_schema, phase_token)
        ):
            with _diagnostic_active_injection(
                unwrapped,
                context=context,
                prior_mean=prior_mean,
                confidence=injection_confidence,
                task_schema=task_schema,
                phase_token=phase_token,
            ):
                student_h50_raw, noisy_embedding = _foundation_raw_flow_and_embedding(
                    unwrapped,
                    processed,
                    rng_state=foundation_rng_state,
                    capture_embedding=True,
                )
                residual_tensors = _diagnostic_residual_tensors(
                    unwrapped,
                    task_schema=task_schema,
                    phase_token=phase_token,
                    context=context,
                    prior_mean=prior_mean,
                    confidence=injection_confidence,
                    action_embedding_dtype=noisy_embedding.dtype,
                    effective_gates=(
                        unwrapped._active_context_gate,  # noqa: SLF001
                        unwrapped._active_prior_gate,  # noqa: SLF001
                    ),
                )

        fixed_h50_raw = None
        if getattr(unwrapped, "_foundation_anchor_parameters", {}):
            # The immutable anchor, when explicitly loaded, is a different
            # scientific comparison from the current residual-off student.
            with _diagnostic_foundation_anchor(unwrapped) as loaded:
                if loaded:
                    with _diagnostic_active_injection(
                        unwrapped,
                        context=None,
                        prior_mean=None,
                        confidence=None,
                        task_schema=None,
                        phase_token=None,
                    ):
                        fixed_h50_raw, _ = _foundation_raw_flow_and_embedding(
                            unwrapped,
                            processed,
                            rng_state=foundation_rng_state,
                            capture_embedding=False,
                        )

        batch_size = current_h50_raw.shape[0]
        valid_mask, mask_source = _diagnostic_action_valid_mask(
            processed,
            batch_size=batch_size,
            horizon=ROBOTWIN_ACTION_HORIZON,
            device=current_h50_raw.device,
        )
        current_h50, valid_steps_h50 = _masked_action_flow_per_sample(
            current_h50_raw,
            horizon=ROBOTWIN_ACTION_HORIZON,
            valid_mask=valid_mask,
        )
        current_h15, valid_steps_h15 = _masked_action_flow_per_sample(
            current_h50_raw,
            horizon=ROBOTWIN_EXECUTED_HORIZON,
            valid_mask=valid_mask[:, :ROBOTWIN_EXECUTED_HORIZON],
        )
        student_h50 = student_h15 = None
        if student_h50_raw is not None:
            student_h50, student_steps_h50 = _masked_action_flow_per_sample(
                student_h50_raw,
                horizon=ROBOTWIN_ACTION_HORIZON,
                valid_mask=valid_mask,
            )
            student_h15, student_steps_h15 = _masked_action_flow_per_sample(
                student_h50_raw,
                horizon=ROBOTWIN_EXECUTED_HORIZON,
                valid_mask=valid_mask[:, :ROBOTWIN_EXECUTED_HORIZON],
            )
            if not torch.equal(student_steps_h50, valid_steps_h50) or not torch.equal(
                student_steps_h15, valid_steps_h15
            ):
                raise RuntimeError("Student and current H-prefix masks diverged.")
        fixed_h50 = fixed_h15 = None
        if fixed_h50_raw is not None:
            fixed_h50, fixed_steps_h50 = _masked_action_flow_per_sample(
                fixed_h50_raw,
                horizon=ROBOTWIN_ACTION_HORIZON,
                valid_mask=valid_mask,
            )
            fixed_h15, fixed_steps_h15 = _masked_action_flow_per_sample(
                fixed_h50_raw,
                horizon=ROBOTWIN_EXECUTED_HORIZON,
                valid_mask=valid_mask[:, :ROBOTWIN_EXECUTED_HORIZON],
            )
            if not torch.equal(fixed_steps_h50, valid_steps_h50) or not torch.equal(
                fixed_steps_h15, valid_steps_h15
            ):
                raise RuntimeError("Fixed-teacher and current H-prefix masks diverged.")

        residual_h50 = residual_h15 = None
        residual_reason = None
        if noisy_embedding is not None and residual_tensors is not None:
            context_delta, prior_delta = residual_tensors
            residual_h50 = _relative_residual_values(
                noisy_embedding,
                context_delta,
                prior_delta,
                valid_mask=valid_mask,
                horizon=ROBOTWIN_ACTION_HORIZON,
            )
            residual_h15 = _relative_residual_values(
                noisy_embedding,
                context_delta,
                prior_delta,
                valid_mask=valid_mask[:, :ROBOTWIN_EXECUTED_HORIZON],
                horizon=ROBOTWIN_EXECUTED_HORIZON,
            )
        else:
            residual_reason = (
                "injection tensors or pre-injection action embedding were not "
                "exposed by this validation path; no gate-only estimate was used"
            )

        unavailable.update(
            {
                "available": True,
                "mask_source": mask_source,
                "reason": residual_reason,
                "_student_h50": student_h50,
                "_student_h15": student_h15,
                "_current_h50": current_h50,
                "_current_h15": current_h15,
                "_fixed_h50": fixed_h50,
                "_fixed_h15": fixed_h15,
                "_valid_steps_h50": valid_steps_h50,
                "_valid_steps_h15": valid_steps_h15,
                "_residual_h50": residual_h50,
                "_residual_h15": residual_h15,
            }
        )
    except (TypeError, ValueError, RuntimeError, KeyError, AttributeError) as error:
        unavailable["reason"] = f"{type(error).__name__}: {error}"
    return unavailable


def _finalize_validation_diagnostics(
    values: dict[str, list[torch.Tensor]],
    *,
    reasons: list[str],
    mask_sources: list[str],
) -> dict[str, Any]:
    """Reduce gathered diagnostic vectors into a stable JSON-safe report."""
    def concat(name: str) -> torch.Tensor | None:
        chunks = values.get(name, [])
        return torch.cat(chunks) if chunks else None

    student_h50 = concat("student_h50")
    student_h15 = concat("student_h15")
    current_h50 = concat("current_h50")
    current_h15 = concat("current_h15")
    fixed_h15 = concat("fixed_h15")
    steps_h50 = concat("valid_steps_h50")
    steps_h15 = concat("valid_steps_h15")

    def unavailable_pair(label: str, reason: str) -> dict[str, Any]:
        return {"available": False, "label": label, "reason": reason}

    paired_current = (
        _diagnostic_pair_summary(student_h15, current_h15, valid_steps=steps_h15)
        if student_h15 is not None and current_h15 is not None
        else unavailable_pair(
            "current_residual_off",
            "student residual-on H15 flow was unavailable",
        )
    )
    paired_current["label"] = "current_residual_off"
    paired_fixed = (
        _diagnostic_pair_summary(student_h15, fixed_h15, valid_steps=steps_h15)
        if student_h15 is not None and fixed_h15 is not None
        else unavailable_pair(
            "fixed_teacher",
            "an explicit immutable foundation anchor was not available",
        )
    )
    paired_fixed["label"] = "fixed_teacher"

    residual = {
        "available": False,
        "measurement": (
            "projected gated context/prior tensors divided by the measured "
            "pre-injection noisy action-embedding norm"
        ),
    }
    for horizon_name, suffix, steps_name in (
        ("h50", "h50", "residual_h50_valid_steps"),
        ("h15", "h15", "residual_h15_valid_steps"),
    ):
        prefix = f"residual_{suffix}_"
        if not any(values.get(prefix + metric, []) for metric in (
            "reference_norm",
            "context_delta_norm",
            "prior_delta_norm",
            "total_delta_norm",
            "context_relative_norm",
            "prior_relative_norm",
            "total_relative_norm",
        )):
            continue
        horizon_steps = concat(steps_name)
        report: dict[str, Any] = {
            "available": True,
            "horizon": ROBOTWIN_ACTION_HORIZON
            if horizon_name == "h50"
            else ROBOTWIN_EXECUTED_HORIZON,
            "reference": "pre_injection_noisy_action_embedding",
        }
        for metric in (
            "reference_norm",
            "context_delta_norm",
            "prior_delta_norm",
            "total_delta_norm",
            "context_relative_norm",
            "prior_relative_norm",
            "total_relative_norm",
        ):
            metric_values = concat(prefix + metric)
            if metric_values is not None:
                report[metric] = _diagnostic_value_summary(
                    metric_values,
                    valid_steps=horizon_steps,
                )
        residual[horizon_name] = report
        residual["available"] = True
    if not residual["available"]:
        residual["reason"] = (
            "measured residual tensors were unavailable; no scalar gate was "
            "used as an amplitude proxy"
        )

    report: dict[str, Any] = {
        "enabled": True,
        "available": any(
            value is not None
            for value in (student_h15, current_h15, fixed_h15)
        ),
        "policy_horizon": ROBOTWIN_ACTION_HORIZON,
        "executed_horizon": ROBOTWIN_EXECUTED_HORIZON,
        "raw_flow_source": "PI05Pytorch.forward unreduced [B,H,D] flow errors",
        "preprocessing_contract": (
            "same processed batch, PI0.5 image preprocessing, action preparation, "
            "and normalization path as the ordinary H50 validation"
        ),
        "mask_source": sorted(set(mask_sources)) if mask_sources else None,
        "flow": {
            "zeva_residual_on_h50": (
                _diagnostic_value_summary(student_h50, valid_steps=steps_h50)
                if student_h50 is not None
                else {"available": False, "reason": "student raw flow unavailable"}
            ),
            "current_residual_off_h50": (
                _diagnostic_value_summary(current_h50, valid_steps=steps_h50)
                if current_h50 is not None
                else {"available": False, "reason": "current raw flow unavailable"}
            ),
            "zeva_residual_on_executed_h15": (
                _diagnostic_value_summary(student_h15, valid_steps=steps_h15)
                if student_h15 is not None
                else {"available": False, "reason": "student raw flow unavailable"}
            ),
            "current_residual_off_executed_h15": (
                _diagnostic_value_summary(current_h15, valid_steps=steps_h15)
                if current_h15 is not None
                else {"available": False, "reason": "current raw flow unavailable"}
            ),
        },
        "paired": {
            "current_residual_off": paired_current,
            "fixed_teacher": paired_fixed,
        },
        "injected_residual_norms": residual,
    }
    if reasons:
        report["unavailable_batches"] = sorted(set(reasons))
    return report


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


def _prepare_cached_output_correction_batch(
    policy: nn.Module,
    raw_batch: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare v13 without decoding images or invoking PI0.5 again."""
    unwrapped = policy.module if hasattr(policy, "module") else policy
    base_actions = raw_batch.pop("zeva.base_actions")
    raw_targets = raw_batch["action"]
    task_goal = unwrapped._task_only_goal_embedding(  # noqa: SLF001
        raw_batch["task"], raw_targets.device
    )
    processed = {"zeva.goal_embedding": task_goal}
    normalized_targets = unwrapped.action_normalizer.normalize(raw_targets)
    return processed, base_actions, normalized_targets, task_goal


def _losses(
    policy: nn.Module,
    processed: dict[str, torch.Tensor],
    bank_batch,
    injection_confidence: torch.Tensor,
    baseline_flow: torch.Tensor | None,
    foundation_rng_state: tuple[torch.Tensor, torch.Tensor | None] | None,
    prior_weight: float,
    preserve_weight: float,
    gate_regularization_weight: float,
    prior_residual_dropout_probability: float,
    paired_improvement_margin: float = 0.0,
    preserve_scale: float = 1.0,
    prior_supervision_horizon: int = ROBOTWIN_ACTION_HORIZON,
    *,
    training: bool,
    return_diagnostics: bool = False,
    prior_nll_detach_context: bool = False,
    prior_residual_mask: torch.Tensor | None = None,
    executed_flow_capture: _ConnectedRawFlowCapture | None = None,
) -> dict[str, torch.Tensor]:
    unwrapped = policy.module if hasattr(policy, "module") else policy
    if foundation_rng_state is not None:
        unwrapped.set_foundation_rng_state(*foundation_rng_state)
    if prior_residual_mask is not None:
        expected_shape = (processed["action"].shape[0],)
        if tuple(prior_residual_mask.shape) != expected_shape:
            raise ValueError(
                "Provided prior residual mask must have one value per example, got "
                f"{tuple(prior_residual_mask.shape)} versus {expected_shape}."
            )
        if prior_residual_mask.device != processed["action"].device:
            raise ValueError("Provided prior residual mask must be on the processed batch device.")
        prior_residual_mask = prior_residual_mask.to(
            device=processed["action"].device,
            dtype=processed["action"].dtype,
        )
    elif training and prior_residual_dropout_probability > 0:
        batch_size = processed["action"].shape[0]
        prior_residual_mask = (
            torch.rand(batch_size, device=processed["action"].device)
            >= prior_residual_dropout_probability
        ).to(processed["action"].dtype)
    if executed_flow_capture is not None:
        executed_flow_capture.begin()
    try:
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
            **({"prior_nll_detach_context": True} if prior_nll_detach_context else {}),
        )
    except BaseException:
        if executed_flow_capture is not None:
            executed_flow_capture.abort()
        raise
    flow_per_sample = _foundation_loss(foundation_output, reduction="none")
    flow = flow_per_sample.mean()
    executed_flow = None
    if executed_flow_capture is not None:
        raw = executed_flow_capture.take()
        executed_per_sample, raw_h50_per_sample = _connected_executed_flow_per_sample(
            raw,
            expected_batch_size=flow_per_sample.shape[0],
        )
        if not executed_flow_capture.h50_equivalence_verified:
            torch.testing.assert_close(
                raw_h50_per_sample.detach(),
                flow_per_sample.detach(),
                rtol=1e-5,
                atol=1e-7,
                msg="Captured PI05 raw H50 differs from ordinary policy H50 loss.",
            )
            executed_flow_capture.h50_equivalence_verified = True
        executed_flow = executed_per_sample.mean()
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
    # The opt-in ZeVA branch is trained on the *deployed* prefix; matched Base
    # and the historical H50 preserve hinge keep their existing definitions.
    total_flow = executed_flow if executed_flow is not None else flow
    total = total_flow + prior_weight * prior + preserve_weight * preserve + gate_regularization_weight * gate
    result = {
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
    if executed_flow is not None:
        result["training_flow_h15"] = executed_flow
    if return_diagnostics:
        # ``policy.forward`` already built the exact context and Gaussian
        # prior used by the residual hook.  Expose detached copies only to the
        # opt-in validation telemetry; the ordinary validation/training
        # result remains byte-for-byte the historical dictionary above.
        causal_context = unwrapped.memory_context_encoder(
            task_schema,
            bank_batch.phase_token,
            bank_batch.brief_signals,
            bank_batch.retrieved_signals,
            bank_batch.brief_mask,
            bank_batch.retrieved_mask,
        )
        result.update(
            {
                "_diagnostic_context": causal_context.detach(),
                "_diagnostic_action_prior_mean": action_prior.mean.detach(),
                "_diagnostic_task_schema": task_schema.detach(),
            }
        )
    return result


def _sample_prior_residual_mask(
    processed: dict[str, torch.Tensor],
    *,
    probability: float,
) -> torch.Tensor | None:
    """Sample the ZeVA-only prior mask once for a routed forward pair."""
    if probability <= 0:
        return None
    batch_size = processed["action"].shape[0]
    return (
        torch.rand(batch_size, device=processed["action"].device) >= probability
    ).to(processed["action"].dtype)


def _gradient_routed_on_forward(
    policy: nn.Module,
    processed: dict[str, torch.Tensor],
    bank_batch,
    injection_confidence: torch.Tensor,
    baseline_flow: torch.Tensor | None,
    foundation_rng_state: tuple[torch.Tensor, torch.Tensor | None] | None,
    prior_weight: float,
    preserve_weight: float,
    gate_regularization_weight: float,
    prior_residual_dropout_probability: float,
    paired_improvement_margin: float,
    preserve_scale: float,
    prior_supervision_horizon: int,
    executed_flow_capture: _ConnectedRawFlowCapture | None = None,
) -> tuple[dict[str, torch.Tensor], tuple[torch.Tensor, torch.Tensor | None]]:
    """Build only the residual-on loss for opt-in action-expert routing.

    The caller must complete this loss's backward before invoking
    :func:`_gradient_routed_off_forward`.  The prior residual dropout mask is
    sampled exactly once here and the foundation RNG state is returned for the
    later residual-off forward.  Retrieval and memory dropout are intentionally
    done by the caller before this helper and are not repeated here.
    """
    if foundation_rng_state is None:
        device = processed["action"].device
        # CPU doubles are useful for unit tests; the policy's production path
        # always supplies a CUDA state through the PI0.5 runtime.
        pair_rng_state = _diagnostic_rng_state(device)
    else:
        pair_rng_state = foundation_rng_state

    # Draw ZeVA-only dropout before restoring the foundation stream.  This
    # keeps the subsequent foundation calls independent of this extra branch.
    prior_residual_mask = _sample_prior_residual_mask(
        processed,
        probability=prior_residual_dropout_probability,
    )
    device = processed["action"].device
    _restore_diagnostic_rng_state(pair_rng_state, device)
    on_losses = _losses(
        policy,
        processed,
        bank_batch,
        injection_confidence,
        baseline_flow,
        pair_rng_state,
        prior_weight,
        preserve_weight,
        gate_regularization_weight,
        prior_residual_dropout_probability,
        paired_improvement_margin=paired_improvement_margin,
        preserve_scale=preserve_scale,
        prior_supervision_horizon=prior_supervision_horizon,
        prior_nll_detach_context=True,
        prior_residual_mask=prior_residual_mask,
        executed_flow_capture=executed_flow_capture,
        training=True,
    )
    return on_losses, pair_rng_state


def _gradient_routed_off_forward(
    policy: nn.Module,
    processed: dict[str, torch.Tensor],
    foundation_rng_state: tuple[torch.Tensor, torch.Tensor | None],
    zeva_parameters: list[nn.Parameter] | tuple[nn.Parameter, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the residual-off loss after the residual-on backward completes.

    ``foundation_only`` is the current student foundation, not the optional
    immutable anchor used by the preserve hinge.  The zero edge marks ZeVA
    parameters used for DDP while contributing no off-branch gradient.  This
    helper deliberately performs no work until the caller has finished the
    residual-on backward; two DDP forwards before the first backward leave
    reducer buckets rank-local.
    """
    device = processed["action"].device
    _restore_diagnostic_rng_state(foundation_rng_state, device)
    off_output = policy(processed, foundation_only=True, foundation_reduction="none")
    off_flow_per_sample = _foundation_loss(off_output, reduction="none")
    off_flow = off_flow_per_sample.mean()
    off_total = off_flow + _zero_parameter_gradient_link(zeva_parameters, off_flow)
    return off_total, off_flow


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


def _action_expert_control_losses(
    policy: nn.Module,
    processed: dict[str, torch.Tensor],
    baseline_flow: torch.Tensor | None,
    foundation_rng_state: tuple[torch.Tensor, torch.Tensor] | None,
    preserve_weight: float,
    paired_improvement_margin: float,
    preserve_scale: float,
) -> dict[str, torch.Tensor]:
    """Flow plus matched immutable-Base PBD for the v12 causal control.

    Unlike ``_losses``, this path never builds a causal bank batch, action
    prior, context residual, or Gaussian NLL.  The only trainable path is the
    PI0.5 action expert; when a teacher loss is sampled, the anchor forward
    saved the rank-local RNG state and this student forward consumes the same
    noise/time draw.  The shared data contract remains H50 prediction with
    H15-spaced decisions and H15 deployment execution.
    """
    if foundation_rng_state is not None:
        cpu_state, cuda_state = foundation_rng_state
        torch.random.set_rng_state(cpu_state)
        torch.cuda.set_rng_state(cuda_state, processed["action"].device)
    foundation_output = policy(processed, foundation_only=True)
    flow_per_sample = _foundation_loss(foundation_output, reduction="none")
    flow = flow_per_sample.mean()
    if baseline_flow is None:
        baseline = flow.detach().new_full((), float("nan"))
        baseline_per_sample = flow_per_sample.detach().new_full(
            flow_per_sample.shape, float("nan")
        )
        preserve = flow.new_zeros(())
        baseline_sampled = flow.new_zeros(())
        paired_improvement = flow.new_full((), float("nan"))
        paired_win_fraction = flow.new_full((), float("nan"))
        paired_degradation = flow.new_full((), float("nan"))
    else:
        baseline_per_sample = baseline_flow.detach()
        if baseline_per_sample.shape != flow_per_sample.shape:
            raise ValueError(
                "Matched immutable-Base and control flow losses must have identical "
                f"per-example shapes, got {baseline_per_sample.shape} and {flow_per_sample.shape}."
            )
        baseline = baseline_per_sample.mean()
        preserve = (
            F.relu(
                flow_per_sample
                - baseline_per_sample
                + float(paired_improvement_margin)
            ).mean()
            * preserve_scale
        )
        baseline_sampled = flow.new_ones(())
        paired_improvement = baseline - flow
        paired_win_fraction = (flow_per_sample < baseline_per_sample).to(flow.dtype).mean()
        paired_degradation = F.relu(flow_per_sample - baseline_per_sample).mean()
    return {
        "total": flow + preserve_weight * preserve,
        "flow": flow,
        "prior": flow.new_zeros(()),
        "baseline": baseline,
        "baseline_sampled": baseline_sampled,
        "preserve": preserve,
        "gate": flow.new_zeros(()),
        "paired_improvement": paired_improvement,
        "paired_win_fraction": paired_win_fraction,
        "paired_degradation": paired_degradation,
        "flow_per_sample": flow_per_sample,
        "baseline_per_sample": baseline_per_sample,
        "prior_std": flow.new_zeros(()),
        "prior_residual_keep": flow.new_zeros(()),
    }


def _output_correction_losses(
    policy: nn.Module,
    processed: dict[str, torch.Tensor],
    bank_batch,
    injection_confidence: torch.Tensor,
    *,
    prior_weight: float,
    preserve_weight: float,
    gate_regularization_weight: float,
    paired_improvement_margin: float,
    correction_horizon: int,
    cached_base_actions: torch.Tensor | None = None,
    task_goal_embedding: torch.Tensor | None = None,
    target_actions: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Train the actual post-diffusion H15 output against its exact Base sample."""
    base, corrected, action_prior, gates = policy(
        processed,
        output_correction=True,
        bank_phase_token=bank_batch.phase_token,
        bank_brief_signals=bank_batch.brief_signals,
        bank_retrieved_signals=bank_batch.retrieved_signals,
        bank_brief_mask=bank_batch.brief_mask,
        bank_retrieved_mask=bank_batch.retrieved_mask,
        injection_confidence=injection_confidence,
        cached_base_actions=cached_base_actions,
        task_goal_embedding=task_goal_embedding,
    )
    horizon = min(int(correction_horizon), base.shape[1])
    if target_actions is None:
        target_actions = processed["action"]
    target = target_actions[:, :horizon, :ROBOTWIN_ACTION_DIM].to(base.dtype)
    base_prefix = base[:, :horizon]
    corrected_prefix = corrected[:, :horizon]
    base_per_sample = (base_prefix - target).square().mean(dim=(1, 2))
    corrected_per_sample = (corrected_prefix - target).square().mean(dim=(1, 2))
    corrected_mse = corrected_per_sample.mean()
    base_mse = base_per_sample.mean()
    preserve = F.relu(
        corrected_per_sample - base_per_sample + float(paired_improvement_margin)
    ).mean()
    supervised_prior = action_prior._replace(
        mean=action_prior.mean[:, :horizon],
        log_std=action_prior.log_std[:, :horizon],
    )
    prior = gaussian_action_prior_nll(supervised_prior, target)
    gate = gates.square().mean()
    total = (
        corrected_mse
        + prior_weight * prior
        + preserve_weight * preserve
        + gate_regularization_weight * gate
    )
    improvement = base_mse - corrected_mse
    return {
        "total": total,
        # Keep legacy metric names consumable by the shared logger, but make
        # their direct action-space meaning explicit in the v13 manifest.
        "flow": corrected_mse,
        "prior": prior,
        "baseline": base_mse,
        "baseline_sampled": corrected_mse.new_ones(()),
        "preserve": preserve,
        "gate": gate,
        "paired_improvement": improvement,
        "paired_win_fraction": (corrected_per_sample < base_per_sample).float().mean(),
        "paired_degradation": F.relu(corrected_per_sample - base_per_sample).mean(),
        "flow_per_sample": corrected_per_sample,
        "baseline_per_sample": base_per_sample,
        "prior_std": supervised_prior.log_std.detach().exp().mean(),
        "prior_residual_keep": gates.detach().mean(),
    }


def _output_residual_losses(
    policy: nn.Module,
    processed: dict[str, torch.Tensor],
    bank_batch,
    injection_confidence: torch.Tensor,
    *,
    prior_weight: float,
    preserve_weight: float,
    gate_regularization_weight: float,
    paired_improvement_margin: float,
    correction_horizon: int,
    residual_regression_weight: float,
    residual_trust_region_weight: float,
    residual_trust_region_radius: float,
    residual_bound: float,
    cached_base_actions: torch.Tensor | None = None,
    task_goal_embedding: torch.Tensor | None = None,
    target_actions: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Train the direct bounded H15 action residual against expert output.

    The exact Base chunk is immutable data.  The main loss supervises the
    final corrected action; a second regression term supervises the raw
    residual against the clipped expert-minus-Base delta.  The per-example
    hinge is retained so a task cannot hide degradation behind another task.
    Gaussian NLL trains the context/prior representation but never supplies
    the deployed action by interpolation.
    """
    base, corrected, action_prior, gates, residual = policy(
        processed,
        output_residual=True,
        bank_phase_token=bank_batch.phase_token,
        bank_brief_signals=bank_batch.brief_signals,
        bank_retrieved_signals=bank_batch.retrieved_signals,
        bank_brief_mask=bank_batch.brief_mask,
        bank_retrieved_mask=bank_batch.retrieved_mask,
        injection_confidence=injection_confidence,
        cached_base_actions=cached_base_actions,
        task_goal_embedding=task_goal_embedding,
    )
    horizon = min(int(correction_horizon), base.shape[1])
    if target_actions is None:
        target_actions = processed["action"]
    target = target_actions[:, :horizon, :ROBOTWIN_ACTION_DIM].to(base.dtype)
    base_prefix = base[:, :horizon]
    corrected_prefix = corrected[:, :horizon]
    corrected_per_sample = (corrected_prefix - target).square().mean(dim=(1, 2))
    base_per_sample = (base_prefix - target).square().mean(dim=(1, 2))
    corrected_mse = corrected_per_sample.mean()
    base_mse = base_per_sample.mean()
    preserve = F.relu(
        corrected_per_sample - base_per_sample + float(paired_improvement_margin)
    ).mean()

    target_delta = (target - base_prefix).clamp(-float(residual_bound), float(residual_bound))
    residual_regression = (residual - target_delta).square().mean()
    effective_delta = corrected_prefix - base_prefix
    trust_excess = F.relu(
        effective_delta.abs() - float(residual_trust_region_radius)
    )
    trust_region = trust_excess.square().mean()
    supervised_prior = action_prior._replace(
        mean=action_prior.mean[:, :horizon],
        log_std=action_prior.log_std[:, :horizon],
    )
    prior = gaussian_action_prior_nll(supervised_prior, target)
    # Centering rather than shrinking the gate avoids the v13 gate-collapse
    # failure mode while leaving the residual head itself fully bounded.
    gate = (gates - 0.5).square().mean()
    total = (
        corrected_mse
        + float(residual_regression_weight) * residual_regression
        + prior_weight * prior
        + preserve_weight * preserve
        + float(residual_trust_region_weight) * trust_region
        + gate_regularization_weight * gate
    )
    improvement = base_mse - corrected_mse
    return {
        "total": total,
        "flow": corrected_mse,
        "prior": prior,
        "baseline": base_mse,
        "baseline_sampled": corrected_mse.new_ones(()),
        "preserve": preserve,
        "gate": gate,
        "residual_regression": residual_regression,
        "trust_region": trust_region,
        "paired_improvement": improvement,
        "paired_win_fraction": (corrected_per_sample < base_per_sample).float().mean(),
        "paired_degradation": F.relu(corrected_per_sample - base_per_sample).mean(),
        "flow_per_sample": corrected_per_sample,
        "baseline_per_sample": base_per_sample,
        "prior_std": supervised_prior.log_std.detach().exp().mean(),
        "prior_residual_keep": gates.detach().mean(),
        "residual_abs": residual.detach().abs().mean(),
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


def _assert_gradient_routed_zeva_gradients(policy: RobotWinZevaPolicy) -> None:
    """Check ZeVA gradients immediately after the residual-on backward.

    This guard runs before the residual-off backward can add its explicit DDP
    zero links.  It therefore proves that the ZeVA optimizer group is sourced
    by the residual-on objective rather than merely being marked used for DDP.
    """
    modules = (
        policy.task_token_projector,
        policy.memory_context_encoder,
        policy.action_prior,
        policy.causal_action_projector,
        policy.prior_action_projector,
        policy.residual_gate_router,
    )
    if any(not any(parameter.grad is not None for parameter in module.parameters()) for module in modules):
        raise RuntimeError(
            "Gradient-routing invariant failed: a ZeVA module received no residual-on gradient."
        )
    if policy.context_gate_logit.grad is None or policy.prior_gate_logit.grad is None:
        raise RuntimeError(
            "Gradient-routing invariant failed: residual-on scalar gates received no gradient."
        )


_ACTION_EXPERT_PARAMETER_PREFIXES = (
    "model.paligemma_with_expert.gemma_expert.model.",
    "model.action_in_proj.",
    "model.action_out_proj.",
    "model.time_mlp_in.",
    "model.time_mlp_out.",
)


def _assert_action_expert_control_parameters(
    policy: RobotWinZevaPolicy,
    trainable: list[nn.Parameter],
) -> tuple[str, ...]:
    """Fail closed unless v12's optimizer candidate is exactly action expert."""
    trainable_ids = {id(parameter) for parameter in trainable}
    named_foundation = dict(policy.foundation.named_parameters())
    named_policy = dict(policy.named_parameters())
    foundation_names = tuple(
        sorted(name for name, parameter in named_foundation.items() if id(parameter) in trainable_ids)
    )
    unexpected_foundation = [
        name
        for name in foundation_names
        if not any(name.startswith(prefix) for prefix in _ACTION_EXPERT_PARAMETER_PREFIXES)
    ]
    if unexpected_foundation:
        raise RuntimeError(
            "action_expert_control has non-action foundation parameters in the optimizer: "
            f"{unexpected_foundation[:8]}"
        )
    non_foundation = sorted(
        name
        for name, parameter in named_policy.items()
        if id(parameter) in trainable_ids and not name.startswith("foundation.")
    )
    if non_foundation:
        raise RuntimeError(
            "action_expert_control has ZeVA parameters in the optimizer: "
            f"{non_foundation[:8]}"
        )
    if not foundation_names:
        raise RuntimeError("action_expert_control has no trainable action-expert parameters.")
    return foundation_names


def _assert_action_expert_control_optimizer(
    optimizer: torch.optim.Optimizer,
    trainable: list[nn.Parameter],
    anchor_parameters: dict[str, torch.Tensor],
) -> None:
    """Ensure the immutable anchor is data-only and never enters AdamW."""
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    trainable_ids = {id(parameter) for parameter in trainable}
    if optimizer_ids != trainable_ids:
        raise RuntimeError(
            "action_expert_control optimizer parameters differ from the audited trainable set."
        )
    teacher_ids = {id(value) for value in anchor_parameters.values()}
    if optimizer_ids.intersection(teacher_ids):
        raise RuntimeError("Immutable action-expert teacher tensor entered the optimizer.")


def _assert_output_correction_gradients(policy: RobotWinZevaPolicy) -> None:
    """Fail closed unless v13 trains only the direct output-correction path."""
    required = (
        policy.task_token_projector,
        policy.memory_context_encoder,
        policy.action_prior,
        policy.output_correction_gate,
    )
    if any(
        not any(parameter.grad is not None for parameter in module.parameters())
        for module in required
    ):
        raise RuntimeError("v13 output correction omitted gradients from a required module.")
    forbidden = (
        policy.foundation,
        policy.causal_transition_encoder,
        policy.causal_action_projector,
        policy.prior_action_projector,
        policy.residual_gate_router,
    )
    if any(parameter.grad is not None for module in forbidden for parameter in module.parameters()):
        raise RuntimeError("v13 output correction leaked gradients into a frozen/token path.")


def _assert_output_residual_gradients(
    policy: RobotWinZevaPolicy, *, allow_zero_init_delay: bool = False
) -> None:
    """Audit corrector on step one and all upstream paths after zero-init lifts."""
    required = {
        "task_token_projector": policy.task_token_projector,
        "memory_context_encoder": policy.memory_context_encoder,
        "action_prior": policy.action_prior,
        "output_residual_corrector": policy.output_residual_corrector,
    }
    names_to_check = (
        ("output_residual_corrector",)
        if allow_zero_init_delay else tuple(required)
    )
    missing = [
        name for name, module in required.items()
        if name in names_to_check
        if not any(parameter.grad is not None for parameter in module.parameters())
    ]
    if missing:
        raise RuntimeError(
            f"v14 output residual omitted gradients from required modules: {missing}; "
            f"allow_zero_init_delay={allow_zero_init_delay}."
        )
    forbidden = (
        policy.foundation,
        policy.causal_transition_encoder,
        policy.causal_action_projector,
        policy.prior_action_projector,
        policy.residual_gate_router,
        policy.output_correction_gate,
    )
    if any(parameter.grad is not None for module in forbidden for parameter in module.parameters()):
        raise RuntimeError("v14 output residual leaked gradients into a frozen/token path.")


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
        "output_correction",
        "output_residual",
    }
    action_expert_control = args.training_variant == "action_expert_control"
    output_correction = args.training_variant == "output_correction"
    output_residual = args.training_variant == "output_residual"
    paired_teacher_enabled = (
        zeva_enabled and not output_correction and not output_residual
    ) or action_expert_control
    prior_only = args.training_variant in {"prior_adapter", "prior_zeva"}
    stage1_prediction_enabled = zeva_enabled
    frozen_foundation = args.training_variant in {
        "adapter",
        "prior_adapter",
        "output_correction",
        "output_residual",
    }
    zte_checkpoint = torch.load(args.zte_checkpoint, map_location="cpu")
    stage1_schema = zte_checkpoint.get("schema")
    if stage1_schema not in LEGACY_SCHEMAS | {V2_SCHEMA}:
        raise ValueError("Aligned Stage 2 requires a supported Stage 1 checkpoint.")
    transition_horizon = stage1_transition_horizon(zte_checkpoint)
    if transition_horizon != 15:
        raise ValueError("RoboTwin Stage 2 integration requires the fixed executed H15 contract.")
    bank_payload = torch.load(args.causal_bank, map_location="cpu", weights_only=False)
    if stage1_schema == V2_SCHEMA:
        validate_stage1_v2_artifact_status(bank_payload, artifact_name="causal bank")
    if bank.manifest["stage1_checkpoint_sha256"] != _sha256(args.zte_checkpoint):
        raise ValueError("The causal bank was not exported from the selected Stage 1 checkpoint.")
    if bank.manifest["statistics_sha256"] != _sha256(handoff.statistics):
        raise ValueError("The causal bank does not match the selected PI0.5 normalization.")
    bank_stage1_schema = stage1_artifact_schema(bank.manifest)
    if stage1_schema == V2_SCHEMA and bank_stage1_schema != V2_SCHEMA:
        raise ValueError(
            "Stage 2 v2 requires a causal bank with explicit "
            "stage1_checkpoint_schema=v2."
        )
    if bank_stage1_schema is not None and bank_stage1_schema != stage1_schema:
        raise ValueError("The causal bank was exported from a different Stage 1 encoder schema.")
    bank_horizon = bank.manifest.get("causal_transition_horizon")
    if bank_horizon is not None and int(bank_horizon) != transition_horizon:
        raise ValueError("The causal bank transition horizon differs from the Stage 1 checkpoint.")
    return {
        "schema": (
            "zeva-robotwin-stage2-action-expert-control-manifest-v1"
            if action_expert_control
            else "zeva-robotwin-stage2-output-correction-manifest-v1"
            if output_correction
            else "zeva-robotwin-stage2-output-residual-manifest-v1"
            if output_residual
            else "zeva-robotwin-stage2-action-expert-manifest-v11"
        ),
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
        "zte_checkpoint_schema": stage1_schema,
        "causal_transition_horizon": transition_horizon,
        "zte_step": int(zte_checkpoint["step"]),
        "causal_bank": str(Path(args.causal_bank).resolve()),
        "causal_bank_sha256": _sha256(args.causal_bank),
        "causal_bank_manifest": bank.manifest,
        "live_queries": str(Path(args.live_queries).resolve()),
        "live_queries_sha256": _sha256(args.live_queries),
        "base_action_cache": (
            str(Path(args.base_action_cache).resolve()) if args.base_action_cache else None
        ),
        "base_action_cache_sha256": (
            _sha256(args.base_action_cache) if args.base_action_cache else None
        ),
        "task_retrieval": str(Path(args.task_retrieval).resolve()),
        "task_retrieval_sha256": _sha256(args.task_retrieval),
        "dataset_adapter": str((Path(args.dataset_root) / "adapter.json").resolve()),
        "train_split": "train95",
        "validation_split": "validation5",
        "retrieval_protocol": (
            "task-language inferred task + recurrent H15 ZTE phase; no oracle progress"
            if stage1_prediction_enabled
            else "loaded for lineage validation only; not used for Base predictions or training"
        ),
        "stage1_usage": {
            "checkpoint_and_artifacts_loaded_for_lineage_validation": True,
            "used_for_predictions": stage1_prediction_enabled,
            "used_for_training": stage1_prediction_enabled,
            "task_language_retrieval_used": stage1_prediction_enabled,
            "memory_streams": {
                "brief": stage1_prediction_enabled,
                "persistent": stage1_prediction_enabled,
            },
            "base_contract": (
                "ordinary_untouched_best_v1_action_expert_flow; H50 flow with H15 decision sampling"
                if not stage1_prediction_enabled
                else None
            ),
        },
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
            "untouched_best_v1_action_expert_only_pbd_control_v12"
            if action_expert_control
            else
            "frozen_paligemma_action_expert_prior_residual_pbd_v11"
            if args.training_variant == "prior_zeva"
            else "frozen_pi05_zeva_prior_only_fixed_gate_v11"
            if args.training_variant == "prior_adapter"
            else "frozen_pi05_post_diffusion_h15_output_correction_v13"
            if args.training_variant == "output_correction"
            else "frozen_pi05_post_diffusion_h15_output_residual_v14"
            if output_residual
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
        "action_expert_gradient_routing": {
            "enabled": args.decouple_action_expert_gradient,
            "zeva_flow_objective": (
                "connected_raw_executed_h15"
                if args.zeva_h15_flow_objective
                else "ordinary_h50"
            ),
            "base_flow_objective": "ordinary_h50",
            "preserve_hinge_objective": "ordinary_h50",
            "variant": "standard_zeva_only" if args.decouple_action_expert_gradient else None,
            "residual_on": "ZeVA-only total loss; action-expert parameter hooks return zero",
            "residual_off": "current student foundation-only flow; action-expert gradients pass through",
            "nll_shared_input_gradient": (
                "detached" if args.decouple_action_expert_gradient else "configured_by_prior_nll_detach_context"
            ),
            "same_processed_batch": True,
            "same_foundation_rng_noise_time": True,
            "post_pair_rng": "one ordinary current-student residual-off foundation forward",
            "ddp_zero_link_for_zeva": args.decouple_action_expert_gradient,
            "hooks_registered_before_accelerator_prepare": args.decouple_action_expert_gradient,
            "gradient_clipping": {
                "max_norm": 1.0,
                "scope": (
                    "independent_optimizer_groups"
                    if args.decouple_action_expert_gradient
                    else "all_trainable_parameters"
                ),
                "action_expert": (
                    "independent_global_norm"
                    if args.decouple_action_expert_gradient
                    else "shared_global_norm"
                ),
                "zeva": (
                    "independent_global_norm"
                    if args.decouple_action_expert_gradient
                    else "shared_global_norm"
                ),
                "matches_ordinary_base_action_expert_clip": args.decouple_action_expert_gradient,
            },
        },
        "video_decode": {
            "backend": args.video_backend,
            "persistent_decoder_cache": args.video_backend == "torchcodec",
        },
        "paired_baseline_preservation": {
            "enabled": paired_teacher_enabled,
            "teacher": (
                "independent_untouched_foundation_path"
                if args.anchor_foundation_checkpoint is not None
                else (
                    "independent_frozen_base_action_path"
                    if args.anchor_stage2_checkpoint is not None
                    else ("current_student_residual_off" if paired_teacher_enabled else "none")
                )
            ),
            "sampling": (
                "fixed_optimizer_step_interval" if paired_teacher_enabled else "none"
            ),
            "interval": args.baseline_preserve_interval if paired_teacher_enabled else None,
            "sampled_loss_scale": (
                args.baseline_preserve_interval if paired_teacher_enabled else None
            ),
            "validation": (
                "every_batch_matched_teacher" if paired_teacher_enabled else "self_flow"
            ),
            "same_noise_rng_replay": paired_teacher_enabled,
            "hinge_scope": "per_example_no_cross_task_cancellation" if paired_teacher_enabled else None,
            "positive_improvement_margin": (
                args.paired_improvement_margin if paired_teacher_enabled else None
            ),
        },
        "causal_context_residual": {
            "source": "memory_context_encoder" if stage1_prediction_enabled else "disabled",
            "target": (
                "disabled"
                if not stage1_prediction_enabled or action_expert_control
                else "post_diffusion_action_residual_h15"
                if output_residual
                else
                "prior_conditioning_only"
                if prior_only
                else "noisy_action_embedding"
            ),
            "direct_injection_enabled": (
                False
                if not stage1_prediction_enabled
                or action_expert_control
                or output_correction
                or output_residual
                else not prior_only
            ),
            "broadcast_horizon": 50,
            "gate": (
                "disabled"
                if not stage1_prediction_enabled or prior_only or action_expert_control
                else "bounded_task_language_plus_recurrent_h15_phase_router"
            ),
        },
        "action_prior": {
            "distribution": "disabled" if not stage1_prediction_enabled or action_expert_control else "diagonal_gaussian",
            "parameterization": None if not stage1_prediction_enabled or action_expert_control else "mean_and_log_std",
            "log_std_range": None if not stage1_prediction_enabled or action_expert_control else [-5.0, 2.0],
            "loss": "disabled" if not stage1_prediction_enabled or action_expert_control else "negative_log_likelihood_sum_action_mean_executed_horizon",
            "supervision_horizon": None if not stage1_prediction_enabled or action_expert_control else args.prior_injection_horizon,
            "residual_source": None if not stage1_prediction_enabled or action_expert_control else "mean",
            "residual_target": (
                None
                if not stage1_prediction_enabled or action_expert_control
                else "post_diffusion_h15_expert_minus_base_delta"
                if output_residual
                else "post_diffusion_h15_output"
                if output_correction
                else "noisy_action_embedding"
            ),
            "residual_dropout_probability": 0.0 if not stage1_prediction_enabled or action_expert_control else args.prior_residual_dropout_probability,
            "injection_horizon": None if not stage1_prediction_enabled or action_expert_control else args.prior_injection_horizon,
            "scalar_gate": (
                "disabled"
                if not stage1_prediction_enabled or action_expert_control
                else "direct_bounded_residual_with_floor"
                if output_residual
                else "learned_per_step_output_gate"
                if output_correction
                else ("fixed_0.5" if prior_only else "learned_sigmoid")
            ),
        },
        "output_action_correction": {
            "enabled": args.training_variant == "output_correction",
            "target": "post_diffusion_normalized_eef16",
            "horizon": args.prior_injection_horizon if args.training_variant == "output_correction" else None,
            "operator": "bounded_convex_interpolation_between_exact_base_and_gaussian_prior_mean",
            "identity_fallback": "gate_zero_is_bitwise_base_before_postprocessor",
            "selection_metric": "heldout_paired_h15_action_mse_not_flow_loss",
            "token_injection": False,
        },
        "output_residual_correction": {
            "enabled": output_residual,
            "target": "post_diffusion_normalized_eef16",
            "horizon": args.prior_injection_horizon if output_residual else None,
            "operator": "base_plus_bounded_direct_residual_no_prior_lerp",
            "residual_bound": args.residual_bound if output_residual else None,
            "gate_floor": 0.05 if output_residual else None,
            "identity_fallback": "zero_initialized_residual_is_exact_base",
            "h35": "exact_base_copy",
            "selection_metric": "heldout_paired_h15_action_mse_not_flow_loss",
            "deployment_scale": "per_task_validation_selected_in_[0,1], zero_is_exact_base",
            "token_injection": False,
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
            "output_correction_gate",
            "output_residual_corrector",
        ] if args.training_variant in {"baseline", "action_expert_control"} else []) + ([
            "pi05_gemma_action_expert",
            "pi05_action_input_output_and_time_projections",
        ] if frozen_foundation else []) + (([
            "causal_action_projector",
            "context_gate_logit",
            "prior_gate_logit",
        ] + (["residual_gate_router"] if args.training_variant == "prior_zeva" else []))
        if prior_only else []) + ([
            "causal_action_projector",
            "prior_action_projector",
            "context_gate_logit",
            "prior_gate_logit",
            "residual_gate_router",
        ] if output_correction else []) + ([
            "causal_action_projector",
            "prior_action_projector",
            "context_gate_logit",
            "prior_gate_logit",
            "residual_gate_router",
            "output_correction_gate",
        ] if output_residual else []),
        "trainable": ([
            "task_token_projector",
            "memory_context_encoder",
            "action_prior",
            "output_correction_gate",
        ] if output_correction else ( [
            "task_token_projector",
            "memory_context_encoder",
            "action_prior",
            "output_residual_corrector",
        ] if output_residual else ([] if frozen_foundation else [
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
        ] if zeva_enabled else [])))),
        "parameter_contract": (
            {
                "trainable_scope": "pi05_action_expert_only",
                "adapter_parameters_frozen": True,
                "teacher_in_optimizer": False,
                "pbd_loss": "per_example_flow_hinge_against_immutable_same_noise_teacher",
                "action_output_horizon": 50,
                "flow_supervision_horizon": 50,
                "pbd_horizon": 50,
                "decision_stride_horizon": 15,
                "supervision_contract": (
                    "same H50 PI flow/PBD objective as v11 action expert; "
                    "dataset decisions and deployment execute every H15"
                ),
                "execution_horizon": 15,
                "gaussian_nll_enabled": False,
                "checkpoint_contract": {
                    "full_model_safetensors": True,
                    "optimizer_state_dict": True,
                    "adapter_checkpoint": False,
                },
            }
            if action_expert_control
            else None
        ),
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
    diagnostics_enabled = bool(getattr(args, "validation_diagnostics", False))
    if diagnostics_enabled and getattr(accelerator, "num_processes", 1) != 1:
        # Optional raw-forward failures may differ across ranks. Until their
        # availability is synchronized, conditional metric gathers are unsafe.
        raise ValueError("Validation diagnostics currently require a single process.")
    diagnostic_values: dict[str, list[torch.Tensor]] = {}
    diagnostic_reasons: list[str] = []
    diagnostic_mask_sources: list[str] = []
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
        "residual_regression": [],
        "trust_region": [],
        "residual_abs": [],
        "prior_residual_keep": [],
        "gate": [],
    }
    paired_by_task: dict[int, dict[str, list[torch.Tensor]]] = {}
    for batch_index, raw_batch in enumerate(loader):
        if batch_index >= args.eval_batches:
            break
        raw_batch.pop("zeva.sample_index")
        task_ids = raw_batch.pop("zeva.task_id")
        phase_queries = raw_batch.pop("zeva.phase_query")
        live_brief = raw_batch.pop("zeva.live_brief")
        live_brief_mask = raw_batch.pop("zeva.live_brief_mask")
        live_retrieved = raw_batch.pop("zeva.live_retrieved")
        live_retrieved_mask = raw_batch.pop("zeva.live_retrieved_mask")
        if args.training_variant in {"output_correction", "output_residual"} and args.base_action_cache is not None:
            processed, cached_base_actions, target_actions, task_goal_embedding = (
                _prepare_cached_output_correction_batch(policy, raw_batch)
            )
        else:
            processed = _preprocess_with_task_only_goal(policy, preprocessor, raw_batch)
            cached_base_actions = target_actions = task_goal_embedding = None
        diagnostic_rng_state = None
        diagnostic_confidence = None
        diagnostic_phase_token = None
        if args.training_variant == "action_expert_control":
            baseline_flow, foundation_rng_state = _matched_baseline_flow(policy, processed)
            diagnostic_rng_state = foundation_rng_state
            losses = _action_expert_control_losses(
                policy,
                processed,
                baseline_flow,
                foundation_rng_state,
                args.preserve_loss_weight,
                paired_improvement_margin=0.0,
                preserve_scale=1.0,
            )
            retrieval_accuracy = losses["flow"].detach().new_full((), float("nan"))
        elif args.training_variant in {
            "zeva",
            "adapter",
            "prior_adapter",
            "prior_zeva",
            "output_correction",
            "output_residual",
        }:
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
            diagnostic_confidence = confidence
            diagnostic_phase_token = bank_batch.phase_token
            if args.training_variant == "output_correction":
                # Fixed validation batches receive a deterministic PI diffusion
                # stream; Base and corrected metrics reuse the same sampled chunk.
                torch.manual_seed(args.seed + 100_000 + batch_index + accelerator.process_index * 10_000)
                torch.cuda.manual_seed_all(
                    args.seed + 100_000 + batch_index + accelerator.process_index * 10_000
                )
                losses = _output_correction_losses(
                    policy,
                    processed,
                    bank_batch,
                    confidence,
                    prior_weight=args.prior_loss_weight,
                    preserve_weight=args.preserve_loss_weight,
                    gate_regularization_weight=args.gate_regularization_weight,
                    paired_improvement_margin=0.0,
                    correction_horizon=args.prior_injection_horizon,
                    cached_base_actions=cached_base_actions,
                    task_goal_embedding=task_goal_embedding,
                    target_actions=target_actions,
                )
            elif args.training_variant == "output_residual":
                losses = _output_residual_losses(
                    policy,
                    processed,
                    bank_batch,
                    confidence,
                    prior_weight=args.prior_loss_weight,
                    preserve_weight=args.preserve_loss_weight,
                    gate_regularization_weight=args.gate_regularization_weight,
                    paired_improvement_margin=0.0,
                    correction_horizon=args.prior_injection_horizon,
                    residual_regression_weight=args.residual_regression_weight,
                    residual_trust_region_weight=args.residual_trust_region_weight,
                    residual_trust_region_radius=args.residual_trust_region_radius,
                    residual_bound=args.residual_bound,
                    cached_base_actions=cached_base_actions,
                    task_goal_embedding=task_goal_embedding,
                    target_actions=target_actions,
                )
            else:
                baseline_flow, foundation_rng_state = _matched_baseline_flow(policy, processed)
                diagnostic_rng_state = foundation_rng_state
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
                    prior_nll_detach_context=args.prior_nll_detach_context,
                    training=False,
                    return_diagnostics=diagnostics_enabled,
                )
        else:
            losses = _baseline_losses(policy, processed)
            retrieval_accuracy = losses["flow"].detach().new_full((), float("nan"))
        if diagnostics_enabled:
            diagnostic = _collect_validation_diagnostics(
                policy,
                processed,
                losses,
                foundation_rng_state=diagnostic_rng_state,
                injection_confidence=diagnostic_confidence,
                phase_token=diagnostic_phase_token,
                training_variant=args.training_variant,
            )
            mask_source = diagnostic.get("mask_source")
            if mask_source is not None:
                diagnostic_mask_sources.append(str(mask_source))
            reason = diagnostic.get("reason")
            if reason:
                diagnostic_reasons.append(str(reason))

            def append_diagnostic(name: str, value: torch.Tensor | None) -> None:
                if value is None:
                    return
                gathered = accelerator.gather_for_metrics(value.detach().reshape(-1)).cpu()
                diagnostic_values.setdefault(name, []).append(gathered)

            append_diagnostic("student_h50", diagnostic.get("_student_h50"))
            append_diagnostic("student_h15", diagnostic.get("_student_h15"))
            append_diagnostic("current_h50", diagnostic.get("_current_h50"))
            append_diagnostic("current_h15", diagnostic.get("_current_h15"))
            append_diagnostic("fixed_h50", diagnostic.get("_fixed_h50"))
            append_diagnostic("fixed_h15", diagnostic.get("_fixed_h15"))
            append_diagnostic("valid_steps_h50", diagnostic.get("_valid_steps_h50"))
            append_diagnostic("valid_steps_h15", diagnostic.get("_valid_steps_h15"))
            for horizon_name in ("h50", "h15"):
                residual_values = diagnostic.get(f"_residual_{horizon_name}")
                if residual_values is None:
                    continue
                for metric in (
                    "reference_norm",
                    "context_delta_norm",
                    "prior_delta_norm",
                    "total_delta_norm",
                    "context_relative_norm",
                    "prior_relative_norm",
                    "total_relative_norm",
                ):
                    append_diagnostic(
                        f"residual_{horizon_name}_{metric}",
                        residual_values.get(metric),
                    )
                append_diagnostic(
                    f"residual_{horizon_name}_valid_steps",
                    residual_values.get("valid_steps"),
                )
        if args.training_variant in {
            "zeva",
            "adapter",
            "prior_adapter",
            "prior_zeva",
            "output_correction",
            "output_residual",
            "action_expert_control",
        }:
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
            "residual_regression",
            "trust_region",
            "residual_abs",
            "prior_residual_keep",
            "gate",
        ):
            if name in losses:
                totals[name].append(
                    accelerator.gather_for_metrics(losses[name].detach().reshape(1))
                )
        totals["retrieval_accuracy"].append(
            accelerator.gather_for_metrics(retrieval_accuracy.detach().reshape(1))
        )
    result = {
        name: float(torch.cat(values).mean()) if values else float("nan")
        for name, values in totals.items()
    }
    per_task = {}
    for task_id, values in sorted(paired_by_task.items()):
        flow = torch.cat(values["flow"])
        baseline = torch.cat(values["baseline"])
        delta = baseline - flow
        per_task[bank.task_names[task_id]] = {
            "count": int(delta.numel()),
            "corrected_h15_mse": float(flow.mean()),
            "base_h15_mse": float(baseline.mean()),
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
    if diagnostics_enabled:
        result["validation_diagnostics"] = _finalize_validation_diagnostics(
            diagnostic_values,
            reasons=diagnostic_reasons,
            mask_sources=diagnostic_mask_sources,
        )
    return result


def main(args: Args) -> None:
    runtime_versions = _validated_runtime_versions()
    if args.training_variant not in {
        "zeva",
        "adapter",
        "prior_adapter",
        "prior_zeva",
            "output_correction",
            "output_residual",
            "action_expert_control",
        "baseline",
    }:
        raise ValueError(
            "training_variant must be 'zeva', 'adapter', 'prior_adapter', 'prior_zeva', "
            "'output_correction', 'output_residual', 'action_expert_control', or 'baseline'."
        )
    zeva_enabled = args.training_variant in {
        "zeva",
        "adapter",
        "prior_adapter",
        "prior_zeva",
        "output_correction",
        "output_residual",
    }
    if args.prior_nll_detach_context and args.training_variant != "zeva":
        raise ValueError("NLL context detachment is restricted to the standard zeva variant")
    _validate_action_expert_gradient_routing_contract(
        args.training_variant,
        args.decouple_action_expert_gradient,
        args.prior_nll_detach_context,
    )
    if args.zeva_h15_flow_objective and not args.decouple_action_expert_gradient:
        raise ValueError("The ZeVA H15 objective requires protected action-expert gradient routing.")
    prior_only = args.training_variant in {"prior_adapter", "prior_zeva"}
    action_expert_control = args.training_variant == "action_expert_control"
    output_correction = args.training_variant == "output_correction"
    output_residual = args.training_variant == "output_residual"
    paired_teacher_enabled = (
        zeva_enabled and not output_correction and not output_residual
    ) or action_expert_control
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
    elif args.training_variant == "output_correction":
        if args.prior_injection_horizon != 15:
            raise ValueError("Formal v13 output correction requires H15.")
        trainable = policy.configure_output_action_correction_stage2(
            correction_horizon=args.prior_injection_horizon
        )
    elif args.training_variant == "output_residual":
        if args.prior_injection_horizon != 15:
            raise ValueError("Formal v14 output residual requires H15.")
        trainable = policy.configure_output_residual_stage2(
            correction_horizon=args.prior_injection_horizon,
            residual_bound=args.residual_bound,
        )
    elif args.training_variant == "action_expert_control":
        trainable = policy.configure_action_expert_only_finetune()
        # Persist the same H15 decision boundary used by the v11 deployment
        # contract even though this control has no residual to truncate.
        policy.configure_prior_injection_horizon(args.prior_injection_horizon)
        with torch.no_grad():
            # Keep the disabled residual path numerically inert even if a
            # future caller accidentally supplies ZeVA inputs to this control.
            policy.causal_action_projector.weight.zero_()
            policy.causal_action_projector.bias.zero_()
            policy.prior_action_projector.weight.zero_()
            policy.prior_action_projector.bias.zero_()
            policy.context_gate_logit.fill_(-20.0)
            policy.prior_gate_logit.fill_(-20.0)
            for parameter in policy.residual_gate_router.parameters():
                parameter.zero_()
        policy._direct_context_injection_enabled = False  # noqa: SLF001
        if policy._direct_context_injection_enabled:  # noqa: SLF001
            raise RuntimeError("action_expert_control invariant failed: direct context is enabled.")
        if any(
            parameter.requires_grad
            for module in (
                policy.task_token_projector,
                policy.memory_context_encoder,
                policy.action_prior,
                policy.causal_action_projector,
                policy.prior_action_projector,
                policy.residual_gate_router,
            )
            for parameter in module.parameters()
        ) or policy.context_gate_logit.requires_grad or policy.prior_gate_logit.requires_grad:
            raise RuntimeError("action_expert_control invariant failed: a ZeVA parameter is trainable.")
    else:
        trainable = policy.configure_action_expert_only_finetune()
    # Prior-only variants set the deployed 0.5 guidance inside their
    # configure_* method.  Do not overwrite that fixed value with the legacy
    # fresh-dual-residual initialization probability.
    if zeva_enabled and not prior_only and not output_residual and args.resume_checkpoint is None:
        policy.initialize_residual_gate_probability(
            args.initial_residual_gate_probability
        )
    if args.anchor_stage2_checkpoint is not None:
        if args.training_variant != "zeva":
            raise ValueError("An independent Stage 2 Base anchor is only valid for joint ZeVA training.")
        policy.load_foundation_anchor(args.anchor_stage2_checkpoint)
    if args.anchor_foundation_checkpoint is not None:
        if args.training_variant not in {"prior_zeva", "action_expert_control"}:
            raise ValueError(
                "An untouched foundation anchor is only valid for prior_zeva or "
                "action_expert_control."
            )
        if Path(args.anchor_foundation_checkpoint).resolve() != handoff.checkpoint.resolve():
            raise ValueError(
                "The immutable anchor must be the same untouched foundation checkpoint "
                "used to initialize the action expert."
            )
        policy.load_foundation_anchor(args.anchor_foundation_checkpoint)
    elif args.training_variant in {"prior_zeva", "action_expert_control"}:
        raise ValueError(
            f"{args.training_variant} requires --anchor-foundation-checkpoint."
        )
    if action_expert_control:
        if args.foundation_checkpoint is None:
            raise ValueError("action_expert_control requires explicit untouched --foundation-checkpoint.")
        foundation_model = Path(args.foundation_checkpoint).resolve() / "model.safetensors"
        if _sha256(foundation_model) != UNTOUCHED_BEST_V1_MODEL_SHA256:
            raise ValueError(
                "action_expert_control requires untouched best-v1 model.safetensors; "
                f"got {_sha256(foundation_model)}."
            )
        if args.initial_stage2_checkpoint is not None:
            raise ValueError(
                "action_expert_control must initialize directly from untouched best-v1; "
                "--initial-stage2-checkpoint is forbidden."
            )
        if args.goal_embedding_checkpoint is not None:
            raise ValueError(
                "action_expert_control must use the untouched best-v1 language table; "
                "--goal-embedding-checkpoint is forbidden."
            )
        if Path(args.anchor_foundation_checkpoint).resolve() != handoff.checkpoint.resolve():
            raise ValueError(
                "action_expert_control requires the immutable teacher to be the same "
                "untouched foundation checkpoint used for initialization."
            )
        if args.prior_loss_weight != 0.0:
            raise ValueError("action_expert_control requires --prior-loss-weight 0.")
        if args.prior_residual_dropout_probability != 0.0:
            raise ValueError(
                "action_expert_control requires --prior-residual-dropout-probability 0."
            )
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
    valid_action_expert_lr = (
        args.action_expert_learning_rate == 0
        if args.zeva_h15_flow_objective
        else 0 < args.action_expert_learning_rate < args.learning_rate
    )
    if not valid_action_expert_lr:
        raise ValueError(
            "Action-expert LR must be zero for the explicitly Base-locked H15 experiment, "
            "or positive and below the ZeVA LR for ordinary Stage 2."
        )
    if not math.isfinite(args.prior_loss_weight) or args.prior_loss_weight < 0 or (
        args.prior_loss_weight == 0 and not (action_expert_control or output_residual)
    ):
        raise ValueError(
            "Gaussian action-prior NLL weight must be non-negative and may be zero "
            "only for action_expert_control or direct output_residual."
        )
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
    if action_expert_control and args.prior_injection_horizon != 15:
        raise ValueError("action_expert_control requires the H15 execution/supervision contract.")
    if output_residual:
        if not math.isfinite(args.residual_bound) or args.residual_bound <= 0:
            raise ValueError("v14 residual_bound must be finite and positive.")
        if not math.isfinite(args.residual_regression_weight) or args.residual_regression_weight < 0:
            raise ValueError("v14 residual_regression_weight must be finite and non-negative.")
        if not math.isfinite(args.residual_trust_region_weight) or args.residual_trust_region_weight < 0:
            raise ValueError("v14 residual_trust_region_weight must be finite and non-negative.")
        if not math.isfinite(args.residual_trust_region_radius) or args.residual_trust_region_radius <= 0:
            raise ValueError("v14 residual_trust_region_radius must be finite and positive.")
    if output_correction and args.base_action_cache is None:
        raise ValueError("Formal v13 output correction requires --base-action-cache.")
    if output_residual and args.base_action_cache is None:
        raise ValueError("Formal v14 output residual requires --base-action-cache.")
    if not (output_correction or output_residual) and args.base_action_cache is not None:
        raise ValueError(
            "A Base action cache is valid only for output_correction or output_residual."
        )
    if output_correction or output_residual:
        base_cache = torch.load(args.base_action_cache, map_location="cpu", weights_only=False)
        cache_schema = base_cache.get("schema")
        if cache_schema not in {
            "zeva-robotwin-untouched-base-action-cache-v1",
            "zeva-robotwin-stage2-base-action-cache-v2",
        }:
            raise ValueError("Output correction Base action cache has the wrong schema.")
        if base_cache.get("foundation_model_sha256") != _sha256(
            handoff.checkpoint / "model.safetensors"
        ):
            raise ValueError("Base action cache was generated by a different PI foundation.")
        if cache_schema == "zeva-robotwin-stage2-base-action-cache-v2":
            if args.initial_stage2_checkpoint is None:
                raise ValueError("Stage2 Base action cache requires the exact initial Base checkpoint.")
            selected_base = Path(args.initial_stage2_checkpoint).resolve()
            declared_base = base_cache.get("stage2_checkpoint")
            if not declared_base or Path(declared_base).resolve() != selected_base:
                raise ValueError("Stage2 Base action cache names a different initial Base checkpoint.")
            if base_cache.get("selected_model_sha256") != _sha256(
                selected_base / "model.safetensors"
            ):
                raise ValueError("Stage2 Base action cache model SHA differs from initial Base.")
            if base_cache.get("goal_embedding_model_sha256") != _sha256(
                handoff.checkpoint / "model.safetensors"
            ):
                raise ValueError("Stage2 Base cache task language differs from deployed Base.")
            manifest["dataset_replica_proof"] = _verify_cached_dataset_replica(
                cache_adapter_sha256=base_cache.get("dataset_adapter_sha256", ""),
                current_adapter=Path(args.dataset_root) / "adapter.json",
                source_report=args.dataset_identity_source_report,
                replica_report=args.dataset_identity_replica_report,
                cache_only=args.cache_only_dataset_replica,
            )
        elif args.dataset_identity_source_report or args.dataset_identity_replica_report:
            raise ValueError("Dataset replica identity reports require the Stage2 Base action cache.")
        if base_cache.get("live_queries_sha256") != _sha256(args.live_queries):
            raise ValueError("v13 Base action cache uses different Stage1 live queries.")
        if args.task_subset is None or base_cache.get("task_subset_sha256") != _sha256(args.task_subset):
            raise ValueError("v13 Base action cache uses a different task subset.")
    if args.compile_mode not in {"default", "reduce-overhead", "max-autotune"}:
        raise ValueError(f"Unsupported torch.compile mode: {args.compile_mode!r}.")
    preprocessor = policy.preprocessor
    config = policy.zeva_config
    adapter_manifest = Path(args.dataset_root) / "adapter.json"
    live_cache = torch.load(args.live_queries, map_location="cpu")
    zte_for_live = torch.load(args.zte_checkpoint, map_location="cpu")
    live_stage1_schema = stage1_artifact_schema(live_cache)
    expected_stage1_schema = zte_for_live.get("schema")
    if expected_stage1_schema not in LEGACY_SCHEMAS | {V2_SCHEMA}:
        raise ValueError("Live queries reference an unsupported Stage 1 checkpoint schema.")
    if expected_stage1_schema == V2_SCHEMA:
        validate_stage1_v2_artifact_status(live_cache, artifact_name="live-query cache")
    if expected_stage1_schema == V2_SCHEMA and live_stage1_schema != V2_SCHEMA:
        raise ValueError(
            "Stage 2 v2 requires live queries with explicit "
            "stage1_checkpoint_schema=v2."
        )
    if live_stage1_schema is not None and live_stage1_schema != expected_stage1_schema:
        raise ValueError("Live queries were exported from a different Stage 1 encoder schema.")
    if live_cache.get("zte_checkpoint_sha256") != _sha256(args.zte_checkpoint):
        raise ValueError("Live queries were not exported from the selected ZTE checkpoint.")
    if live_cache.get("statistics_sha256") != _sha256(handoff.statistics):
        raise ValueError("Live queries use different PI0.5 normalization statistics.")
    live_horizon = live_cache.get("transition_horizon")
    expected_horizon = stage1_transition_horizon(zte_for_live)
    if expected_stage1_schema == V2_SCHEMA and live_horizon is None:
        raise ValueError("Stage 2 v2 requires live queries with an explicit transition horizon.")
    if live_horizon is not None and int(live_horizon) != expected_horizon:
        raise ValueError("Live queries use a different Stage 1 transition horizon.")
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
        base_action_cache=args.base_action_cache if (output_correction or output_residual) else None,
    )
    validation_dataset = RobotWinStage2Dataset(
        adapter_manifest,
        args.live_queries,
        subset="validation",
        config=config,
        selected_tasks=selected_tasks,
        video_backend=args.video_backend,
        decoder_threads=args.decoder_threads,
        base_action_cache=args.base_action_cache if (output_correction or output_residual) else None,
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
    action_expert_gradient_router: _ActionExpertGradientRouter | None = None
    if args.training_variant not in {
        "adapter",
        "prior_adapter",
        "output_correction",
        "output_residual",
    } and not action_expert_parameters:
        raise RuntimeError("Stage 2 requires a non-empty action-expert optimizer group.")
    if zeva_enabled and not zeva_parameters:
        raise RuntimeError("Zeva Stage 2 requires a non-empty Zeva optimizer group.")
    if args.training_variant == "baseline" and zeva_parameters:
        raise RuntimeError("Matched PI baseline must not contain trainable Zeva parameters.")
    if args.decouple_action_expert_gradient:
        # Register on the unwrapped parameters before ``accelerator.prepare``
        # adds DDP's reducer hooks.  This ordering is part of the routing
        # contract: DDP observes the action expert as used, but receives zero
        # from the residual-on backward.
        action_expert_gradient_router = _ActionExpertGradientRouter(
            action_expert_parameters
        )
    if action_expert_control:
        if zeva_parameters:
            raise RuntimeError(
                "action_expert_control invariant failed: adapter/prior parameters are trainable."
            )
        trainable_names = _assert_action_expert_control_parameters(policy, trainable)
        manifest["parameter_contract"]["actual_trainable_parameter_count"] = len(trainable_names)
        manifest["parameter_contract"]["actual_trainable_parameter_names_sha256"] = hashlib.sha256(
            "\n".join(trainable_names).encode("utf-8")
        ).hexdigest()
        manifest["parameter_contract"]["teacher_parameter_count"] = len(
            policy._foundation_anchor_parameters  # noqa: SLF001
        )
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
    if action_expert_control:
        _assert_action_expert_control_optimizer(
            optimizer,
            trainable,
            policy._foundation_anchor_parameters,  # noqa: SLF001
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
            "output_correction": "zeva-robotwin-stage2-output-correction-training-state-v1",
            "output_residual": "zeva-robotwin-stage2-output-residual-training-state-v1",
            "action_expert_control": "zeva-robotwin-stage2-action-expert-control-training-state-v1",
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
        checkpoint_routing = bool(
            checkpoint["manifest"].get("train_args", {}).get(
                "decouple_action_expert_gradient", False
            )
        )
        if checkpoint_routing != args.decouple_action_expert_gradient:
            raise ValueError(
                "Stage 2 resume checkpoint uses a different action-expert gradient-routing "
                f"contract ({checkpoint_routing} versus {args.decouple_action_expert_gradient})."
            )
        checkpoint_h15_objective = bool(
            checkpoint["manifest"].get("train_args", {}).get("zeva_h15_flow_objective", False)
        )
        if checkpoint_h15_objective != args.zeva_h15_flow_objective:
            raise ValueError("Stage 2 resume checkpoint uses a different ZeVA flow horizon objective.")
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

    # PI05Policy calls core.forward directly, bypassing module forward hooks.
    # Install this training-only capture outside the compiled callable so the
    # connected H15 loss is from the *same* PI forward, noise, and timestep.
    executed_flow_capture = (
        _ConnectedRawFlowCapture(core) if args.zeva_h15_flow_objective else None
    )

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
        if action_expert_gradient_router is not None and step == start_step:
            action_expert_gradient_router.begin_first_step_audit()
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
        sample_baseline = paired_teacher_enabled and step % args.baseline_preserve_interval == 0
        for micro_step in range(args.gradient_accumulation_steps):
            try:
                raw_batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                raw_batch = next(iterator)
            raw_batch.pop("zeva.sample_index")
            task_ids = raw_batch.pop("zeva.task_id")
            phase_queries = raw_batch.pop("zeva.phase_query")
            live_brief = raw_batch.pop("zeva.live_brief")
            live_brief_mask = raw_batch.pop("zeva.live_brief_mask")
            live_retrieved = raw_batch.pop("zeva.live_retrieved")
            live_retrieved_mask = raw_batch.pop("zeva.live_retrieved_mask")
            if args.training_variant in {"output_correction", "output_residual"} and args.base_action_cache is not None:
                processed, cached_base_actions, target_actions, task_goal_embedding = (
                    _prepare_cached_output_correction_batch(policy, raw_batch)
                )
            else:
                processed = _preprocess_with_task_only_goal(policy, preprocessor, raw_batch)
                cached_base_actions = target_actions = task_goal_embedding = None
            # Capture before Stage-1 phase/memory dropout.  The routed pair
            # restores this state before each foundation call and leaves the
            # global stream after exactly one current-student Base forward.
            routing_rng_state = (
                _diagnostic_rng_state(processed["action"].device)
                if action_expert_gradient_router is not None
                else None
            )
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
            # Routed residual-on backward must never synchronize: the matching
            # residual-off backward is the one that performs the final DDP
            # reduction for this micro-step (and any accumulated micro-steps).
            # The ordinary path keeps the historical final-micro-step sync.
            sync_context = (
                policy.no_sync()
                if hasattr(policy, "no_sync")
                and (
                    action_expert_gradient_router is not None
                    or micro_step + 1 < args.gradient_accumulation_steps
                )
                else nullcontext()
            )
            with sync_context:
                if args.training_variant == "baseline":
                    micro_losses = _baseline_losses(policy, processed)
                elif args.training_variant == "action_expert_control":
                    if sample_baseline:
                        baseline_flow, foundation_rng_state = _matched_baseline_flow(policy, processed)
                    else:
                        baseline_flow, foundation_rng_state = None, None
                    micro_losses = _action_expert_control_losses(
                        policy,
                        processed,
                        baseline_flow,
                        foundation_rng_state,
                        args.preserve_loss_weight,
                        paired_improvement_margin=args.paired_improvement_margin,
                        preserve_scale=(
                            float(args.baseline_preserve_interval) if sample_baseline else 1.0
                        ),
                    )
                elif args.training_variant == "output_correction":
                    micro_losses = _output_correction_losses(
                        policy,
                        processed,
                        bank_batch,
                        confidence,
                        prior_weight=args.prior_loss_weight,
                        preserve_weight=args.preserve_loss_weight,
                        gate_regularization_weight=args.gate_regularization_weight,
                        paired_improvement_margin=args.paired_improvement_margin,
                        correction_horizon=args.prior_injection_horizon,
                        cached_base_actions=cached_base_actions,
                        task_goal_embedding=task_goal_embedding,
                        target_actions=target_actions,
                    )
                elif args.training_variant == "output_residual":
                    micro_losses = _output_residual_losses(
                        policy,
                        processed,
                        bank_batch,
                        confidence,
                        prior_weight=args.prior_loss_weight,
                        preserve_weight=args.preserve_loss_weight,
                        gate_regularization_weight=args.gate_regularization_weight,
                        paired_improvement_margin=args.paired_improvement_margin,
                        correction_horizon=args.prior_injection_horizon,
                        residual_regression_weight=args.residual_regression_weight,
                        residual_trust_region_weight=args.residual_trust_region_weight,
                        residual_trust_region_radius=args.residual_trust_region_radius,
                        residual_bound=args.residual_bound,
                        cached_base_actions=cached_base_actions,
                        task_goal_embedding=task_goal_embedding,
                        target_actions=target_actions,
                    )
                else:
                    if sample_baseline:
                        if routing_rng_state is not None:
                            _restore_diagnostic_rng_state(
                                routing_rng_state, processed["action"].device
                            )
                        baseline_flow, foundation_rng_state = _matched_baseline_flow(policy, processed)
                    else:
                        baseline_flow, foundation_rng_state = None, routing_rng_state
                    if action_expert_gradient_router is None:
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
                            prior_nll_detach_context=args.prior_nll_detach_context,
                            training=True,
                        )
                        accelerator.backward(micro_losses["total"] / args.gradient_accumulation_steps)
                    else:
                        action_expert_gradient_router.set_phase("residual_on")
                        micro_losses, _pair_rng_state = _gradient_routed_on_forward(
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
                            args.paired_improvement_margin,
                            float(args.baseline_preserve_interval)
                            if sample_baseline
                            else 1.0,
                            args.prior_injection_horizon,
                            executed_flow_capture=executed_flow_capture,
                        )
                        # Complete the residual-on backward before constructing
                        # the second DDP forward.  Building two forwards first
                        # leaves reducer buckets rank-local under NCCL.
                        accelerator.backward(
                            micro_losses["total"] / args.gradient_accumulation_steps
                        )
                        if step == start_step and micro_step == 0:
                            _assert_gradient_routed_zeva_gradients(
                                accelerator.unwrap_model(policy)
                            )
                if args.training_variant in {
                    "baseline",
                    "action_expert_control",
                    "output_correction",
                    "output_residual",
                }:
                    # These direct-loss branches do not call backward inside
                    # their own branch.  Without this, optimizer.step() is a
                    # silent no-op and the output corrector never trains.
                    accelerator.backward(
                        micro_losses["total"] / args.gradient_accumulation_steps
                    )
            if action_expert_gradient_router is not None:
                action_expert_gradient_router.set_phase("residual_off")
                # Only the final off backward synchronizes.  Its zero-valued
                # ZeVA links mark the parameters as used while preserving the
                # ZeVA gradients accumulated by the residual-on backward.
                off_sync_context = (
                    policy.no_sync()
                    if micro_step + 1 < args.gradient_accumulation_steps
                    and hasattr(policy, "no_sync")
                    else nullcontext()
                )
                with off_sync_context:
                    off_total, off_flow = _gradient_routed_off_forward(
                        policy,
                        processed,
                        _pair_rng_state,
                        zeva_parameters,
                    )
                    accelerator.backward(off_total / args.gradient_accumulation_steps)
                micro_losses["residual_off_flow"] = off_flow.detach()
            if step == start_step and micro_step == 0:
                if action_expert_gradient_router is not None:
                    action_expert_gradient_router.assert_first_step_audit()
                    action_expert_gradient_router.end_first_step_audit()
                if args.training_variant == "output_correction":
                    _assert_output_correction_gradients(accelerator.unwrap_model(policy))
                elif args.training_variant == "output_residual":
                    # The delta head starts exactly zero; its first update may
                    # be the only connected nonzero path into the corrector.
                    # Require every upstream ZTE-conditioned module at step 2.
                    _assert_output_residual_gradients(
                        accelerator.unwrap_model(policy), allow_zero_init_delay=True
                    )
                else:
                    _assert_action_expert_finetune_gradients(
                        accelerator.unwrap_model(policy),
                        include_zeva=zeva_enabled,
                        action_expert_trainable=args.training_variant
                        not in {"adapter", "prior_adapter"},
                        prior_only=args.training_variant in {"prior_adapter", "prior_zeva"},
                    )
            elif step == start_step + 1 and micro_step == 0 and args.training_variant == "output_residual":
                _assert_output_residual_gradients(accelerator.unwrap_model(policy))
            for name, value in micro_losses.items():
                accumulated_losses.setdefault(name, []).append(value.detach())
            retrieval_accuracies.append(retrieval_accuracy.detach())
        losses = {name: torch.stack(values).mean() for name, values in accumulated_losses.items()}
        retrieval_accuracy = torch.stack(retrieval_accuracies).mean()
        _clip_stage2_gradients(
            accelerator,
            trainable,
            action_expert_parameters=action_expert_parameters,
            zeva_parameters=zeva_parameters,
            decouple_action_expert_gradient=args.decouple_action_expert_gradient,
        )
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
            off_flow = losses.get("residual_off_flow")
            progress_bar.set_postfix(
                loss=f"{float(losses['total'].detach()):.4f}",
                flow=f"{float(losses['flow'].detach()):.4f}",
                h15=(
                    f"{float(losses['training_flow_h15'].detach()):.4f}"
                    if "training_flow_h15" in losses else "off"
                ),
                base=(f"{float(losses['baseline']):.4f}" if baseline_sampled else "skip"),
                off=(f"{float(off_flow.detach()):.4f}" if off_flow is not None else "skip"),
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
                        "output_correction": "zeva-robotwin-stage2-output-correction-training-state-v1",
                        "output_residual": "zeva-robotwin-stage2-output-residual-training-state-v1",
                        "action_expert_control": "zeva-robotwin-stage2-action-expert-control-training-state-v1",
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
                            "train_flow_h15": (
                                float(losses["training_flow_h15"])
                                if "training_flow_h15" in losses else None
                            ),
                            "train_residual_off_flow": (
                                float(losses["residual_off_flow"])
                                if "residual_off_flow" in losses
                                else None
                            ),
                            "train_baseline_flow": (
                                float(losses["baseline"])
                                if bool(float(losses["baseline_sampled"]))
                                else None
                            ),
                            "train_baseline_sampled": bool(float(losses["baseline_sampled"])),
                            "train_prior_nll": float(losses["prior"]),
                            "train_prior_std": float(losses["prior_std"]),
                            "train_prior_residual_keep": float(losses["prior_residual_keep"]),
                            "train_residual_regression": (
                                float(losses["residual_regression"])
                                if "residual_regression" in losses
                                else None
                            ),
                            "train_residual_trust_region": (
                                float(losses["trust_region"])
                                if "trust_region" in losses
                                else None
                            ),
                            "train_residual_abs": (
                                float(losses["residual_abs"])
                                if "residual_abs" in losses
                                else None
                            ),
                            "validation": validation,
                        },
                        indent=2,
                    )
                    + "\n"
                )
            accelerator.wait_for_everyone()
    if action_expert_gradient_router is not None:
        action_expert_gradient_router.close()
    accelerator.end_training()


if __name__ == "__main__":
    main(tyro.cli(Args))
