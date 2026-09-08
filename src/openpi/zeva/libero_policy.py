"""ZeVA wrapper around the selected OpenPI LIBERO PI0.5 checkpoint."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any

import safetensors.torch
import torch
from torch import nn
import torch.nn.functional as F
from tokenizers import Tokenizer

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.zeva.config import ZevaConfig
from openpi.zeva.context import MemoryContextEncoder
from openpi.zeva.memory import CausalMemoryManager
from openpi.zeva.libero_bank import LiberoCausalBank
from openpi.zeva.libero_contract import LIBERO_ACTION_DIM
from openpi.zeva.libero_contract import LIBERO_EXECUTION_HORIZON
from openpi.zeva.libero_contract import LIBERO_MODEL_ACTION_DIM
from openpi.zeva.libero_contract import LIBERO_POLICY_HORIZON
from openpi.zeva.libero_contract import LiberoHandoff
from openpi.zeva.libero_contract import QuantileActionNormalizer
from openpi.zeva.retrieval import CausalRetrievalHead
from openpi.zeva.transition_encoder import CausalTransitionEncoder


@dataclass
class LiberoObservation:
    """Torch-only PI0.5 observation; avoids importing the unused JAX stack."""

    images: dict[str, torch.Tensor]
    image_masks: dict[str, torch.Tensor]
    state: torch.Tensor
    tokenized_prompt: torch.Tensor
    tokenized_prompt_mask: torch.Tensor
    episode_index: torch.Tensor | None = None
    frame_index: torch.Tensor | None = None
    previous_action: torch.Tensor | None = None
    has_previous_action: torch.Tensor | None = None
    token_ar_mask: torch.Tensor | None = None
    token_loss_mask: torch.Tensor | None = None
    task_only_tokens: torch.Tensor | None = None
    task_only_token_mask: torch.Tensor | None = None


@dataclass(frozen=True)
class LiberoPi0Config:
    """The selected handoff's Torch-only PI0.5 architecture."""

    dtype: str = "bfloat16"
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    action_dim: int = LIBERO_MODEL_ACTION_DIM
    action_horizon: int = LIBERO_POLICY_HORIZON
    pi05: bool = True
    use_zeva: bool = False
    use_action_prior: bool = False
    schema_dim: int = 256


class LiberoBatchProcessor:
    """Exact task-only PI0.5 preprocessing with frozen quantile statistics."""

    def __init__(self, statistics: str | Path, tokenizer_path: str | Path):
        payload = json.loads(Path(statistics).read_text(encoding="utf-8"))["norm_stats"]
        self.state_q01 = torch.tensor(payload["state"]["q01"], dtype=torch.float32)
        self.state_q99 = torch.tensor(payload["state"]["q99"], dtype=torch.float32)
        self.action_normalizer = QuantileActionNormalizer.from_stats_file(statistics)
        self.tokenizer_path = Path(tokenizer_path).resolve()
        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path / "tokenizer.json"))
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()

    def tokenize(self, prompts: list[str] | tuple[str, ...], states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rows, masks = [], []
        pad = self.tokenizer.token_to_id("<pad>") or 0
        bos = self.tokenizer.token_to_id("<bos>")
        if bos is None:
            raise ValueError("PI0.5 tokenizer has no BOS token.")
        bins = torch.linspace(-1.0, 1.0, 257, device=states.device)[:-1]
        discretized = torch.bucketize(
            states[..., :LIBERO_ACTION_DIM].clamp(-1.0, 1.0), bins, right=True
        ).sub(1).cpu().tolist()
        for prompt, state in zip(prompts, discretized, strict=True):
            cleaned = str(prompt).strip().replace("_", " ").replace("\n", " ")
            state_string = " ".join(map(str, state))
            full_prompt = f"Task: {cleaned}, State: {state_string};\nAction: "
            ids = [bos] + self.tokenizer.encode(full_prompt, add_special_tokens=False).ids
            ids = ids[:200]
            mask = [True] * len(ids)
            rows.append(ids + [pad] * (200 - len(ids)))
            masks.append(mask + [False] * (200 - len(mask)))
        return torch.tensor(rows, dtype=torch.long), torch.tensor(masks, dtype=torch.bool)

    def tokenize_task_only(self, prompts: list[str] | tuple[str, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        rows, masks = [], []
        pad = self.tokenizer.token_to_id("<pad>") or 0
        bos = self.tokenizer.token_to_id("<bos>")
        if bos is None:
            raise ValueError("PI0.5 tokenizer has no BOS token.")
        newline = self.tokenizer.encode("\n", add_special_tokens=False).ids
        for prompt in prompts:
            cleaned = str(prompt).strip().replace("_", " ").replace("\n", " ")
            ids = ([bos] + self.tokenizer.encode(cleaned, add_special_tokens=False).ids + newline)[:200]
            mask = [True] * len(ids)
            rows.append(ids + [pad] * (200 - len(ids)))
            masks.append(mask + [False] * (200 - len(mask)))
        return torch.tensor(rows, dtype=torch.long), torch.tensor(masks, dtype=torch.bool)

    @staticmethod
    def _image(value: torch.Tensor) -> torch.Tensor:
        image = value.float()
        if image.ndim == 4 and image.shape[-1] == 3:
            image = image.permute(0, 3, 1, 2).contiguous()
        if image.amax() > 2.0:
            image = image / 127.5 - 1.0
        elif image.amin() >= 0.0:
            image = image * 2.0 - 1.0
        return image

    def preprocess_observation(
        self,
        batch: dict[str, Any],
        *,
        device: torch.device | str,
    ) -> LiberoObservation:
        """Apply the frozen PI0.5 transforms without requiring action labels."""
        base = self._image(batch["observation.image"]).to(device)
        wrist = self._image(batch["observation.wrist_image"]).to(device)
        state = batch["observation.state"].float().to(device)
        q01 = self.state_q01.to(device)
        q99 = self.state_q99.to(device)
        state = (state - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        tokens, masks = self.tokenize(list(batch["prompt"]), state)
        task_tokens, task_masks = self.tokenize_task_only(list(batch["prompt"]))
        state = F.pad(state, (0, LIBERO_MODEL_ACTION_DIM - LIBERO_ACTION_DIM))
        tokens, masks = tokens.to(device), masks.to(device)
        task_tokens, task_masks = task_tokens.to(device), task_masks.to(device)
        batch_size = len(state)
        observation = LiberoObservation(
            images={"base_0_rgb": base, "left_wrist_0_rgb": wrist, "right_wrist_0_rgb": torch.zeros_like(base)},
            image_masks={
                "base_0_rgb": torch.ones(batch_size, dtype=torch.bool, device=device),
                "left_wrist_0_rgb": torch.ones(batch_size, dtype=torch.bool, device=device),
                "right_wrist_0_rgb": torch.zeros(batch_size, dtype=torch.bool, device=device),
            },
            state=state,
            tokenized_prompt=tokens,
            tokenized_prompt_mask=masks,
            episode_index=(
                batch["episode_index"].to(device) if "episode_index" in batch else None
            ),
            frame_index=(batch["frame_index"].to(device) if "frame_index" in batch else None),
            task_only_tokens=task_tokens,
            task_only_token_mask=task_masks,
        )
        return observation

    def __call__(self, batch: dict[str, Any], *, device: torch.device | str) -> tuple[LiberoObservation, torch.Tensor]:
        observation = self.preprocess_observation(batch, device=device)
        actions = self.action_normalizer.normalize(batch["actions"].float().to(device))
        actions = F.pad(actions, (0, LIBERO_MODEL_ACTION_DIM - LIBERO_ACTION_DIM))
        return observation, actions


class LiberoActionPrior(nn.Module):
    def __init__(self, task_dim: int, phase_dim: int, context_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(task_dim + phase_dim + context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, LIBERO_POLICY_HORIZON * LIBERO_ACTION_DIM),
        )

    def forward(self, task, phase, context):
        return self.network(torch.cat([task, phase, context], -1)).view(-1, LIBERO_POLICY_HORIZON, LIBERO_ACTION_DIM)


class LiberoZevaPolicy(nn.Module):
    def __init__(self, foundation: PI0Pytorch, processor: LiberoBatchProcessor):
        super().__init__()
        self.foundation = foundation
        self.processor = processor
        self.action_normalizer = processor.action_normalizer
        self.zeva_config = ZevaConfig(
            action_dim=16,
            action_horizon=10,
            vision_pretrained=False,
            num_views=2,
            task_count=40,
        )
        self.causal_transition_encoder = CausalTransitionEncoder(self.zeva_config)
        self.task_token_projector = nn.Sequential(nn.Linear(2048, 256), nn.LayerNorm(256))
        self.memory_context_encoder = MemoryContextEncoder(task_dim=256, phase_dim=128, signal_dim=256, context_dim=256)
        self.action_prior = LiberoActionPrior(256, 128, 256)
        self.causal_prefix_projector = nn.Linear(256, 2048)
        self.prior_action_projector = nn.Linear(16, 1024)
        initial_gate = torch.logit(torch.tensor(0.01))
        self.prefix_gate_logit = nn.Parameter(initial_gate.clone())
        self.action_gate_logit = nn.Parameter(initial_gate.clone())
        nn.init.zeros_(self.causal_prefix_projector.weight)
        nn.init.zeros_(self.causal_prefix_projector.bias)
        nn.init.zeros_(self.prior_action_projector.weight)
        nn.init.zeros_(self.prior_action_projector.bias)
        self.register_buffer("frozen_goal_embedding_table", None, persistent=False)
        self.register_buffer("canonical_goal_embedding_table", None, persistent=False)
        self.register_buffer("retrieval_task_prototypes", None, persistent=False)
        self._active_context = None
        self._active_prior = None
        self._active_injection_confidence = None
        self._foundation_rng_state_override = None
        self._full_finetune = False
        self._selective_action_finetune = False
        self._stage3_trainable_names: tuple[str, ...] = ()
        self.retrieval_head: CausalRetrievalHead | None = None
        self.causal_bank: LiberoCausalBank | None = None
        self.retrieval_task_names: tuple[str, ...] = ()
        self._install_hooks()
        self.foundation.requires_grad_(False).eval()
        self.reset(scope="episode")

    @classmethod
    def from_handoff(
        cls,
        handoff_root: str | Path,
        *,
        tokenizer_path: str | Path,
        device: str = "cuda",
        zte_checkpoint: str | Path | None = None,
        stage2_checkpoint: str | Path | None = None,
        adapter_checkpoint: str | Path | None = None,
        stage3_action_checkpoint: str | Path | None = None,
        retrieval_checkpoint: str | Path | None = None,
        causal_bank: str | Path | None = None,
    ) -> "LiberoZevaPolicy":
        handoff = LiberoHandoff.from_root(handoff_root)
        config = LiberoPi0Config()
        foundation = PI0Pytorch(config).to(device)
        safetensors.torch.load_model(foundation, handoff.checkpoint / "model.safetensors", strict=True)
        processor = LiberoBatchProcessor(handoff.statistics, tokenizer_path)
        wrapper = cls(foundation, processor).to(device)
        wrapper.frozen_goal_embedding_table = (
            foundation.paligemma_with_expert.paligemma.language_model.embed_tokens.weight.detach().clone()
        )
        if zte_checkpoint:
            wrapper.load_zte(zte_checkpoint)
        if stage2_checkpoint:
            safetensors.torch.load_model(foundation, Path(stage2_checkpoint) / "model.safetensors", strict=True)
        if adapter_checkpoint:
            wrapper.load_adapter(adapter_checkpoint)
        if stage3_action_checkpoint:
            wrapper.load_stage3_action(stage3_action_checkpoint)
        if (retrieval_checkpoint is None) != (causal_bank is None):
            raise ValueError("LIBERO retrieval checkpoint and causal bank must be supplied together.")
        if retrieval_checkpoint is not None:
            wrapper.load_retrieval(retrieval_checkpoint, causal_bank)
        return wrapper.eval()

    def _install_hooks(self):
        original_prefix = self.foundation.embed_prefix
        original_suffix = self.foundation.embed_suffix
        owner = self

        def prefix(_model, *args, **kwargs):
            embeddings, masks, attention = original_prefix(*args, **kwargs)
            if owner._active_context is not None:
                delta = owner.causal_prefix_projector(owner._active_context).to(embeddings.dtype)[:, None]
                confidence = owner._active_injection_confidence
                if confidence is None:
                    confidence = delta.new_ones(delta.shape[0])
                gate = torch.sigmoid(owner.prefix_gate_logit).to(delta.dtype) * confidence.to(delta.dtype)
                delta = delta * gate[:, None, None]
                embeddings = torch.cat((embeddings[:, :1] + delta, embeddings[:, 1:]), 1)
            return embeddings, masks, attention

        def suffix(_model, *args, **kwargs):
            embeddings, masks, attention, adarms = original_suffix(*args, **kwargs)
            if owner._active_prior is not None:
                confidence = owner._active_injection_confidence
                if confidence is None:
                    confidence = owner._active_prior.new_ones(owner._active_prior.shape[0])
                gate = torch.sigmoid(owner.action_gate_logit).to(owner._active_prior.dtype)
                gate = gate * confidence.to(owner._active_prior.dtype)
                delta = owner.prior_action_projector(owner._active_prior).to(embeddings.dtype)
                embeddings = embeddings + delta * gate[:, None, None]
            return embeddings, masks, attention, adarms

        self.foundation.embed_prefix = MethodType(prefix, self.foundation)
        self.foundation.embed_suffix = MethodType(suffix, self.foundation)

    def train(self, mode: bool = True):
        super().train(mode)
        self.foundation.train(mode if self._full_finetune else False)
        self.causal_transition_encoder.eval()
        self.causal_transition_encoder.target_vision_encoder.eval()
        if self._selective_action_finetune:
            for module in (
                self.task_token_projector,
                self.memory_context_encoder,
                self.action_prior,
                self.causal_prefix_projector,
                self.prior_action_projector,
            ):
                module.eval()
            if self.retrieval_head is not None:
                self.retrieval_head.eval()
        return self

    def raw_task_embedding(self, observation):
        tokens = observation.task_only_tokens
        if tokens is None or observation.task_only_token_mask is None:
            raise RuntimeError("LIBERO task-only tokens are unavailable.")
        masks = observation.task_only_token_mask.float()[..., None]
        table = self.foundation.paligemma_with_expert.paligemma.language_model.embed_tokens.weight
        return (F.embedding(tokens, table) * masks).sum(1) / masks.sum(1).clamp_min(1.0)

    def raw_goal_embedding(self, observation):
        if self.frozen_goal_embedding_table is None:
            raise RuntimeError("Frozen LIBERO goal table is unavailable.")
        if observation.task_only_tokens is None or observation.task_only_token_mask is None:
            raise RuntimeError("LIBERO task-only tokens are unavailable.")
        masks = observation.task_only_token_mask.float()[..., None]
        values = F.embedding(observation.task_only_tokens, self.frozen_goal_embedding_table)
        return (values * masks).sum(1) / masks.sum(1).clamp_min(1.0)

    def stage2_forward(self, observation, actions, bank_batch, injection_confidence=None):
        raw = self.raw_task_embedding(observation)
        task = self.task_token_projector(raw.to(next(self.task_token_projector.parameters()).dtype))
        context = self.memory_context_encoder(
            task,
            bank_batch.phase_token,
            bank_batch.brief_signals,
            bank_batch.retrieved_signals,
            getattr(bank_batch, "brief_mask", None),
            getattr(bank_batch, "retrieved_mask", None),
        )
        prior = self.action_prior(task, bank_batch.phase_token, context)
        self._active_context, self._active_prior = context, prior
        self._active_injection_confidence = injection_confidence
        try:
            if self._foundation_rng_state_override is not None:
                cpu_state, cuda_state = self._foundation_rng_state_override
                torch.random.set_rng_state(cpu_state)
                torch.cuda.set_rng_state(cuda_state, actions.device)
                self._foundation_rng_state_override = None
            output = self.foundation(observation, actions)
        finally:
            self._active_context = self._active_prior = None
            self._active_injection_confidence = None
        return output, prior

    def forward(self, observation, actions, bank_batch, injection_confidence=None):
        """DDP-visible Stage 2 entry point."""
        return self.stage2_forward(observation, actions, bank_batch, injection_confidence)

    def extract_vlm_features(self, observation):
        return self.foundation.extract_vlm_features(observation)

    def load_zte(self, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("schema") != "zeva-libero-zte-stage1-checkpoint-v3":
            raise ValueError("LIBERO policy requires a LIBERO ZTE checkpoint.")
        expected = dict(vars(self.zeva_config))
        actual = dict(checkpoint["zte_config"])
        actual.pop("vision_pretrained", None)
        expected.pop("vision_pretrained", None)
        if actual != expected:
            raise ValueError("LIBERO ZTE architecture differs from the policy wrapper.")
        self.causal_transition_encoder.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.causal_transition_encoder.requires_grad_(False).eval()

    def configure_full_finetune_stage2(self):
        self.requires_grad_(False)
        self.foundation.requires_grad_(True)
        for module in (self.task_token_projector, self.memory_context_encoder, self.action_prior,
                       self.causal_prefix_projector, self.prior_action_projector):
            module.requires_grad_(True)
        self._full_finetune = True
        self.foundation.train()
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def configure_adapter_stage2(self):
        """Freeze PI0.5/ZTE/retrieval and train only protected ZeVA residuals."""
        self.requires_grad_(False)
        for module in (self.task_token_projector, self.memory_context_encoder, self.action_prior,
                       self.causal_prefix_projector, self.prior_action_projector):
            module.requires_grad_(True)
        self.prefix_gate_logit.requires_grad_(True)
        self.action_gate_logit.requires_grad_(True)
        self._full_finetune = False
        self._selective_action_finetune = False
        self.foundation.eval()
        self.causal_transition_encoder.eval()
        if self.retrieval_head is not None:
            self.retrieval_head.requires_grad_(False).eval()
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def configure_selective_action_stage3(self, final_action_layers: int = 2):
        """Freeze Stage 2A and tune only the final action-expert blocks/output head."""
        self.requires_grad_(False)
        layers = self.foundation.paligemma_with_expert.gemma_expert.model.layers
        if not 1 <= final_action_layers <= len(layers):
            raise ValueError(f"final_action_layers must be in [1,{len(layers)}].")
        for layer in layers[-final_action_layers:]:
            layer.requires_grad_(True)
        self.foundation.action_out_proj.requires_grad_(True)
        self._full_finetune = False
        self._selective_action_finetune = True
        self.foundation.eval()
        self.causal_transition_encoder.eval()
        if self.retrieval_head is not None:
            self.retrieval_head.requires_grad_(False).eval()
        self._stage3_trainable_names = tuple(
            name for name, parameter in self.foundation.named_parameters() if parameter.requires_grad
        )
        if not self._stage3_trainable_names:
            raise RuntimeError("LIBERO Stage 3 selected no action parameters.")
        return [parameter for parameter in self.foundation.parameters() if parameter.requires_grad]

    def set_foundation_rng_state(self, cpu_state, cuda_state):
        self._foundation_rng_state_override = (cpu_state.cpu(), cuda_state.cpu())

    def injection_gate_regularizer(self):
        gates = torch.stack((torch.sigmoid(self.prefix_gate_logit), torch.sigmoid(self.action_gate_logit)))
        return gates.square().mean()

    def retrieve_task_ids_from_language(self, observation):
        if self.retrieval_head is None or self.causal_bank is None:
            raise RuntimeError("LIBERO task-language retrieval and causal bank are not loaded.")
        query = self.retrieval_head(self.raw_goal_embedding(observation))
        scores = query @ self.retrieval_task_prototypes.T
        confidence, task_ids = scores.max(dim=1)
        return task_ids, confidence

    @staticmethod
    def calibrate_retrieval_confidence(scores, floor=0.2):
        return ((scores - floor) / max(1e-6, 1.0 - floor)).clamp(0.0, 1.0)

    @staticmethod
    def _multiview_image(observation: LiberoObservation) -> torch.Tensor:
        """Build the exact two-view width-concatenated ZTE input."""
        views = []
        for key in ("base_0_rgb", "left_wrist_0_rgb"):
            image = observation.images[key]
            if image.ndim != 4:
                raise ValueError(f"{key} must be BCHW or BHWC, got {tuple(image.shape)}.")
            if image.shape[-1] == 3:
                image = image.permute(0, 3, 1, 2)
            if image.shape[1] != 3:
                raise ValueError(f"{key} must contain RGB images.")
            views.append(image.contiguous())
        if views[0].shape[-2:] != views[1].shape[-2:]:
            raise ValueError("LIBERO agent and wrist image sizes differ.")
        return torch.cat(views, dim=-1)

    def _new_memory(self) -> CausalMemoryManager:
        return CausalMemoryManager(
            brief_size=self.zeva_config.brief_memory_size,
            persistent_size=self.zeva_config.persistent_memory_size,
            retrieval_top_k=self.zeva_config.retrieval_top_k,
            merge_phase_weight=self.zeva_config.merge_phase_weight,
            merge_signal_weight=self.zeva_config.merge_signal_weight,
            merge_threshold=self.zeva_config.merge_threshold,
            use_brief_memory=self.zeva_config.use_brief_memory,
            use_persistent_memory=self.zeva_config.use_persistent_memory,
        )

    def reset(self, scope: str = "episode") -> None:
        """Reset recurrent deployment state at a fixed LIBERO episode boundary."""
        if scope not in {"attempt", "episode"}:
            raise ValueError("LIBERO ZeVA reset scope must be 'attempt' or 'episode'.")
        if hasattr(self, "_memories"):
            for memory in self._memories:
                memory.reset_attempt() if scope == "attempt" else memory.reset_episode()
        else:
            self._memories: list[CausalMemoryManager] = []
        self._previous_image: torch.Tensor | None = None
        self._current_phase: torch.Tensor | None = None
        self._pending_normalized_actions: torch.Tensor | None = None
        self._cte_inference_params = None
        self._retrieved_task_ids: torch.Tensor | None = None
        self._retrieved_task_scores: torch.Tensor | None = None
        self._action_queue: deque[torch.Tensor] = deque(maxlen=LIBERO_POLICY_HORIZON)

    def _ensure_batch_state(self, batch_size: int) -> None:
        if not self._memories:
            self._memories = [self._new_memory() for _ in range(batch_size)]
        elif len(self._memories) != batch_size:
            raise ValueError("LIBERO ZeVA batch size changed without an episode reset.")

    def _infer_task_once(self, observation: LiberoObservation) -> None:
        if self._retrieved_task_ids is not None:
            return
        self._retrieved_task_ids, self._retrieved_task_scores = self.retrieve_task_ids_from_language(
            observation
        )

    def _build_live_context(
        self,
        task_schema: torch.Tensor,
        phase_token: torch.Tensor,
        offline,
    ) -> torch.Tensor:
        contexts = []
        for index, memory in enumerate(self._memories):
            brief = memory.brief_tensor(device=phase_token.device)
            retrieved = memory.retrieve(phase_token[index], device=phase_token.device)
            offline_brief = offline.brief_signals[index : index + 1]
            offline_retrieved = offline.retrieved_signals[index : index + 1]
            brief = offline_brief if brief is None else torch.cat((offline_brief, brief), dim=1)
            retrieved = (
                offline_retrieved
                if retrieved is None
                else torch.cat((offline_retrieved, retrieved), dim=1)
            )
            contexts.append(
                self.memory_context_encoder(
                    task_schema[index : index + 1],
                    offline.phase_token[index : index + 1],
                    brief,
                    retrieved,
                )
            )
        return torch.cat(contexts, dim=0)

    @torch.inference_mode()
    def observe_causal_transition(
        self,
        observation: LiberoObservation,
        executed_actions: torch.Tensor,
    ) -> None:
        """Advance deployment memory by one observed action-effect interval.

        LIBERO training uses consecutive H5 transitions.  This explicit hook
        lets an evaluator keep the benchmark's R10 control plan while
        committing the midpoint and endpoint as two separate H5 transitions.
        """
        if self._previous_image is None or self._current_phase is None:
            raise RuntimeError("A LIBERO action chunk must be sampled before a causal commit.")
        image = self._multiview_image(observation)
        if image.shape[0] != self._previous_image.shape[0]:
            raise ValueError("LIBERO ZeVA batch size changed during a causal transition.")
        actions = torch.as_tensor(
            executed_actions, dtype=torch.float32, device=observation.state.device
        )
        if actions.ndim == 2:
            actions = actions.unsqueeze(0)
        if actions.ndim != 3 or not 1 <= actions.shape[1] <= LIBERO_EXECUTION_HORIZON:
            raise ValueError(
                "A LIBERO causal transition must contain between one and "
                f"{LIBERO_EXECUTION_HORIZON} actions, got {tuple(actions.shape)}."
            )
        normalized = self.action_normalizer.normalize(actions)
        causal, self._cte_inference_params = self.causal_transition_encoder.step(
            self._previous_image,
            normalized,
            image,
            inference_params=self._cte_inference_params,
        )
        self._current_phase = causal.phase_token
        for index, memory in enumerate(self._memories):
            memory.update(self._current_phase[index], causal.causal_signal[index])
        self._previous_image = image.detach().clone()
        # An explicit observed transition supersedes the legacy deferred chunk.
        self._pending_normalized_actions = None

    @torch.inference_mode()
    def sample_action_chunk(
        self,
        observation: LiberoObservation,
        *,
        use_memory: bool,
        noise: torch.Tensor | None = None,
        executed_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample raw H10 EEF16 actions for paired baseline/ZeVA rollout."""
        device = observation.state.device
        if not use_memory:
            normalized = self.foundation.sample_actions(device, observation, noise=noise)
            return self.action_normalizer.unnormalize(normalized[..., :LIBERO_ACTION_DIM])
        if self.retrieval_head is None or self.causal_bank is None:
            raise RuntimeError("ZeVA rollout requires task retrieval and the frozen causal bank.")

        image = self._multiview_image(observation)
        self._ensure_batch_state(image.shape[0])
        self._infer_task_once(observation)
        goal = self.canonical_goal_embedding_table[self._retrieved_task_ids].to(
            device=device, dtype=self.frozen_goal_embedding_table.dtype
        )
        task = self.task_token_projector(
            self.raw_task_embedding(observation).to(next(self.task_token_projector.parameters()).dtype)
        )
        if self._previous_image is None:
            phase, self._cte_inference_params = self.causal_transition_encoder.initialize_phase_state(
                image, goal
            )
            self._current_phase = phase
        else:
            if executed_actions is not None:
                self.observe_causal_transition(observation, executed_actions)
            elif self._pending_normalized_actions is not None:
                causal, self._cte_inference_params = self.causal_transition_encoder.step(
                    self._previous_image,
                    self._pending_normalized_actions,
                    image,
                    inference_params=self._cte_inference_params,
                )
                self._current_phase = causal.phase_token
                for index, memory in enumerate(self._memories):
                    memory.update(
                        self._current_phase[index], causal.causal_signal[index]
                    )
                self._previous_image = image.detach().clone()
                self._pending_normalized_actions = None
            if self._current_phase is None:
                raise RuntimeError("LIBERO ZeVA phase state is unavailable.")
            phase = self._current_phase

        offline = self.causal_bank.retrieve(
            self._retrieved_task_ids,
            phase,
            brief_size=self.zeva_config.brief_memory_size,
            retrieval_top_k=self.zeva_config.retrieval_top_k,
        )
        context = self._build_live_context(task, phase, offline)
        self._active_context = context
        self._active_prior = self.action_prior(task, offline.phase_token, context)
        self._active_injection_confidence = self.calibrate_retrieval_confidence(
            self._retrieved_task_scores
        )
        try:
            normalized = self.foundation.sample_actions(device, observation, noise=noise)
        finally:
            self._active_context = None
            self._active_prior = None
            self._active_injection_confidence = None
        self._previous_image = image.detach().clone()
        self._pending_normalized_actions = normalized[
            ..., :LIBERO_EXECUTION_HORIZON, :LIBERO_ACTION_DIM
        ].detach().clone()
        return self.action_normalizer.unnormalize(normalized[..., :LIBERO_ACTION_DIM])

    def retrieval_diagnostics(self) -> list[dict[str, Any]]:
        if self._retrieved_task_ids is None or self._retrieved_task_scores is None:
            return []
        return [
            {
                "task_id": int(task_id),
                "task": self.retrieval_task_names[int(task_id)],
                "score": float(score),
            }
            for task_id, score in zip(
                self._retrieved_task_ids.detach().cpu(),
                self._retrieved_task_scores.detach().cpu(),
                strict=True,
            )
        ]

    def adapter_state_dict(self, manifest=None):
        return {
            "schema": "zeva-libero-stage2-adapter-v2",
            "physical_contract": "libero-camera-t0-relative-eef16-h10-execute5-quantile",
            "task_token_projector": self.task_token_projector.state_dict(),
            "memory_context_encoder": self.memory_context_encoder.state_dict(),
            "action_prior": self.action_prior.state_dict(),
            "causal_prefix_projector": self.causal_prefix_projector.state_dict(),
            "prior_action_projector": self.prior_action_projector.state_dict(),
            "prefix_gate_logit": self.prefix_gate_logit.detach().cpu(),
            "action_gate_logit": self.action_gate_logit.detach().cpu(),
            "action_normalization": self.action_normalizer.metadata(),
            "manifest": manifest,
        }

    def load_adapter(self, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("schema") not in {
            "zeva-libero-stage2-adapter-v1", "zeva-libero-stage2-adapter-v2"
        }:
            raise ValueError("Unsupported LIBERO Stage 2 adapter.")
        for name in ("task_token_projector", "memory_context_encoder", "action_prior",
                     "causal_prefix_projector", "prior_action_projector"):
            getattr(self, name).load_state_dict(checkpoint[name], strict=True)
        if "prefix_gate_logit" in checkpoint:
            self.prefix_gate_logit.data.copy_(checkpoint["prefix_gate_logit"])
        if "action_gate_logit" in checkpoint:
            self.action_gate_logit.data.copy_(checkpoint["action_gate_logit"])

    def stage3_action_state_dict(self, manifest=None):
        if not self._stage3_trainable_names:
            raise RuntimeError("Stage 3 action parameter selection has not been configured.")
        parameters = dict(self.foundation.named_parameters())
        return {
            "schema": "zeva-libero-stage3-selective-action-v1",
            "physical_contract": "libero-camera-t0-relative-eef16-h10-execute5-quantile",
            "trainable_parameter_names": self._stage3_trainable_names,
            "model_state_dict": {
                name: parameters[name].detach().cpu() for name in self._stage3_trainable_names
            },
            "manifest": manifest,
        }

    def load_stage3_action(self, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("schema") != "zeva-libero-stage3-selective-action-v1":
            raise ValueError("Unsupported LIBERO selective-action Stage 3 checkpoint.")
        parameters = dict(self.foundation.named_parameters())
        state = checkpoint["model_state_dict"]
        if set(state) != set(checkpoint["trainable_parameter_names"]):
            raise ValueError("LIBERO Stage 3 parameter table is inconsistent.")
        unknown = set(state) - set(parameters)
        if unknown:
            raise ValueError(f"LIBERO Stage 3 contains unknown action parameters: {sorted(unknown)}")
        with torch.no_grad():
            for name, value in state.items():
                parameters[name].copy_(value.to(device=parameters[name].device, dtype=parameters[name].dtype))

    def load_retrieval(self, path, causal_bank):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("schema") != "zeva-libero-task-language-retrieval-v1":
            raise ValueError("Unsupported LIBERO task-language retrieval checkpoint.")
        bank = LiberoCausalBank.load(causal_bank, device=next(self.parameters()).device)
        task_names = tuple(checkpoint["task_names"])
        if task_names != bank.task_names:
            raise ValueError("LIBERO retrieval task table differs from the causal bank.")
        prototypes = F.normalize(torch.as_tensor(checkpoint["task_prototypes"]), dim=-1)
        state = checkpoint["model_state_dict"]
        head = CausalRetrievalHead(
            input_dim=state["network.0.weight"].shape[1],
            hidden_dim=state["network.0.weight"].shape[0],
            output_dim=state["network.4.weight"].shape[0],
            dropout=0.0,
        ).to(next(self.parameters()).device)
        head.load_state_dict(state, strict=True)
        canonical = torch.as_tensor(checkpoint["canonical_goal_embeddings"])
        expected = (len(task_names), self.zeva_config.goal_dim)
        if tuple(canonical.shape) != expected:
            raise ValueError(f"LIBERO canonical goal table must have shape {expected}.")
        self.retrieval_head = head.requires_grad_(False).eval()
        self.causal_bank = bank
        self.retrieval_task_prototypes = prototypes.to(next(self.parameters()).device)
        self.canonical_goal_embedding_table = canonical.to(next(self.parameters()).device)
        self.retrieval_task_names = task_names
