"""Zeva adapter around the exact released LeRobot RoboTwin PI0.5 policy."""

from __future__ import annotations

from collections import deque
import dataclasses
import inspect
import math
from pathlib import Path
from types import MethodType
from typing import Any, NamedTuple

from safetensors import safe_open
import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812

from openpi.zeva.causal_bank import CausalBankBatch
from openpi.zeva.causal_bank import RobotWinCausalBank
from openpi.zeva.config import ZevaConfig
from openpi.zeva.context import MemoryContextEncoder
from openpi.zeva.memory import CausalMemoryManager
from openpi.zeva.retrieval import CausalRetrievalHead
from openpi.zeva.robotwin_contract import ROBOTWIN_ACTION_DIM
from openpi.zeva.robotwin_contract import ROBOTWIN_ACTION_HORIZON
from openpi.zeva.robotwin_contract import ROBOTWIN_CAMERA_KEYS
from openpi.zeva.robotwin_contract import MeanStdActionNormalizer
from openpi.zeva.robotwin_contract import RobotWinHandoff
from openpi.zeva.transition_encoder import CausalTransitionEncoder


def robotwin_multiview_image(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Build a lossless three-view canvas; CTE splits it before per-view resize."""
    views = []
    for key in ROBOTWIN_CAMERA_KEYS:
        image = batch[key]
        if image.ndim != 4:
            raise ValueError(f"{key} must be BCHW or BHWC, got {tuple(image.shape)}.")
        if image.shape[-1] == 3:
            image = image.permute(0, 3, 1, 2)
        if image.shape[1] != 3:
            raise ValueError(f"{key} must have three RGB channels, got {tuple(image.shape)}.")
        views.append(image.contiguous())
    spatial_shapes = {tuple(view.shape[-2:]) for view in views}
    if len(spatial_shapes) != 1:
        raise ValueError(f"RoboTwin camera shapes differ: {sorted(spatial_shapes)!r}.")
    return torch.cat(views, dim=-1)


class RobotWinGaussianActionPrior(NamedTuple):
    """Diagonal Gaussian parameters in the normalized RoboTwin EEF16 domain."""

    mean: torch.Tensor
    log_std: torch.Tensor


def gaussian_action_prior_nll(
    prior: RobotWinGaussianActionPrior,
    target: torch.Tensor,
) -> torch.Tensor:
    """BehaviorVLA-style NLL: sum over EEF16, then mean over batch and horizon."""
    if prior.mean.shape != target.shape or prior.log_std.shape != target.shape:
        raise ValueError(
            "Gaussian action-prior tensors and target must have identical shapes, got "
            f"mean={tuple(prior.mean.shape)}, log_std={tuple(prior.log_std.shape)}, "
            f"target={tuple(target.shape)}."
        )
    inverse_variance = torch.exp(-2.0 * prior.log_std)
    elementwise_nll = 0.5 * (
        (target - prior.mean).square() * inverse_variance
        + 2.0 * prior.log_std
        + math.log(2.0 * math.pi)
    )
    return elementwise_nll.sum(dim=-1).mean()


class RobotWinActionPrior(nn.Module):
    """Phase-conditioned diagonal Gaussian over normalized H50 EEF16 actions."""

    def __init__(
        self,
        *,
        task_dim: int,
        phase_dim: int,
        context_dim: int,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(task_dim + phase_dim + context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, ROBOTWIN_ACTION_HORIZON * ROBOTWIN_ACTION_DIM * 2),
        )
        nn.init.normal_(self.network[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self,
        task_schema: torch.Tensor,
        phase_token: torch.Tensor,
        causal_context: torch.Tensor,
    ) -> RobotWinGaussianActionPrior:
        parameters = self.network(torch.cat([task_schema, phase_token, causal_context], dim=-1))
        parameters = parameters.view(-1, ROBOTWIN_ACTION_HORIZON, ROBOTWIN_ACTION_DIM, 2)
        mean = parameters[..., 0]
        log_std = parameters[..., 1].clamp(min=-5.0, max=2.0)
        return RobotWinGaussianActionPrior(mean=mean, log_std=log_std)


class RobotWinZevaPolicy(nn.Module):
    """RoboTwin PI0.5 plus Zeva causal conditioning for staged fine-tuning.

    The wrapped policy continues to consume and return the baseline LeRobot
    processor domain. Causal transitions use raw EEF16 commands by default and
    are normalized with the checkpoint's frozen mean/std artifact.
    """

    def __init__(
        self,
        foundation_policy: nn.Module,
        action_normalizer: MeanStdActionNormalizer,
        *,
        preprocessor: Any | None = None,
        postprocessor: Any | None = None,
        zeva_config: ZevaConfig | None = None,
    ):
        super().__init__()
        self.foundation = foundation_policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.action_normalizer = action_normalizer
        self.zeva_config = zeva_config or ZevaConfig(
            action_dim=ROBOTWIN_ACTION_DIM,
            action_horizon=ROBOTWIN_ACTION_HORIZON,
            vision_pretrained=False,
        )
        if self.zeva_config.action_dim != ROBOTWIN_ACTION_DIM:
            raise ValueError("RoboTwin Zeva CTE must consume EEF16 actions.")
        if self.zeva_config.action_horizon != ROBOTWIN_ACTION_HORIZON:
            raise ValueError("RoboTwin Zeva action horizon must remain 50.")
        self._validate_foundation_contract()

        for parameter in self.foundation.parameters():
            parameter.requires_grad = False
        self.foundation.eval()

        self.causal_transition_encoder = CausalTransitionEncoder(self.zeva_config)
        self.task_token_projector = nn.Sequential(
            nn.Linear(2048, self.zeva_config.model_dim),
            nn.LayerNorm(self.zeva_config.model_dim),
        )
        self.memory_context_encoder = MemoryContextEncoder(
            task_dim=self.zeva_config.model_dim,
            phase_dim=self.zeva_config.phase_dim,
            signal_dim=self.zeva_config.signal_dim,
            context_dim=self.zeva_config.model_dim,
        )
        self.action_prior = RobotWinActionPrior(
            task_dim=self.zeva_config.model_dim,
            phase_dim=self.zeva_config.phase_dim,
            context_dim=self.zeva_config.model_dim,
        )
        self.causal_action_projector = nn.Linear(self.zeva_config.model_dim, 1024)
        self.prior_action_projector = nn.Linear(ROBOTWIN_ACTION_DIM, 1024)
        # The projectors start at zero, so the wrapper is exactly the released
        # RoboTwin policy at construction time.  The small learned gates bound
        # how much the memory path can change that policy after training.
        initial_gate = torch.logit(torch.tensor(0.01))
        self.context_gate_logit = nn.Parameter(initial_gate.clone())
        self.prior_gate_logit = nn.Parameter(initial_gate.clone())
        # Task/phase-conditioned safety routing.  The final linear layer is
        # zero-initialized, so ``2 * sigmoid(router) == 1`` and a fresh wrapper
        # is numerically identical to the scalar-gate implementation.  The
        # multiplier is bounded to [0, 2], keeping either residual below 2% at
        # the initial scalar gate while allowing harmful tasks/phases to route
        # the residual toward zero.
        self.residual_gate_router = nn.Sequential(
            nn.Linear(self.zeva_config.model_dim + self.zeva_config.phase_dim, 2),
        )
        nn.init.zeros_(self.residual_gate_router[0].weight)
        nn.init.zeros_(self.residual_gate_router[0].bias)
        # A newly constructed wrapper is exactly the released foundation model.
        # Causal influence is learned gradually by the two zero-init injection heads.
        nn.init.zeros_(self.causal_action_projector.weight)
        nn.init.zeros_(self.causal_action_projector.bias)
        nn.init.zeros_(self.prior_action_projector.weight)
        nn.init.zeros_(self.prior_action_projector.bias)

        self._active_causal_context: torch.Tensor | None = None
        self._active_action_prior: torch.Tensor | None = None
        self._active_injection_confidence: torch.Tensor | None = None
        self._active_prior_residual_mask: torch.Tensor | None = None
        self._active_context_gate: torch.Tensor | None = None
        self._active_prior_gate: torch.Tensor | None = None
        # Optional deployment-only safety calibration.  A single trust scale is
        # shared by the context and action-prior residuals so calibration cannot
        # change the learned dual-residual semantics.  Task identity comes from
        # the task-language retriever; no episode/progress oracle is involved.
        self._deployment_task_residual_scales: dict[str, float] = {}
        self._deployment_default_residual_scale = 1.0
        self._deployment_residual_calibration: dict[str, Any] | None = None
        self._foundation_rng_state_override: tuple[torch.Tensor, torch.Tensor] | None = None
        # Optional immutable action-path weights used by Stage 2 to compare a
        # jointly tuned ZeVA student against the independently trained Base.
        # These tensors are deliberately outside state_dict: the source Base
        # checkpoint is recorded in the run manifest and remains authoritative.
        self._foundation_anchor_parameters: dict[str, torch.Tensor] = {}
        self._full_pi05_finetune = False
        self._action_expert_finetune = False
        self.retrieval_head: CausalRetrievalHead | None = None
        self.causal_bank: RobotWinCausalBank | None = None
        self.register_buffer("retrieval_task_prototypes", None, persistent=False)
        self.register_buffer("canonical_goal_embedding_table", None, persistent=False)
        self.register_buffer("frozen_goal_embedding_table", None, persistent=False)
        self._task_only_tokenizer: Any | None = None
        self.retrieval_task_names: tuple[str, ...] = ()
        self._install_foundation_injection_hooks()
        self.reset(scope="episode")

    def train(self, mode: bool = True):  # noqa: FBT001, FBT002 - matches torch.nn.Module API.
        super().train(mode)
        if self._full_pi05_finetune or self._action_expert_finetune:
            self.foundation.train(mode)
        else:
            self.foundation.eval()
        if self._action_expert_finetune:
            self.foundation.model.paligemma_with_expert.paligemma.eval()
        self.causal_transition_encoder.eval()
        self.causal_transition_encoder.target_vision_encoder.eval()
        return self

    @classmethod
    def from_handoff(
        cls,
        handoff_root: str | Path,
        *,
        device: str = "cuda",
        foundation_checkpoint: str | Path | None = None,
        goal_embedding_checkpoint: str | Path | None = None,
        zte_checkpoint: str | Path | None = None,
        adapter_checkpoint: str | Path | None = None,
        stage2_checkpoint: str | Path | None = None,
        retrieval_checkpoint: str | Path | None = None,
        causal_bank: str | Path | None = None,
    ) -> RobotWinZevaPolicy:
        """Load the exact selected PI0.5 weights and their saved processors."""
        handoff = RobotWinHandoff.from_root(handoff_root)
        if foundation_checkpoint is not None:
            handoff = dataclasses.replace(handoff, checkpoint=Path(foundation_checkpoint).resolve())
            handoff.validate()
        # Some H100 hosts expose the release's Torch runtime through Python
        # 3.10 while the frozen LeRobot source uses typing.Self (3.11). The
        # symbol is type-only, so provide its standard typing_extensions alias
        # without changing the released source tree.
        import typing  # noqa: PLC0415

        import typing_extensions  # noqa: PLC0415

        for typing_name in ("Self", "Unpack", "NotRequired"):
            if not hasattr(typing, typing_name):
                setattr(typing, typing_name, getattr(typing_extensions, typing_name))
        try:
            import sys  # noqa: PLC0415
            from types import ModuleType  # noqa: PLC0415

            import lerobot  # noqa: PLC0415

            # LeRobot's aggregate policies package imports every optional policy
            # (and their environment dependencies). Register the real policies
            # directory as a package and load only PI0.5 from the frozen source.
            if "lerobot.policies" not in sys.modules:
                policies_package = ModuleType("lerobot.policies")
                policies_package.__path__ = [str(Path(next(iter(lerobot.__path__))) / "policies")]
                policies_package.__package__ = "lerobot.policies"
                sys.modules["lerobot.policies"] = policies_package
            import lerobot.policies.pi_gemma as pi_gemma_runtime  # noqa: I001, PLC0415
            import lerobot.policies.pi05.modeling_pi05 as modeling_pi05_runtime  # noqa: PLC0415
            from lerobot.configs import PreTrainedConfig  # noqa: PLC0415
            from lerobot.policies.pi05.modeling_pi05 import PI05Policy  # noqa: PLC0415
            from lerobot.processor import AbsoluteActionsProcessorStep  # noqa: PLC0415
            from lerobot.processor import PolicyProcessorPipeline  # noqa: PLC0415
            from lerobot.processor import RelativeActionsProcessorStep  # noqa: PLC0415
            from lerobot.processor import batch_to_transition  # noqa: PLC0415
            from lerobot.processor import policy_action_to_transition  # noqa: PLC0415
            from lerobot.processor import transition_to_batch  # noqa: PLC0415
            from lerobot.processor import transition_to_policy_action  # noqa: PLC0415
            from safetensors.torch import load_file  # noqa: PLC0415
            from safetensors.torch import load_model  # noqa: PLC0415
            import transformers as transformers_runtime  # noqa: PLC0415
            from transformers import AutoTokenizer  # noqa: PLC0415
            from transformers.cache_utils import DynamicCache  # noqa: PLC0415
        except ImportError as error:  # pragma: no cover - depends on the H100 release runtime.
            raise RuntimeError(
                "RobotWinZevaPolicy must run with the handoff's LeRobot runtime on PYTHONPATH."
            ) from error
        transformers_major = int(transformers_runtime.__version__.split(".", 1)[0])
        native_handoff_runtime = transformers_major >= 5

        # Compatibility shims are needed only on the fallback Transformers 4
        # deployment.  The released checkpoint's frozen runtime is
        # Transformers 5; modifying its cache, attention, or image feature
        # path changes PI0.5 numerics and can collapse closed-loop success.
        original_create_causal_mask = pi_gemma_runtime.create_causal_mask
        if (
            not native_handoff_runtime
            and "input_embeds" in inspect.signature(original_create_causal_mask).parameters
        ):

            def create_causal_mask_compat(*args, **kwargs):
                if "inputs_embeds" in kwargs and "input_embeds" not in kwargs:
                    kwargs["input_embeds"] = kwargs.pop("inputs_embeds")
                return original_create_causal_mask(*args, **kwargs)

            pi_gemma_runtime.create_causal_mask = create_causal_mask_compat

        if not native_handoff_runtime:
            def clone_past_key_values_compat(past_key_values):
                layers = []
                for layer_cache in past_key_values:
                    if len(layer_cache) < 2:
                        raise ValueError("PI0.5 DynamicCache layer must contain key and value tensors.")
                    layers.append((layer_cache[0].clone(), layer_cache[1].clone()))
                return DynamicCache(tuple(layers))

            modeling_pi05_runtime.clone_past_key_values = clone_past_key_values_compat

        config = PreTrainedConfig.from_pretrained(str(handoff.checkpoint), local_files_only=True)
        config.device = device
        # Install Zeva hooks before the first model call; avoiding torch.compile
        # also keeps adapter parameters visible to autograd and checkpointing.
        config.compile_model = False
        foundation = PI05Policy.from_pretrained(
            str(handoff.checkpoint),
            config=config,
            strict=True,
            local_files_only=True,
        )
        reference = load_file(
            handoff.checkpoint / "model.safetensors",
            device="cpu",
        )["model.action_in_proj.weight"]
        torch.testing.assert_close(
            foundation.model.action_in_proj.weight.detach().cpu(),
            reference,
            rtol=0,
            atol=0,
        )
        # LeRobot's PI0.5 port historically expected PaliGemma
        # ``get_image_features`` to return an object with ``pooler_output``;
        # the pinned HF release returns the projected feature tensor directly.
        # Normalize only that return-type difference without changing weights,
        # pixels, or gradients.
        paligemma_with_expert = foundation.model.paligemma_with_expert
        original_embed_image = paligemma_with_expert.embed_image

        # The handoff decoder layer forwards ``past_key_values`` while the
        # pinned HF attention implementation accepts singular
        # ``past_key_value``. Keep inference prefix caches connected without
        # modifying the frozen runtime source.
        if not native_handoff_runtime:
            for decoder in (
                paligemma_with_expert.paligemma.model.language_model,
                paligemma_with_expert.gemma_expert.model,
            ):
                for layer in decoder.layers:
                    attention = layer.self_attn
                    original_attention_forward = attention.forward

                    def attention_forward_compat(_attention, *args, _forward=original_attention_forward, **kwargs):
                        if "past_key_values" in kwargs and "past_key_value" not in kwargs:
                            kwargs["past_key_value"] = kwargs.pop("past_key_values")
                        return _forward(*args, **kwargs)

                    attention.forward = MethodType(attention_forward_compat, attention)

        def embed_image_compat(model, image: torch.Tensor, **kwargs):
            if image.ndim == 5:
                return original_embed_image(image, **kwargs)
            output_dtype = image.dtype
            model_input = image.float() if image.dtype != torch.float32 else image
            features = model.paligemma.model.get_image_features(model_input)
            if hasattr(features, "pooler_output"):
                features = features.pooler_output
            if not torch.is_tensor(features):
                raise TypeError(f"Unexpected PaliGemma image feature output: {type(features)!r}.")
            return features.to(output_dtype) if features.dtype != output_dtype else features

        if not native_handoff_runtime:
            paligemma_with_expert.embed_image = MethodType(embed_image_compat, paligemma_with_expert)
        # The released tokenizer config stores ``extra_special_tokens`` as a
        # vocabulary list. Transformers 4.53+ interprets that keyword as a
        # mapping of model-specific token attributes. Supplying an empty
        # mapping preserves the tokenizer vocabulary and the ordinary
        # bos/eos/pad settings while avoiding that incompatible reinterpretation.
        tokenizer_kwargs = {"local_files_only": True}
        if not native_handoff_runtime:
            tokenizer_kwargs["extra_special_tokens"] = {}
        tokenizer = AutoTokenizer.from_pretrained(
            str(handoff.checkpoint / "tokenizer"),
            **tokenizer_kwargs,
        )
        preprocessor = PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=str(handoff.checkpoint),
            config_filename="policy_preprocessor.json",
            overrides={"tokenizer_processor": {"tokenizer": tokenizer}},
            to_transition=batch_to_transition,
            to_output=transition_to_batch,
        )
        postprocessor = PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=str(handoff.checkpoint),
            config_filename="policy_postprocessor.json",
            overrides={},
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        )
        relative_step = next(
            (step for step in preprocessor.steps if isinstance(step, RelativeActionsProcessorStep)),
            None,
        )
        if relative_step is not None:
            for step in postprocessor.steps:
                if isinstance(step, AbsoluteActionsProcessorStep) and step.relative_step is None:
                    step.relative_step = relative_step
        zeva_config = None
        if zte_checkpoint is not None:
            zte_payload = torch.load(zte_checkpoint, map_location="cpu", weights_only=False)
            if zte_payload.get("schema") not in {
                "zeva-robotwin-zte-stage1-checkpoint-v4",
                "zeva-robotwin-zte-stage1-checkpoint-v5",
            }:
                raise ValueError("Task-language deployment requires a Stage 1 v4/v5 checkpoint.")
            zeva_config = ZevaConfig(**zte_payload["zte_config"])
            # Pretrained weights are already present in the checkpoint; do not
            # trigger a network/cache lookup while constructing the wrapper.
            zeva_config = dataclasses.replace(zeva_config, vision_pretrained=False)
        wrapper = cls(
            foundation,
            MeanStdActionNormalizer.from_stats_file(handoff.statistics),
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            zeva_config=zeva_config,
        ).to(device)
        if goal_embedding_checkpoint is None:
            wrapper.frozen_goal_embedding_table = (
                wrapper.foundation.model.paligemma_with_expert.paligemma.language_model.embed_tokens.weight
                .detach()
                .clone()
            )
        else:
            goal_embedding_checkpoint = Path(goal_embedding_checkpoint).resolve()
            goal_config = PreTrainedConfig.from_pretrained(
                str(goal_embedding_checkpoint), local_files_only=True
            )
            if goal_config.tokenizer_max_length != config.tokenizer_max_length:
                raise ValueError("Foundation and frozen Zeva goal encoders use different token lengths.")
            goal_tokenizer_kwargs = {"local_files_only": True}
            if not native_handoff_runtime:
                goal_tokenizer_kwargs["extra_special_tokens"] = {}
            goal_tokenizer = AutoTokenizer.from_pretrained(
                str(goal_embedding_checkpoint / "tokenizer"),
                **goal_tokenizer_kwargs,
            )
            if goal_tokenizer.get_vocab() != tokenizer.get_vocab():
                raise ValueError("Foundation and frozen Zeva goal encoders use different tokenizers.")
            embedding_key = (
                "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
            )
            with safe_open(
                goal_embedding_checkpoint / "model.safetensors",
                framework="pt",
                device="cpu",
            ) as goal_weights:
                wrapper.frozen_goal_embedding_table = goal_weights.get_tensor(embedding_key).to(device)
        wrapper._task_only_tokenizer = tokenizer  # noqa: SLF001
        if zte_checkpoint is not None:
            wrapper.load_zte(zte_checkpoint)
        if adapter_checkpoint is not None:
            wrapper.load_adapter(adapter_checkpoint)
        if stage2_checkpoint is not None:
            load_model(
                wrapper.foundation,
                Path(stage2_checkpoint) / "model.safetensors",
                strict=True,
            )
        if (retrieval_checkpoint is None) != (causal_bank is None):
            raise ValueError("Stage 3 retrieval checkpoint and causal bank must be supplied together.")
        if retrieval_checkpoint is not None:
            wrapper.load_retrieval(retrieval_checkpoint, causal_bank)
        wrapper.eval()
        return wrapper

    def _validate_foundation_contract(self) -> None:
        config = self.foundation.config
        if config.chunk_size != ROBOTWIN_ACTION_HORIZON or config.n_action_steps != ROBOTWIN_ACTION_HORIZON:
            raise ValueError("Foundation PI0.5 must use chunk_size=n_action_steps=50.")
        if tuple(config.output_features["action"].shape) != (ROBOTWIN_ACTION_DIM,):
            raise ValueError("Foundation PI0.5 must output EEF16.")
        if tuple(config.input_features["observation.state"].shape) != (14,):
            raise ValueError("Foundation PI0.5 must consume absolute Joint14 state.")
        if tuple(config.image_features) != ROBOTWIN_CAMERA_KEYS:
            raise ValueError(f"Foundation camera order drifted: {tuple(config.image_features)!r}.")

    def _install_foundation_injection_hooks(self) -> None:
        model = self.foundation.model
        original_embed_suffix = model.embed_suffix
        owner = self

        def embed_suffix_with_zeva(_model, *args, **kwargs):
            action_embeddings, pad_masks, attention_masks, adarms = original_embed_suffix(*args, **kwargs)
            context = owner._active_causal_context
            confidence = owner._active_injection_confidence
            if context is not None:
                if confidence is None:
                    confidence = context.new_ones((context.shape[0],))
                gate = owner._active_context_gate
                if gate is None:
                    gate = torch.sigmoid(owner.context_gate_logit).expand(context.shape[0])
                gate = gate.to(context.dtype) * confidence.to(context.dtype)
                context_delta = owner.causal_action_projector(context).to(action_embeddings.dtype)
                action_embeddings = action_embeddings + context_delta[:, None, :] * gate[:, None, None]
            prior = owner._active_action_prior
            if prior is not None:
                confidence = owner._active_injection_confidence
                if confidence is None:
                    confidence = prior.new_ones((prior.shape[0],))
                gate = owner._active_prior_gate
                if gate is None:
                    gate = torch.sigmoid(owner.prior_gate_logit).expand(prior.shape[0])
                gate = gate.to(prior.dtype) * confidence.to(prior.dtype)
                residual_mask = owner._active_prior_residual_mask
                if residual_mask is not None:
                    gate = gate * residual_mask.to(device=gate.device, dtype=gate.dtype)
                prior_delta = owner.prior_action_projector(prior).to(action_embeddings.dtype)
                action_embeddings = action_embeddings + prior_delta * gate[:, None, None]
            return action_embeddings, pad_masks, attention_masks, adarms

        model.embed_suffix = MethodType(embed_suffix_with_zeva, model)

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
        if scope not in {"attempt", "episode"}:
            raise ValueError("Zeva reset scope must be 'attempt' or 'episode'.")
        if hasattr(self, "_memories"):
            for memory in self._memories:
                memory.reset_attempt() if scope == "attempt" else memory.reset_episode()
        else:
            self._memories: list[CausalMemoryManager] = []
        self._previous_image: torch.Tensor | None = None
        self._pending_normalized_actions: torch.Tensor | None = None
        self._cte_inference_params = None
        self._retrieved_task_ids: torch.Tensor | None = None
        self._retrieved_task_scores: torch.Tensor | None = None
        self._active_prior_residual_mask = None
        self._active_context_gate = None
        self._active_prior_gate = None
        self._action_queue: deque[torch.Tensor] = deque(maxlen=ROBOTWIN_ACTION_HORIZON)
        if hasattr(self.foundation, "reset"):
            self.foundation.reset()

    def _ensure_batch_state(self, batch_size: int) -> None:
        if not self._memories:
            self._memories = [self._new_memory() for _ in range(batch_size)]
        elif len(self._memories) != batch_size:
            raise ValueError("RoboTwin Zeva batch size changed without an episode reset.")

    def _raw_task_embedding(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        tokens = batch["observation.language.tokens"]
        masks = batch["observation.language.attention_mask"].bool()
        with torch.no_grad():
            if self.frozen_goal_embedding_table is None:
                raise RuntimeError("The frozen Zeva language table was not initialized.")
            token_embeddings = F.embedding(tokens, self.frozen_goal_embedding_table)
        weights = masks.to(token_embeddings.dtype).unsqueeze(-1)
        return (token_embeddings * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def _raw_goal_embedding(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        supplied = batch.get("zeva.goal_embedding")
        if supplied is not None:
            return supplied
        if self.frozen_goal_embedding_table is None:
            raise RuntimeError("The frozen PI0.5 goal embedding table was not initialized from the handoff.")
        tokens = batch["observation.language.tokens"]
        masks = batch["observation.language.attention_mask"].bool()
        token_embeddings = F.embedding(tokens, self.frozen_goal_embedding_table)
        weights = masks.to(token_embeddings.dtype).unsqueeze(-1)
        return (token_embeddings * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def _task_only_goal_embedding(
        self,
        task: str | list[str] | tuple[str, ...],
        device: torch.device,
    ) -> torch.Tensor:
        if self._task_only_tokenizer is None or self.frozen_goal_embedding_table is None:
            raise RuntimeError("Task-only PI0.5 goal encoder was not initialized from the handoff.")
        tasks = [task] if isinstance(task, str) else list(task)
        prompts = [item.strip().replace("_", " ").replace("\n", " ") + "\n" for item in tasks]
        encoded = self._task_only_tokenizer(
            prompts,
            max_length=200,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tokens = encoded["input_ids"].to(self.frozen_goal_embedding_table.device)
        masks = encoded["attention_mask"].to(self.frozen_goal_embedding_table.device).bool()
        token_embeddings = F.embedding(tokens, self.frozen_goal_embedding_table)
        weights = masks.to(token_embeddings.dtype).unsqueeze(-1)
        pooled = (token_embeddings * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return pooled.to(device=device)

    def _task_schema(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        pooled = self._raw_task_embedding(batch)
        projector_dtype = next(self.task_token_projector.parameters()).dtype
        return self.task_token_projector(pooled.to(projector_dtype))

    @torch.inference_mode()
    def extract_vlm_features(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract the exact first-frame PI0.5 EOS feature used by Stage 3."""
        from lerobot.policies.common.vla_utils import make_att_2d_masks  # noqa: PLC0415
        from lerobot.policies.common.vla_utils import prepare_attention_masks_4d  # noqa: PLC0415
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK  # noqa: PLC0415
        from lerobot.utils.constants import OBS_LANGUAGE_TOKENS  # noqa: PLC0415

        foundation = self.foundation
        core = foundation.model
        images, image_masks = foundation._preprocess_images(batch)  # noqa: SLF001
        states, state_masks = foundation._prepare_memory_states(batch)  # noqa: SLF001
        embeddings, pad_masks, attention_masks = core.embed_prefix(
            images,
            image_masks,
            batch[OBS_LANGUAGE_TOKENS],
            batch[OBS_LANGUAGE_ATTENTION_MASK],
            states,
            state_masks,
        )
        q_proj = core.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj
        embeddings = embeddings.to(q_proj.weight.dtype)
        attention_2d = make_att_2d_masks(pad_masks, attention_masks)
        attention_4d = prepare_attention_masks_4d(attention_2d, dtype=embeddings.dtype)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        outputs, _ = core.paligemma_with_expert.forward(
            attention_mask=attention_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[embeddings, None],
            use_cache=False,
        )
        hidden = outputs[0].float()
        last_valid = pad_masks.long().sum(dim=1).sub(1).clamp_min(0)
        gather_index = last_valid[:, None, None].expand(-1, 1, hidden.shape[-1])
        return F.normalize(hidden.gather(1, gather_index).squeeze(1), dim=-1)

    def _offline_bank_batch(
        self,
        batch: dict[str, torch.Tensor],
        live_phase: torch.Tensor,
    ) -> CausalBankBatch | None:
        if self.retrieval_head is None or self.causal_bank is None:
            return None
        if self._retrieved_task_ids is None:
            self._infer_task_once(batch)
        return self.causal_bank.retrieve(
            self._retrieved_task_ids,
            live_phase,
            brief_size=self.zeva_config.brief_memory_size,
            retrieval_top_k=self.zeva_config.retrieval_top_k,
        )

    def _infer_task_once(self, batch: dict[str, torch.Tensor]) -> None:
        if self.retrieval_head is None or self.causal_bank is None:
            return
        if self._retrieved_task_ids is not None:
            return
        if getattr(self, "retrieval_source", "vlm_eos") == "task_language":
            source = self._raw_goal_embedding(batch)
        else:
            source = self.extract_vlm_features(batch)
        query = self.retrieval_head(source)
        scores = query @ self.retrieval_task_prototypes.T
        self._retrieved_task_scores, self._retrieved_task_ids = scores.max(dim=1)

    def retrieve_task_ids_from_language(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Use the frozen task-language retrieval path shared by train and deployment."""
        if self.retrieval_head is None or self.causal_bank is None:
            raise RuntimeError("Task-language retrieval and the causal bank must be loaded.")
        if getattr(self, "retrieval_source", None) != "task_language":
            raise RuntimeError("The loaded retrieval head was not trained from task language.")
        query = self.retrieval_head(self._raw_goal_embedding(batch))
        scores = query @ self.retrieval_task_prototypes.T
        confidence, task_ids = scores.max(dim=1)
        return task_ids, confidence

    @staticmethod
    def calibrate_retrieval_confidence(scores: torch.Tensor, floor: float = 0.2) -> torch.Tensor:
        """Map cosine retrieval scores to a bounded residual-injection confidence."""
        return ((scores - floor) / max(1e-6, 1.0 - floor)).clamp(0.0, 1.0)

    def set_foundation_rng_state(
        self, cpu_state: torch.Tensor, cuda_state: torch.Tensor
    ) -> None:
        """Set a rank-local one-shot RNG override for matched baseline loss."""
        self._foundation_rng_state_override = (cpu_state.cpu(), cuda_state.cpu())

    def load_foundation_anchor(self, checkpoint: str | Path) -> None:
        """Load immutable Base weights for the currently trainable action path."""
        model_path = Path(checkpoint).resolve() / "model.safetensors"
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing Base anchor weights: {model_path}")
        trainable = {
            name: parameter
            for name, parameter in self.foundation.named_parameters()
            if parameter.requires_grad
        }
        if not trainable:
            raise RuntimeError("Base anchor requires a trainable PI0.5 action path.")
        anchors: dict[str, torch.Tensor] = {}
        with safe_open(model_path, framework="pt", device="cpu") as weights:
            available = set(weights.keys())
            missing = sorted(set(trainable).difference(available))
            if missing:
                raise KeyError(f"Base anchor is missing action-path tensors: {missing[:8]}")
            for name, parameter in trainable.items():
                anchors[name] = weights.get_tensor(name).to(
                    device=parameter.device, dtype=parameter.dtype
                )
                if not torch.equal(parameter.detach(), anchors[name]):
                    raise ValueError(
                        f"ZeVA student did not start from its declared Base anchor: {name}"
                    )
        self._foundation_anchor_parameters = anchors

    def foundation_anchor_forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        reduction: str = "none",
    ):
        """Run the immutable Base action path without duplicating PaliGemma."""
        if not self._foundation_anchor_parameters:
            raise RuntimeError("No independent Base anchor has been loaded.")
        named = dict(self.foundation.named_parameters())
        original: dict[str, torch.Tensor] = {}
        try:
            with torch.no_grad():
                for name, anchor in self._foundation_anchor_parameters.items():
                    parameter = named[name]
                    original[name] = parameter.data
                    parameter.data = anchor
                return self.foundation(batch, reduction=reduction)
        finally:
            with torch.no_grad():
                for name, value in original.items():
                    named[name].data = value

    def _build_context(
        self,
        task_schema: torch.Tensor,
        phase_token: torch.Tensor,
        offline: CausalBankBatch | None = None,
    ) -> torch.Tensor:
        contexts = []
        for index, memory in enumerate(self._memories):
            brief = memory.brief_tensor(device=phase_token.device)
            retrieved = memory.retrieve(phase_token[index], device=phase_token.device)
            if offline is not None:
                offline_brief = offline.brief_signals[index : index + 1]
                offline_retrieved = offline.retrieved_signals[index : index + 1]
                brief = offline_brief if brief is None else torch.cat([offline_brief, brief], dim=1)
                retrieved = (
                    offline_retrieved
                    if retrieved is None
                    else torch.cat([offline_retrieved, retrieved], dim=1)
                )
            contexts.append(
                self.memory_context_encoder(
                    task_schema[index : index + 1],
                    phase_token[index : index + 1],
                    brief,
                    retrieved,
                )
            )
        return torch.cat(contexts, dim=0)

    def residual_injection_gates(
        self, task_schema: torch.Tensor, phase_token: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return bounded per-example gates from task language and recurrent phase."""
        router_dtype = next(self.residual_gate_router.parameters()).dtype
        router_input = torch.cat(
            [task_schema.to(router_dtype), phase_token.to(router_dtype)], dim=-1
        )
        multipliers = 2.0 * torch.sigmoid(self.residual_gate_router(router_input))
        context = torch.sigmoid(self.context_gate_logit) * multipliers[:, 0]
        prior = torch.sigmoid(self.prior_gate_logit) * multipliers[:, 1]
        return context, prior

    def initialize_residual_gate_probability(self, probability: float) -> None:
        """Set the fresh dual-residual gate strength without changing step-zero outputs.

        Both residual projectors are zero-initialized, so changing the scalar
        gate before Stage 2 training keeps the wrapped PI0.5 exactly equal to
        its foundation while avoiding vanishingly small projector gradients.
        This initializer is intentionally separate from adapter loading: it is
        only valid for a fresh residual branch.
        """
        probability = float(probability)
        if not math.isfinite(probability) or not 0.0 < probability < 1.0:
            raise ValueError(
                "initial residual gate probability must be finite and in (0, 1)."
            )
        with torch.no_grad():
            value = torch.logit(
                torch.tensor(
                    probability,
                    device=self.context_gate_logit.device,
                    dtype=self.context_gate_logit.dtype,
                )
            )
            self.context_gate_logit.copy_(value)
            self.prior_gate_logit.copy_(value)

    def _activate_residual_gates(
        self, task_schema: torch.Tensor, phase_token: torch.Tensor
    ) -> None:
        self._active_context_gate, self._active_prior_gate = self.residual_injection_gates(
            task_schema, phase_token
        )
        if (
            self._deployment_task_residual_scales
            or self._deployment_default_residual_scale != 1.0
        ):
            if self._retrieved_task_ids is None or not self.retrieval_task_names:
                raise RuntimeError(
                    "Deployment residual calibration requires task-language retrieval."
                )
            scales = torch.as_tensor(
                [
                    self._deployment_task_residual_scales.get(
                        self.retrieval_task_names[int(task_id)],
                        self._deployment_default_residual_scale,
                    )
                    for task_id in self._retrieved_task_ids.detach().cpu()
                ],
                device=self._active_context_gate.device,
                dtype=self._active_context_gate.dtype,
            )
            self._active_context_gate = self._active_context_gate * scales
            self._active_prior_gate = self._active_prior_gate * scales.to(
                dtype=self._active_prior_gate.dtype
            )

    def configure_deployment_residual_scales(
        self,
        scales: dict[str, float] | None,
        *,
        default_scale: float = 1.0,
        calibration: dict[str, Any] | None = None,
    ) -> None:
        """Install validation-selected conservative residual scales for deployment."""
        default_scale = float(default_scale)
        if not math.isfinite(default_scale) or not 0.0 <= default_scale <= 1.0:
            raise ValueError("Deployment default residual scale must be finite and in [0, 1].")
        normalized: dict[str, float] = {}
        for task_name, raw_value in (scales or {}).items():
            value = float(raw_value)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"Deployment residual scale for {task_name!r} must be finite and in [0, 1]."
                )
            normalized[str(task_name)] = value
        if self.retrieval_task_names:
            unknown = sorted(set(normalized).difference(self.retrieval_task_names))
            if unknown:
                raise ValueError(f"Residual calibration contains unknown tasks: {unknown}")
        self._deployment_task_residual_scales = normalized
        self._deployment_default_residual_scale = default_scale
        self._deployment_residual_calibration = calibration

    def _clear_active_residuals(self) -> None:
        self._active_causal_context = None
        self._active_action_prior = None
        self._active_injection_confidence = None
        self._active_prior_residual_mask = None
        self._active_context_gate = None
        self._active_prior_gate = None

    def _prepare_causal_conditioning(
        self,
        batch: dict[str, torch.Tensor],
        *,
        executed_actions: torch.Tensor | None,
        actions_normalized: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image = robotwin_multiview_image(batch)
        batch_size = image.shape[0]
        self._ensure_batch_state(batch_size)
        raw_task_embedding = self._raw_task_embedding(batch)
        raw_goal_embedding = self._raw_goal_embedding(batch)
        if getattr(self, "retrieval_source", None) == "task_language":
            self._infer_task_once(batch)
            if self.canonical_goal_embedding_table is None or self._retrieved_task_ids is None:
                raise RuntimeError("Task-language retrieval omitted canonical ZTE goal embeddings.")
            raw_goal_embedding = self.canonical_goal_embedding_table[self._retrieved_task_ids].to(
                device=raw_goal_embedding.device, dtype=raw_goal_embedding.dtype
            )
        projector_dtype = next(self.task_token_projector.parameters()).dtype
        task_schema = self.task_token_projector(raw_task_embedding.to(projector_dtype))

        if self._previous_image is None:
            phase_token, self._cte_inference_params = self.causal_transition_encoder.initialize_phase_state(
                image,
                raw_goal_embedding,
            )
        else:
            if executed_actions is None:
                if self._pending_normalized_actions is None:
                    raise RuntimeError("Missing both executed actions and the previous PI0.5 action chunk.")
                normalized_actions = self._pending_normalized_actions
            else:
                executed_actions = torch.as_tensor(
                    executed_actions,
                    dtype=torch.float32,
                    device=image.device,
                )
                if executed_actions.ndim == 2:
                    executed_actions = executed_actions.unsqueeze(0)
                normalized_actions = (
                    executed_actions[..., :ROBOTWIN_ACTION_DIM]
                    if actions_normalized
                    else self.action_normalizer.normalize(executed_actions)
                )
            causal, self._cte_inference_params = self.causal_transition_encoder.step(
                self._previous_image,
                normalized_actions,
                image,
                inference_params=self._cte_inference_params,
            )
            phase_token = causal.phase_token
            for index, memory in enumerate(self._memories):
                memory.update(phase_token[index], causal.causal_signal[index])

        offline = self._offline_bank_batch(batch, phase_token)
        conditioning_phase = offline.phase_token if offline is not None else phase_token
        causal_context = self._build_context(task_schema, conditioning_phase, offline)
        return task_schema, conditioning_phase, causal_context

    @torch.no_grad()
    def predict_action_chunk(
        self,
        batch: dict[str, torch.Tensor],
        *,
        executed_actions: torch.Tensor | None = None,
        actions_normalized: bool = False,
        reset_scope: str | None = None,
    ) -> torch.Tensor:
        """Return normalized `[B, 50, 16]` actions like baseline PI0.5."""
        if reset_scope is not None:
            self.reset(scope=reset_scope)
        task_schema, phase_token, causal_context = self._prepare_causal_conditioning(
            batch,
            executed_actions=executed_actions,
            actions_normalized=actions_normalized,
        )
        self._active_causal_context = causal_context
        self._active_action_prior = self.action_prior(task_schema, phase_token, causal_context).mean
        self._activate_residual_gates(task_schema, phase_token)
        self._active_prior_residual_mask = None
        if self._retrieved_task_scores is None:
            self._active_injection_confidence = causal_context.new_ones(causal_context.shape[0])
        else:
            self._active_injection_confidence = self.calibrate_retrieval_confidence(
                self._retrieved_task_scores
            )
        try:
            actions = self.foundation.predict_action_chunk(batch)
        finally:
            self._clear_active_residuals()
        self._previous_image = robotwin_multiview_image(batch).detach().clone()
        self._pending_normalized_actions = actions.detach().clone()
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """LeRobot-compatible one-step normalized action API."""
        if not self._action_queue:
            chunk = self.predict_action_chunk(batch)
            self._action_queue.extend(chunk.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def infer_chunk(
        self,
        raw_observation: dict[str, Any],
        *,
        executed_actions: torch.Tensor | None = None,
        reset_scope: str | None = None,
    ) -> torch.Tensor:
        """Run the saved baseline processors and return raw EEF16 commands."""
        if self.preprocessor is None or self.postprocessor is None:
            raise RuntimeError("Raw inference requires processors loaded by from_handoff().")
        if "task" not in raw_observation:
            raise ValueError("Raw RoboTwin inference requires the invariant task instruction.")
        batch = self.preprocessor(raw_observation)
        token_device = batch["observation.language.tokens"].device
        batch["zeva.goal_embedding"] = self._task_only_goal_embedding(
            raw_observation["task"], token_device
        )
        normalized = self.predict_action_chunk(
            batch,
            executed_actions=executed_actions,
            reset_scope=reset_scope,
        )
        return self.postprocessor(normalized)

    @torch.no_grad()
    def observe_final(
        self,
        batch: dict[str, torch.Tensor],
        executed_actions: torch.Tensor,
        *,
        actions_normalized: bool = False,
    ) -> list[dict[str, int]]:
        """Commit the terminal action-effect transition without planning again."""
        self._prepare_causal_conditioning(
            batch,
            executed_actions=executed_actions,
            actions_normalized=actions_normalized,
        )
        self._previous_image = robotwin_multiview_image(batch).detach().clone()
        self._pending_normalized_actions = None
        return [memory.snapshot() for memory in self._memories]

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        previous_image: torch.Tensor | None = None,
        effect_image: torch.Tensor | None = None,
        executed_actions: torch.Tensor | None = None,
        actions_normalized: bool = False,
        return_causal: bool = False,
        bank_phase_token: torch.Tensor | None = None,
        bank_brief_signals: torch.Tensor | None = None,
        bank_retrieved_signals: torch.Tensor | None = None,
        bank_brief_mask: torch.Tensor | None = None,
        bank_retrieved_mask: torch.Tensor | None = None,
        injection_confidence: torch.Tensor | None = None,
        prior_residual_mask: torch.Tensor | None = None,
        foundation_only: bool = False,
        foundation_reduction: str = "mean",
    ):
        """Train the PI0.5 flow policy with Zeva Gaussian-prior conditioning."""
        if foundation_only:
            return self.foundation(batch, reduction=foundation_reduction)
        if bank_phase_token is not None:
            task_schema = self._task_schema(batch)
            causal_context = self.memory_context_encoder(
                task_schema,
                bank_phase_token,
                bank_brief_signals,
                bank_retrieved_signals,
                bank_brief_mask,
                bank_retrieved_mask,
            )
            action_prior = self.action_prior(task_schema, bank_phase_token, causal_context)
            self._active_causal_context = causal_context
            self._active_action_prior = action_prior.mean
            self._activate_residual_gates(task_schema, bank_phase_token)
            self._active_injection_confidence = injection_confidence
            self._active_prior_residual_mask = prior_residual_mask
            try:
                if self._foundation_rng_state_override is not None:
                    cpu_state, cuda_state = self._foundation_rng_state_override
                    torch.random.set_rng_state(cpu_state)
                    torch.cuda.set_rng_state(cuda_state, batch["action"].device)
                    self._foundation_rng_state_override = None
                foundation_output = self.foundation(batch, reduction=foundation_reduction)
            finally:
                self._clear_active_residuals()
            return foundation_output, action_prior
        if previous_image is None or effect_image is None or executed_actions is None:
            raise ValueError("Joint causal training requires previous/effect images and executed actions.")
        normalized_actions = (
            executed_actions[..., :ROBOTWIN_ACTION_DIM]
            if actions_normalized
            else self.action_normalizer.normalize(executed_actions)
        )
        raw_task_embedding = self._raw_task_embedding(batch)
        raw_goal_embedding = self._raw_goal_embedding(batch)
        causal = self.causal_transition_encoder(
            previous_image,
            normalized_actions,
            effect_image,
            raw_goal_embedding,
        )
        phase_token = causal.phase_token[:, -1]
        causal_signal = causal.causal_signal[:, -1].unsqueeze(1)
        projector_dtype = next(self.task_token_projector.parameters()).dtype
        task_schema = self.task_token_projector(raw_task_embedding.to(projector_dtype))
        causal_context = self.memory_context_encoder(
            task_schema,
            phase_token,
            causal_signal,
            causal_signal,
        )
        self._active_causal_context = causal_context
        self._active_action_prior = self.action_prior(task_schema, phase_token, causal_context).mean
        self._activate_residual_gates(task_schema, phase_token)
        self._active_injection_confidence = injection_confidence
        self._active_prior_residual_mask = prior_residual_mask
        try:
            foundation_output = self.foundation(batch, reduction=foundation_reduction)
        finally:
            self._clear_active_residuals()
        return (foundation_output, causal) if return_causal else foundation_output

    def load_zte(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location="cpu")
        if checkpoint.get("schema") not in {
            "zeva-robotwin-zte-stage1-checkpoint-v4",
            "zeva-robotwin-zte-stage1-checkpoint-v5",
        }:
            raise ValueError("Formal Stage 2 requires a task-language Stage 1 v4/v5 ZTE checkpoint.")
        checkpoint_config = checkpoint.get("zte_config", {})
        expected_config = vars(self.zeva_config)
        comparable_checkpoint = {key: value for key, value in checkpoint_config.items() if key != "vision_pretrained"}
        comparable_expected = {key: value for key, value in expected_config.items() if key != "vision_pretrained"}
        if comparable_checkpoint != comparable_expected:
            raise ValueError("Stage 1 ZTE architecture differs from the RoboTwin Zeva policy.")
        self.causal_transition_encoder.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.causal_transition_encoder.requires_grad_(False).eval()

    def configure_full_finetune_stage2(self) -> list[nn.Parameter]:
        """Freeze ZTE and full-finetune PI0.5 plus Zeva, matching BehaviorVLA."""
        self.requires_grad_(False)
        self.foundation.requires_grad_(True)
        modules = (
            self.task_token_projector,
            self.memory_context_encoder,
            self.action_prior,
            self.causal_action_projector,
            self.prior_action_projector,
            self.residual_gate_router,
        )
        for module in modules:
            module.requires_grad_(True)
        self.context_gate_logit.requires_grad_(True)
        self.prior_gate_logit.requires_grad_(True)
        self.causal_transition_encoder.eval()
        if self.retrieval_head is not None:
            self.retrieval_head.requires_grad_(False).eval()
        self._full_pi05_finetune = True
        self._action_expert_finetune = False
        self.foundation.train()
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def configure_action_expert_finetune_stage2(self) -> list[nn.Parameter]:
        """Freeze PaliGemma and tune the PI0.5 action expert plus Zeva residuals."""
        self.requires_grad_(False)
        core = self.foundation.model
        action_modules = [
            core.paligemma_with_expert.gemma_expert.model,
            core.action_in_proj,
            core.action_out_proj,
            core.time_mlp_in,
            core.time_mlp_out,
        ]
        for module in action_modules:
            module.requires_grad_(True)

        zeva_modules = (
            self.task_token_projector,
            self.memory_context_encoder,
            self.action_prior,
            self.causal_action_projector,
            self.prior_action_projector,
            self.residual_gate_router,
        )
        for module in zeva_modules:
            module.requires_grad_(True)
        self.context_gate_logit.requires_grad_(True)
        self.prior_gate_logit.requires_grad_(True)

        self._full_pi05_finetune = False
        self._action_expert_finetune = True
        self.foundation.train()
        core.paligemma_with_expert.paligemma.eval()
        self.causal_transition_encoder.eval()
        if self.retrieval_head is not None:
            self.retrieval_head.requires_grad_(False).eval()

        frozen_backbone = core.paligemma_with_expert.paligemma
        if any(parameter.requires_grad for parameter in frozen_backbone.parameters()):
            raise RuntimeError("Stage 2 invariant failed: PaliGemma backbone is trainable.")
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def configure_action_expert_only_finetune(self) -> list[nn.Parameter]:
        """Matched PI0.5 baseline: tune the action expert without any Zeva path."""
        self.requires_grad_(False)
        core = self.foundation.model
        action_modules = (
            core.paligemma_with_expert.gemma_expert.model,
            core.action_in_proj,
            core.action_out_proj,
            core.time_mlp_in,
            core.time_mlp_out,
        )
        for module in action_modules:
            module.requires_grad_(True)
        self._full_pi05_finetune = False
        self._action_expert_finetune = True
        self.foundation.train()
        core.paligemma_with_expert.paligemma.eval()
        self.causal_transition_encoder.eval()
        if self.retrieval_head is not None:
            self.retrieval_head.requires_grad_(False).eval()
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def enforce_action_expert_stage2_mode(self) -> None:
        """Restore eval mode for frozen modules after the wrapper enters train mode."""
        if not getattr(self, "_action_expert_finetune", False):
            return
        self.foundation.model.paligemma_with_expert.paligemma.eval()
        self.causal_transition_encoder.eval()
        if self.retrieval_head is not None:
            self.retrieval_head.eval()

    def configure_adapter_stage2(self) -> list[nn.Parameter]:
        """Freeze the trained RoboTwin PI0.5 and ZTE; train only residual ZeVA fusion."""
        self.requires_grad_(False)
        modules = (
            self.task_token_projector,
            self.memory_context_encoder,
            self.action_prior,
            self.causal_action_projector,
            self.prior_action_projector,
            self.residual_gate_router,
        )
        for module in modules:
            module.requires_grad_(True)
        self.context_gate_logit.requires_grad_(True)
        self.prior_gate_logit.requires_grad_(True)
        self._full_pi05_finetune = False
        self._action_expert_finetune = False
        self.foundation.eval()
        self.causal_transition_encoder.eval()
        if self.retrieval_head is not None:
            self.retrieval_head.requires_grad_(False).eval()
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def configure_prior_only_adapter_stage2(
        self, *, prior_gate_probability: float = 0.5
    ) -> list[nn.Parameter]:
        """Train the action-prior branch at its deployed strength with PI0.5 frozen.

        The causal-context projector is kept identically zero, so this mode
        isolates the BehaviorVLA-style action-prior residual.  The prior gate
        is fixed rather than optimized: the zero-initialized prior projector
        still makes step zero exactly equal to the untouched PI0.5, while all
        subsequent projector gradients are trained at the same magnitude used
        for deployment instead of being attenuated by the legacy 1% gate.
        """
        probability = float(prior_gate_probability)
        if not math.isfinite(probability) or not 0.0 < probability < 1.0:
            raise ValueError("prior_gate_probability must be finite and in (0, 1).")

        self.requires_grad_(False)
        with torch.no_grad():
            self.causal_action_projector.weight.zero_()
            self.causal_action_projector.bias.zero_()
            self.context_gate_logit.fill_(-20.0)
            self.prior_gate_logit.copy_(
                torch.logit(
                    torch.tensor(
                        probability,
                        device=self.prior_gate_logit.device,
                        dtype=self.prior_gate_logit.dtype,
                    )
                )
            )
        modules = (
            self.task_token_projector,
            self.memory_context_encoder,
            self.action_prior,
            self.prior_action_projector,
            self.residual_gate_router,
        )
        for module in modules:
            module.requires_grad_(True)
        self._full_pi05_finetune = False
        self._action_expert_finetune = False
        self.foundation.eval()
        self.causal_transition_encoder.eval()
        if self.retrieval_head is not None:
            self.retrieval_head.requires_grad_(False).eval()
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def injection_gate_regularizer(
        self, task_schema: torch.Tensor | None = None, phase_token: torch.Tensor | None = None
    ) -> torch.Tensor:
        if task_schema is None or phase_token is None:
            gates = torch.stack(
                [torch.sigmoid(self.context_gate_logit), torch.sigmoid(self.prior_gate_logit)]
            )
        else:
            gates = torch.stack(self.residual_injection_gates(task_schema, phase_token), dim=-1)
        return gates.square().mean()

    def stage2_state_dict(self, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "schema": "zeva-robotwin-stage2-adapter-v5",
            "physical_contract": "joint14-relative-eef16-h50-meanstd",
            "action_prior_distribution": "diagonal_gaussian_log_std_clamped_-5_2",
            "task_token_projector": self.task_token_projector.state_dict(),
            "memory_context_encoder": self.memory_context_encoder.state_dict(),
            "action_prior": self.action_prior.state_dict(),
            "causal_action_projector": self.causal_action_projector.state_dict(),
            "prior_action_projector": self.prior_action_projector.state_dict(),
            "residual_gate_router": self.residual_gate_router.state_dict(),
            "context_gate_logit": self.context_gate_logit.detach().cpu(),
            "prior_gate_logit": self.prior_gate_logit.detach().cpu(),
            "action_normalization": self.action_normalizer.metadata(),
            "deployment_task_residual_scales": dict(self._deployment_task_residual_scales),
            "deployment_default_residual_scale": self._deployment_default_residual_scale,
            "deployment_residual_calibration": self._deployment_residual_calibration,
            "manifest": manifest,
        }

    def adapter_state_dict(self) -> dict[str, Any]:
        return {
            "schema": "zeva-robotwin-pi05-adapter-v4",
            "physical_contract": "joint14-relative-eef16-h50-meanstd",
            "action_prior_distribution": "diagonal_gaussian_log_std_clamped_-5_2",
            "causal_transition_encoder": self.causal_transition_encoder.state_dict(),
            "task_token_projector": self.task_token_projector.state_dict(),
            "memory_context_encoder": self.memory_context_encoder.state_dict(),
            "action_prior": self.action_prior.state_dict(),
            "causal_action_projector": self.causal_action_projector.state_dict(),
            "prior_action_projector": self.prior_action_projector.state_dict(),
            "residual_gate_router": self.residual_gate_router.state_dict(),
            "context_gate_logit": self.context_gate_logit.detach().cpu(),
            "prior_gate_logit": self.prior_gate_logit.detach().cpu(),
            "action_normalization": self.action_normalizer.metadata(),
            "deployment_task_residual_scales": dict(self._deployment_task_residual_scales),
            "deployment_default_residual_scale": self._deployment_default_residual_scale,
            "deployment_residual_calibration": self._deployment_residual_calibration,
        }

    def save_adapter(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.adapter_state_dict(), path)

    def load_adapter(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location="cpu")
        schema = checkpoint.get("schema")
        if schema not in {
            "zeva-robotwin-pi05-adapter-v3",
            "zeva-robotwin-pi05-adapter-v4",
            "zeva-robotwin-stage2-adapter-v4",
            "zeva-robotwin-stage2-adapter-v5",
        }:
            raise ValueError(
                "Unsupported or legacy deterministic Zeva RoboTwin adapter checkpoint. "
                "Gaussian-prior Stage 2 must start from the original PI base."
            )
        if checkpoint.get("physical_contract") != "joint14-relative-eef16-h50-meanstd":
            raise ValueError("Zeva adapter was trained for a different physical contract.")
        names = [
            "task_token_projector",
            "memory_context_encoder",
            "action_prior",
            "causal_action_projector",
            "prior_action_projector",
        ]
        if schema == "zeva-robotwin-pi05-adapter-v3":
            names.insert(0, "causal_transition_encoder")
        for name in names:
            getattr(self, name).load_state_dict(checkpoint[name])
        if "residual_gate_router" in checkpoint:
            self.residual_gate_router.load_state_dict(checkpoint["residual_gate_router"])
        self.context_gate_logit.data.copy_(checkpoint["context_gate_logit"])
        self.prior_gate_logit.data.copy_(checkpoint["prior_gate_logit"])
        self.configure_deployment_residual_scales(
            checkpoint.get("deployment_task_residual_scales"),
            default_scale=checkpoint.get("deployment_default_residual_scale", 1.0),
            calibration=checkpoint.get("deployment_residual_calibration"),
        )

    def load_retrieval(self, path: str | Path, causal_bank: str | Path) -> None:
        """Load the formal Stage 3 head and its immutable train95 bank."""
        checkpoint = torch.load(path, map_location="cpu")
        if checkpoint.get("schema") not in {
            "zeva-robotwin-stage3-retrieval-head-v1",
            "zeva-robotwin-task-language-retrieval-v1",
        }:
            raise ValueError("Unsupported Zeva RoboTwin Stage 3 checkpoint.")
        bank = RobotWinCausalBank.load(causal_bank, device=next(self.parameters()).device)
        task_names = tuple(checkpoint["task_names"])
        if task_names != bank.task_names:
            raise ValueError("Stage 3 task table differs from the causal bank.")
        prototypes = F.normalize(torch.as_tensor(checkpoint["task_prototypes"]), dim=-1)
        if prototypes.shape != (len(bank.task_names), bank.task_prototype.shape[-1]):
            raise ValueError("Stage 3 task prototypes have the wrong shape.")
        state = checkpoint["model_state_dict"]
        first_weight = state["network.0.weight"]
        last_weight = state["network.4.weight"]
        head = CausalRetrievalHead(
            input_dim=first_weight.shape[1],
            hidden_dim=first_weight.shape[0],
            output_dim=last_weight.shape[0],
            dropout=0.0,
        ).to(next(self.parameters()).device)
        head.load_state_dict(state, strict=True)
        self.retrieval_head = head.requires_grad_(False).eval()
        self.retrieval_source = checkpoint.get("source_feature", "vlm_eos")
        if self.retrieval_source == "task_language":
            canonical = checkpoint.get("canonical_goal_embeddings")
            if canonical is None:
                raise ValueError("Task-language retrieval omitted canonical ZTE goal embeddings.")
            canonical = torch.as_tensor(canonical)
            expected = (len(task_names), self.zeva_config.goal_dim)
            if tuple(canonical.shape) != expected:
                raise ValueError(f"Canonical ZTE goal table must have shape {expected}.")
            self.canonical_goal_embedding_table = canonical.to(next(self.parameters()).device)
        self.causal_bank = bank
        self.retrieval_task_prototypes = prototypes.to(next(self.parameters()).device)
        self.retrieval_task_names = task_names
        if self._deployment_task_residual_scales:
            unknown = sorted(
                set(self._deployment_task_residual_scales).difference(self.retrieval_task_names)
            )
            if unknown:
                raise ValueError(f"Residual calibration contains unknown tasks: {unknown}")

    def retrieval_diagnostics(self) -> list[dict[str, Any]]:
        if self._retrieved_task_ids is None or self._retrieved_task_scores is None:
            return []
        return [
            {"task": self.retrieval_task_names[int(task_id)], "score": float(score)}
            for task_id, score in zip(
                self._retrieved_task_ids.detach().cpu(),
                self._retrieved_task_scores.detach().cpu(),
                strict=True,
            )
        ]
