"""Frozen PI0.5 provider for the RoboTwin handoff HTTP protocol.

This module deliberately contains no ZeVA compatibility hooks.  It loads the
released LeRobot PI0.5 checkpoint with the handoff's native Transformers 5
runtime and returns the checkpoint's raw chunk-start-relative EEF16 actions.
It is used as an anchor to prove that a simulator/evaluator deployment still
reproduces the published RoboTwin baseline before comparing trained models.
"""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
import sys
import typing
from types import ModuleType
from typing import Any

import numpy as np
import torch
import typing_extensions

try:
    from outcome_trace import validate_decision_trace
except ImportError:  # pragma: no cover - package import path in unit tests.
    from .outcome_trace import validate_decision_trace


for _name in ("Self", "Unpack", "NotRequired"):
    if not hasattr(typing, _name):
        setattr(typing, _name, getattr(typing_extensions, _name))


def _load_pi05_symbols():
    """Import PI0.5 without importing every optional LeRobot policy."""
    import lerobot

    if "lerobot.policies" not in sys.modules:
        package = ModuleType("lerobot.policies")
        package.__path__ = [str(Path(next(iter(lerobot.__path__))) / "policies")]
        package.__package__ = "lerobot.policies"
        sys.modules["lerobot.policies"] = package

    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.processor import AbsoluteActionsProcessorStep
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor import RelativeActionsProcessorStep
    from lerobot.processor import batch_to_transition
    from lerobot.processor import policy_action_to_transition
    from lerobot.processor import transition_to_batch
    from lerobot.processor import transition_to_policy_action

    return {
        "PreTrainedConfig": PreTrainedConfig,
        "PI05Policy": PI05Policy,
        "AbsoluteActionsProcessorStep": AbsoluteActionsProcessorStep,
        "PolicyProcessorPipeline": PolicyProcessorPipeline,
        "RelativeActionsProcessorStep": RelativeActionsProcessorStep,
        "batch_to_transition": batch_to_transition,
        "policy_action_to_transition": policy_action_to_transition,
        "transition_to_batch": transition_to_batch,
        "transition_to_policy_action": transition_to_policy_action,
    }


def _image_tensor(image: Any) -> torch.Tensor:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"RoboTwin RGB image must be HWC RGB, got {array.shape}.")
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and array.size and array.max() <= 1.0:
            array = np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            array = np.clip(array, 0, 255).astype(np.uint8)
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).float().div_(255.0)


class Provider:
    def __init__(self, args: Mapping[str, Any]) -> None:
        symbols = _load_pi05_symbols()
        checkpoint = Path(str(args["checkpoint_path"])).resolve()
        device = str(args.get("device", "cuda:0"))
        config = symbols["PreTrainedConfig"].from_pretrained(
            str(checkpoint), local_files_only=True
        )
        config.device = device
        self.policy = symbols["PI05Policy"].from_pretrained(
            str(checkpoint), config=config, strict=True, local_files_only=True
        )
        self.policy.eval()
        self.preprocessor = symbols["PolicyProcessorPipeline"].from_pretrained(
            pretrained_model_name_or_path=str(checkpoint),
            config_filename="policy_preprocessor.json",
            overrides={},
            to_transition=symbols["batch_to_transition"],
            to_output=symbols["transition_to_batch"],
        )
        self.postprocessor = symbols["PolicyProcessorPipeline"].from_pretrained(
            pretrained_model_name_or_path=str(checkpoint),
            config_filename="policy_postprocessor.json",
            overrides={},
            to_transition=symbols["policy_action_to_transition"],
            to_output=symbols["transition_to_policy_action"],
        )
        relative_step = next(
            (
                step
                for step in self.preprocessor.steps
                if isinstance(step, symbols["RelativeActionsProcessorStep"])
            ),
            None,
        )
        if relative_step is not None:
            for step in self.postprocessor.steps:
                if (
                    isinstance(step, symbols["AbsoluteActionsProcessorStep"])
                    and step.relative_step is None
                ):
                    step.relative_step = relative_step

        trace_value = args.get("outcome_trace_enabled", os.environ.get("ZEVA_OUTCOME_TRACE", "0"))
        self.trace_enabled = str(trace_value).strip().lower() in {"1", "true", "yes", "on"}
        self.replan_index = 0
        self.previous_executed_h15: np.ndarray | None = None
        self.last_trace: dict[str, Any] | None = None

    def reset_episode(self, context: Mapping[str, Any]) -> None:
        del context
        self.policy.reset()
        self.replan_index = 0
        self.previous_executed_h15 = None
        self.last_trace = None

    @torch.inference_mode()
    def _extract_vlm_eos_feature(self, batch: Mapping[str, Any]) -> np.ndarray:
        """Extract the same normalized first-frame PI VLM feature as ZeVA."""
        from lerobot.policies.common.vla_utils import make_att_2d_masks
        from lerobot.policies.common.vla_utils import prepare_attention_masks_4d
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK
        from lerobot.utils.constants import OBS_LANGUAGE_TOKENS

        foundation = self.policy
        core = foundation.model
        images, image_masks = foundation._preprocess_images(batch)
        states, state_masks = foundation._prepare_memory_states(batch)
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
        feature = torch.nn.functional.normalize(hidden.gather(1, gather_index).squeeze(1), dim=-1)
        if tuple(feature.shape) != (1, 2048) or not torch.isfinite(feature).all():
            raise ValueError(f"PI VLM EOS feature must be finite [1,2048], got {tuple(feature.shape)}")
        return feature[0].detach().cpu().numpy()

    @torch.inference_mode()
    def predict_action_chunk(
        self,
        observation: Mapping[str, Any],
        instruction: str,
        context: Mapping[str, Any],
    ) -> np.ndarray:
        del context
        cameras = observation["observation"]
        state = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        if state.shape != (14,):
            raise ValueError(f"RoboTwin state must be Joint14, got {state.shape}.")
        raw = {
            "observation.state": torch.from_numpy(state),
            "observation.images.cam_high": _image_tensor(cameras["head_camera"]["rgb"]),
            "observation.images.cam_left_wrist": _image_tensor(cameras["left_camera"]["rgb"]),
            "observation.images.cam_right_wrist": _image_tensor(cameras["right_camera"]["rgb"]),
            "task": str(instruction),
        }
        batch = self.preprocessor(raw)
        normalized = self.policy.predict_action_chunk(batch)
        actions = torch.as_tensor(self.postprocessor(normalized)).detach().cpu()
        if actions.ndim == 3:
            actions = actions[0]
        if tuple(actions.shape) != (50, 16) or not torch.isfinite(actions).all():
            raise ValueError(f"PI0.5 must return finite [50,16], got {tuple(actions.shape)}.")
        self.last_trace = None
        if self.trace_enabled:
            eos = self._extract_vlm_eos_feature(batch) if self.replan_index == 0 else None
            trace = {
                "replan_index": self.replan_index,
                "retrieved_task": None,
                "candidate_h15_actions": actions[:15].numpy()[None, ...],
                "pairwise_distances": np.zeros((1, 1), dtype=np.float32),
                "selected_candidate": 0,
                "pi_vlm_eos_feature": eos,
                "previous_executed_h15": self.previous_executed_h15,
            }
            self.last_trace = validate_decision_trace(trace)
            self.replan_index += 1
        return actions.float().numpy()

    def begin_attempt(self, context: Mapping[str, Any]) -> None:
        del context

    def observe_transition(self, transition: Mapping[str, Any]) -> None:
        del transition

    def finalize_attempt(self, context: Mapping[str, Any]) -> None:
        del context

    def close_episode(self, context: Mapping[str, Any]) -> None:
        del context


def create_provider(args: Mapping[str, Any]) -> Provider:
    return Provider(args)


class StockSocketModel:
    """Compatibility surface for RoboTwin's lightweight TCP model server."""

    def __init__(self, args: Mapping[str, Any]) -> None:
        self.provider = Provider(args)

    def reset_model(self, payload: Mapping[str, Any] | None = None) -> None:
        # The frozen evaluator does not reseed diffusion noise per episode.
        del payload
        self.provider.reset_episode({})

    def predict(self, request: Mapping[str, Any]) -> dict[str, Any]:
        actions = self.provider.predict_action_chunk(
            request,
            str(request["task"]),
            {},
        )
        response: dict[str, Any] = {"actions": actions, "retrieval": []}
        if self.provider.trace_enabled:
            if self.provider.last_trace is None:
                raise RuntimeError("Outcome tracing enabled but PI baseline produced no trace")
            response["trace"] = self.provider.last_trace
        return response

    def commit_executed_actions(self, actions: np.ndarray) -> None:
        if not self.provider.trace_enabled:
            return
        value = np.asarray(actions, dtype=np.float32)
        if value.shape != (15, 16) or not np.isfinite(value).all():
            raise ValueError(f"Executed actions must be finite [15,16], got {value.shape}")
        self.provider.previous_executed_h15 = value.copy()


def get_model(args: Mapping[str, Any]) -> StockSocketModel:
    return StockSocketModel(args)
