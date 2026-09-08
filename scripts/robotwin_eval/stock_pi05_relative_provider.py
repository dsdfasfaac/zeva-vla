"""Frozen PI0.5 provider for the RoboTwin handoff HTTP protocol.

This module deliberately contains no ZeVA compatibility hooks.  It loads the
released LeRobot PI0.5 checkpoint with the handoff's native Transformers 5
runtime and returns the checkpoint's raw chunk-start-relative EEF16 actions.
It is used as an anchor to prove that a simulator/evaluator deployment still
reproduces the published RoboTwin baseline before comparing trained models.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import sys
import typing
from types import ModuleType
from typing import Any

import numpy as np
import torch
import typing_extensions


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

    def reset_episode(self, context: Mapping[str, Any]) -> None:
        del context
        self.policy.reset()

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

    def predict(self, request: Mapping[str, Any]) -> dict[str, np.ndarray]:
        actions = self.provider.predict_action_chunk(
            request,
            str(request["task"]),
            {},
        )
        return {"actions": actions, "retrieval": []}

    def commit_executed_actions(self, actions: np.ndarray) -> None:
        del actions


def get_model(args: Mapping[str, Any]) -> StockSocketModel:
    return StockSocketModel(args)
