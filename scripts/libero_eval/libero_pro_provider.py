"""LIBERO-Pro lifecycle provider for the matched PI0.5 and ZeVA Stage 3.

The frozen LIBERO-Pro client owns image/state preprocessing, R10 execution,
EEF conversion, OSC tracking, and success predicates.  This provider owns
only model inference and episode-local ZeVA state.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from openpi.zeva.libero_contract import LIBERO_ACTION_DIM
from openpi.zeva.libero_contract import LIBERO_MODEL_ACTION_DIM
from openpi.zeva.libero_contract import LIBERO_POLICY_HORIZON
from openpi.zeva.libero_policy import LiberoZevaPolicy


def _required(config: Mapping[str, Any], key: str) -> str:
    value = str(config.get(key) or "")
    if not value:
        raise ValueError(f"provider config requires {key!r}")
    return value


def _noise_seed(context: Mapping[str, Any]) -> int:
    """Return one stable, model-independent seed for a paired replan query."""
    identity = "|".join(
        str(context.get(key, ""))
        for key in (
            "seed",
            "base_suite",
            "perturbation",
            "task_id",
            "episode_index",
            "replan_id",
        )
    )
    digest = hashlib.blake2b(identity.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


class Provider:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self.mode = str(config.get("mode", "stage3"))
        if self.mode not in {"baseline", "stage3"}:
            raise ValueError("LIBERO-Pro provider mode must be baseline or stage3")
        self.device = torch.device(str(config.get("device", "cuda:0")))
        use_memory = self.mode == "stage3"
        stage2_root = Path(str(config.get("stage2_checkpoint", "")))
        stage3_root = Path(str(config.get("stage3_checkpoint", "")))
        self.policy = LiberoZevaPolicy.from_handoff(
            _required(config, "checkpoint_path"),
            tokenizer_path=_required(config, "tokenizer_path"),
            device=str(self.device),
            zte_checkpoint=_required(config, "zte_checkpoint") if use_memory else None,
            adapter_checkpoint=(
                str(stage2_root / "zeva_adapter.pth") if use_memory else None
            ),
            stage3_action_checkpoint=(
                str(stage3_root / "stage3_action.pth") if use_memory else None
            ),
            retrieval_checkpoint=(
                _required(config, "retrieval_checkpoint") if use_memory else None
            ),
            causal_bank=_required(config, "causal_bank") if use_memory else None,
        )
        compiled_sampler = self.policy.foundation.sample_actions
        eager_sampler = getattr(compiled_sampler, "_torchdynamo_orig_callable", None)
        if eager_sampler is not None:
            self.policy.foundation.sample_actions = eager_sampler
        self.use_memory = use_memory
        self._logged_retrieval = False

    def reset_episode(self, context: Mapping[str, Any]) -> None:
        self.policy.reset(scope="episode")
        self._logged_retrieval = False

    def begin_attempt(self, context: Mapping[str, Any]) -> None:
        if int(context.get("attempt_id", -1)) != 0:
            raise ValueError("LIBERO-Pro formal evaluation requires attempt_id=0")

    @torch.inference_mode()
    def predict_action_chunk(
        self,
        observation: Mapping[str, np.ndarray],
        instruction: str,
        context: Mapping[str, Any],
    ) -> np.ndarray:
        processed = self._preprocess_observation(observation, instruction)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(_noise_seed(context))
        noise = torch.randn(
            (1, LIBERO_POLICY_HORIZON, LIBERO_MODEL_ACTION_DIM),
            generator=generator,
            dtype=torch.float32,
            device=self.device,
        )
        actions = self.policy.sample_action_chunk(
            processed,
            use_memory=self.use_memory,
            noise=noise,
        )[0, :, :LIBERO_ACTION_DIM]
        result = np.ascontiguousarray(actions.float().cpu().numpy(), dtype=np.float32)
        result[:, 7:14] = 0.0
        result[:, 15] = 0.0
        result[:, 14] = np.clip(result[:, 14], -1.0, 1.0)
        if self.use_memory and not self._logged_retrieval:
            logging.info("LIBERO-Pro ZeVA retrieval: %s", self.policy.retrieval_diagnostics())
            self._logged_retrieval = True
        return result

    def _preprocess_observation(
        self, observation: Mapping[str, np.ndarray], instruction: str
    ):
        batch = {
            "observation.image": torch.as_tensor(
                np.asarray(observation["observation/image"]).copy()
            ).unsqueeze(0),
            "observation.wrist_image": torch.as_tensor(
                np.asarray(observation["observation/wrist_image"]).copy()
            ).unsqueeze(0),
            "observation.state": torch.as_tensor(
                np.asarray(observation["observation/state"]).copy(),
                dtype=torch.float32,
            ).unsqueeze(0),
            "prompt": [str(instruction)],
        }
        return self.policy.processor.preprocess_observation(
            batch, device=self.device
        )

    def observe_transition(self, transition: Mapping[str, Any]) -> None:
        if not self.use_memory:
            return
        executed = np.asarray(transition["executed_model_targets"], dtype=np.float32)
        causal_actions = np.asarray(transition["causal_h5_actions"], dtype=np.float32)
        if executed.ndim != 2 or executed.shape[1] != LIBERO_ACTION_DIM:
            raise ValueError(f"invalid committed model targets: {executed.shape}")
        if causal_actions.shape != executed.shape or not np.isfinite(causal_actions).all():
            raise ValueError(
                f"invalid H5-rebased causal actions: {causal_actions.shape}"
            )
        if not bool(transition.get("success", False)) and len(executed) != 10:
            raise ValueError("non-terminal LIBERO-Pro transitions must commit exact R10")
        midpoint = transition.get("causal_mid_observation")
        instruction = str(transition["context"].get("instruction") or "")
        if len(executed) >= 5:
            if midpoint is None:
                raise ValueError("R10 LIBERO-Pro commits must include the H5 midpoint observation")
            self.policy.observe_causal_transition(
                self._preprocess_observation(midpoint, instruction),
                causal_actions[:5],
            )
        if len(executed) == 10:
            self.policy.observe_causal_transition(
                self._preprocess_observation(transition["end_observation"], instruction),
                causal_actions[5:10],
            )

    def finalize_attempt(self, context: Mapping[str, Any]) -> None:
        pass

    def close_episode(self, context: Mapping[str, Any]) -> None:
        self.policy.reset(scope="episode")


def create_provider(config: Mapping[str, Any]) -> Provider:
    return Provider(config)
