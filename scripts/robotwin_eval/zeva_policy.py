"""RoboTwin 2.0 client/server adapter for the formal three-stage Zeva policy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

_WORLD_FROM_CAMERA_ROTATION = np.stack(
    (
        np.asarray([1.0, 0.0, 0.0]),
        -np.cross(
            np.asarray([0.0, 0.6, -0.8]),
            np.asarray([-1.0, 0.0, 0.0]),
        ),
        np.asarray([0.0, 0.6, -0.8]),
    ),
    axis=1,
)


def _xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64)
    value /= np.linalg.norm(value)
    x, y, z, w = value
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    return _xyzw_to_matrix(np.asarray(quaternion)[[1, 2, 3, 0]])


def _matrix_to_wxyz(matrix: np.ndarray) -> np.ndarray:
    """Match the frozen baseline evaluator's canonical matrix conversion."""
    matrix = np.asarray(matrix, dtype=np.float64)
    r00 = matrix[0, 0]
    r11 = matrix[1, 1]
    r22 = matrix[2, 2]
    xyzw = np.asarray(
        [
            np.copysign(np.sqrt(max(0.0, 1.0 + r00 - r11 - r22)) * 0.5, matrix[2, 1] - matrix[1, 2]),
            np.copysign(np.sqrt(max(0.0, 1.0 - r00 + r11 - r22)) * 0.5, matrix[0, 2] - matrix[2, 0]),
            np.copysign(np.sqrt(max(0.0, 1.0 - r00 - r11 + r22)) * 0.5, matrix[1, 0] - matrix[0, 1]),
            np.sqrt(max(0.0, 1.0 + r00 + r11 + r22)) * 0.5,
        ],
        dtype=np.float64,
    )
    xyzw /= np.linalg.norm(xyzw)
    if xyzw[3] < 0.0 or (
        abs(xyzw[3]) <= 1e-12
        and next((value for value in xyzw[:3] if abs(value) > 1e-12), 0.0) < 0.0
    ):
        xyzw *= -1.0
    return xyzw[[3, 0, 1, 2]]


def _relative_chunk_to_robotwin(
    task_env: Any,
    relative_chunk: np.ndarray,
) -> np.ndarray:
    """Map camera-axis chunk-start deltas to RoboTwin world EEF targets."""
    actions = np.asarray(relative_chunk, dtype=np.float64)
    if actions.shape != (50, 16) or not np.isfinite(actions).all():
        raise ValueError(f"Zeva must return finite [50,16], got {actions.shape}.")
    starts = (
        np.asarray(task_env.get_arm_pose("left"), dtype=np.float64),
        np.asarray(task_env.get_arm_pose("right"), dtype=np.float64),
    )
    robotwin = np.empty_like(actions)
    slots = (
        (slice(0, 3), slice(3, 7), 14, slice(0, 3), slice(3, 7), 7),
        (slice(7, 10), slice(10, 14), 15, slice(8, 11), slice(11, 15), 15),
    )
    camera_from_world = _WORLD_FROM_CAMERA_ROTATION.T
    for start, (
        position,
        quaternion,
        gripper,
        output_position,
        output_quaternion,
        output_gripper,
    ) in zip(starts, slots, strict=True):
        start_rotation_world = _wxyz_to_matrix(start[3:7])
        robotwin[:, output_position] = (
            start[:3] + actions[:, position] @ _WORLD_FROM_CAMERA_ROTATION.T
        )
        delta_camera = np.stack([_xyzw_to_matrix(value) for value in actions[:, quaternion]])
        delta_world = _WORLD_FROM_CAMERA_ROTATION @ delta_camera @ camera_from_world
        target_rotation = delta_world @ start_rotation_world
        robotwin[:, output_quaternion] = np.stack([_matrix_to_wxyz(value) for value in target_rotation])
        robotwin[:, output_gripper] = np.clip(actions[:, gripper], 0.0, 1.0)
    return robotwin.astype(np.float32)


class ZevaModel:
    def __init__(self, args: dict[str, Any]):
        import torch  # noqa: PLC0415

        from openpi.zeva.robotwin_contract import prepare_robotwin_pi_image  # noqa: PLC0415
        from openpi.zeva.robotwin_policy import RobotWinZevaPolicy  # noqa: PLC0415

        self._torch = torch
        self._prepare_robotwin_pi_image = prepare_robotwin_pi_image
        self._baseline_only = bool(args.get("baseline_only", False))
        self._candidate_selector = args.get("candidate_selector")
        self._candidate_count = int(args.get("candidate_count", 4))
        self._phase_confidence_floor = float(args.get("phase_confidence_floor", 0.85))
        if self._candidate_selector not in {None, "phase_gated_consensus_medoid"}:
            raise ValueError(f"Unsupported candidate selector: {self._candidate_selector!r}.")
        if self._candidate_selector and self._candidate_count != 4:
            raise ValueError("The frozen-PI consensus protocol requires exactly four candidates.")
        stage2_checkpoint = args.get("stage2_checkpoint")
        if not self._baseline_only and not stage2_checkpoint and not self._candidate_selector:
            raise ValueError("ZeVA evaluation requires stage2_checkpoint.")
        self.policy = RobotWinZevaPolicy.from_handoff(
            args["handoff_root"],
            device=args.get("device", "cuda"),
            foundation_checkpoint=args.get("foundation_checkpoint"),
            goal_embedding_checkpoint=(
                None if self._baseline_only else args.get("goal_embedding_checkpoint")
            ),
            zte_checkpoint=None if self._baseline_only else args["zte_checkpoint"],
            adapter_checkpoint=(
                None
                if self._baseline_only or not stage2_checkpoint
                else str(Path(stage2_checkpoint) / "zeva_adapter.pth")
            ),
            # Stage 2 saves the complete action-expert PI0.5 weights separately
            # from the ZeVA adapter.  Both trained variants must load them here;
            # omitting this silently evaluates the original foundation model.
            stage2_checkpoint=stage2_checkpoint,
            retrieval_checkpoint=None if self._baseline_only else args["retrieval_checkpoint"],
            causal_bank=None if self._baseline_only else args["causal_bank"],
        )
        # Keep the frozen PI0.5 evaluator's continuous diffusion stream, while
        # making the process-initial stream reproducible across paired
        # Anchor/Base/ZeVA conditions.  Loading can consume RNG while modules
        # are constructed, so seed only after every checkpoint is restored.
        self._model_rng_seed = (
            None if args.get("model_rng_seed") is None else int(args["model_rng_seed"])
        )
        if self._model_rng_seed is not None:
            self._torch.manual_seed(self._model_rng_seed)
            if self._torch.cuda.is_available():
                self._torch.cuda.manual_seed_all(self._model_rng_seed)
        self._previous_commands: torch.Tensor | None = None
        self._proposal_cpu_rng_state = None
        self._proposal_cuda_rng_state = None
        self._reset_proposal_rng(self._model_rng_seed or 0)

    def _reset_proposal_rng(self, seed: int) -> None:
        """Create an isolated proposal stream without moving the Base stream."""
        base_cpu = self._torch.random.get_rng_state()
        base_cuda = (
            self._torch.cuda.get_rng_state_all() if self._torch.cuda.is_available() else None
        )
        self._torch.manual_seed(int(seed) ^ 0x5EEDC0DE)
        if self._torch.cuda.is_available():
            self._torch.cuda.manual_seed_all(int(seed) ^ 0x5EEDC0DE)
        self._proposal_cpu_rng_state = self._torch.random.get_rng_state()
        self._proposal_cuda_rng_state = (
            self._torch.cuda.get_rng_state_all() if self._torch.cuda.is_available() else None
        )
        self._torch.random.set_rng_state(base_cpu)
        if base_cuda is not None:
            self._torch.cuda.set_rng_state_all(base_cuda)

    def reset_model(self, payload: dict[str, Any] | None = None) -> None:
        if payload is not None and "seed" in payload:
            seed = int(payload["seed"])
            self._torch.manual_seed(seed)
            if self._torch.cuda.is_available():
                self._torch.cuda.manual_seed_all(seed)
            self._reset_proposal_rng(seed)
        self.policy.reset(scope="episode")
        self._previous_commands = None

    def _frozen_pi_candidates(self, batch: dict[str, Any]):
        """Draw K candidates while preserving candidate-0's future Base RNG."""
        candidate_zero = self.policy.foundation.predict_action_chunk(batch)
        after_base_cpu = self._torch.random.get_rng_state()
        after_base_cuda = (
            self._torch.cuda.get_rng_state_all() if self._torch.cuda.is_available() else None
        )
        self._torch.random.set_rng_state(self._proposal_cpu_rng_state)
        if self._proposal_cuda_rng_state is not None:
            self._torch.cuda.set_rng_state_all(self._proposal_cuda_rng_state)
        candidates = [candidate_zero]
        for _ in range(self._candidate_count - 1):
            candidates.append(self.policy.foundation.predict_action_chunk(batch))
        self._proposal_cpu_rng_state = self._torch.random.get_rng_state()
        self._proposal_cuda_rng_state = (
            self._torch.cuda.get_rng_state_all() if self._torch.cuda.is_available() else None
        )
        self._torch.random.set_rng_state(after_base_cpu)
        if after_base_cuda is not None:
            self._torch.cuda.set_rng_state_all(after_base_cuda)
        return self._torch.stack(candidates, dim=1)

    def _phase_gated_consensus_chunk(self, raw: dict[str, Any]):
        from openpi.zeva.robotwin_policy import robotwin_multiview_image  # noqa: PLC0415

        batch = self.policy.preprocessor(raw)
        batch["zeva.goal_embedding"] = self.policy._task_only_goal_embedding(  # noqa: SLF001
            raw["task"], batch["observation.state"].device
        )
        self.policy._prepare_causal_conditioning(  # noqa: SLF001
            batch,
            executed_actions=self._previous_commands,
            actions_normalized=False,
        )
        candidates = self._frozen_pi_candidates(batch)
        prefix = candidates[:, :, :15]
        pairwise = (prefix[:, :, None] - prefix[:, None, :]).square().mean(dim=(3, 4))
        medoid_index = pairwise.sum(dim=-1).argmin(dim=-1)

        task_ids = self.policy._retrieved_task_ids  # noqa: SLF001
        live_phase = self.policy._last_live_phase_token  # noqa: SLF001
        bank = self.policy.causal_bank
        if task_ids is None or live_phase is None or bank is None:
            raise RuntimeError("Consensus selector requires task-language retrieval and live phase.")
        phase_table = bank.phase_key[task_ids]
        valid = bank.count[task_ids] > 0
        phase_score = self._torch.einsum(
            "bpd,bd->bp",
            self._torch.nn.functional.normalize(phase_table.float(), dim=-1),
            self._torch.nn.functional.normalize(live_phase.float(), dim=-1),
        ).masked_fill(~valid, -self._torch.inf)
        confidence = phase_score.max(dim=1).values
        selected_index = self._torch.where(
            confidence >= self._phase_confidence_floor,
            medoid_index,
            self._torch.zeros_like(medoid_index),
        )
        selected = candidates[
            self._torch.arange(len(candidates), device=candidates.device), selected_index
        ]
        self.policy._previous_image = robotwin_multiview_image(batch).detach().clone()  # noqa: SLF001
        self.policy._pending_normalized_actions = selected.detach().clone()  # noqa: SLF001
        return self.policy.postprocessor(selected), confidence, selected_index

    def commit_executed_actions(self, value: np.ndarray) -> None:
        """Record only the H15 controls that RoboTwin actually executed."""
        actions = self._torch.as_tensor(value, dtype=self._torch.float32)
        if actions.ndim != 2 or actions.shape[1] != 16 or not 1 <= actions.shape[0] <= 15:
            raise ValueError(f"Executed actions must have shape [1..15,16], got {tuple(actions.shape)}.")
        if not self._torch.isfinite(actions).all():
            raise ValueError("Executed actions contain non-finite values.")
        self._previous_commands = actions.clone()

    def _model_image(self, value: np.ndarray, name: str):
        """Match the training loader's CHW float32 [0,1] image contract."""
        return self._prepare_robotwin_pi_image(value, name=name)

    def predict(self, observation: dict[str, Any]) -> dict[str, Any]:
        source = observation["observation"]
        state = self._torch.as_tensor(
            observation["joint_action"]["vector"], dtype=self._torch.float32
        )
        if state.shape != (14,) or not self._torch.isfinite(state).all():
            raise ValueError(f"RoboTwin state must be finite Joint14, got {tuple(state.shape)}.")
        raw = {
            "observation.state": state,
            "observation.images.cam_high": self._model_image(
                source["head_camera"]["rgb"], "head_camera"
            ),
            "observation.images.cam_left_wrist": self._model_image(
                source["left_camera"]["rgb"], "left_camera"
            ),
            "observation.images.cam_right_wrist": self._model_image(
                source["right_camera"]["rgb"], "right_camera"
            ),
            "task": observation["task"],
        }
        selector_diagnostics = None
        if self._baseline_only:
            batch = self.policy.preprocessor(raw)
            normalized = self.policy.foundation.predict_action_chunk(batch)
            chunk = self.policy.postprocessor(normalized)
        elif self._candidate_selector == "phase_gated_consensus_medoid":
            chunk, phase_confidence, selected_index = self._phase_gated_consensus_chunk(raw)
            selector_diagnostics = {
                "phase_confidence": float(phase_confidence[0].detach().cpu()),
                "selected_candidate": int(selected_index[0].detach().cpu()),
                "fallback_to_base": bool(int(selected_index[0].detach().cpu()) == 0),
            }
        else:
            chunk = self.policy.infer_chunk(raw, executed_actions=self._previous_commands)
        chunk = self._torch.as_tensor(chunk).detach().cpu()
        if chunk.ndim == 3:
            if chunk.shape[0] != 1:
                raise ValueError(f"Formal RoboTwin eval requires batch size one, got {tuple(chunk.shape)}.")
            chunk = chunk[0]
        return {
            "actions": chunk.numpy().astype(np.float32),
            "retrieval": [] if self._baseline_only else self.policy.retrieval_diagnostics(),
            "selector": selector_diagnostics,
        }


def get_model(args: dict[str, Any]) -> ZevaModel:
    return ZevaModel(args)


def eval(task_env: Any, model: Any, observation: dict[str, Any]) -> None:
    """Execute one predicted chunk, stopping immediately on success/step limit."""
    observation = dict(observation)
    observation["task"] = task_env.get_instruction()
    response = model.call(func_name="predict", obs=observation)
    actions = _relative_chunk_to_robotwin(task_env, response["actions"])
    if response.get("retrieval"):
        item = response["retrieval"][0]
        print(f"Zeva task-language retrieval: {item['task']} score={item['score']:.4f}")
    if response.get("selector"):
        item = response["selector"]
        print(
            "Zeva PI consensus: "
            f"candidate={item['selected_candidate']} "
            f"phase_score={item['phase_confidence']:.4f} "
            f"base_fallback={item['fallback_to_base']}"
        )
    executed_model_actions = []
    for model_action, action in zip(response["actions"][:15], actions[:15], strict=True):
        if task_env.take_action_cnt >= task_env.step_lim or task_env.eval_success:
            break
        task_env.take_action(action, action_type="ee")
        executed_model_actions.append(model_action)
    # The frozen comparison protocol commits only complete H15 transitions.
    if len(executed_model_actions) == 15:
        model.call(
            func_name="commit_executed_actions",
            obs=np.asarray(executed_model_actions, dtype=np.float32),
        )
