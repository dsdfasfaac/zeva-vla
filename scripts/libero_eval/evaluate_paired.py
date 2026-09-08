#!/usr/bin/env python3
"""Paired official LIBERO rollout with the baseline EEF16 OSC protocol."""

from __future__ import annotations

import argparse
import collections
import json
import logging
import math
from pathlib import Path
import importlib.util
import sys
import time

import imageio.v2 as imageio
import numpy as np
import torch

# The pinned simulator venv deliberately contains only LIBERO dependencies;
# reuse the host's websocket client dependency without replacing its NumPy stack.
if importlib.util.find_spec("websockets") is None:
    sys.path.extend(
        (
            "/mnt/100T/users/dingxin/VLA/zeva-eval/runtime-deps-py310",
            "/usr/local/lib/python3.10/dist-packages",
        )
    )
if importlib.util.find_spec("msgpack") is None:
    sys.path.extend(
        (
            "/mnt/100T/users/dingxin/VLA/zeva-eval/runtime-deps-py310",
            "/data1/dingxin/lingbot_va_robotwin10/robotwin_ws",
        )
    )

# LIBERO init files are trusted NumPy artifacts. Torch 2.6 changed this default.
_torch_load = torch.load
torch.load = lambda *args, **kwargs: _torch_load(*args, weights_only=False, **kwargs)

from egoscale.pi05.eef_contract import quaternion_xyzw_to_rotation_matrix
from egoscale.pi05.libero_camera_frame import (
    libero_stage1_actions_to_world_command_targets,
    libero_states_world_to_stage1_slots,
    libero_world_target_with_gripper_command_to_osc_action,
    world_from_live_opencv_camera,
)
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client.websocket_client_policy import WebsocketClientPolicy


DUMMY_ACTION = [0.0] * 6 + [-1.0]
MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("baseline", "stage2", "stage3"), required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--suite", choices=tuple(MAX_STEPS), required=True)
    parser.add_argument("--task-ids", default="0,5")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--max-osc-steps", type=int, default=20)
    parser.add_argument("--position-tolerance-m", type=float, default=0.005)
    parser.add_argument("--rotation-tolerance-deg", type=float, default=3.0)
    parser.add_argument("--max-model-targets", type=int)
    args = parser.parse_args()
    args.task_ids = tuple(int(value) for value in args.task_ids.split(",") if value)
    if not args.task_ids or any(value < 0 or value >= 10 for value in args.task_ids):
        parser.error("--task-ids must be comma-separated values in [0,9].")
    if not 1 <= args.episodes <= 50:
        parser.error("--episodes must be in [1,50].")
    if args.replan_steps != 5:
        parser.error("The selected checkpoint requires H10 prediction / H5 execution.")
    return args


def quat_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64).copy()
    value[3] = np.clip(value[3], -1.0, 1.0)
    denominator = math.sqrt(max(0.0, 1.0 - value[3] * value[3]))
    if math.isclose(denominator, 0.0):
        return np.zeros(3, dtype=np.float64)
    return value[:3] * (2.0 * math.acos(value[3]) / denominator)


def target_errors(observation, position: np.ndarray, quaternion: np.ndarray) -> tuple[float, float]:
    position_error = float(np.linalg.norm(position - observation["robot0_eef_pos"]))
    current = quaternion_xyzw_to_rotation_matrix(observation["robot0_eef_quat"])
    target = quaternion_xyzw_to_rotation_matrix(quaternion)
    cosine = float(np.clip((np.trace(target @ current.T) - 1.0) / 2.0, -1.0, 1.0))
    return position_error, math.degrees(math.acos(cosine))


def model_state(observation, world_from_camera: np.ndarray) -> np.ndarray:
    raw = np.concatenate(
        (
            observation["robot0_eef_pos"],
            quat_to_axis_angle(observation["robot0_eef_quat"]),
            observation["robot0_gripper_qpos"],
        )
    )
    return libero_states_world_to_stage1_slots(
        raw[None], world_from_camera=world_from_camera
    )[0]


def images(observation) -> tuple[np.ndarray, np.ndarray]:
    agent = np.ascontiguousarray(observation["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(observation["robot0_eye_in_hand_image"][::-1, ::-1])
    return (
        image_tools.convert_to_uint8(image_tools.resize_with_pad(agent, 224, 224)),
        image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, 224, 224)),
    )


def noise_seed(base: int, suite: str, task_id: int, episode: int, query: int) -> int:
    suite_id = tuple(MAX_STEPS).index(suite)
    return int(base * 10_000_000 + suite_id * 1_000_000 + task_id * 10_000 + episode * 100 + query)


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    args.video_dir.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.unlink()
    client = WebsocketClientPolicy(args.host, args.port)
    suite = benchmark.get_benchmark_dict()[args.suite]()
    total_successes = 0
    total_episodes = 0

    for task_id in args.task_ids:
        task = suite.get_task(task_id)
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(
            bddl_file_name=bddl,
            camera_heights=256,
            camera_widths=256,
            ignore_done=True,
        )
        env.seed(args.seed)
        initial_states = suite.get_task_init_states(task_id)
        camera_id = env.env.sim.model.camera_name2id("agentview")
        world_from_camera = world_from_live_opencv_camera(
            env.env.sim.data.cam_xpos[camera_id],
            env.env.sim.data.cam_xmat[camera_id].reshape(3, 3),
        )
        output_max = np.asarray(env.env.robots[0].controller.output_max, dtype=np.float32)

        for episode in range(args.episodes):
            start_time = time.monotonic()
            env.reset()
            observation = env.set_init_state(initial_states[episode])
            for _ in range(10):
                observation, _, _, _ = env.step(DUMMY_ACTION)
            action_plan: collections.deque = collections.deque()
            replay = []
            targets = reached = osc_steps = queries = 0
            done = False
            first_query = True
            retrieval = []

            target_budget = MAX_STEPS[args.suite]
            if args.max_model_targets is not None:
                target_budget = min(target_budget, args.max_model_targets)
            while targets < target_budget and not done:
                agent, wrist = images(observation)
                replay.append(agent)
                if not action_plan:
                    request = {
                        "observation/image": agent,
                        "observation/wrist_image": wrist,
                        "observation/state": model_state(observation, world_from_camera),
                        "prompt": str(task.language),
                        "noise_seed": noise_seed(args.seed, args.suite, task_id, episode, queries),
                    }
                    if first_query:
                        request["reset_scope"] = "episode"
                        first_query = False
                    response = client.infer(request)
                    chunk = np.asarray(response["actions"], dtype=np.float32)
                    if chunk.shape != (10, 16) or not np.isfinite(chunk).all():
                        raise ValueError(f"Policy returned invalid EEF16 chunk {chunk.shape}.")
                    positions, quaternions, grippers = (
                        libero_stage1_actions_to_world_command_targets(
                            chunk,
                            start_position_world=observation["robot0_eef_pos"],
                            start_quaternion_xyzw=observation["robot0_eef_quat"],
                            world_from_camera=world_from_camera,
                        )
                    )
                    action_plan.extend(
                        zip(
                            positions[: args.replan_steps],
                            quaternions[: args.replan_steps],
                            grippers[: args.replan_steps],
                            strict=True,
                        )
                    )
                    retrieval = response.get("retrieval", retrieval)
                    queries += 1

                target_position, target_quaternion, target_gripper = action_plan.popleft()
                target_reached = False
                for _ in range(args.max_osc_steps):
                    action = libero_world_target_with_gripper_command_to_osc_action(
                        target_position_world=target_position,
                        target_quaternion_xyzw=target_quaternion,
                        target_gripper_command=float(target_gripper),
                        current_position_world=observation["robot0_eef_pos"],
                        current_quaternion_xyzw=observation["robot0_eef_quat"],
                        controller_output_max=output_max,
                    )
                    observation, _, done, _ = env.step(action.tolist())
                    osc_steps += 1
                    if done:
                        break
                    position_error, rotation_error = target_errors(
                        observation, target_position, target_quaternion
                    )
                    if (
                        position_error <= args.position_tolerance_m
                        and rotation_error <= args.rotation_tolerance_deg
                    ):
                        target_reached = True
                        break
                reached += int(target_reached)
                targets += 1

            total_episodes += 1
            total_successes += int(done)
            record = {
                "schema": "zeva-libero-paired-pilot-v1",
                "model": args.model,
                "suite": args.suite,
                "task_id": task_id,
                "task": str(task.language),
                "episode": episode,
                "seed": args.seed,
                "success": bool(done),
                "targets": targets,
                "targets_reached": reached,
                "osc_steps": osc_steps,
                "queries": queries,
                "elapsed_seconds": time.monotonic() - start_time,
                "retrieval": retrieval,
            }
            append_jsonl(args.output, record)
            suffix = "success" if done else "failure"
            safe_task = "_".join(str(task.language).split())
            imageio.mimwrite(
                args.video_dir
                / f"{args.model}_{args.suite}_task{task_id:02d}_ep{episode:02d}_{suffix}.mp4",
                replay,
                fps=10,
            )
            logging.info(
                "%s %s task=%d episode=%d success=%s aggregate=%d/%d",
                args.model,
                args.suite,
                task_id,
                episode,
                done,
                total_successes,
                total_episodes,
            )
        env.close()

    summary = {
        "event": "summary",
        "model": args.model,
        "suite": args.suite,
        "task_ids": args.task_ids,
        "episodes_per_task": args.episodes,
        "successes": total_successes,
        "episodes": total_episodes,
        "success_rate": total_successes / total_episodes,
    }
    append_jsonl(args.output, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
