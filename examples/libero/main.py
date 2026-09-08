import collections
import dataclasses
import fcntl
import json
import logging
import math
import os
import pathlib
import time
from typing import Any

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data

LOG_TS_FMT = "%Y-%m-%d %H:%M:%S"

def _now_str():
    return time.strftime(LOG_TS_FMT, time.localtime())

def safe_append(log_file: str, record: dict[str, Any]) -> None:
    path = pathlib.Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    log_file: str = "./libero_benchmark_object.txt"
    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_object"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task
    max_attempts_per_episode: int = 4

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos
    causal_rollout_dir: str | None = None

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    safe_append(args.log_file, {
        "ts": _now_str(),
        "event": "process_start",
        "pid": os.getpid(),
        "task_suite": args.task_suite_name,
        "port": args.port,
        "seed": args.seed,
        "replan_steps": args.replan_steps,
    })

    # Zeva evaluates fixed episodes. Each fixed initialization may be retried
    # while PIM persists, whereas independently randomized episodes never share memory.
    total_episodes, total_successes = 0, 0
    cumulative_successes = np.zeros(args.max_attempts_per_episode, dtype=np.int64)
    transition_index = 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start fixed episodes
        task_episodes, task_successes = 0, 0
        num_fixed_episodes = min(args.num_trials_per_task, len(initial_states))
        for episode_idx in tqdm.tqdm(range(num_fixed_episodes)):
            logging.info("Task: %s | fixed episode: %d", task_description, episode_idx)
            success_attempt = None
            episode_replay = []

            for attempt_idx in range(args.max_attempts_per_episode):
                env.reset()
                obs = env.set_init_state(initial_states[episode_idx])
                action_plan = collections.deque()
                executed_actions = []
                previous_query_image = None
                previous_query_wrist = None
                previous_query_state = None
                previous_planned_actions = None
                attempt_replay = []
                reset_scope = "episode" if attempt_idx == 0 else "attempt"
                t = 0
                done = False

                logging.info("Starting Zeva attempt %d/%d", attempt_idx + 1, args.max_attempts_per_episode)
                while t < max_steps + args.num_steps_wait:
                    try:
                        if t < args.num_steps_wait:
                            obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                            t += 1
                            continue

                        img, wrist_img = _preprocess_images(obs, args.resize_size)
                        attempt_replay.append(img)

                        if not action_plan:
                            if previous_query_image is not None and executed_actions and args.causal_rollout_dir:
                                transition_index = _save_causal_transition(
                                    args.causal_rollout_dir,
                                    transition_index,
                                    image_before=previous_query_image,
                                    image_after=img,
                                    wrist_before=previous_query_wrist,
                                    wrist_after=wrist_img,
                                    state_before=previous_query_state,
                                    state_after=_robot_state(obs),
                                    executed_actions=executed_actions,
                                    planned_actions=previous_planned_actions,
                                    task_id=task_id,
                                    episode_id=episode_idx,
                                    attempt_id=attempt_idx,
                                    progress=np.clip((t - args.num_steps_wait) / max_steps, 0.0, 1.0),
                                    task_description=task_description,
                                )

                            element = _make_policy_observation(obs, img, wrist_img, task_description)
                            element.update(
                                {
                                    "episode_id": episode_idx,
                                    "attempt_id": attempt_idx,
                                    "reset_scope": reset_scope,
                                }
                            )
                            reset_scope = None
                            if executed_actions:
                                element["executed_actions"] = np.asarray(executed_actions, dtype=np.float32)
                                element["executed_steps"] = len(executed_actions)

                            response = client.infer(element)
                            action_chunk = response["actions"]
                            if len(action_chunk) < args.replan_steps:
                                raise ValueError(
                                    f"Policy returned {len(action_chunk)} actions, expected at least {args.replan_steps}."
                                )
                            action_plan.extend(action_chunk[: args.replan_steps])
                            previous_query_image = img.copy()
                            previous_query_wrist = wrist_img.copy()
                            previous_query_state = _robot_state(obs).copy()
                            previous_planned_actions = np.asarray(action_chunk, dtype=np.float32).copy()
                            executed_actions = []

                        action = np.asarray(action_plan.popleft(), dtype=np.float32)
                        executed_actions.append(action.copy())
                        obs, _, done, _ = env.step(action.tolist())
                        t += 1
                        if done:
                            break
                    except Exception:
                        logging.exception("Zeva attempt failed with an exception")
                        break

                # Commit the final observed action-effect transition before any reset.
                if previous_query_image is not None and executed_actions:
                    final_img, final_wrist = _preprocess_images(obs, args.resize_size)
                    final_element = _make_policy_observation(obs, final_img, final_wrist, task_description)
                    final_element.update(
                        {
                            "executed_actions": np.asarray(executed_actions, dtype=np.float32),
                            "executed_steps": len(executed_actions),
                            "episode_id": episode_idx,
                            "attempt_id": attempt_idx,
                            "observe_only": True,
                        }
                    )
                    client.infer(final_element)
                    if args.causal_rollout_dir:
                        transition_index = _save_causal_transition(
                            args.causal_rollout_dir,
                            transition_index,
                            image_before=previous_query_image,
                            image_after=final_img,
                            wrist_before=previous_query_wrist,
                            wrist_after=final_wrist,
                            state_before=previous_query_state,
                            state_after=_robot_state(obs),
                            executed_actions=executed_actions,
                            planned_actions=previous_planned_actions,
                            task_id=task_id,
                            episode_id=episode_idx,
                            attempt_id=attempt_idx,
                            progress=np.clip((t - args.num_steps_wait) / max_steps, 0.0, 1.0),
                            task_description=task_description,
                        )

                episode_replay.extend(attempt_replay)
                suffix = "success" if done else "failure"
                task_segment = task_description.replace(" ", "_")
                imageio.mimwrite(
                    pathlib.Path(args.video_out_path)
                    / f"zeva_{task_segment}_ep{episode_idx:03d}_attempt{attempt_idx + 1}_{suffix}.mp4",
                    [np.asarray(image) for image in attempt_replay],
                    fps=10,
                )
                if done:
                    success_attempt = attempt_idx + 1
                    break

            task_episodes += 1
            total_episodes += 1
            if success_attempt is not None:
                task_successes += 1
                total_successes += 1
                cumulative_successes[success_attempt - 1 :] += 1

            logging.info(
                "Fixed episode result: success_attempt=%s, CSR=%s",
                success_attempt,
                (cumulative_successes / total_episodes).round(4).tolist(),
            )

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")

    safe_append(args.log_file, {
        "ts": _now_str(),
        "event": "run_summary",
        "pid": os.getpid(),
        "task_suite": args.task_suite_name,
        "port": args.port,
        "total_episodes": int(total_episodes),
        "total_successes": int(total_successes),
        "total_success_rate": float(float(total_successes) / float(total_episodes)),
        "max_attempts_per_episode": args.max_attempts_per_episode,
        "csr_at_k": (cumulative_successes / total_episodes).tolist(),
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "unknown"),
    })


def _preprocess_images(obs, resize_size):
    image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_image = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    image = image_tools.convert_to_uint8(image_tools.resize_with_pad(image, resize_size, resize_size))
    wrist_image = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist_image, resize_size, resize_size)
    )
    return image, wrist_image


def _make_policy_observation(obs, image, wrist_image, task_description):
    return {
        "observation/image": image,
        "observation/wrist_image": wrist_image,
        "observation/state": _robot_state(obs),
        "prompt": str(task_description),
    }


def _robot_state(obs):
    return np.concatenate(
        (
            obs["robot0_eef_pos"],
            _quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    )


def _save_causal_transition(
    output_dir,
    index,
    *,
    image_before,
    image_after,
    wrist_before,
    wrist_after,
    state_before,
    state_after,
    executed_actions,
    planned_actions,
    task_id,
    episode_id,
    attempt_id,
    progress,
    task_description,
):
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path / f"transition_{index:08d}.npz",
        image_before=np.asarray(image_before, dtype=np.uint8),
        image_after=np.asarray(image_after, dtype=np.uint8),
        wrist_before=np.asarray(wrist_before, dtype=np.uint8),
        wrist_after=np.asarray(wrist_after, dtype=np.uint8),
        state_before=np.asarray(state_before, dtype=np.float32),
        state_after=np.asarray(state_after, dtype=np.float32),
        executed_actions=np.asarray(executed_actions, dtype=np.float32),
        planned_actions=np.asarray(planned_actions, dtype=np.float32),
        task_id=np.int64(task_id),
        episode_id=np.int64(episode_id),
        attempt_id=np.int64(attempt_id),
        progress=np.float32(progress),
        task_description=np.asarray(str(task_description)),
    )
    return index + 1


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
