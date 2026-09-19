import sys
import os
import subprocess

sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import numpy as np
from pathlib import Path
from collections import deque
import traceback

import yaml
from datetime import datetime
import importlib
import argparse
import pdb

from generate_episode_instructions import *


import sys
import os
import subprocess
import socket
import json
import threading
import time
import random
import re
import traceback
import yaml
from datetime import datetime
import importlib
import argparse
from pathlib import Path
from collections import deque

import numpy as np
import json
from typing import Any

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)

import numpy as np
import json
from typing import Any
import base64

try:
    from outcome_trace import OutcomeTraceError
    from outcome_trace import atomic_write_episode_trace
    from outcome_trace import bind_episode_trace
    from outcome_trace import trace_filename
    from outcome_trace import validate_decision_trace
except ImportError:  # pragma: no cover - package import path in unit tests.
    from .outcome_trace import OutcomeTraceError
    from .outcome_trace import atomic_write_episode_trace
    from .outcome_trace import bind_episode_trace
    from .outcome_trace import trace_filename
    from .outcome_trace import validate_decision_trace

class NumpyEncoder(json.JSONEncoder):
    """Enhanced json encoder for numpy types with array reconstruction info"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            if obj.dtype == np.float32:
                dtype = 'float32'
            elif obj.dtype == np.float64:
                dtype = 'float64'
            elif obj.dtype == np.int32:
                dtype = 'int32'
            elif obj.dtype == np.int64:
                dtype = 'int64'
            else:
                dtype = str(obj.dtype)

            return {
                '__numpy_array__': True,
                'data': base64.b64encode(obj.tobytes()).decode('ascii'),
                'dtype': dtype,
                'shape': obj.shape
            }
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

def numpy_to_json(data: Any) -> str:
    """Convert numpy-containing data to JSON string with reconstruction info"""
    return json.dumps(data, cls=NumpyEncoder)

def json_to_numpy(json_str: str) -> Any:
    """Convert JSON string back to Python objects with numpy arrays"""
    def object_hook(dct):
        if '__numpy_array__' in dct:
            data = base64.b64decode(dct['data'])
            return np.frombuffer(data, dtype=dct['dtype']).reshape(dct['shape'])
        return dct

    return json.loads(json_str, object_hook=object_hook)


def as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _configure_sapien_renderer(config: dict[str, Any]) -> str | None:
    """Optionally pin SAPIEN's Vulkan renderer to the worker's visible device.

    RoboTwin's ``CUDA_VISIBLE_DEVICES`` only constrains CUDA; SAPIEN's Vulkan
    device enumeration otherwise starts at physical GPU 0.  The hook is
    deliberately opt-in so ordinary evaluation keeps the historical import
    and renderer behavior.  When enabled, callers must use a SAPIEN device
    alias such as ``cuda:0`` (the local ordinal after CUDA visibility mapping).
    """
    raw_spec = os.environ.get("ZEVA_SAPIEN_RENDER_DEVICE", "")
    device_spec = str(raw_spec).strip()
    if not device_spec:
        return None
    if device_spec != "cpu" and not re.fullmatch(r"cuda:[0-9]+", device_spec):
        raise ValueError(
            "ZEVA_SAPIEN_RENDER_DEVICE must be 'cpu' or a cuda:N alias, "
            f"got {raw_spec!r}"
        )

    # If a future RoboTwin task config grows an explicit renderer setting,
    # never silently replace a conflicting value with the process override.
    for key in ("sapien_render_device", "render_device"):
        configured = config.get(key)
        if configured is not None and str(configured).strip() != device_spec:
            raise ValueError(
                f"{key}={configured!r} conflicts with "
                f"ZEVA_SAPIEN_RENDER_DEVICE={device_spec!r}"
            )

    import sapien
    import sapien.core as sapien_core

    renderer_class = getattr(sapien, "SapienRenderer")
    device = sapien.Device(device_spec)

    def device_key(value):
        if isinstance(value, str):
            return value.strip()
        try:
            return str(value)
        except Exception:
            return repr(value)

    expected_key = device_key(device)

    class ZevaPinnedSapienRenderer(renderer_class):
        def __init__(self, *args, **kwargs):
            if args:
                raise RuntimeError(
                    "ZEVA_SAPIEN_RENDER_DEVICE cannot safely override a "
                    "positional SAPIEN renderer device"
                )
            provided = kwargs.get("device")
            if provided is not None and device_key(provided) != expected_key:
                raise RuntimeError(
                    "SAPIEN renderer device conflicts with "
                    f"ZEVA_SAPIEN_RENDER_DEVICE={device_spec!r}: {provided!r}"
                )
            kwargs["device"] = device
            super().__init__(**kwargs)

    # Base_Task imports both spellings (``sapien`` and ``sapien.core``).
    sapien.SapienRenderer = ZevaPinnedSapienRenderer
    sapien_core.SapienRenderer = ZevaPinnedSapienRenderer
    print(
        "Using explicit SAPIEN renderer device "
        f"{device_spec!r} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')!r})"
    )
    return device_spec


def model_seed_policy(value: Any) -> str:
    """Validate the policy RNG reset behavior used by the evaluator.

    ``continuous`` matches the frozen RoboTwin PI0.5 evaluation handoff: an
    episode reset clears policy state but does not overwrite the model RNG with
    the environment seed. ``episode_seed`` remains available for diagnostics
    and for resuming older runs that used that nonstandard setting.
    """
    policy = str(value or "continuous").strip().lower()
    if policy not in {"continuous", "episode_seed"}:
        raise ValueError(
            "model_seed_policy must be 'continuous' or 'episode_seed', "
            f"got {value!r}."
        )
    return policy


def _load_fixed_episodes(path, task_name, test_num):
    """Load one task's exact sparse seeds and instructions from a paired manifest."""
    if not path:
        return None
    manifest_path = Path(str(path)).expanduser()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    mapping = payload.get("tasks", payload) if isinstance(payload, dict) else None
    if not isinstance(mapping, dict) or task_name not in mapping:
        raise RuntimeError(f"Seed manifest {manifest_path} has no entry for task {task_name!r}.")
    entries = mapping[task_name]
    if entries and isinstance(entries[0], dict):
        seeds = [int(entry["seed"]) for entry in entries]
        instructions = [str(entry["instruction"]) for entry in entries]
    else:
        seeds = [int(seed) for seed in entries]
        instructions = None
    if len(seeds) != test_num:
        raise RuntimeError(
            f"Seed manifest {manifest_path} has {len(seeds)} seeds for {task_name}; expected {test_num}."
        )
    if len(set(seeds)) != len(seeds) or any(seed < 1000 for seed in seeds):
        raise RuntimeError(f"Seed manifest {manifest_path} contains invalid seeds for {task_name}: {seeds}")
    if any(right <= left for left, right in zip(seeds, seeds[1:])):
        raise RuntimeError(f"Frozen expert-valid seeds must be strictly increasing for {task_name}: {seeds}")
    if instructions is not None and any(not instruction.strip() for instruction in instructions):
        raise RuntimeError(f"Seed manifest {manifest_path} has an empty instruction for {task_name}.")
    return seeds, instructions


def _atomic_write_json(path, payload):
    """Atomically replace a JSON file without removing the previous good copy first."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _build_resume_identity(usr_args, task_name, task_config, ckpt_setting, test_num):
    checkpoint = usr_args.get("checkpoint_path")
    if checkpoint is None:
        checkpoint = ckpt_setting
    return {
        "task_name": str(task_name),
        "task_config": str(task_config),
        "checkpoint": str(checkpoint),
        "use_ema": as_bool(usr_args.get("use_ema_weights"), True),
        "instruction_type": str(usr_args["instruction_type"]),
        "execute_horizon": int(usr_args.get("execute_horizon", 8)),
        "num_steps": int(usr_args.get("num_steps", 30)),
        "seed": int(usr_args["seed"]),
        "absolute_start_seed": int(usr_args.get("absolute_start_seed", -1)),
        "fixed_seed_sequence": as_bool(usr_args.get("fixed_seed_sequence"), False),
        "fixed_seed_values": usr_args.get("fixed_seed_values"),
        "fixed_instructions": usr_args.get("fixed_instructions"),
        "model_seed_policy": model_seed_policy(usr_args.get("model_seed_policy")),
        "max_policy_attempts": int(usr_args.get("max_policy_attempts", 1)),
        "attempt_reset_scope": str(usr_args.get("attempt_reset_scope", "episode")),
        "target_episodes": int(test_num),
    }


def _load_resume_progress(progress_path, identity, initial_seed):
    """Load and validate resumable episode state, or create an empty progress file."""
    progress_path = Path(progress_path).expanduser()
    if progress_path.exists():
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
        except Exception as error:
            raise RuntimeError(
                f"Cannot read RoboTwin resume progress {progress_path}: {error}. "
                "The existing file was left unchanged."
            ) from error

        if not isinstance(progress, dict):
            raise RuntimeError(
                f"Invalid RoboTwin resume progress {progress_path}: top-level JSON value must be an object. "
                "The existing file was left unchanged."
            )

        stored_identity = progress.get("identity")
        if not isinstance(stored_identity, dict):
            raise RuntimeError(
                f"Invalid RoboTwin resume progress {progress_path}: missing object field 'identity'. "
                "The existing file was left unchanged."
            )
        mismatches = []
        for key, expected_value in identity.items():
            actual_value = stored_identity.get(key, "<missing>")
            if actual_value != expected_value:
                mismatches.append(f"{key}: progress={actual_value!r}, current={expected_value!r}")
        if mismatches:
            raise RuntimeError(
                f"RoboTwin resume identity mismatch for {progress_path}; refusing to overwrite existing progress:\n  - "
                + "\n  - ".join(mismatches)
            )

        episode_results = progress.get("episode_results")
        if not isinstance(episode_results, list):
            raise RuntimeError(
                f"Invalid RoboTwin resume progress {progress_path}: 'episode_results' must be a list."
            )
        target_episodes = identity["target_episodes"]
        if len(episode_results) > target_episodes:
            raise RuntimeError(
                f"Invalid RoboTwin resume progress {progress_path}: found {len(episode_results)} episodes, "
                f"but this run targets {target_episodes}."
            )
        for expected_index, result in enumerate(episode_results):
            if not isinstance(result, dict) or result.get("episode_index") != expected_index:
                raise RuntimeError(
                    f"Invalid RoboTwin resume progress {progress_path}: episode_results must have contiguous "
                    f"episode_index values starting at 0 (bad entry at position {expected_index})."
                )
            if not isinstance(result.get("success"), bool):
                raise RuntimeError(
                    f"Invalid RoboTwin resume progress {progress_path}: episode {expected_index} has no boolean "
                    "'success' field."
                )
            try:
                episode_seed = int(result["seed"])
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError(
                    f"Invalid RoboTwin resume progress {progress_path}: episode {expected_index} has no integer "
                    "'seed' field."
                ) from error
            if expected_index and episode_seed <= int(episode_results[expected_index - 1]["seed"]):
                raise RuntimeError(
                    f"Invalid RoboTwin resume progress {progress_path}: policy episode seeds must be strictly "
                    f"increasing (bad entry at position {expected_index})."
                )

        try:
            next_seed = int(progress["next_seed"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                f"Invalid RoboTwin resume progress {progress_path}: missing integer field 'next_seed'."
            ) from error
        if episode_results and next_seed <= int(episode_results[-1]["seed"]):
            raise RuntimeError(
                f"Invalid RoboTwin resume progress {progress_path}: next_seed={next_seed} must be greater than "
                f"the last completed episode seed={episode_results[-1]['seed']}."
            )
        if not episode_results and next_seed != int(initial_seed):
            raise RuntimeError(
                f"Invalid RoboTwin resume progress {progress_path}: an empty run must start at "
                f"next_seed={initial_seed}, found {next_seed}."
            )
        if as_bool(progress.get("complete")) and len(episode_results) != target_episodes:
            raise RuntimeError(
                f"Invalid RoboTwin resume progress {progress_path}: complete=true but only "
                f"{len(episode_results)}/{target_episodes} episodes are present."
            )
        print(
            f"Resuming RoboTwin evaluation from {progress_path}: "
            f"{len(episode_results)}/{target_episodes} episodes, next_seed={next_seed}"
        )
        return progress_path, progress, next_seed, list(episode_results)

    now = datetime.now().isoformat(timespec="seconds")
    progress = {
        "version": 1,
        "identity": identity,
        "complete": False,
        "next_seed": int(initial_seed),
        "completed_episodes": 0,
        "successes": 0,
        "episode_results": [],
        "created_at": now,
        "updated_at": now,
    }
    _atomic_write_json(progress_path, progress)
    print(f"Initialized RoboTwin resume progress at {progress_path}")
    return progress_path, progress, int(initial_seed), []


def _make_progress_writer(progress_path, progress, identity):
    def write_progress(next_seed, episode_results, complete=False, summary_path=None):
        progress.update(
            {
                "version": 1,
                "identity": identity,
                "complete": bool(complete),
                "next_seed": int(next_seed),
                "completed_episodes": len(episode_results),
                "successes": sum(bool(result["success"]) for result in episode_results),
                "episode_results": list(episode_results),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        if summary_path is not None:
            progress["summary_path"] = str(summary_path)
        _atomic_write_json(progress_path, progress)

    return write_progress


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance


def eval_function_decorator(policy_name, model_name, conda_env=None):
    # conda_env is abandoned
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e


def get_camera_config(camera_type):
    # Resolve from the imported RoboTwin environment package instead of from
    # this script's location.  Formal runs execute this maintained client from
    # the shared ZeVA repository while the simulator remains on the render
    # host, so a relative sibling ``task_config`` directory is not guaranteed.
    camera_config_path = os.path.join(CONFIGS_PATH, "_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args

class ModelClient:
    def __init__(
        self,
        host='localhost',
        port=9999,
        timeout=600,
        max_attempts=120,
        retry_delay=5,
        outcome_trace_dir=None,
        outcome_trace_condition=None,
        outcome_trace_policy_name=None,
        outcome_trace_split=None,
        outcome_trace_protocol=None,
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.retry_delay = retry_delay
        self.sock = None
        self._outcome_trace_dir = Path(outcome_trace_dir).expanduser() if outcome_trace_dir else None
        self._outcome_trace_condition = (
            str(outcome_trace_condition).strip() if outcome_trace_condition else None
        )
        self._outcome_trace_policy_name = (
            str(outcome_trace_policy_name).strip() if outcome_trace_policy_name else None
        )
        self._outcome_trace_split = (
            str(outcome_trace_split).strip() if outcome_trace_split else None
        )
        self._outcome_trace_protocol = (
            str(outcome_trace_protocol).strip() if outcome_trace_protocol else None
        )
        if self._outcome_trace_dir is not None and not self._outcome_trace_condition:
            raise ValueError("outcome_trace_condition is required when outcome tracing is enabled")
        if self._outcome_trace_dir is not None and not self._outcome_trace_policy_name:
            raise ValueError("outcome_trace_policy_name is required when outcome tracing is enabled")
        if self._outcome_trace_dir is not None and not self._outcome_trace_split:
            raise ValueError("outcome_trace_split is required when outcome tracing is enabled")
        if self._outcome_trace_dir is not None and not self._outcome_trace_protocol:
            raise ValueError("outcome_trace_protocol is required when outcome tracing is enabled")
        self._trace_episode: dict[str, Any] | None = None
        self._trace_replans: list[dict[str, Any]] = []
        self._connect()

    def _connect(self):
        attempts = 0
        max_attempts = self.max_attempts
        retry_delay = self.retry_delay

        while attempts < max_attempts:
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.settimeout(self.timeout)
                self.sock.connect((self.host, self.port))
                print(f"🔗 Connected to model server at {self.host}:{self.port}")
                return
            except Exception as e:
                attempts += 1
                if self.sock:
                    self.sock.close()
                if attempts < max_attempts:
                    print(f"⚠️ Connection attempt {attempts} failed: {str(e)}")
                    print(f"🔄 Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    raise ConnectionError(
                        f"Failed to connect to server after {max_attempts} attempts: {str(e)}"
                    )

    def _send_recv(self, data):
        """Send request and receive response with numpy array support"""
        try:
            # Serialize with numpy support
            json_data = numpy_to_json(data).encode('utf-8')

            # Send data length and data
            self.sock.sendall(len(json_data).to_bytes(4, 'big'))
            self.sock.sendall(json_data)

            # Receive and deserialize response
            response = self._recv_response()
            return response

        except Exception as e:
            self.close()
            raise ConnectionError(f"Communication error: {str(e)}")

    def _recv_response(self):
        """Receive response with numpy array reconstruction"""
        # Read response length
        len_data = self.sock.recv(4)
        if not len_data:
            raise ConnectionError("Connection closed by server")

        size = int.from_bytes(len_data, 'big')

        # Read complete response
        chunks = []
        received = 0
        while received < size:
            chunk = self.sock.recv(min(size - received, 4096))
            if not chunk:
                raise ConnectionError("Incomplete response received")
            chunks.append(chunk)
            received += len(chunk)

        # Deserialize with numpy reconstruction
        return json_to_numpy(b''.join(chunks).decode('utf-8'))

    def begin_episode_trace(
        self,
        *,
        task: str,
        seed: int,
        episode_index: int,
        instruction: str,
    ) -> None:
        """Start an opt-in trace; no-op for ordinary evaluation."""
        if self._outcome_trace_dir is None:
            return
        if self._trace_episode is not None:
            raise RuntimeError("Cannot begin an outcome trace before finalizing the previous episode")
        self._trace_episode = {
            "task": str(task),
            "seed": int(seed),
            "episode_index": int(episode_index),
            "instruction": str(instruction),
        }
        self._trace_replans = []

    def _record_prediction_trace(self, response: dict[str, Any]) -> None:
        if self._outcome_trace_dir is None:
            return
        if self._trace_episode is None:
            raise RuntimeError("Received a prediction before begin_episode_trace")
        raw_trace = response.get("trace")
        if raw_trace is None:
            raise OutcomeTraceError(
                "Outcome tracing is enabled, but the model server returned no 'trace' payload"
            )
        trace = validate_decision_trace(raw_trace)
        expected_index = len(self._trace_replans)
        if trace["replan_index"] != expected_index:
            raise OutcomeTraceError(
                "Model server returned non-contiguous replan_index: "
                f"expected {expected_index}, got {trace['replan_index']}"
            )
        self._trace_replans.append(trace)

    def finalize_episode_trace(
        self,
        *,
        success: bool,
        steps: int,
        step_limit: int,
    ) -> Path | None:
        """Validate and atomically commit one completed episode trace."""
        if self._outcome_trace_dir is None:
            return None
        if self._trace_episode is None:
            raise RuntimeError("Cannot finalize an outcome trace before begin_episode_trace")
        payload = bind_episode_trace(
            **self._trace_episode,
            condition=self._outcome_trace_condition,
            policy_name=self._outcome_trace_policy_name,
            split=self._outcome_trace_split,
            protocol=self._outcome_trace_protocol,
            success=bool(success),
            steps=int(steps),
            step_limit=int(step_limit),
            replans=self._trace_replans,
        )
        path = self._outcome_trace_dir / trace_filename(
            payload["task"], payload["episode_index"], payload["seed"]
        )
        result = atomic_write_episode_trace(path, payload)
        self._trace_episode = None
        self._trace_replans = []
        print(f"Outcome trace committed atomically: {result}")
        return result

    def call(self, func_name=None, obs=None):
        response = self._send_recv({"cmd": func_name, "obs": obs})
        if "error" in response:
            raise RuntimeError(f"Model server error: {response['error']}\n{response.get('traceback', '')}")
        if "res" not in response:
            raise RuntimeError(f"Malformed model server response: {response!r}")
        if func_name == "predict":
            self._record_prediction_trace(response["res"])
        return response['res']

    def close(self):
        """Close the connection"""
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
            finally:
                self.sock = None
                print("🔌 Connection closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def main(usr_args):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    # checkpoint_num = usr_args['checkpoint_num']
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]
    port = usr_args["port"]
    save_dir = None
    video_save_dir = None
    video_size = None

    policy_conda_env = usr_args.get("policy_conda_env", None)

    get_model = eval_function_decorator(policy_name, "get_model", conda_env=policy_conda_env)

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting
    args["absolute_start_seed"] = int(usr_args.get("absolute_start_seed", -1))
    args["fixed_seed_sequence"] = as_bool(usr_args.get("fixed_seed_sequence"), False)
    args["model_seed_policy"] = model_seed_policy(usr_args.get("model_seed_policy"))
    args["max_policy_attempts"] = int(usr_args.get("max_policy_attempts", 1))
    args["attempt_reset_scope"] = str(usr_args.get("attempt_reset_scope", "episode"))
    sapien_render_device = _configure_sapien_renderer(args)
    if sapien_render_device is not None:
        args["sapien_render_device"] = sapien_render_device

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    result_dir = usr_args.get("result_dir")
    if result_dir:
        save_dir = Path(str(result_dir)).expanduser()
    else:
        result_root = Path(str(usr_args.get("eval_result_root", "eval_result"))).expanduser()
        save_dir = result_root / task_name / policy_name / task_config / ckpt_setting / current_time
    save_dir.mkdir(parents=True, exist_ok=True)

    if "eval_video_log" in usr_args:
        args["eval_video_log"] = as_bool(usr_args["eval_video_log"])

    if args["eval_video_log"]:
        video_save_dir = save_dir
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir

    # output camera config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    seed = usr_args["seed"]

    st_seed = int(usr_args.get("absolute_start_seed", 100000 * (1 + seed)))
    requested_start_seed = st_seed
    suc_nums = []
    test_num = int(usr_args.get("test_num", 100))
    topk = int(usr_args.get("topk", 1))
    fixed_episode_values = _load_fixed_episodes(
        usr_args.get("seed_manifest"), task_name=task_name, test_num=test_num
    )
    if fixed_episode_values is not None:
        fixed_seed_values, fixed_instructions = fixed_episode_values
        args["fixed_seed_sequence"] = True
        args["fixed_seed_values"] = fixed_seed_values
        args["fixed_instructions"] = fixed_instructions
        usr_args["fixed_seed_sequence"] = True
        usr_args["fixed_seed_values"] = fixed_seed_values
        usr_args["fixed_instructions"] = fixed_instructions
        st_seed = fixed_seed_values[0]
        requested_start_seed = st_seed

    resume_progress_path = usr_args.get("resume_progress_path")
    progress_writer = None
    resumed_episode_results = []
    if resume_progress_path:
        resume_identity = _build_resume_identity(
            usr_args,
            task_name=task_name,
            task_config=task_config,
            ckpt_setting=ckpt_setting,
            test_num=test_num,
        )
        (
            resume_progress_path,
            progress_state,
            st_seed,
            resumed_episode_results,
        ) = _load_resume_progress(resume_progress_path, resume_identity, st_seed)
        progress_writer = _make_progress_writer(
            resume_progress_path,
            progress_state,
            resume_identity,
        )

    outcome_trace_enabled = as_bool(usr_args.get("outcome_trace_enabled"), False)
    outcome_trace_dir = usr_args.get("outcome_trace_dir")
    if outcome_trace_enabled and not outcome_trace_dir:
        raise ValueError("outcome_trace_enabled=True requires outcome_trace_dir")
    if not outcome_trace_enabled:
        outcome_trace_dir = None
    outcome_trace_condition = str(
        usr_args.get("outcome_trace_condition", policy_name)
    )
    outcome_trace_split = (
        str(usr_args.get("outcome_trace_split", "")).strip()
        if outcome_trace_enabled
        else None
    )
    outcome_trace_protocol = (
        str(usr_args.get("outcome_trace_protocol", "")).strip()
        if outcome_trace_enabled
        else None
    )
    if outcome_trace_enabled and not outcome_trace_split:
        raise ValueError("outcome_trace_enabled=True requires outcome_trace_split")
    if outcome_trace_enabled and not outcome_trace_protocol:
        raise ValueError("outcome_trace_enabled=True requires outcome_trace_protocol")

    # model = get_model(usr_args)
    if len(resumed_episode_results) < test_num:
        model = ModelClient(
            host=str(usr_args.get("server_host", "127.0.0.1")),
            port=port,
            timeout=float(usr_args.get("client_timeout", 600)),
            max_attempts=int(usr_args.get("connect_attempts", 120)),
            retry_delay=float(usr_args.get("connect_retry_seconds", 5)),
            outcome_trace_dir=outcome_trace_dir,
            outcome_trace_condition=outcome_trace_condition,
            outcome_trace_policy_name=policy_name,
            outcome_trace_split=outcome_trace_split,
            outcome_trace_protocol=outcome_trace_protocol,
        )
        try:
            st_seed, suc_num, episode_results = eval_policy(
                task_name,
                TASK_ENV,
                args,
                model,
                st_seed,
                test_num=test_num,
                video_size=video_size,
                instruction_type=instruction_type,
                policy_conda_env=policy_conda_env,
                episode_results=resumed_episode_results,
                progress_writer=progress_writer,
            )
        finally:
            model.close()
    else:
        episode_results = resumed_episode_results
        suc_num = sum(bool(result["success"]) for result in episode_results)
    suc_nums.append(suc_num)

    topk_success_rate = sorted(suc_nums, reverse=True)[:topk]

    result_name = "_result_random.txt" if "randomized" in task_config else "_result_clean.txt"
    file_path = os.path.join(save_dir, result_name)
    with open(file_path, "w") as file:
        file.write(f"Timestamp: {current_time}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        # file.write(str(task_reward) + '\n')
        file.write("\n".join(map(str, np.array(suc_nums) / test_num)))

    summary = {
        "task_name": task_name,
        "task_config": task_config,
        "policy_name": policy_name,
        "ckpt_setting": ckpt_setting,
        "checkpoint_path": usr_args.get("checkpoint_path"),
        "config_file": usr_args.get("config_file"),
        "use_ema_weights": as_bool(usr_args.get("use_ema_weights"), True),
        "instruction_type": instruction_type,
        "seed": int(seed),
        "absolute_start_seed": int(requested_start_seed),
        "fixed_seed_sequence": as_bool(usr_args.get("fixed_seed_sequence"), False),
        "seed_manifest": usr_args.get("seed_manifest"),
        "evaluated_seeds": [int(result["seed"]) for result in episode_results],
        "model_seed_policy": model_seed_policy(usr_args.get("model_seed_policy")),
        "episodes": int(test_num),
        "successes": int(suc_num),
        "success_rate": float(suc_num / test_num),
        "max_policy_attempts": int(usr_args.get("max_policy_attempts", 1)),
        "attempt_reset_scope": str(usr_args.get("attempt_reset_scope", "episode")),
        "total_attempts_executed": sum(len(row.get("attempts", [row])) for row in episode_results),
        "successes_by_attempt": [
            sum(bool(row.get("attempts", [row])[index]["success"])
                for row in episode_results if len(row.get("attempts", [row])) > index)
            for index in range(int(usr_args.get("max_policy_attempts", 1)))
        ],
        "domain_id": int(usr_args.get("domain_id", 10)),
        "fps": float(usr_args.get("fps", 30)),
        "resolution": str(usr_args.get("resolution", "480")),
        "view_resize_size": str(usr_args.get("view_resize_size", "480x640")),
        "chunk_length": int(usr_args.get("chunk_length", 32)),
        "execute_horizon": int(usr_args.get("execute_horizon", 8)),
        "action_dim": int(usr_args.get("action_dim", 14)),
        "num_steps": int(usr_args.get("num_steps", 30)),
        "guidance": float(usr_args.get("guidance", 1.0)),
        "shift": float(usr_args.get("shift", 10.0)),
        "sigma_max": float(usr_args.get("sigma_max", 80.0)),
        "eval_video_log": bool(args["eval_video_log"]),
        "episode_results": episode_results,
    }
    summary_path = save_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if progress_writer is not None:
        progress_writer(
            st_seed,
            episode_results,
            complete=True,
            summary_path=summary_path,
        )

    print(f"Data has been saved to {file_path} and {summary_path}")
    # return task_reward


def eval_policy(task_name,
                TASK_ENV,
                args,
                model,
                st_seed,
                test_num=100,
                video_size=None,
                instruction_type=None,
                policy_conda_env=None,
                episode_results=None,
                progress_writer=None):
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    expert_check = True
    fixed_seed_sequence = as_bool(args.get("fixed_seed_sequence"), False)
    fixed_seed_values = args.get("fixed_seed_values")
    fixed_instructions = args.get("fixed_instructions")
    # A seed that passed expert filtering can still be reported unstable in a
    # later process because RoboTwin GPU physics is not bit deterministic.
    # For paired evaluation, retry the *same* frozen seed.  Substituting the
    # next seed would silently destroy Base/Anchor/ZeVA pairing, while these
    # setup failures happen before any policy query and therefore do not
    # advance the model's continuous diffusion RNG stream.
    fixed_seed_init_max_attempts = int(args.get("fixed_seed_init_max_attempts", 20))
    fixed_seed_init_attempts = 0
    episode_results = list(episode_results or [])
    TASK_ENV.suc = sum(bool(result["success"]) for result in episode_results)
    TASK_ENV.test_num = len(episode_results)

    now_id = len(episode_results)
    succ_seed = len(episode_results)
    suc_test_seed_list = [int(result["seed"]) for result in episode_results]

    policy_name = args["policy_name"]
    eval_func = eval_function_decorator(policy_name, "eval", conda_env=policy_conda_env)

    now_seed = st_seed
    task_total_reward = 0
    clear_cache_freq = args["clear_cache_freq"]

    args["eval_mode"] = True

    while succ_seed < test_num:
        if fixed_seed_values is not None:
            now_seed = int(fixed_seed_values[succ_seed])
        render_freq = args["render_freq"]
        args["render_freq"] = 0

        if expert_check:
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except UnStableError as e:
                print(" -------------")
                print("Error: ", e)
                print(" -------------")
                TASK_ENV.close_env()
                if fixed_seed_sequence:
                    fixed_seed_init_attempts += 1
                    args["render_freq"] = render_freq
                    if fixed_seed_init_attempts >= fixed_seed_init_max_attempts:
                        raise RuntimeError(
                            f"Frozen RoboTwin seed {now_seed} remained unstable for "
                            f"{fixed_seed_init_attempts} attempts; refusing seed substitution."
                        ) from e
                    print(
                        f"Retrying frozen seed {now_seed} without substitution "
                        f"({fixed_seed_init_attempts}/{fixed_seed_init_max_attempts})."
                    )
                    continue
                now_seed += 1
                args["render_freq"] = render_freq
                continue
            except Exception as e:
                stack_trace = traceback.format_exc()
                print(" -------------")
                print("Error: ", stack_trace)
                print(" -------------")
                TASK_ENV.close_env()
                if fixed_seed_sequence:
                    fixed_seed_init_attempts += 1
                    args["render_freq"] = render_freq
                    if fixed_seed_init_attempts >= fixed_seed_init_max_attempts:
                        raise RuntimeError(
                            f"Frozen RoboTwin seed {now_seed} failed initialization for "
                            f"{fixed_seed_init_attempts} attempts; refusing seed substitution."
                        ) from e
                    print(
                        f"Retrying frozen seed {now_seed} after initialization failure "
                        f"({fixed_seed_init_attempts}/{fixed_seed_init_max_attempts})."
                    )
                    continue
                now_seed += 1
                args["render_freq"] = render_freq
                print("error occurs !")
                continue

        fixed_seed_init_attempts = 0

        if fixed_seed_sequence or (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed += 1
            suc_test_seed_list.append(now_seed)
        else:
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq

        # RoboTwin initializes the environment twice: once for expert
        # validation above and once again for the actual policy rollout.  GPU
        # physics can mark the second initialization unstable even when the
        # expert-validation initialization succeeded.  With a frozen paired
        # manifest, retry that same seed here as well; changing the seed would
        # break Base/ZeVA pairing.  This is still before reset_model or any
        # policy query, so a retry does not advance the model RNG stream.
        rollout_init_attempt = 0
        while True:
            try:
                TASK_ENV.setup_demo(
                    now_ep_num=now_id, seed=now_seed, is_test=True, **args
                )
                break
            except UnStableError as e:
                TASK_ENV.close_env()
                rollout_init_attempt += 1
                if rollout_init_attempt >= fixed_seed_init_max_attempts:
                    raise RuntimeError(
                        f"Selected RoboTwin seed {now_seed} remained unstable during "
                        f"rollout initialization for {rollout_init_attempt} attempts; "
                        "refusing seed substitution."
                    ) from e
                print(
                    f"Retrying selected seed {now_seed} during rollout initialization "
                    f"without substitution ({rollout_init_attempt}/"
                    f"{fixed_seed_init_max_attempts})."
                )
        episode_info_list = [episode_info["info"]]
        if fixed_instructions is None:
            results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)
            instruction = np.random.choice(results[0][instruction_type])
        else:
            instruction = fixed_instructions[now_id]
        TASK_ENV.set_instruction(instruction=instruction)  # set language instruction

        max_policy_attempts = int(args.get("max_policy_attempts", 1))
        attempt_reset_scope = str(args.get("attempt_reset_scope", "episode"))
        if max_policy_attempts < 1 or max_policy_attempts > 4:
            raise ValueError("max_policy_attempts must be in [1,4].")
        if attempt_reset_scope not in {"episode", "attempt"}:
            raise ValueError("attempt_reset_scope must be episode or attempt.")
        if attempt_reset_scope == "attempt" and max_policy_attempts == 1:
            raise ValueError("Attempt-persistent reset is meaningless for one attempt.")

        attempt_results = []
        succ = False
        for attempt_index in range(max_policy_attempts):
            if attempt_index:
                # Recreate exactly the same scene/robot/object initialization;
                # only the model-side reset scope differs across conditions.
                TASK_ENV.close_env()
                retry = 0
                while True:
                    try:
                        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                        break
                    except UnStableError as e:
                        TASK_ENV.close_env()
                        retry += 1
                        if retry >= fixed_seed_init_max_attempts:
                            raise RuntimeError(
                                f"Frozen seed {now_seed} stayed unstable for multi-attempt "
                                f"retry {attempt_index + 1}."
                            ) from e
                TASK_ENV.set_instruction(instruction=instruction)

            if TASK_ENV.eval_video_path is not None:
                ffmpeg = subprocess.Popen(
                    [
                        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
                        "-pixel_format", "rgb24", "-video_size", video_size,
                        "-framerate", "10", "-i", "-", "-pix_fmt", "yuv420p",
                        "-vcodec", "libx264", "-crf", "23",
                        f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4",
                    ],
                    stdin=subprocess.PIPE,
                )
                TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

            if hasattr(model, "begin_episode_trace"):
                model.begin_episode_trace(
                    task=task_name,
                    seed=now_seed,
                    episode_index=now_id,
                    instruction=instruction,
                )

            # Attempt one is an ordinary episode reset.  Later attempts either
            # commit BIT into PIM (PIM condition) or fully forget history
            # (Base/parent controls), without reseeding the diffusion stream.
            if attempt_index == 0:
                if model_seed_policy(args.get("model_seed_policy")) == "episode_seed":
                    model.call(func_name="reset_model", obs={"seed": int(now_seed), "scope": "episode"})
                else:
                    model.call(func_name="reset_model", obs={"scope": "episode"})
            else:
                model.call(func_name="reset_model", obs={"scope": attempt_reset_scope})

            attempt_success = False
            while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
                observation = TASK_ENV.get_obs()
                eval_func(TASK_ENV, model, observation)
                if TASK_ENV.eval_success:
                    attempt_success = True
                    break

            if TASK_ENV.eval_video_path is not None:
                original_video = Path(TASK_ENV.eval_video_path) / f"episode{TASK_ENV.test_num}.mp4"
                TASK_ENV._del_eval_video_ffmpeg()
                suffix = "" if max_policy_attempts == 1 else f"_attempt-{attempt_index + 1}"
                renamed_video = Path(TASK_ENV.eval_video_path) / (
                    f"episode{TASK_ENV.test_num}{suffix}_randomized-true_"
                    f"success-{str(bool(attempt_success)).lower()}.mp4"
                )
                if not original_video.is_file():
                    raise FileNotFoundError(f"RoboTwin evaluator did not produce {original_video}")
                os.replace(original_video, renamed_video)

            attempt_results.append({
                "attempt": attempt_index + 1,
                "success": bool(attempt_success),
                "steps": int(TASK_ENV.take_action_cnt),
                "step_limit": int(TASK_ENV.step_lim),
                "reset_scope": "episode" if attempt_index == 0 else attempt_reset_scope,
            })
            if hasattr(model, "finalize_episode_trace"):
                model.finalize_episode_trace(
                    success=bool(attempt_success),
                    steps=int(TASK_ENV.take_action_cnt),
                    step_limit=int(TASK_ENV.step_lim),
                )
            if attempt_success:
                succ = True
                break

        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        episode_results.append({
            "episode_index": int(now_id),
            "seed": int(now_seed),
            "success": bool(succ),
            "steps": sum(item["steps"] for item in attempt_results),
            "step_limit": sum(item["step_limit"] for item in attempt_results),
            "instruction": str(instruction),
            "attempts": attempt_results,
        })

        now_id += 1

        if progress_writer is not None:
            progress_writer(now_seed + 1, episode_results, complete=False)

        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1

        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | \033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => \033[95m{round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%\033[0m, current seed: \033[90m{now_seed}\033[0m\n"
        )
        # TASK_ENV._take_picture()
        now_seed += 1

    return now_seed, TASK_ENV.suc, episode_results


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config['port'] = args.port

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except:
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    return config


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    usr_args = parse_args_and_config()

    main(usr_args)
