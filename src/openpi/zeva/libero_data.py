"""LIBERO EEF16 dataset and fixed-camera action conversion utilities."""

from __future__ import annotations

import bisect
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from openpi.zeva.libero_contract import LIBERO_ACTION_DIM
from openpi.zeva.libero_contract import LIBERO_CAMERA_KEYS
from openpi.zeva.libero_contract import LIBERO_EXECUTION_HORIZON


def _normalize_quaternion_xyzw(quaternion: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    if np.any(norm <= 1e-12):
        raise ValueError("Quaternion must have nonzero norm.")
    value = value / norm
    return np.where(value[..., 3:4] < 0.0, -value, value)


def quaternion_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    q = _normalize_quaternion_xyzw(quaternion)
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def matrix_to_quaternion_xyzw(rotation: np.ndarray) -> np.ndarray:
    """Stable batched rotation-matrix to canonical xyzw quaternion conversion."""
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape[-2:] != (3, 3):
        raise ValueError("Rotation matrices must end in [3,3].")
    flat = matrix.reshape(-1, 3, 3)
    result = np.empty((len(flat), 4), dtype=np.float64)
    for index, value in enumerate(flat):
        trace = np.trace(value)
        if trace > 0.0:
            scale = np.sqrt(trace + 1.0) * 2.0
            result[index] = (
                (value[2, 1] - value[1, 2]) / scale,
                (value[0, 2] - value[2, 0]) / scale,
                (value[1, 0] - value[0, 1]) / scale,
                0.25 * scale,
            )
        else:
            axis = int(np.argmax(np.diag(value)))
            if axis == 0:
                scale = np.sqrt(1.0 + value[0, 0] - value[1, 1] - value[2, 2]) * 2.0
                result[index] = (0.25 * scale, (value[0, 1] + value[1, 0]) / scale,
                                 (value[0, 2] + value[2, 0]) / scale, (value[2, 1] - value[1, 2]) / scale)
            elif axis == 1:
                scale = np.sqrt(1.0 + value[1, 1] - value[0, 0] - value[2, 2]) * 2.0
                result[index] = ((value[0, 1] + value[1, 0]) / scale, 0.25 * scale,
                                 (value[1, 2] + value[2, 1]) / scale, (value[0, 2] - value[2, 0]) / scale)
            else:
                scale = np.sqrt(1.0 + value[2, 2] - value[0, 0] - value[1, 1]) * 2.0
                result[index] = ((value[0, 2] + value[2, 0]) / scale,
                                 (value[1, 2] + value[2, 1]) / scale, 0.25 * scale,
                                 (value[1, 0] - value[0, 1]) / scale)
    return _normalize_quaternion_xyzw(result).reshape(*matrix.shape[:-2], 4).astype(np.float32)


def absolute_pose_command_horizon_to_actions(state: np.ndarray, future: np.ndarray) -> np.ndarray:
    """Convert stored absolute future poses to chunk-start-relative EEF16."""
    current = np.asarray(state, dtype=np.float64)
    targets = np.asarray(future, dtype=np.float64)
    if current.shape != (LIBERO_ACTION_DIM,) or targets.ndim != 2 or targets.shape[1] != LIBERO_ACTION_DIM:
        raise ValueError("Expected state [16] and future [H,16].")
    output = np.zeros_like(targets, dtype=np.float32)
    output[:, :3] = targets[:, :3] - current[None, :3]
    current_rotation = quaternion_xyzw_to_matrix(current[3:7])
    future_rotation = quaternion_xyzw_to_matrix(targets[:, 3:7])
    output[:, 3:7] = matrix_to_quaternion_xyzw(future_rotation @ current_rotation.T)
    commands = targets[:, 14]
    if not np.all(np.isclose(np.abs(commands), 1.0, atol=1e-6)):
        raise ValueError("LIBERO gripper targets must be official -1/+1 commands.")
    output[:, 14] = commands
    return output


def libero_multiview_image(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Concatenate agent and wrist RGB views along width for the ZTE."""
    views = []
    for key in LIBERO_CAMERA_KEYS:
        image = batch[key]
        if image.ndim not in (4, 5):
            raise ValueError(f"{key} must be BCHW/BTCHW or channels-last, got {tuple(image.shape)}.")
        if image.shape[-1] == 3:
            order = (0, 3, 1, 2) if image.ndim == 4 else (0, 1, 4, 2, 3)
            image = image.permute(*order)
        channel_axis = 1 if image.ndim == 4 else 2
        if image.shape[channel_axis] != 3:
            raise ValueError(f"{key} must contain RGB images.")
        views.append(image.contiguous())
    if {tuple(value.shape[-2:]) for value in views}.__len__() != 1:
        raise ValueError("LIBERO agent and wrist image sizes differ.")
    return torch.cat(views, dim=-1)


class LiberoEpisodeTable:
    def __init__(self, dataset_root: str | Path, subset: str):
        self.subset = subset
        self.root = Path(dataset_root).resolve() / subset / "physical-intelligence" / "libero"
        info = json.loads((self.root / "meta" / "info.json").read_text(encoding="utf-8"))
        self.episodes = [json.loads(line) for line in (self.root / "meta" / "episodes.jsonl").read_text().splitlines()]
        self.tasks = {
            int(item["task_index"]): item["task"]
            for item in (json.loads(line) for line in (self.root / "meta" / "tasks.jsonl").read_text().splitlines())
        }
        if int(info["total_episodes"]) != len(self.episodes):
            raise ValueError(f"LIBERO {subset} episode metadata is incomplete.")
        self.paths = tuple(
            self.root / "data" / f"chunk-{int(item['episode_index']) // 1000:03d}" / f"episode_{int(item['episode_index']):06d}.parquet"
            for item in self.episodes
        )
        if any(not path.is_file() for path in self.paths):
            raise FileNotFoundError("LIBERO parquet episode staging is incomplete.")

    def task(self, record_index: int) -> str:
        episode = self.episodes[record_index]
        return str(episode["tasks"][0])

    def record_id(self, record_index: int) -> str:
        return f"{self.subset}:{int(self.episodes[record_index]['episode_index'])}:{self.task(record_index)}"


def _decode_png(value: dict[str, Any]) -> torch.Tensor:
    with Image.open(io.BytesIO(value["bytes"])) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1)


class LiberoZTEEpisodeDataset(Dataset):
    """Complete episodes sampled at the official five-step replanning boundary."""

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        subset: str,
        goal_embeddings: str | Path,
        transition_horizon: int = LIBERO_EXECUTION_HORIZON,
    ):
        if transition_horizon != LIBERO_EXECUTION_HORIZON:
            raise ValueError("Formal LIBERO ZTE training requires H5 transitions.")
        self.table = LiberoEpisodeTable(dataset_root, subset)
        self.transition_horizon = transition_horizon
        payload = torch.load(goal_embeddings, map_location="cpu", weights_only=False)
        if payload.get("schema") != "zeva-libero-pi05-goal-embeddings-v1":
            raise ValueError("Unsupported LIBERO goal embedding table.")
        split = payload["splits"][subset]
        expected = tuple(self.table.record_id(index) for index in range(len(self.table.episodes)))
        if tuple(split["record_ids"]) != expected:
            raise ValueError(f"LIBERO {subset} goal record order differs from the dataset.")
        self.goals = torch.as_tensor(split["embeddings"], dtype=torch.float32)
        task_names = sorted(set(self.table.tasks.values()))
        self.task_names = tuple(task_names)
        self.task_count = len(task_names)
        self.task_ids = {task: index for index, task in enumerate(task_names)}
        self.indices_by_task = tuple(
            tuple(
                record_index
                for record_index in range(len(self.table.episodes))
                if self.task_ids[self.table.task(record_index)] == task_id
            )
            for task_id in range(self.task_count)
        )

    def __len__(self) -> int:
        return len(self.table.episodes)

    def __getitem__(self, record_index: int) -> dict[str, torch.Tensor]:
        path = self.table.paths[record_index]
        table = pq.read_table(path, columns=["image", "wrist_image", "state", "actions"])
        length = len(table)
        starts = list(range(0, length - self.transition_horizon, self.transition_horizon))
        after = [frame + self.transition_horizon for frame in starts]
        selected = starts + after
        agent = torch.stack([_decode_png(table["image"][frame].as_py()) for frame in selected])
        wrist = torch.stack([_decode_png(table["wrist_image"][frame].as_py()) for frame in selected])
        count = len(starts)
        before_views = {
            LIBERO_CAMERA_KEYS[0]: agent[:count],
            LIBERO_CAMERA_KEYS[1]: wrist[:count],
        }
        after_views = {
            LIBERO_CAMERA_KEYS[0]: agent[count:],
            LIBERO_CAMERA_KEYS[1]: wrist[count:],
        }
        states = np.asarray(table["state"].to_pylist(), dtype=np.float32)
        stored_actions = np.asarray(table["actions"].to_pylist(), dtype=np.float32)
        action_chunks = np.stack(
            [
                absolute_pose_command_horizon_to_actions(
                    states[frame], stored_actions[frame : frame + self.transition_horizon]
                )
                for frame in starts
            ]
        )
        task = self.table.task(record_index)
        return {
            "images_before": libero_multiview_image(before_views),
            "images_after": libero_multiview_image(after_views),
            "actions": torch.from_numpy(action_chunks),
            "progress": torch.tensor(
                [(frame + self.transition_horizon) / max(1, length - 1) for frame in starts],
                dtype=torch.float32,
            ),
            "initial_progress": torch.tensor(0.0, dtype=torch.float32),
            "task_id": torch.tensor(self.task_ids[task], dtype=torch.long),
            "goal_embedding": self.goals[record_index],
        }


class LiberoStage2Dataset(Dataset):
    """Natural valid-row PI0.5 decisions from the official episode split."""

    def __init__(self, dataset_root: str | Path, *, subset: str):
        self.table = LiberoEpisodeTable(dataset_root, subset)
        counts = np.asarray(
            [int(episode["length"]) - 10 for episode in self.table.episodes],
            dtype=np.int64,
        )
        if np.any(counts <= 0):
            raise ValueError("Every LIBERO episode must contain one H10 target.")
        self.cumulative = np.concatenate(([0], np.cumsum(counts)))
        self.task_names = tuple(sorted(set(self.table.tasks.values())))
        self.task_ids = {task: index for index, task in enumerate(self.task_names)}

    def __len__(self):
        return int(self.cumulative[-1])

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        record = bisect.bisect_right(self.cumulative, index) - 1
        frame = index - int(self.cumulative[record])
        table = pq.read_table(
            self.table.paths[record],
            columns=["image", "wrist_image", "state", "actions", "episode_index", "frame_index"],
        )
        state = np.asarray(table["state"][frame].as_py(), dtype=np.float32)
        future = np.asarray(table["actions"].slice(frame, 10).to_pylist(), dtype=np.float32)
        actions = absolute_pose_command_horizon_to_actions(state, future)
        task = self.table.task(record)
        return {
            "observation.image": _decode_png(table["image"][frame].as_py()),
            "observation.wrist_image": _decode_png(table["wrist_image"][frame].as_py()),
            "observation.state": torch.from_numpy(state),
            "actions": torch.from_numpy(actions),
            "prompt": task,
            "episode_index": torch.tensor(record, dtype=torch.long),
            "frame_index": torch.tensor(frame, dtype=torch.long),
            "zeva.task_id": torch.tensor(self.task_ids[task], dtype=torch.long),
            "zeva.progress": torch.tensor(frame / max(1, len(table) - 1), dtype=torch.float32),
        }
