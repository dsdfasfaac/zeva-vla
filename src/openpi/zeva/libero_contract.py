"""Frozen LIBERO contract for the selected PI0.5 training handoff."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


LIBERO_HANDOFF_SCHEMA = "egoscale-libero-memory-baseline-handoff-v1"
LIBERO_DATASET_SCHEMA = "pi05-libero-stage1-eef16-dataset-v2"
LIBERO_ACTION_DIM = 16
LIBERO_MODEL_ACTION_DIM = 32
LIBERO_POLICY_HORIZON = 10
LIBERO_EXECUTION_HORIZON = 5
LIBERO_IMAGE_SIZE = 256
LIBERO_MODEL_IMAGE_SIZE = 224
LIBERO_CAMERA_KEYS = ("observation.image", "observation.wrist_image")
LIBERO_TASK_COUNT = 40
LIBERO_CHECKPOINT_SHA256 = "d6eabd264bb4b7b4fbde795068a1615ba6a2ce5c18fefcbefb0c5498b5582d92"
LIBERO_PHYSICAL_CONTRACT = (
    "dual-eef-camera-t0-spatial-delta-quat-xyzw-official-gripper-command-tail-16d-v2"
)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class QuantileActionNormalizer:
    """Exact q01/q99 action transform shipped with the LIBERO checkpoint."""

    q01: torch.Tensor
    q99: torch.Tensor
    source: str

    @classmethod
    def from_stats_file(cls, path: str | Path) -> "QuantileActionNormalizer":
        resolved = Path(path).resolve()
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        entry = payload.get("norm_stats", {}).get("actions", {})
        q01 = torch.as_tensor(entry.get("q01"), dtype=torch.float32)
        q99 = torch.as_tensor(entry.get("q99"), dtype=torch.float32)
        if q01.shape != (LIBERO_ACTION_DIM,) or q99.shape != (LIBERO_ACTION_DIM,):
            raise ValueError("LIBERO quantile statistics must describe EEF16 actions.")
        if not torch.isfinite(q01).all() or not torch.isfinite(q99).all():
            raise ValueError("LIBERO action quantiles must be finite.")
        # Padding dimensions legitimately use [-1, 1], so every range remains positive.
        if torch.any(q99 <= q01):
            raise ValueError("LIBERO q99 must be strictly greater than q01.")
        return cls(q01=q01, q99=q99, source=str(resolved))

    def normalize(self, actions: torch.Tensor) -> torch.Tensor:
        q01 = self.q01.to(device=actions.device, dtype=actions.dtype)
        q99 = self.q99.to(device=actions.device, dtype=actions.dtype)
        return (actions[..., :LIBERO_ACTION_DIM] - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0

    def unnormalize(self, actions: torch.Tensor) -> torch.Tensor:
        q01 = self.q01.to(device=actions.device, dtype=actions.dtype)
        q99 = self.q99.to(device=actions.device, dtype=actions.dtype)
        return (actions[..., :LIBERO_ACTION_DIM] + 1.0) * 0.5 * (q99 - q01 + 1e-6) + q01

    def metadata(self) -> dict[str, Any]:
        return {
            "type": "quantile",
            "q01": self.q01.cpu().clone(),
            "q99": self.q99.cpu().clone(),
            "source": self.source,
        }


@dataclass(frozen=True)
class LiberoHandoff:
    root: Path
    checkpoint: Path
    contract: Path
    statistics: Path

    @classmethod
    def from_root(cls, root: str | Path, *, verify_checkpoint_hash: bool = False) -> "LiberoHandoff":
        root = Path(root).resolve()
        packaged_contract = root / "runtime" / "baseline" / "contract.json"
        staged_contract = root / "contract.json"
        packaged_stats = (
            root
            / "reference"
            / "assets"
            / "pi05_libero_stage1_eef16"
            / "physical-intelligence"
            / "libero"
            / "norm_stats.json"
        )
        staged_stats = (
            root
            / "assets"
            / "pi05_libero_stage1_eef16"
            / "physical-intelligence"
            / "libero"
            / "norm_stats.json"
        )
        handoff = cls(
            root=root,
            checkpoint=root / "checkpoint" / "pretrained_model",
            contract=packaged_contract if packaged_contract.is_file() else staged_contract,
            statistics=packaged_stats if packaged_stats.is_file() else staged_stats,
        )
        handoff.validate(verify_checkpoint_hash=verify_checkpoint_hash)
        return handoff

    def validate(self, *, verify_checkpoint_hash: bool = False) -> None:
        for path in (self.contract, self.statistics, self.checkpoint / "model.safetensors"):
            if not path.is_file() or path.stat().st_size <= 0:
                raise FileNotFoundError(path)
        contract = json.loads(self.contract.read_text(encoding="utf-8"))
        if contract.get("schema") != LIBERO_HANDOFF_SCHEMA:
            raise ValueError("Unsupported LIBERO handoff schema.")
        dataset = contract.get("dataset", {})
        expected_counts = {
            "episodes_total": 1693,
            "train_episodes": 1614,
            "validation_episodes": 79,
            "train_valid_rows": 244430,
            "validation_valid_rows": 12105,
        }
        if any(dataset.get(key) != value for key, value in expected_counts.items()):
            raise ValueError(f"LIBERO dataset counts drifted: {dataset!r}.")
        physical = contract.get("physical_contract", {})
        if physical.get("action") != "chunk-start-relative-eef16":
            raise ValueError("LIBERO action representation drifted.")
        if physical.get("action_horizon") != LIBERO_POLICY_HORIZON:
            raise ValueError("LIBERO PI0.5 must retain H10 prediction.")
        if tuple(physical.get("image_inputs", ())) != ("agentview_rgb", "eye_in_hand_rgb"):
            raise ValueError("LIBERO camera contract drifted.")
        normalization = contract.get("normalization", {})
        if normalization.get("type") != "quantile" or normalization.get("recompute") is not False:
            raise ValueError("The selected LIBERO PI0.5 requires frozen quantile statistics.")
        checkpoint = self.checkpoint / "model.safetensors"
        if checkpoint.stat().st_size <= 7_000_000_000:
            raise ValueError("LIBERO PI0.5 checkpoint is incomplete.")
        if verify_checkpoint_hash and sha256(checkpoint) != LIBERO_CHECKPOINT_SHA256:
            raise ValueError("LIBERO PI0.5 checkpoint hash differs from the selected release.")
        QuantileActionNormalizer.from_stats_file(self.statistics)
