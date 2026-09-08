from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path

import torch

from openpi.shared import normalize as openpi_normalize


@dataclass(frozen=True)
class QuantileActionNormalizer:
    q01: torch.Tensor
    q99: torch.Tensor
    source: str | None = None

    @classmethod
    def from_source(cls, source, action_dim: int = 7):
        if isinstance(source, (str, os.PathLike)):
            source_path = Path(source).expanduser().resolve()
            stats = openpi_normalize.load(source_path)["actions"]
            q01, q99, source_name = stats.q01, stats.q99, str(source_path)
        elif isinstance(source, Mapping):
            if source.get("type", "quantile") != "quantile":
                raise ValueError("Zeva CTE actions must use quantile normalization.")
            q01, q99, source_name = source.get("q01"), source.get("q99"), source.get("source")
        else:
            raise TypeError("Expected an OpenPI stats directory or normalization metadata mapping.")
        q01 = torch.as_tensor(q01, dtype=torch.float32).flatten()[:action_dim]
        q99 = torch.as_tensor(q99, dtype=torch.float32).flatten()[:action_dim]
        if q01.numel() != action_dim or q99.numel() != action_dim or torch.any(q99 <= q01):
            raise ValueError(f"Invalid quantile statistics for {action_dim} action dimensions.")
        return cls(q01=q01, q99=q99, source=source_name)

    def normalize(self, actions: torch.Tensor) -> torch.Tensor:
        q01 = self.q01.to(device=actions.device, dtype=actions.dtype)
        q99 = self.q99.to(device=actions.device, dtype=actions.dtype)
        return (actions - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0

    def metadata(self):
        return {
            "type": "quantile",
            "q01": self.q01.cpu().clone(),
            "q99": self.q99.cpu().clone(),
            "source": self.source,
        }
