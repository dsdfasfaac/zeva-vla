"""Frozen official-train LIBERO causal bank shared by ZeVA Stages 2 and 3."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from openpi.zeva.causal_bank import CAUSAL_BANK_SCHEMA
from openpi.zeva.causal_bank import RobotWinCausalBank


LIBERO_CAUSAL_BANK_SCHEMA = "zeva-libero-train-causal-bank-v2"


class LiberoCausalBank(RobotWinCausalBank):
    """Read-only phase-indexed ZTE keys and values from 1,614 train episodes."""

    def __init__(self, payload: dict[str, Any], *, device: torch.device | str = "cpu"):
        if payload.get("schema") != LIBERO_CAUSAL_BANK_SCHEMA:
            raise ValueError("Unsupported LIBERO causal bank schema.")
        if payload.get("split") != "official-train-1614":
            raise ValueError("LIBERO Stage 2 may load only the official train bank.")
        compatible = dict(payload)
        compatible["schema"] = CAUSAL_BANK_SCHEMA
        compatible["split"] = "train95"
        super().__init__(compatible, device=device)

    @classmethod
    def load(cls, path: str | Path, *, device: torch.device | str = "cpu") -> "LiberoCausalBank":
        return cls(torch.load(path, map_location="cpu", weights_only=False), device=device)
