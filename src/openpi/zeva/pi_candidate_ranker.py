"""ZeVA value head for choosing among frozen PI0.5 action chunks."""

from __future__ import annotations

import torch
from torch import nn


class RobotWinPICandidateRanker(nn.Module):
    """Predict candidate H15 log error without modifying any PI action.

    The scorer is permutation-equivariant over candidates.  Candidate zero is
    treated specially only by the deployment fallback rule, never by the
    learned value function.
    """

    def __init__(
        self,
        *,
        static_dim: int,
        horizon: int = 15,
        action_dim: int = 16,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.static_dim = int(static_dim)
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        action_features = 2 * self.horizon * self.action_dim
        self.static_encoder = nn.Sequential(
            nn.LayerNorm(self.static_dim),
            nn.Linear(self.static_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.candidate_encoder = nn.Sequential(
            nn.LayerNorm(action_features),
            nn.Linear(action_features, 2 * hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.value_head = nn.Sequential(
            nn.LayerNorm(4 * hidden_dim),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        static_features: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        """Return predicted log MSE `[batch, candidates]`; lower is better."""
        if static_features.ndim != 2 or static_features.shape[-1] != self.static_dim:
            raise ValueError(
                f"Expected static features [B,{self.static_dim}], got "
                f"{tuple(static_features.shape)}."
            )
        if candidates.ndim != 4 or candidates.shape[-2:] != (
            self.horizon,
            self.action_dim,
        ):
            raise ValueError(
                "Expected candidate actions [B,K,H,D] with "
                f"H={self.horizon}, D={self.action_dim}; got {tuple(candidates.shape)}."
            )
        if candidates.shape[0] != static_features.shape[0]:
            raise ValueError("Static and candidate batch sizes differ.")
        base = candidates[:, :1]
        candidate_input = torch.cat([candidates, candidates - base], dim=-1)
        candidate_input = candidate_input.flatten(start_dim=2)
        candidate_hidden = self.candidate_encoder(candidate_input)
        static_hidden = self.static_encoder(static_features)
        static_hidden = static_hidden[:, None].expand_as(candidate_hidden)
        fused = torch.cat(
            [
                candidate_hidden,
                static_hidden,
                candidate_hidden * static_hidden,
                (candidate_hidden - static_hidden).abs(),
            ],
            dim=-1,
        )
        return self.value_head(fused).squeeze(-1)
