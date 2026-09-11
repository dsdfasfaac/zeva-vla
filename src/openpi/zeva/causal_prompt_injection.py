"""Minimal ZeVA causal-prompt injection for a frozen PI0.5.

This module deliberately implements the *prompt* side of ZeVA only.  It does
not add a residual to noisy actions, replace an action chunk, or own an action
prior.  Equation (8) of the ZeVA paper is represented as

    M = F_mem([g, p, P_brief(H_brief), P_proj(R)]).

The default integration mechanism is a zero-initialized *additive residual on
the existing prefix embeddings*.  This is a baseline-preserving adaptation,
not an appended or prepended new prompt token.  The residual is broadcast to
all existing prefix positions.  This is important for PI0.5: its
``embed_prefix`` result determines prefix length, padding/attention masks,
position IDs, and the prefix KV cache.  Appending an all-zero token still
changes all of those quantities and is therefore not an exact Base fallback.
Adding a zero residual leaves the tuple's shapes and masks untouched, is
bit-equivalent to Base at construction time, and still gives the final
projector a nonzero first gradient through a frozen foundation.

``append_prefix_prompt`` and the hook's ``mode="append"`` are retained as an
explicit legacy mechanism for controlled ablations.  They must not be used to
claim exact Base equivalence, including when the appended token is zero.
The hook is intentionally independent from ``RobotWinZevaPolicy``: it wraps
only ``core.embed_prefix`` and can be removed without changing the foundation.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from types import MethodType
from typing import Any, Literal

import torch
from torch import Tensor
from torch import nn

_BRANCH_NAMES = ("global", "phase", "brief", "retrieved")
_PROMPT_MODES = ("residual", "append")
PromptMode = Literal["residual", "append"]


def tensor_rms(value: Tensor) -> Tensor:
    """Return a numerically stable root-mean-square magnitude."""

    if not isinstance(value, Tensor):
        raise TypeError(f"Expected a tensor, got {type(value)!r}.")
    return value.float().square().mean().sqrt()


@dataclass(frozen=True)
class PromptBranches:
    """Branch switches used for causal attribution and ablation.

    ``global_enabled`` and ``phase_enabled`` control the task/phase anchors
    ``g`` and ``p``.  ``brief_enabled`` and ``retrieved_enabled`` control the
    short-term BIT and phase-matched PIM streams respectively.
    """

    global_enabled: bool = True
    phase_enabled: bool = True
    brief_enabled: bool = True
    retrieved_enabled: bool = True

    @classmethod
    def from_value(
        cls,
        value: PromptBranches | Mapping[str, bool] | None,
    ) -> PromptBranches:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("branches must be PromptBranches, a mapping, or None.")
        unknown = sorted(set(value).difference(_BRANCH_NAMES))
        if unknown:
            raise ValueError(f"Unknown prompt branches: {unknown!r}.")
        return cls(
            global_enabled=bool(value.get("global", True)),
            phase_enabled=bool(value.get("phase", True)),
            brief_enabled=bool(value.get("brief", True)),
            retrieved_enabled=bool(value.get("retrieved", True)),
        )

    def enabled(self, name: str) -> bool:
        if name not in _BRANCH_NAMES:
            raise ValueError(f"Unknown prompt branch: {name!r}.")
        return bool(getattr(self, f"{name}_enabled"))

    def as_dict(self) -> dict[str, bool]:
        return {name: self.enabled(name) for name in _BRANCH_NAMES}

    @classmethod
    def only(cls, name: str) -> PromptBranches:
        if name not in _BRANCH_NAMES:
            raise ValueError(f"Unknown prompt branch: {name!r}.")
        return cls(**{f"{candidate}_enabled": candidate == name for candidate in _BRANCH_NAMES})


@dataclass
class CausalPromptResult:
    """Structured output of :class:`CausalPromptInjection.build_prompt`."""

    tokens: Tensor
    memory_tokens: Tensor
    gate: Tensor
    branches: PromptBranches
    branch_rms: dict[str, float]


class CausalPromptInjection(nn.Module):
    """Build a ZeVA memory prompt without touching the action output.

    Args:
        global_dim: Dimension of the task/global token ``g``.
        phase_dim: Dimension of the live phase token ``p``.
        signal_dim: Dimension of each BIT/PIM causal signal.
        context_dim: Internal ``F_mem`` width.
        foundation_dim: PI0.5 prefix embedding width (2048 for PaliGemma).
        prompt_tokens: Number of prefix tokens emitted for ``M``.  Eq. (8) is
            a single vector, so the default is one token.
        num_heads: Number of attention heads in the small memory fuser.
        gate_init: Direct prompt gate.  The default ``1.0`` leaves the
            zero-initialized residual projector learnable at step zero;
            ``0.0`` is an explicitly closed/no-gradient ablation.
        zero_init_projection: Zero-initialize the final context-to-prefix
            projection.  This keeps a newly created adapter conservative once
            the hook is opened, while the closed hook remains exactly Base.
    """

    def __init__(
        self,
        *,
        global_dim: int,
        phase_dim: int,
        signal_dim: int,
        context_dim: int = 256,
        foundation_dim: int = 2048,
        prompt_tokens: int = 1,
        num_heads: int = 4,
        dropout: float = 0.0,
        gate_init: float = 1.0,
        zero_init_projection: bool = True,
    ):
        super().__init__()
        for name, value in (
            ("global_dim", global_dim),
            ("phase_dim", phase_dim),
            ("signal_dim", signal_dim),
            ("context_dim", context_dim),
            ("foundation_dim", foundation_dim),
            ("prompt_tokens", prompt_tokens),
            ("num_heads", num_heads),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive, got {value!r}.")
        if context_dim % num_heads:
            raise ValueError("context_dim must be divisible by num_heads.")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if not torch.isfinite(torch.tensor(float(gate_init))):
            raise ValueError("gate_init must be finite.")

        self.global_dim = int(global_dim)
        self.phase_dim = int(phase_dim)
        self.signal_dim = int(signal_dim)
        self.context_dim = int(context_dim)
        self.foundation_dim = int(foundation_dim)
        self.prompt_tokens = int(prompt_tokens)

        # These are the four explicit projections in Eq. (8).  Keeping brief
        # and retrieved projections separate makes branch attribution honest.
        self.global_projector = nn.Linear(global_dim, context_dim)
        self.phase_projector = nn.Linear(phase_dim, context_dim)
        self.brief_projector = nn.Linear(signal_dim, context_dim)
        self.retrieved_projector = nn.Linear(signal_dim, context_dim)

        self.memory_queries = nn.Parameter(torch.empty(prompt_tokens, context_dim))
        nn.init.normal_(self.memory_queries, mean=0.0, std=context_dim**-0.5)
        self.memory_attention = nn.MultiheadAttention(
            context_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.memory_norm = nn.LayerNorm(context_dim)
        self.memory_mlp = nn.Sequential(
            nn.Linear(context_dim, 2 * context_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * context_dim, context_dim),
        )
        self.prompt_projection = nn.Linear(context_dim, foundation_dim)
        if zero_init_projection:
            nn.init.zeros_(self.prompt_projection.weight)
            nn.init.zeros_(self.prompt_projection.bias)

        # A direct gate is deliberately used instead of sigmoid(logit), whose
        # zero value would be 0.5.  The gate is open by default because the
        # zero-initialized prompt projection already gives an exact Base path;
        # setting the gate to zero is an explicit closed/no-gradient ablation.
        self.gate = nn.Parameter(torch.tensor(float(gate_init), dtype=torch.float32))
        self._injection_enabled = True

    @property
    def injection_enabled(self) -> bool:
        """Whether callers are allowed to materialize prefix prompt tokens."""

        return self._injection_enabled

    @property
    def gate_value(self) -> float:
        return float(self.gate.detach().float().item())

    @property
    def is_closed(self) -> bool:
        """True when integration should bypass prefix concatenation entirely."""

        return not self._injection_enabled or self.gate_value == 0.0

    def set_injection_enabled(self, *, enabled: bool) -> None:
        self._injection_enabled = bool(enabled)

    def open_gate(self, value: float = 1.0) -> None:
        value = float(value)
        if not torch.isfinite(torch.tensor(value)) or value <= 0.0:
            raise ValueError("An open gate must be finite and positive.")
        with torch.no_grad():
            self.gate.fill_(value)

    def close_gate(self) -> None:
        with torch.no_grad():
            self.gate.zero_()

    @staticmethod
    def _batch_size(values: dict[str, Tensor | None]) -> int:
        sizes = {int(value.shape[0]) for value in values.values() if value is not None}
        if not sizes:
            raise ValueError("At least one prompt input tensor is required.")
        if len(sizes) != 1:
            raise ValueError(f"Prompt input batch sizes disagree: {sorted(sizes)!r}.")
        return sizes.pop()

    @staticmethod
    def _vector(
        value: Tensor | None,
        *,
        name: str,
        width: int,
        batch_size: int,
    ) -> Tensor:
        if value is None:
            raise ValueError(f"Prompt branch {name!r} is enabled but has no tensor.")
        if value.ndim != 2 or value.shape[0] != batch_size or value.shape[1] != width:
            raise ValueError(
                f"{name} must have shape [{batch_size}, {width}], got {tuple(value.shape)}."
            )
        return value

    @staticmethod
    def _sequence(
        value: Tensor | None,
        *,
        name: str,
        width: int,
        batch_size: int,
    ) -> Tensor:
        if value is None:
            raise ValueError(f"Prompt branch {name!r} is enabled but has no tensor.")
        if value.ndim != 3 or value.shape[0] != batch_size or value.shape[2] != width:
            raise ValueError(
                f"{name} must have shape [{batch_size}, L, {width}], got {tuple(value.shape)}."
            )
        return value

    @staticmethod
    def _sequence_mask(
        value: Tensor | None,
        *,
        batch_size: int,
        length: int,
        name: str,
        device: torch.device,
    ) -> Tensor:
        if value is None:
            return torch.ones(batch_size, length, dtype=torch.bool, device=device)
        if value.shape != (batch_size, length):
            raise ValueError(
                f"{name} mask must have shape [{batch_size}, {length}], got {tuple(value.shape)}."
            )
        return value.to(device=device, dtype=torch.bool)

    def _memory_tokens(
        self,
        *,
        global_token: Tensor | None,
        phase_token: Tensor | None,
        brief_signals: Tensor | None,
        retrieved_signals: Tensor | None,
        brief_mask: Tensor | None,
        retrieved_mask: Tensor | None,
        branches: PromptBranches,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        values: dict[str, Tensor | None] = {
            "global": global_token,
            "phase": phase_token,
            "brief": brief_signals,
            "retrieved": retrieved_signals,
        }
        batch_size = self._batch_size(values)
        projected: list[Tensor] = []
        masks: list[Tensor] = []
        branch_contexts: dict[str, Tensor] = {}

        if branches.global_enabled:
            token = self._vector(
                global_token,
                name="global_token",
                width=self.global_dim,
                batch_size=batch_size,
            )
            projected_token = self.global_projector(token.to(self.global_projector.weight.dtype)).unsqueeze(1)
            projected.append(projected_token)
            masks.append(torch.ones(batch_size, 1, dtype=torch.bool, device=projected_token.device))
            branch_contexts["global"] = projected_token

        if branches.phase_enabled:
            token = self._vector(
                phase_token,
                name="phase_token",
                width=self.phase_dim,
                batch_size=batch_size,
            )
            projected_token = self.phase_projector(token.to(self.phase_projector.weight.dtype)).unsqueeze(1)
            projected.append(projected_token)
            masks.append(torch.ones(batch_size, 1, dtype=torch.bool, device=projected_token.device))
            branch_contexts["phase"] = projected_token

        if branches.brief_enabled:
            signals = self._sequence(
                brief_signals,
                name="brief_signals",
                width=self.signal_dim,
                batch_size=batch_size,
            )
            projected_signals = self.brief_projector(signals.to(self.brief_projector.weight.dtype))
            projected.append(projected_signals)
            masks.append(
                self._sequence_mask(
                    brief_mask,
                    batch_size=batch_size,
                    length=signals.shape[1],
                    name="brief_signals",
                    device=projected_signals.device,
                )
            )
            branch_contexts["brief"] = projected_signals

        if branches.retrieved_enabled:
            signals = self._sequence(
                retrieved_signals,
                name="retrieved_signals",
                width=self.signal_dim,
                batch_size=batch_size,
            )
            projected_signals = self.retrieved_projector(
                signals.to(self.retrieved_projector.weight.dtype)
            )
            projected.append(projected_signals)
            masks.append(
                self._sequence_mask(
                    retrieved_mask,
                    batch_size=batch_size,
                    length=signals.shape[1],
                    name="retrieved_signals",
                    device=projected_signals.device,
                )
            )
            branch_contexts["retrieved"] = projected_signals

        if not projected:
            # An all-off branch ablation is a true no-prompt condition.  The
            # caller can use ``prompt_tokens_or_none`` to bypass prefix changes.
            empty = self.memory_queries.new_zeros((batch_size, 0, self.context_dim))
            return empty, torch.zeros(batch_size, 0, dtype=torch.bool, device=empty.device), branch_contexts

        memory = torch.cat(projected, dim=1)
        valid = torch.cat(masks, dim=1)
        return memory, valid, branch_contexts

    def build_prompt(
        self,
        global_token: Tensor | None,
        phase_token: Tensor | None,
        brief_signals: Tensor | None,
        retrieved_signals: Tensor | None,
        *,
        brief_mask: Tensor | None = None,
        retrieved_mask: Tensor | None = None,
        branches: PromptBranches | Mapping[str, bool] | None = None,
    ) -> CausalPromptResult:
        """Construct ``M`` and project it to PI0.5 prefix-token width."""

        selected = PromptBranches.from_value(branches)
        memory, valid, branch_contexts = self._memory_tokens(
            global_token=global_token,
            phase_token=phase_token,
            brief_signals=brief_signals,
            retrieved_signals=retrieved_signals,
            brief_mask=brief_mask,
            retrieved_mask=retrieved_mask,
            branches=selected,
        )
        if memory.shape[1] == 0:
            batch_size = self._batch_size(
                {
                    "global": global_token,
                    "phase": phase_token,
                    "brief": brief_signals,
                    "retrieved": retrieved_signals,
                }
            )
            memory_output = self.memory_queries.new_zeros(
                (batch_size, self.prompt_tokens, self.context_dim)
            )
            # An all-off ablation is a true no-prompt condition.  Do not pass
            # the synthetic zero through a projection bias: a nonzero bias
            # would make the branch ablation look causal despite having no
            # enabled evidence.
            prompt_tokens = memory_output.new_zeros(
                (batch_size, self.prompt_tokens, self.foundation_dim)
            )
        else:
            query = self.memory_queries.to(device=memory.device, dtype=memory.dtype)
            query = query.unsqueeze(0).expand(memory.shape[0], -1, -1)
            safe_valid = valid.clone()
            all_invalid = ~safe_valid.any(dim=1)
            if all_invalid.any():
                # MultiheadAttention would produce NaNs for an all-masked row.
                # A zero fallback key leaves that row deterministic and marks it
                # as having no usable causal evidence.
                safe_valid[all_invalid, 0] = True
                memory = memory.clone()
                memory[all_invalid, 0] = 0.0
            attended, _ = self.memory_attention(
                query,
                memory,
                memory,
                key_padding_mask=~safe_valid,
                need_weights=False,
            )
            memory_output = self.memory_norm(attended + query)
            memory_output = memory_output + self.memory_mlp(memory_output)
            prompt_tokens = self.prompt_projection(memory_output)

        gate = self.gate.to(device=prompt_tokens.device, dtype=prompt_tokens.dtype).clamp(0.0, 1.0)
        prompt_tokens = prompt_tokens * gate
        branch_rms = {
            name: float(tensor_rms(value).detach()) for name, value in branch_contexts.items()
        }
        for name in _BRANCH_NAMES:
            branch_rms.setdefault(name, 0.0)
        return CausalPromptResult(
            tokens=prompt_tokens,
            memory_tokens=memory_output,
            gate=gate,
            branches=selected,
            branch_rms=branch_rms,
        )

    def forward(
        self,
        global_token: Tensor | None,
        phase_token: Tensor | None,
        brief_signals: Tensor | None,
        retrieved_signals: Tensor | None,
        *,
        brief_mask: Tensor | None = None,
        retrieved_mask: Tensor | None = None,
        branches: PromptBranches | Mapping[str, bool] | None = None,
    ) -> Tensor:
        """Return prefix tokens only; no action tensor is accepted or returned."""

        return self.build_prompt(
            global_token,
            phase_token,
            brief_signals,
            retrieved_signals,
            brief_mask=brief_mask,
            retrieved_mask=retrieved_mask,
            branches=branches,
        ).tokens

    def prompt_tokens_or_none(self, result: CausalPromptResult) -> Tensor | None:
        """Convert a prompt result into a legacy append-mode hook input.

        A closed gate returns ``None`` rather than a zero token.  Appending a
        zero prefix token would still change attention normalization and
        position IDs, so this method is not an exact-Base mechanism.  Use
        :meth:`prompt_residual_or_none` for the default, identity-preserving
        integration.
        """

        if self.is_closed:
            return None
        return result.tokens

    def prompt_residual_or_none(self, result: CausalPromptResult) -> Tensor | None:
        """Return the identity-preserving residual for ``embed_prefix``.

        The residual path intentionally emits one vector per example and
        broadcasts it over the existing prefix sequence.  Broadcasting over
        the existing image/language tokens changes no prefix lengths, masks,
        or position IDs.  ``prompt_tokens`` greater than one is meaningful
        for legacy append-mode experiments but is ambiguous for this residual
        path and is rejected.
        """

        if self.is_closed:
            return None
        if result.tokens.ndim != 3 or result.tokens.shape[1] != 1:
            raise ValueError(
                "The exact-Base residual path requires exactly one prompt token; "
                f"got {tuple(result.tokens.shape)}."
            )
        return result.tokens[:, 0, :]


def append_prefix_prompt(
    prefix_output: tuple[Tensor, ...],
    prompt_tokens: Tensor | None,
) -> tuple[Tensor, ...]:
    """Append prompt tokens to a PI0.5 ``embed_prefix`` result.

    The first three values are the standard ``(embeddings, pad_masks,
    attention_masks)`` tuple.  Extra return values are preserved.  Prompt
    tokens receive ``ar_mask=0`` so they participate in the full-attention
    prefix block and are visible to every subsequent diffusion/action-expert
    query through the normal prefix-to-suffix mask.
    """

    if prompt_tokens is None:
        return prefix_output
    if len(prefix_output) < 3:
        raise ValueError("PI0.5 embed_prefix must return at least three tensors.")
    embeddings, pad_masks, attention_masks = prefix_output[:3]
    if embeddings.ndim != 3:
        raise ValueError(f"Prefix embeddings must be [B,L,D], got {tuple(embeddings.shape)}.")
    if prompt_tokens.ndim == 2:
        prompt_tokens = prompt_tokens.unsqueeze(1)
    if prompt_tokens.ndim != 3:
        raise ValueError(f"Prompt tokens must be [B,P,D], got {tuple(prompt_tokens.shape)}.")
    if prompt_tokens.shape[0] != embeddings.shape[0] or prompt_tokens.shape[2] != embeddings.shape[2]:
        raise ValueError(
            "Prompt and prefix dimensions disagree: "
            f"prefix={tuple(embeddings.shape)}, prompt={tuple(prompt_tokens.shape)}."
        )
    prompt_tokens = prompt_tokens.to(device=embeddings.device, dtype=embeddings.dtype)
    prompt_pad = torch.ones(
        embeddings.shape[0],
        prompt_tokens.shape[1],
        dtype=pad_masks.dtype,
        device=pad_masks.device,
    )
    if attention_masks.ndim == 1:
        prompt_att = torch.zeros(
            prompt_tokens.shape[1],
            dtype=attention_masks.dtype,
            device=attention_masks.device,
        )
        new_attention = torch.cat([attention_masks, prompt_att], dim=0)
    elif attention_masks.ndim == 2:
        prompt_att = torch.zeros(
            embeddings.shape[0],
            prompt_tokens.shape[1],
            dtype=attention_masks.dtype,
            device=attention_masks.device,
        )
        new_attention = torch.cat([attention_masks, prompt_att], dim=1)
    else:
        raise ValueError(
            f"Prefix attention masks must be [L] or [B,L], got {tuple(attention_masks.shape)}."
        )
    return (
        torch.cat([embeddings, prompt_tokens], dim=1),
        torch.cat([pad_masks, prompt_pad], dim=1),
        new_attention,
        *prefix_output[3:],
    )


def add_prefix_residual(
    prefix_output: tuple[Tensor, ...],
    residual: Tensor | None,
) -> tuple[Tensor, ...]:
    """Add a learned residual to every existing PI0.5 prefix embedding.

    ``residual`` is ``[B, D]`` (or ``[B, 1, D]``) and is broadcast over the
    existing prefix sequence.  No new prompt token, sequence dimension, mask,
    or position ID is created or modified, so a zero residual is an exact Base
    computation up to the identity floating-point addition.  This is the
    mechanism used by the new causal-prompt path.
    """

    if residual is None:
        return prefix_output
    if len(prefix_output) < 3:
        raise ValueError("PI0.5 embed_prefix must return at least three tensors.")
    embeddings = prefix_output[0]
    if embeddings.ndim != 3:
        raise ValueError(f"Prefix embeddings must be [B,L,D], got {tuple(embeddings.shape)}.")
    if residual.ndim == 3:
        if residual.shape[1] != 1:
            raise ValueError(
                "Prefix residual must have shape [B,D] or [B,1,D], "
                f"got {tuple(residual.shape)}."
            )
        residual = residual[:, 0, :]
    if residual.ndim != 2:
        raise ValueError(
            "Prefix residual must have shape [B,D] or [B,1,D], "
            f"got {tuple(residual.shape)}."
        )
    if residual.shape != (embeddings.shape[0], embeddings.shape[2]):
        raise ValueError(
            "Prefix residual and embedding dimensions disagree: "
            f"prefix={tuple(embeddings.shape)}, residual={tuple(residual.shape)}."
        )
    residual = residual.to(device=embeddings.device, dtype=embeddings.dtype)
    return (embeddings + residual[:, None, :], *prefix_output[1:])


class PrefixPromptHook:
    """Temporarily expose a causal residual through ``core.embed_prefix``.

    ``core`` may be the LeRobot PI05 model or the local PyTorch PI0.5 model as
    long as it has an ``embed_prefix`` method with the standard three leading
    return values.  Residual mode modifies existing prefix embeddings only;
    explicit append mode is retained for legacy ablations.  The hook never
    touches ``embed_suffix`` or action outputs.
    """

    def __init__(self, core: nn.Module, *, method_name: str = "embed_prefix"):
        if not hasattr(core, method_name):
            raise AttributeError(f"Foundation core has no {method_name!r} method.")
        self.core = core
        self.method_name = method_name
        self._original: Callable[..., Any] | None = None
        self._active_prompt: Tensor | None = None
        self._active_mode: PromptMode = "residual"

    @property
    def installed(self) -> bool:
        return self._original is not None

    def install(self) -> PrefixPromptHook:
        if self.installed:
            return self
        original = getattr(self.core, self.method_name)
        owner = self

        def wrapped(_core: nn.Module, *args: Any, **kwargs: Any):
            result = original(*args, **kwargs)
            if owner._active_prompt is None:
                return result
            if not isinstance(result, tuple):
                raise TypeError("PI0.5 embed_prefix must return a tuple.")
            if owner._active_mode == "residual":
                return add_prefix_residual(result, owner._active_prompt)
            return append_prefix_prompt(result, owner._active_prompt)

        self._original = original
        setattr(self.core, self.method_name, MethodType(wrapped, self.core))
        return self

    def uninstall(self) -> None:
        if self._active_prompt is not None:
            raise RuntimeError("Cannot uninstall PrefixPromptHook while it is active.")
        if self._original is not None:
            setattr(self.core, self.method_name, self._original)
            self._original = None

    @contextmanager
    def activate(self, prompt_tokens: Tensor | None, *, mode: PromptMode = "residual") -> Iterator[None]:
        if not self.installed:
            raise RuntimeError("Install PrefixPromptHook before activation.")
        if self._active_prompt is not None:
            raise RuntimeError("Nested PrefixPromptHook activation is not supported.")
        if mode not in _PROMPT_MODES:
            raise ValueError(f"Unknown prefix prompt mode {mode!r}; expected {_PROMPT_MODES!r}.")
        self._active_prompt = prompt_tokens
        self._active_mode = mode  # type: ignore[assignment]
        try:
            yield
        finally:
            self._active_prompt = None
            self._active_mode = "residual"

    def run(
        self,
        forward: Callable[[], Any],
        prompt_tokens: Tensor | None,
        *,
        mode: PromptMode = "residual",
    ) -> Any:
        """Run a frozen-foundation callable with one prompt condition."""

        with self.activate(prompt_tokens, mode=mode):
            return forward()


@dataclass
class PromptGradientAudit:
    """Gradient/RMS summary produced by a frozen-foundation probe."""

    loss: float
    prompt_rms: float
    gradient_norms: dict[str, float]
    frozen_parameter_gradients: dict[str, float]


def audit_prompt_gradients(
    injector: CausalPromptInjection,
    result: CausalPromptResult,
    forward_with_prompt: Callable[[Tensor], Tensor],
    *,
    loss_fn: Callable[[Tensor], Tensor] | None = None,
    frozen_parameters: Mapping[str, nn.Parameter] | None = None,
) -> PromptGradientAudit:
    """Backpropagate one scalar through a frozen foundation and audit paths.

    ``forward_with_prompt`` must run the frozen PI0.5 with the supplied prefix
    tokens through :class:`PrefixPromptHook`.  The helper does not call an
    optimizer and never changes foundation parameters.  A non-empty gradient
    on the prompt projections is the minimum signal required before any
    Stage-2 unfreezing experiment.
    """

    injector.zero_grad(set_to_none=True)
    output = forward_with_prompt(result.tokens)
    loss = output.square().mean() if loss_fn is None else loss_fn(output)
    if loss.ndim != 0:
        raise ValueError("loss_fn must return a scalar tensor.")
    loss.backward()
    gradient_norms = {
        name: float(parameter.grad.detach().float().norm())
        for name, parameter in injector.named_parameters()
        if parameter.grad is not None
    }
    frozen_gradient_norms: dict[str, float] = {}
    if frozen_parameters is not None:
        frozen_gradient_norms = {
            name: float(parameter.grad.detach().float().norm())
            for name, parameter in frozen_parameters.items()
            if parameter.grad is not None
        }
    return PromptGradientAudit(
        loss=float(loss.detach().float()),
        prompt_rms=float(tensor_rms(result.tokens).detach()),
        gradient_norms=gradient_norms,
        frozen_parameter_gradients=frozen_gradient_norms,
    )


__all__ = [
    "CausalPromptInjection",
    "CausalPromptResult",
    "PrefixPromptHook",
    "PromptBranches",
    "PromptGradientAudit",
    "PromptMode",
    "add_prefix_residual",
    "append_prefix_prompt",
    "audit_prompt_gradients",
    "tensor_rms",
]
