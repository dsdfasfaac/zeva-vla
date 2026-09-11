import unittest
import torch
from torch import nn

from openpi.zeva.causal_prompt_injection import CausalPromptInjection
from openpi.zeva.causal_prompt_injection import PrefixPromptHook
from openpi.zeva.causal_prompt_injection import PromptBranches
from openpi.zeva.causal_prompt_injection import add_prefix_residual
from openpi.zeva.causal_prompt_injection import append_prefix_prompt
from openpi.zeva.causal_prompt_injection import audit_prompt_gradients


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(
        unittest.FunctionTestCase(value)
        for name, value in globals().items()
        if name.startswith("test_") and callable(value)
    )


def _features(batch: int = 3):
    return {
        "global_token": torch.randn(batch, 6),
        "phase_token": torch.randn(batch, 4),
        "brief_signals": torch.randn(batch, 2, 5),
        "retrieved_signals": torch.randn(batch, 3, 5),
        "brief_mask": torch.tensor([[True, True], [True, False], [True, True]]),
        "retrieved_mask": torch.tensor(
            [[True, True, False], [True, False, False], [True, True, True]]
        ),
    }


def _injector(*, gate: float = 1.0, zero_init_projection: bool = False):
    return CausalPromptInjection(
        global_dim=6,
        phase_dim=4,
        signal_dim=5,
        context_dim=8,
        foundation_dim=12,
        num_heads=2,
        gate_init=gate,
        zero_init_projection=zero_init_projection,
    )


def test_prompt_formula_and_branch_ablations_are_shape_safe():
    torch.manual_seed(3)
    injector = _injector()
    values = _features()

    full = injector.build_prompt(**values)
    assert full.tokens.shape == (3, 1, 12)
    assert full.memory_tokens.shape == (3, 1, 8)
    assert set(full.branch_rms) == {"global", "phase", "brief", "retrieved"}

    for name in ("global", "phase", "brief", "retrieved"):
        ablated = injector.build_prompt(**values, branches=PromptBranches.only(name))
        assert ablated.tokens.shape == full.tokens.shape
        assert ablated.branches.enabled(name)
        assert sum(ablated.branches.as_dict().values()) == 1

    no_branches = injector.build_prompt(
        **values,
        branches=PromptBranches(
            global_enabled=False,
            phase_enabled=False,
            brief_enabled=False,
            retrieved_enabled=False,
        ),
    )
    assert torch.count_nonzero(no_branches.tokens) == 0


def test_closed_gate_bypasses_prefix_and_is_bit_equivalent():
    class Core(nn.Module):
        def embed_prefix(self, embeddings):
            batch, length, _ = embeddings.shape
            return (
                embeddings,
                torch.ones(batch, length, dtype=torch.bool),
                torch.zeros(length, dtype=torch.bool),
            )

    core = Core()
    hook = PrefixPromptHook(core).install()
    injector = _injector(gate=0.0, zero_init_projection=False)
    values = _features(batch=2)
    result = injector.build_prompt(**{key: value[:2] for key, value in values.items()})
    base = core.embed_prefix(torch.randn(2, 4, 12))
    inputs = torch.randn(2, 4, 12)
    base = core.embed_prefix(inputs)
    with hook.activate(injector.prompt_tokens_or_none(result)):
        closed = core.embed_prefix(inputs)

    for expected, actual in zip(base, closed, strict=True):
        assert torch.equal(expected, actual)
    hook.uninstall()


def test_open_prompt_is_prefix_visible_and_not_an_action_residual():
    core = type(
        "Core",
        (nn.Module,),
        {
            "embed_prefix": lambda self, embeddings: (
                embeddings,
                torch.ones(embeddings.shape[0], embeddings.shape[1], dtype=torch.bool),
                torch.zeros(embeddings.shape[1], dtype=torch.bool),
            )
        },
    )()
    hook = PrefixPromptHook(core).install()
    injector = _injector(gate=1.0)
    values = _features(batch=2)
    result = injector.build_prompt(**{key: value[:2] for key, value in values.items()})
    inputs = torch.randn(2, 4, 12)
    # Token append is retained only as an explicit legacy ablation.  The
    # default causal path below uses ``mode="residual"`` and preserves length.
    with hook.activate(result.tokens, mode="append"):
        embeddings, pad_masks, attention_masks = core.embed_prefix(inputs)

    assert embeddings.shape == (2, 5, 12)
    assert pad_masks.shape == (2, 5)
    assert attention_masks.shape == (5,)
    assert torch.all(pad_masks[:, -1])
    assert torch.count_nonzero(attention_masks[-1]) == 0
    hook.uninstall()


def test_zero_init_residual_is_exact_base_and_has_first_projector_gradient():
    class Core(nn.Module):
        def __init__(self):
            super().__init__()
            self.frozen = nn.Linear(12, 12, bias=False)
            self.frozen.requires_grad_(requires_grad=False)

        def embed_prefix(self, embeddings):
            transformed = self.frozen(embeddings)
            batch, length, _ = transformed.shape
            return (
                transformed,
                torch.ones(batch, length, dtype=torch.bool),
                torch.zeros(length, dtype=torch.bool),
            )

    torch.manual_seed(11)
    core = Core()
    hook = PrefixPromptHook(core).install()
    injector = _injector(gate=1.0, zero_init_projection=True)
    values = _features(batch=2)
    result = injector.build_prompt(**{key: value[:2] for key, value in values.items()})
    inputs = torch.randn(2, 4, 12)

    with torch.no_grad():
        base = core.embed_prefix(inputs)[0]
    with hook.activate(injector.prompt_residual_or_none(result), mode="residual"):
        injected = core.embed_prefix(inputs)[0]

    assert torch.equal(base, injected)
    assert result.tokens.shape == (2, 1, 12)
    assert torch.count_nonzero(result.tokens) == 0

    injector.zero_grad(set_to_none=True)
    with hook.activate(injector.prompt_residual_or_none(result), mode="residual"):
        output = core.embed_prefix(inputs)[0]
    loss = (output.square()).mean()
    loss.backward()
    assert injector.prompt_projection.weight.grad is not None
    assert torch.count_nonzero(injector.prompt_projection.weight.grad) > 0
    assert core.frozen.weight.grad is None
    hook.uninstall()


def test_residual_hook_preserves_prefix_length_and_extra_return_values():
    embeddings = torch.randn(2, 3, 7)
    pad = torch.ones(2, 3, dtype=torch.bool)
    attention = torch.zeros(3, dtype=torch.bool)
    extra = torch.tensor(4)
    residual = torch.randn(2, 7)
    output = add_prefix_residual((embeddings, pad, attention, extra), residual)
    assert output[0].shape == (2, 3, 7)
    assert output[1] is pad
    assert output[2] is attention
    assert output[3] is extra
    assert torch.equal(output[0], embeddings + residual[:, None])


def test_prefix_hook_gradient_audit_keeps_frozen_parameters_clean():
    class Core(nn.Module):
        def embed_prefix(self, embeddings):
            batch, length, _ = embeddings.shape
            return (
                embeddings,
                torch.ones(batch, length, dtype=torch.bool),
                torch.zeros(length, dtype=torch.bool),
            )

    core = Core()
    hook = PrefixPromptHook(core).install()
    injector = _injector(gate=1.0)
    values = _features(batch=2)
    result = injector.build_prompt(**{key: value[:2] for key, value in values.items()})
    base = torch.randn(2, 4, 12)
    frozen = nn.Parameter(torch.ones(1), requires_grad=False)

    def forward_with_prompt(prompt):
        with hook.activate(prompt):
            return core.embed_prefix(base)[0].sum(dim=(1, 2))

    audit = audit_prompt_gradients(
        injector,
        result,
        forward_with_prompt,
        frozen_parameters={"frozen": frozen},
    )
    assert audit.prompt_rms > 0.0
    assert audit.gradient_norms
    assert not audit.frozen_parameter_gradients
    hook.uninstall()


def test_append_prefix_prompt_preserves_extra_return_values_and_1d_mask():
    embeddings = torch.randn(2, 3, 7)
    pad = torch.ones(2, 3, dtype=torch.bool)
    attention = torch.zeros(3, dtype=torch.bool)
    extra = torch.tensor(4)
    output = append_prefix_prompt((embeddings, pad, attention, extra), torch.randn(2, 2, 7))
    assert output[0].shape == (2, 5, 7)
    assert output[1].shape == (2, 5)
    assert output[2].shape == (5,)
    assert output[3] is extra
