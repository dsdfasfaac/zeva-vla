import pytest

torch = pytest.importorskip("torch")

from openpi.zeva.memory import CausalMemoryManager  # noqa: E402


def test_attempt_reset_preserves_persistent_memory():
    memory = CausalMemoryManager(brief_size=2, merge_threshold=0.99)
    memory.update(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]))
    memory.reset_attempt()

    assert memory.snapshot() == {
        "brief_size": 0,
        "persistent_size": 1,
        "consolidated_observations": 1,
    }


def test_episode_reset_clears_both_timescales():
    memory = CausalMemoryManager()
    memory.update(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]))
    memory.reset_episode()

    assert memory.snapshot() == {
        "brief_size": 0,
        "persistent_size": 0,
        "consolidated_observations": 0,
    }


def test_similar_interactions_merge_and_retrieve():
    memory = CausalMemoryManager(merge_threshold=0.8, retrieval_top_k=1)
    memory.update(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0]))
    merged = memory.update(torch.tensor([0.99, 0.01]), torch.tensor([0.98, 0.02]))
    memory.update(torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0]))

    retrieved = memory.retrieve(torch.tensor([[1.0, 0.0]]), device="cpu")
    assert merged
    assert memory.snapshot()["persistent_size"] == 2
    assert torch.cosine_similarity(retrieved[0, 0], torch.tensor([1.0, 0.0]), dim=0) > 0.99
