import unittest
from scripts.behavior_effect_validation_contract import (
    check_cache_coverage, decision_noise_seed, enumerate_decisions, within_task_permutation,
)


class FakeTensor:
    def __init__(self, values=None, shape=None):
        self.values, self.shape = values, shape

    def tolist(self):
        return self.values


class ContractTests(unittest.TestCase):
    def test_independent_grid_boundaries(self):
        records = [{"key": ("Clean", "task"), "episode_index": i, "length": length}
                   for i, length in enumerate((15, 16, 30, 31, 45, 46))]
        rows = enumerate_decisions(records, ["task"])
        counts = {i: sum(r["record_index"] == i for r in rows) for i in range(6)}
        self.assertEqual(counts, {0: 0, 1: 2, 2: 2, 3: 3, 4: 3, 5: 4})
        self.assertTrue(all(r["frame"] < records[r["record_index"]]["length"] for r in rows))
        with self.assertRaises(ValueError):
            enumerate_decisions(records + records, ["task"])

    def test_fixed_permutation_and_noise(self):
        rows = [{"task": task, "sample_id": f"{task}:{i}"} for task in ("a", "b") for i in range(7)]
        perm = within_task_permutation(rows)
        self.assertEqual(perm, within_task_permutation(list(reversed(rows))))
        self.assertEqual(set(perm), set(perm.values()))
        self.assertTrue(all(a != b and a.split(":")[0] == b.split(":")[0] for a, b in perm.items()))
        self.assertEqual(decision_noise_seed("a:0"), decision_noise_seed("a:0"))
        self.assertNotEqual(decision_noise_seed("a:0"), decision_noise_seed("b:0"))

    def test_partial_cache_cannot_redefine_expected_coverage(self):
        rows = enumerate_decisions([{"key": ("Clean", "a"), "episode_index": 0, "length": 31}], ["a"])
        entry = {"frames": FakeTensor([0, 15, 30]), "phase": FakeTensor(shape=(3, 256)),
                 "effect": FakeTensor(shape=(3, 256))}
        check_cache_coverage(rows, {"Clean:a:0": entry})
        for cache in ({}, {"Clean:a:0": {**entry, "frames": FakeTensor([0, 15])}},
                      {"Clean:a:0": {**entry, "effect": FakeTensor(shape=(3, 128))}}):
            with self.assertRaises(ValueError):
                check_cache_coverage(rows, cache)


if __name__ == "__main__":
    unittest.main()
