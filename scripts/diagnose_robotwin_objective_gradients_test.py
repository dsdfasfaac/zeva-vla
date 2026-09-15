"""CPU/stdlib tests for gradient summaries, independent of the ML runtime."""
import ast
from pathlib import Path
import unittest


source = Path(__file__).with_name("diagnose_robotwin_objective_gradients.py").read_text()
node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "gradient_stats")
scope = {}
exec(compile(ast.Module(body=[node], type_ignores=[]), "gradient_stats", "exec"), scope)
stats = scope["gradient_stats"]


class Vector:
    def __init__(self, values):
        self.values = values

    def detach(self):
        return self

    def double(self):
        return self

    def square(self):
        return Vector([x*x for x in self.values])

    def sum(self):
        return sum(self.values)

    def __mul__(self, other):
        if len(self.values) != len(other.values):
            raise ValueError("Vector lengths differ")
        return Vector([x*y for x, y in zip(self.values, other.values)])


class GradientStatsTest(unittest.TestCase):
    def test_opposed_gradients(self):
        result = stats([Vector([3, 4])], [Vector([-6, -8])])
        self.assertEqual(result["flow_gradient_norm"], 5)
        self.assertEqual(result["nll_to_flow_norm_ratio"], 2)
        self.assertEqual(result["cosine"], -1)

    def test_orthogonal_gradients(self):
        self.assertEqual(stats([Vector([1, 0])], [Vector([0, 1])])["cosine"], 0)

    def test_unused_and_zero_are_not_assumed_aligned(self):
        result = stats([None], [Vector([1])])
        self.assertIsNone(result["cosine"])
        self.assertIsNone(result["nll_to_flow_norm_ratio"])
        self.assertIsNone(stats([Vector([0])], [None])["cosine"])

    def test_no_optimizer_or_checkpoint_writer(self):
        self.assertNotIn("torch.optim", source)
        self.assertNotIn("torch.save", source)
        self.assertIn('subset="train"', source)
        self.assertIn('seed, batch_size, batches = 20260915, 8, 4', source)


if __name__ == "__main__":
    unittest.main()
