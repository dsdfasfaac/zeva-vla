import copy
import unittest

from scripts.select_robotwin_behavior_effect import select


class GateTests(unittest.TestCase):
    def setUp(self):
        self.tasks = [f"task{i}" for i in range(10)]
        self.plan = {"stage2":{"selection":"step5000"}, "gates":{"validation_h15_mse_relative_gain":.03,
                                                                 "validation_nonworse_tasks":8}}
        self.plan["comparison_base_model_sha256"] = "frozen-base-hash"
        self.rows = [{"sample_id":f"{t}:{i}", "task":t, "shuffled_sample_id":f"{t}:{1-i}",
                      "valid_action_steps":15, "base":1., "aligned":.9, "within_task_shuffled":.95, "effect_off":.92}
                     for t in self.tasks for i in range(2)]
        self.ids = [r["sample_id"] for r in self.rows]
        self.report = {"schema":"zeva-behavior-effect-validation5-v1", "split":"validation",
                       "formal_labels_used":False, "checkpoint_step":5000, "checkpoint":"/example/005000",
                       "output_horizon":50, "execution_horizon":15,
                       "metric":"sample_mean_normalized_executed_h15_action_mse", "matched_noise":True,
                       "rows":self.rows}
        self.report["baseline_model_sha256"] = "frozen-base-hash"

    def test_pass(self):
        self.assertTrue(select(self.report,self.ids,self.plan,self.tasks)["passed"])

    def test_coverage_and_leakage_fail_closed(self):
        for mutate in (lambda x:x["rows"].pop(), lambda x:x.update(formal_labels_used=True),
                       lambda x:x.update(split="formal"), lambda x:x.update(checkpoint_step=2500),
                       lambda x:x.update(baseline_model_sha256="weakened-base"),
                       lambda x:x["rows"][0].update(shuffled_sample_id="task1:0"),
                       lambda x:x["rows"][0].update(base=float("nan"))):
            report = copy.deepcopy(self.report)
            mutate(report)
            with self.assertRaises(ValueError):
                select(report,self.ids,self.plan,self.tasks)

    def test_alignment_or_effect_regression_blocks_promotion(self):
        for field in ("within_task_shuffled", "effect_off"):
            report = copy.deepcopy(self.report)
            for row in report["rows"]:
                row[field] = .8
            result = select(report,self.ids,self.plan,self.tasks)
            self.assertFalse(result["passed"])
            self.assertIsNone(result["selected_checkpoint"])


if __name__ == "__main__":
    unittest.main()
