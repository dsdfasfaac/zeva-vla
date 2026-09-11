import copy
import unittest

from scripts.audit_robotwin_zte_loss_control_pair import audit


class PairAuditTest(unittest.TestCase):
    def pair(self):
        first = {
            "schema": "zeva-robotwin-zte-stage1-v2",
            "source_files": {"trainer": "same", "encoder": "same"},
            "statistics_sha256": "same-stats",
            "goal_embeddings_sha256": "same-goals",
            "zte_config": {"action_prediction_context": "phase"},
            "contract": {"executed_horizon": 15, "policy_horizon": 50},
            "grouping": {"task_names": ["task"]},
            "train_args": {
                "steps": 4096, "batch_size": 8, "action_prediction_context": "phase",
                "task_paired_batches": True, "prediction_loss_reduction": "vector_mse",
                "save_dir": "/vector", "learning_rate": 1e-4,
            },
        }
        second = copy.deepcopy(first)
        second["train_args"].update(save_dir="/huber", prediction_loss_reduction="mean_coordinate_huber")
        return first, second

    def test_only_declared_loss_and_output_differences_pass(self):
        report = audit(*self.pair())
        self.assertTrue(report["passed"])
        self.assertFalse(report["stage1_gate_passed"])

    def test_lr_source_stats_and_missing_field_fail_closed(self):
        for field in ("lr", "source", "stats", "missing"):
            first, second = self.pair()
            if field == "lr": second["train_args"]["learning_rate"] = 1e-3
            if field == "source": second["source_files"]["trainer"] = "different"
            if field == "stats": second["statistics_sha256"] = "different"
            if field == "missing": del second["train_args"]["learning_rate"]
            self.assertFalse(audit(first, second)["passed"], field)

    def test_same_loss_or_shared_output_or_short_budget_fails(self):
        for key, value in (("prediction_loss_reduction", "vector_mse"), ("save_dir", "/vector"), ("steps", 256)):
            first, second = self.pair()
            second["train_args"][key] = value
            self.assertFalse(audit(first, second)["passed"])

    def test_both_missing_required_lineage_still_fails(self):
        first, second = self.pair()
        del first["goal_embeddings_sha256"]
        del second["goal_embeddings_sha256"]
        report = audit(first, second)
        self.assertFalse(report["passed"])
        self.assertEqual(len(report["missing_required_fields"]), 2)


if __name__ == "__main__":
    unittest.main()
