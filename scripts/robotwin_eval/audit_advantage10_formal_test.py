"""Regression tests for JSON-formatted .yml staging configs."""
import json
from pathlib import Path
import tempfile
import unittest

from audit_advantage10_formal import EXPECTED_BESTV1, load_flat_yaml


class PolicyConfigParsingTest(unittest.TestCase):
    def parse(self, content):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "policy.yml"
            path.write_text(content, encoding="utf-8")
            return load_flat_yaml(path)

    def test_json_keys_and_boolean_are_normalized(self):
        parsed = self.parse(json.dumps({
            "foundation_checkpoint": str(EXPECTED_BESTV1),
            "baseline_only": True, "stage2_checkpoint": "/trained/001000",
        }, indent=2))
        self.assertEqual(parsed["foundation_checkpoint"], str(EXPECTED_BESTV1))
        self.assertEqual(parsed["baseline_only"], "true")
        self.assertEqual(parsed["stage2_checkpoint"], "/trained/001000")

    def test_false_and_wrong_foundation_not_accepted_as_true_or_bestv1(self):
        parsed = self.parse('{"baseline_only": false, "foundation_checkpoint": "/wrong"}')
        self.assertNotEqual(parsed["baseline_only"], "true")
        self.assertNotEqual(parsed["foundation_checkpoint"], str(EXPECTED_BESTV1))

    def test_legacy_flat_yaml_is_preserved(self):
        parsed = self.parse("foundation_checkpoint: '/legacy/model' # comment\nbaseline_only: true\n")
        self.assertEqual(parsed, {"foundation_checkpoint": "/legacy/model", "baseline_only": "true"})

    def test_json_strings_are_not_mangled(self):
        self.assertEqual(self.parse('{"path": "/path/with#hash:colon"}')["path"], "/path/with#hash:colon")

    def test_json_non_mapping_rejected(self):
        with self.assertRaises(ValueError):
            self.parse('[]')


if __name__ == "__main__":
    unittest.main()
