import json
from pathlib import Path
import tempfile
import unittest

from scripts.audit_robotwin_dataset_identity import audit, component
from scripts.resume_robotwin_stage2_verified import cli_args


class DatasetIdentityTest(unittest.TestCase):
    def test_relocation_preserves_content_but_not_adapter_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            reports = []
            for name in ("old", "new"):
                root = Path(temporary) / name
                data = root / "data"
                data.mkdir(parents=True)
                adapter = {"action_horizon": 50}
                for key, folder in (("dataset_root", "source"), ("eef_cache_root", "eef-index"),
                                    ("joint_cache_root", "joint14-index")):
                    path = root / folder
                    path.mkdir()
                    (path / "sample").write_bytes(b"identical")
                    adapter[key] = str(path)
                stats = root / "stats.json"
                stats.write_text("{}")
                adapter["stats_path"] = str(stats)
                (data / "adapter.json").write_text(json.dumps(adapter))
                reports.append(audit(data))
            self.assertEqual(reports[0]["components"], reports[1]["components"])
            self.assertEqual(reports[0]["semantic_adapter"], reports[1]["semantic_adapter"])
            self.assertNotEqual(reports[0]["adapter_sha256"], reports[1]["adapter_sha256"])
            (Path(temporary) / "new" / "source" / "sample").write_bytes(b"different")
            self.assertNotEqual(reports[0]["components"]["source"], component(Path(temporary) / "new" / "source"))

    def test_cli_flags_do_not_shell_interpolate(self):
        self.assertEqual(cli_args({"resume_checkpoint": "/path with spaces", "compile_model": True,
                                   "save_checkpoints": False, "initial_stage2_checkpoint": None}),
                         ["--resume-checkpoint", "/path with spaces", "--compile-model", "--no-save-checkpoints"])


if __name__ == "__main__":
    unittest.main()
