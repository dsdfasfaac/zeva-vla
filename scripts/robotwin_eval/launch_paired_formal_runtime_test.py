"""Exercise the runtime setup branch without SSH or filesystem mutations."""

from pathlib import Path
import subprocess
import unittest


SCRIPT = Path(__file__).with_name("launch_paired_formal_eval.sh")


class RuntimeSetupTest(unittest.TestCase):
    def run_setup(self, readonly):
        source = SCRIPT.read_text()
        block = source.split('if [[ "$read_only_runtime" == true ]]; then', 1)[1]
        block = 'if [[ "$read_only_runtime" == true ]]; then' + block.split(
            "# Put the experiment's adapter", 1
        )[0]
        result = subprocess.run(
            ["bash", "-c", "\n".join([
                "set -euo pipefail",
                "ssh() { printf '%s\\n' \"$*\"; }",
                "model_host=test-host",
                "native_transformers=/verified/runtime",
                "release_runtime=/handoff/runtime",
                f"read_only_runtime={readonly}",
                block,
            ])],
            check=True, text=True, capture_output=True,
        )
        return result.stdout

    def test_readonly_checks_without_mutation(self):
        output = self.run_setup("true")
        self.assertIn("test -d '/verified/runtime/transformers'", output)
        self.assertIn("transformers-5.5.4.dist-info", output)
        self.assertNotIn("ln -sfn", output)
        self.assertNotIn("mkdir", output)

    def test_legacy_default_setup_preserved(self):
        output = self.run_setup("false")
        self.assertIn("mkdir -p '/verified/runtime'", output)
        self.assertIn("ln -sfn", output)

    def test_shell_syntax(self):
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)

    def test_extra_dependencies_path_validation(self):
        source = SCRIPT.read_text()
        block = source[source.index("model_extra_dependencies="):source.index("model_ld_library_path=")]
        for path, passes in (("/verified/mamba", True), ("relative/path", False), ("/bad'path", False)):
            result = subprocess.run(
                ["bash", "-c", "set -eu\n" + block],
                env={"MODEL_EXTRA_DEPENDENCIES": path}, capture_output=True,
            )
            self.assertEqual(result.returncode == 0, passes)

    def test_optional_overlay_preserves_source_and_native_precedence(self):
        source = SCRIPT.read_text()
        setup = source.split("model_pythonpath=$zeva_root/scripts/robotwin_eval:", 1)[1]
        block = 'if [[ -n "$model_dependency_overlay" ]]; then' + setup.split(
            'if [[ -n "$model_dependency_overlay" ]]; then', 1
        )[1].split("\nfi", 1)[0] + "\nfi"
        for overlay, expected in (
            ("", "/legacy"),
            ("/verified", "/release/scripts/robotwin_eval:/native:/verified:/legacy"),
        ):
            result = subprocess.run(
                ["bash", "-c", "\n".join([
                    "set -euo pipefail", "ssh() { return 0; }",
                    "model_host=test", "zeva_root=/release", "native_transformers=/native",
                    "model_pythonpath=/legacy", f"model_dependency_overlay={overlay}",
                    block, 'printf "%s" "$model_pythonpath"',
                ])], text=True, capture_output=True, check=True,
            )
            self.assertEqual(result.stdout, expected)


if __name__ == "__main__":
    unittest.main()
