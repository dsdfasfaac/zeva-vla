"""Contract tests for per-slot model/renderer GPU placement.

The full launcher needs real checkpoints, RoboTwin, SSH, and output storage.
These tests execute only its self-contained mapping block with mocked shell
inputs, so they never start a server or mutate an evaluation directory.
"""

from pathlib import Path
import os
import subprocess
import unittest


SCRIPT = Path(__file__).with_name("launch_paired_formal_eval.sh")
START_MARKER = "# Each logical slot can be placed on an independently chosen physical GPU."
END_MARKER = "episodes=${EPISODES:-20}"


def mapping_block():
    source = SCRIPT.read_text(encoding="utf-8")
    start = source.index(START_MARKER)
    end = source.index(END_MARKER, start)
    return source[start:end]


class GpuMappingContractTest(unittest.TestCase):
    def run_mapping(self, *, slots="8", model=None, render=None):
        environment = os.environ.copy()
        environment["SLOTS"] = slots
        if model is None:
            environment.pop("MODEL_GPU_IDS", None)
        else:
            environment["MODEL_GPU_IDS"] = model
        if render is None:
            environment.pop("RENDER_GPU_IDS", None)
        else:
            environment["RENDER_GPU_IDS"] = render
        command = "\n".join(
            [
                "set -euo pipefail",
                'slots="${SLOTS}"',
                mapping_block(),
                'printf "model=%s\\nrender=%s\\n" "$model_gpu_ids_json" "$render_gpu_ids_json"',
            ]
        )
        return subprocess.run(
            ["bash", "-c", command],
            env=environment,
            text=True,
            capture_output=True,
        )

    def test_default_mapping_is_one_gpu_per_slot(self):
        result = self.run_mapping(slots="3")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "model=0,1,2\nrender=0,1,2\n")

    def test_explicit_mapping_allows_repeated_physical_ids(self):
        result = self.run_mapping(
            model="2,6,2,6,2,6,2,6",
            render="6,2,6,2,6,2,6,2",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "model=2,6,2,6,2,6,2,6\nrender=6,2,6,2,6,2,6,2\n")

    def test_mapping_length_and_token_are_checked(self):
        bad_length = self.run_mapping(slots="3", model="2,6")
        self.assertNotEqual(bad_length.returncode, 0)
        self.assertIn("MODEL_GPU_IDS", bad_length.stderr)

        bad_token = self.run_mapping(slots="2", render="2,-1")
        self.assertNotEqual(bad_token.returncode, 0)
        self.assertIn("RENDER_GPU_IDS", bad_token.stderr)

    def test_shell_syntax_and_use_sites_are_mapped(self):
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('CUDA_VISIBLE_DEVICES=\'$model_gpu\'', source)
        self.assertIn('CUDA_VISIBLE_DEVICES=\'$render_gpu\'', source)
        self.assertIn('"model_gpu_ids": [$model_gpu_ids_json]', source)
        self.assertIn('"render_gpu_ids": [$render_gpu_ids_json]', source)


if __name__ == "__main__":
    unittest.main()
