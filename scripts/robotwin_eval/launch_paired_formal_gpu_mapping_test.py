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
    def run_mapping(
        self,
        *,
        slots="8",
        model=None,
        render=None,
        model_cuda=None,
        render_cuda=None,
    ):
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
        if model_cuda is None:
            environment.pop("MODEL_CUDA_DEVICES", None)
        else:
            environment["MODEL_CUDA_DEVICES"] = model_cuda
        if render_cuda is None:
            environment.pop("RENDER_CUDA_DEVICES", None)
        else:
            environment["RENDER_CUDA_DEVICES"] = render_cuda
        command = "\n".join(
            [
                "set -euo pipefail",
                'slots="${SLOTS}"',
                mapping_block(),
                'printf "model=%s\\nrender=%s\\nmodel_cuda=%s\\nrender_cuda=%s\\n" "$model_gpu_ids_json" "$render_gpu_ids_json" "$model_cuda_devices_json" "$render_cuda_devices_json"',
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
        self.assertEqual(
            result.stdout,
            'model=0,1,2\nrender=0,1,2\nmodel_cuda=["0","1","2"]\nrender_cuda=["0","1","2"]\n',
        )

    def test_explicit_mapping_allows_repeated_physical_ids(self):
        result = self.run_mapping(
            model="2,6,2,6,2,6,2,6",
            render="6,2,6,2,6,2,6,2",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout,
            'model=2,6,2,6,2,6,2,6\nrender=6,2,6,2,6,2,6,2\nmodel_cuda=["2","6","2","6","2","6","2","6"]\nrender_cuda=["6","2","6","2","6","2","6","2"]\n',
        )

    def test_uuid_mapping_uses_uuid_arrays_and_preserves_numeric_mapping(self):
        model_cuda = "GPU-01234567-89ab-cdef-0123-456789abcdef,GPU-fedcba98-7654-3210-fedc-ba9876543210"
        render_cuda = "GPU-fedcba98-7654-3210-fedc-ba9876543210,GPU-01234567-89ab-cdef-0123-456789abcdef"
        result = self.run_mapping(
            slots="2",
            model="6,2",
            render="2,6",
            model_cuda=model_cuda,
            render_cuda=render_cuda,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout,
            'model=6,2\nrender=2,6\nmodel_cuda=["GPU-01234567-89ab-cdef-0123-456789abcdef","GPU-fedcba98-7654-3210-fedc-ba9876543210"]\nrender_cuda=["GPU-fedcba98-7654-3210-fedc-ba9876543210","GPU-01234567-89ab-cdef-0123-456789abcdef"]\n',
        )

    def test_mapping_length_and_token_are_checked(self):
        bad_length = self.run_mapping(slots="3", model="2,6")
        self.assertNotEqual(bad_length.returncode, 0)
        self.assertIn("MODEL_GPU_IDS", bad_length.stderr)

        bad_token = self.run_mapping(slots="2", render="2,-1")
        self.assertNotEqual(bad_token.returncode, 0)
        self.assertIn("RENDER_GPU_IDS", bad_token.stderr)

        bad_uuid_length = self.run_mapping(
            slots="2", model_cuda="GPU-01234567-89ab-cdef-0123-456789abcdef"
        )
        self.assertNotEqual(bad_uuid_length.returncode, 0)
        self.assertIn("MODEL_CUDA_DEVICES", bad_uuid_length.stderr)

        bad_uuid_token = self.run_mapping(
            slots="2",
            render_cuda="GPU-01234567-89ab-cdef-0123-456789abcde,GPU-01234567-89ab-cdef-0123-456789abcdef",
        )
        self.assertNotEqual(bad_uuid_token.returncode, 0)
        self.assertIn("RENDER_CUDA_DEVICES", bad_uuid_token.stderr)

        numeric_uuid_token = self.run_mapping(slots="1", model_cuda="6")
        self.assertNotEqual(numeric_uuid_token.returncode, 0)
        self.assertIn("MODEL_CUDA_DEVICES", numeric_uuid_token.stderr)

    def test_shell_syntax_and_use_sites_are_mapped(self):
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('CUDA_VISIBLE_DEVICES=\'$model_cuda_device\'', source)
        self.assertIn('CUDA_VISIBLE_DEVICES=\'$render_cuda_device\'', source)
        self.assertIn('"model_gpu_ids": [$model_gpu_ids_json]', source)
        self.assertIn('"render_gpu_ids": [$render_gpu_ids_json]', source)
        self.assertIn('"model_cuda_devices": $model_cuda_devices_json', source)
        self.assertIn('"render_cuda_devices": $render_cuda_devices_json', source)


if __name__ == "__main__":
    unittest.main()
