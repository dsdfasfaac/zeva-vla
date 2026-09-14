"""CPU-only tests for placement and residual GPU allocation safeguards."""
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).with_name("launch_fixed_anchor_pair_20260914.sh")


class PlacementTest(unittest.TestCase):
    def branch(self, host):
        source = SCRIPT.read_text()
        block = source[source.index("zeva_placement="):source.index('[[ $(hostname -s)')]
        return subprocess.run(
            ["bash", "-c", "set -eu\nzeva_release=/release\n" + block
             + '\nprintf "%s %s %s %s" "$zeva_model_ip" "$zeva_uuid6" "$zeva_memory_limit" "$zeva_icd"'],
            env={**os.environ, "EVAL_PLACEMENT": host, "MODEL_DEPENDENCY_OVERLAY": "/verified", "MODEL_LD_LIBRARY_PATH": "/verified/torch/lib"},
            text=True, capture_output=True,
        )

    def test_a24_placement(self):
        result = self.branch("aigc24")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("172.16.80.158 GPU-2e145d56-", result.stdout)
        self.assertIn("1024 /usr/share/vulkan/icd.d/nvidia_icd.json", result.stdout)

    def test_a31_placement(self):
        result = self.branch("aigc31")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("172.16.80.165 GPU-18077766-", result.stdout)
        self.assertIn("6144 /etc/vulkan/icd.d/nvidia_icd.json", result.stdout)

    def test_unknown_host_rejected(self):
        self.assertNotEqual(self.branch("unexpected").returncode, 0)

    def test_a28_placement_requires_no_residual_allocation(self):
        result = self.branch("aigc28")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("172.16.80.162 GPU-3ccf761f-", result.stdout)
        self.assertIn("1024 /etc/vulkan/icd.d/nvidia_icd.json", result.stdout)

    def processes(self, pid, name, *, allowed="", exists=False):
        source = SCRIPT.read_text()
        code = source.split("-q -x | python3 -c '\n", 1)[1].split(
            "' \"$zeva_allowed_stale_pid\")", 1
        )[0]
        xml = ("<nvidia_smi_log><gpu><processes><process_info>"
               f"<pid>{pid}</pid><process_name>{name}</process_name>"
               "</process_info></processes></gpu></nvidia_smi_log>")
        output = io.StringIO()
        with patch.object(sys, "stdin", io.StringIO(xml)), \
             patch.object(sys, "argv", ["-c", allowed]), \
             patch.object(Path, "exists", return_value=exists), \
             contextlib.redirect_stdout(output):
            exec(compile(code, str(SCRIPT), "exec"), {})
        return output.getvalue()

    def test_xorg_allowed_but_graphics_python_blocked(self):
        self.assertEqual(self.processes("123", "/usr/lib/xorg/Xorg"), "")
        self.assertIn("123 python", self.processes("123", "python"))

    def test_only_verified_absent_residual_pid_allowed(self):
        self.assertEqual(self.processes("2369486", "", allowed="2369486"), "")
        self.assertIn("2369486", self.processes("2369486", "", allowed="2369486", exists=True))
        self.assertIn("999", self.processes("999", "", allowed="2369486"))

    def test_shell_syntax(self):
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


if __name__ == "__main__":
    unittest.main()
