"""The training waiter must not identify its own probe as a training worker."""
import pathlib
import subprocess
import unittest


class ProcessDetectionTest(unittest.TestCase):
    def test_only_actual_python_training_commands_match(self):
        script = pathlib.Path(__file__).with_name("wait_and_launch_robotwin_ztev2_formal.sh").read_text()
        function = script.split("training_children_live() {", 1)[1].split("\n}", 1)[0]
        shell = """
train_root=/example/pair
ps() {
  printf '%s\\n' \\
    '101 python3 python3 train_robotwin_stage2.py --save-dir /example/pair/zeva' \\
    '102 awk awk -v root=/example/pair index(train_robotwin_stage2.py)' \\
    '103 bash bash -c ps /example/pair train_robotwin_stage2.py' \\
    '104 python3 python3 unrelated.py --save-dir /example/pair' \\
    '105 python3 python3 train_robotwin_stage2.py --save-dir /other/pair'
}
""" + function
        result = subprocess.run(["bash", "-c", shell], check=True, text=True, capture_output=True)
        self.assertEqual(result.stdout.strip(), "101")


if __name__ == "__main__":
    unittest.main()
