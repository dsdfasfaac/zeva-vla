"""Synthetic serving smoke for a fixed evaluation config, not an accuracy eval."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from zeva_policy import get_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    config_bytes = Path(args.config).read_bytes()
    config = yaml.safe_load(config_bytes)
    started = time.monotonic()
    model = get_model(config)
    print("SELECTED_MODEL_LOADED", config.get("stage2_checkpoint"), flush=True)
    observation = {
        "observation": {
            camera: {"rgb": np.zeros((480, 640, 3), dtype=np.uint8)}
            for camera in ("head_camera", "left_camera", "right_camera")
        },
        "joint_action": {"vector": np.zeros(14, dtype=np.float32)},
        "task": "Pick up both bottles.",
    }
    model.reset_model()
    with torch.no_grad():
        first = model.predict(observation)["actions"]
        assert first.shape == (50, 16) and np.isfinite(first).all()
        model.commit_executed_actions(first[:15])
        second = model.predict(observation)["actions"]
        assert second.shape == (50, 16) and np.isfinite(second).all()
    state = model.policy._stage1_state
    transitions = None if state is None else int(state.transition_count)
    if not config.get("baseline_only", False):
        assert transitions == 1, transitions
    props = torch.cuda.get_device_properties(0)
    report = {
        "passed": True,
        "closed_loop_evaluation": False,
        "synthetic_observations": True,
        "trained_checkpoint_unchanged": True,
        "config": args.config,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "stage2_checkpoint": config.get("stage2_checkpoint"),
        "torch_version": torch.__version__,
        "torch_path": torch.__file__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_uuid": str(getattr(props, "uuid", "unavailable")),
        "first_shape": list(first.shape),
        "second_shape": list(second.shape),
        "committed_horizon": 15,
        "recurrent_transitions": transitions,
        "elapsed_seconds": time.monotonic() - started,
    }
    with output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
