"""One-GPU contract and zero-initialization smoke for RoboTwin Zeva."""

from __future__ import annotations

import dataclasses
import json

import torch
import tyro

from openpi.zeva.robotwin_contract import ROBOTWIN_CAMERA_KEYS
from openpi.zeva.robotwin_contract import prepare_robotwin_pi_image
from openpi.zeva.robotwin_policy import RobotWinZevaPolicy


@dataclasses.dataclass
class Args:
    handoff_root: str = (
        "/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1"
    )
    device: str = "cuda:0"
    zte_checkpoint: str = (
        "/data1/dingxin/zeva-runs/robotwin-v5-h15-tasklang/stage1-zte/zte_best.pth"
    )
    seed: int = 1000


def main(args: Args) -> None:
    policy = RobotWinZevaPolicy.from_handoff(
        args.handoff_root,
        device=args.device,
        zte_checkpoint=args.zte_checkpoint,
    )
    raw_observation = {
        "observation.state": torch.zeros(14, dtype=torch.float32),
        "task": "pick up the object",
        **{
            key: prepare_robotwin_pi_image(
                torch.zeros(3, 480, 640, dtype=torch.uint8), name=key
            )
            for key in ROBOTWIN_CAMERA_KEYS
        },
    }
    processed = policy.preprocessor(raw_observation)
    processed["zeva.goal_embedding"] = policy._task_only_goal_embedding(  # noqa: SLF001
        raw_observation["task"], processed["observation.language.tokens"].device
    )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    with torch.no_grad():
        baseline = policy.foundation.predict_action_chunk(processed)

    policy.reset(scope="episode")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    zeva_zero_init = policy.predict_action_chunk(processed)
    if not torch.equal(baseline, zeva_zero_init):
        difference = float((baseline - zeva_zero_init).abs().max())
        raise AssertionError(f"Zero-init Zeva changed the selected PI0.5 output (max_abs={difference}).")

    raw_actions = policy.postprocessor(zeva_zero_init)
    second_chunk = policy.predict_action_chunk(processed, executed_actions=raw_actions[:, :15])
    if second_chunk.shape != (1, 50, 16):
        raise AssertionError(f"Online Mamba step returned {tuple(second_chunk.shape)}.")
    report = {
        "schema": "zeva-robotwin-pi05-smoke-v1",
        "passed": True,
        "foundation_output_shape": list(baseline.shape),
        "raw_output_shape": list(raw_actions.shape),
        "online_mamba_step_shape": list(second_chunk.shape),
        "zero_init_bit_exact": True,
        "camera_count": 3,
        "state_dim": 14,
        "action_dim": 16,
        "action_horizon": 50,
        "executed_action_horizon": 15,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(tyro.cli(Args))
