"""Verify real FFmpeg frames have identical Stage 2 train/eval PI inputs."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import torch
import tyro

from openpi.zeva.robotwin_contract import ROBOTWIN_CAMERA_KEYS
from openpi.zeva.robotwin_contract import prepare_robotwin_pi_image

try:
    from scripts.train_robotwin_zte import FFmpegRoboTwinDataset
except ModuleNotFoundError:  # Direct `python scripts/...py` execution.
    from train_robotwin_zte import FFmpegRoboTwinDataset


@dataclasses.dataclass
class Args:
    dataset_root: str = "/data1/huangbingjia/robotwin-lerobot-sidney-eef16-v1/data"
    subset: str = "train"
    record_index: int = 0
    frame: int = 0


def main(args: Args) -> None:
    source = FFmpegRoboTwinDataset(Path(args.dataset_root) / "adapter.json", subset=args.subset)
    records = source.dataset._records  # noqa: SLF001
    if not 0 <= args.record_index < len(records):
        raise IndexError(f"record_index {args.record_index} is outside [0,{len(records)}).")
    record = records[args.record_index]
    if not 0 <= args.frame < int(record["length"]):
        raise IndexError(f"frame {args.frame} is outside record length {record['length']}.")
    decoded = source.read_images(record, [args.frame])
    cameras = {}
    for key in ROBOTWIN_CAMERA_KEYS:
        ffmpeg_chw = decoded[key][0]
        if ffmpeg_chw.dtype != torch.uint8:
            raise AssertionError(f"Expected FFmpeg uint8 input for {key}, got {ffmpeg_chw.dtype}.")
        train_input = prepare_robotwin_pi_image(ffmpeg_chw, name=f"train:{key}")
        simulated_eval_hwc = ffmpeg_chw.permute(1, 2, 0).cpu().numpy()
        eval_input = prepare_robotwin_pi_image(simulated_eval_hwc, name=f"eval:{key}")
        torch.testing.assert_close(train_input, eval_input, rtol=0, atol=0)
        cameras[key] = {
            "decoded_dtype": str(ffmpeg_chw.dtype),
            "decoded_range": [int(ffmpeg_chw.min()), int(ffmpeg_chw.max())],
            "pi_dtype": str(train_input.dtype),
            "pi_shape": list(train_input.shape),
            "pi_range": [float(train_input.min()), float(train_input.max())],
            "train_eval_max_abs": float((train_input - eval_input).abs().max()),
        }
    print(
        json.dumps(
            {
                "schema": "zeva-robotwin-stage2-image-contract-v1",
                "passed": True,
                "subset": args.subset,
                "record_index": args.record_index,
                "frame": args.frame,
                "episode_index": int(record["episode_index"]),
                "cameras": cameras,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
