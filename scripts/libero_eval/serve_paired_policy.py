#!/usr/bin/env python3
"""Serve the exact selected LIBERO PI0.5 or a ZeVA stage for paired rollout."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.zeva.libero_contract import LIBERO_MODEL_ACTION_DIM
from openpi.zeva.libero_contract import LIBERO_POLICY_HORIZON
from openpi.zeva.libero_policy import LiberoZevaPolicy


class PairedLiberoPolicy:
    def __init__(self, args: argparse.Namespace):
        use_memory = args.mode != "baseline"
        self.mode = args.mode
        self.device = torch.device(args.device)
        self.policy = LiberoZevaPolicy.from_handoff(
            args.handoff_root,
            tokenizer_path=args.tokenizer_path,
            device=args.device,
            zte_checkpoint=args.zte_checkpoint if use_memory else None,
            adapter_checkpoint=(
                str(Path(args.stage2_checkpoint) / "zeva_adapter.pth") if use_memory else None
            ),
            stage3_action_checkpoint=(
                str(Path(args.stage3_checkpoint) / "stage3_action.pth")
                if args.mode == "stage3"
                else None
            ),
            retrieval_checkpoint=args.retrieval_checkpoint if use_memory else None,
            causal_bank=args.causal_bank if use_memory else None,
        )
        # The training class wraps sampling in max-autotune compilation at
        # construction time. Deployment uses eager sampling: it is numerically
        # identical, starts immediately, and avoids requiring a host CUDA linker.
        compiled_sampler = self.policy.foundation.sample_actions
        eager_sampler = getattr(compiled_sampler, "_torchdynamo_orig_callable", None)
        if eager_sampler is not None:
            self.policy.foundation.sample_actions = eager_sampler
        self.use_memory = use_memory

    def reset(self, scope: str = "episode") -> None:
        self.policy.reset(scope=scope)

    def infer(self, raw: dict) -> dict:
        noise_seed = int(raw.pop("noise_seed"))
        executed_actions = raw.pop("executed_actions", None)
        batch = {
            "observation.image": torch.as_tensor(raw["observation/image"]).unsqueeze(0),
            "observation.wrist_image": torch.as_tensor(
                raw["observation/wrist_image"]
            ).unsqueeze(0),
            "observation.state": torch.as_tensor(
                raw["observation/state"], dtype=torch.float32
            ).unsqueeze(0),
            "prompt": [str(raw["prompt"])],
        }
        observation = self.policy.processor.preprocess_observation(batch, device=self.device)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(noise_seed)
        noise = torch.randn(
            (1, LIBERO_POLICY_HORIZON, LIBERO_MODEL_ACTION_DIM),
            generator=generator,
            dtype=torch.float32,
            device=self.device,
        )
        chunk = self.policy.sample_action_chunk(
            observation,
            use_memory=self.use_memory,
            noise=noise,
            executed_actions=executed_actions,
        )[0]
        result = {"actions": chunk.float().cpu().numpy().astype(np.float32)}
        if self.use_memory:
            result["retrieval"] = self.policy.retrieval_diagnostics()
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("baseline", "stage2", "stage3"), required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--handoff-root", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--zte-checkpoint")
    parser.add_argument("--stage2-checkpoint")
    parser.add_argument("--stage3-checkpoint")
    parser.add_argument("--retrieval-checkpoint")
    parser.add_argument("--causal-bank")
    args = parser.parse_args()
    required = (
        "zte_checkpoint",
        "stage2_checkpoint",
        "retrieval_checkpoint",
        "causal_bank",
    )
    if args.mode != "baseline" and any(getattr(args, name) is None for name in required):
        parser.error("Stage2/3 requires ZTE, adapter, retrieval, and causal-bank paths.")
    if args.mode == "stage3" and args.stage3_checkpoint is None:
        parser.error("Stage3 requires --stage3-checkpoint.")
    return args


def main() -> None:
    args = parse_args()
    policy = PairedLiberoPolicy(args)
    logging.info("LIBERO paired policy ready: mode=%s port=%d", args.mode, args.port)
    WebsocketPolicyServer(
        policy,
        host="0.0.0.0",
        port=args.port,
        metadata={"mode": args.mode, "action_contract": "eef16-h10-execute5"},
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
