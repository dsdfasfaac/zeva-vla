"""Precompute frozen task-only PI0.5 embeddings for RoboTwin B0."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open
import torch
from transformers import AutoTokenizer
import tyro

from openpi.zeva.robotwin_contract import RobotWinHandoff
try:
    from scripts.train_robotwin_zte import FFmpegRoboTwinDataset
except ModuleNotFoundError:  # Direct `python scripts/...py` execution.
    from train_robotwin_zte import FFmpegRoboTwinDataset


GOAL_EMBEDDING_SCHEMA = "zeva-robotwin-pi05-goal-embeddings-v2"
EMBEDDING_KEY = "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"


@dataclasses.dataclass
class Args:
    handoff_root: str = (
        "/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1"
    )
    dataset_root: str = "/data1/huangbingjia/robotwin-lerobot-sidney-eef16-v1/data"
    output_path: str = "/data1/dingxin/zeva-runs/robotwin-v5-h15-tasklang/pi05_task_embeddings.pt"
    batch_size: int = 64


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prompt(task: str) -> str:
    """The paper's g is the invariant task instruction, not observation state."""
    return task.strip().replace("_", " ").replace("\n", " ") + "\n"


def _record_id(record: dict) -> str:
    return f"{record['key'][0]}:{record['key'][1]}:{int(record['episode_index'])}"


def main(args: Args) -> None:
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    handoff = RobotWinHandoff.from_root(args.handoff_root)
    tokenizer_path = handoff.checkpoint / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    checkpoint_path = handoff.checkpoint / "model.safetensors"
    with safe_open(checkpoint_path, framework="pt", device="cpu") as handle:
        embedding_weight = handle.get_tensor(EMBEDDING_KEY)
    if tuple(embedding_weight.shape) != (257_152, 2048):
        raise ValueError(f"Unexpected PI0.5 language embedding shape: {tuple(embedding_weight.shape)}")

    splits = {}
    adapter_manifest = Path(args.dataset_root) / "adapter.json"
    for subset in ("train", "validation"):
        dataset = FFmpegRoboTwinDataset(adapter_manifest, subset).dataset
        prompts = []
        record_ids = []
        for record_index, record in enumerate(dataset._records):  # noqa: SLF001
            prompts.append(_prompt(record["key"][1]))
            record_ids.append(_record_id(record))

        pooled_batches = []
        for start in range(0, len(prompts), args.batch_size):
            encoded = tokenizer(
                prompts[start : start + args.batch_size],
                max_length=200,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            token_embeddings = embedding_weight[encoded["input_ids"]].float()
            mask = encoded["attention_mask"].float().unsqueeze(-1)
            pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            pooled_batches.append(pooled.to(torch.float16))
        embeddings = torch.cat(pooled_batches, dim=0)
        if embeddings.shape != (len(dataset._records), 2048):  # noqa: SLF001
            raise RuntimeError(f"Incomplete {subset} goal embedding table: {tuple(embeddings.shape)}")
        prompt_digest = hashlib.sha256("\n".join(prompts).encode("utf-8")).hexdigest()
        splits[subset] = {
            "record_ids": tuple(record_ids),
            "prompt_sha256": prompt_digest,
            "embeddings": embeddings,
        }

    payload = {
        "schema": GOAL_EMBEDDING_SCHEMA,
        "embedding_dim": 2048,
        "embedding_key": EMBEDDING_KEY,
        "handoff_root": str(handoff.root),
        "foundation_checkpoint_sha256": _sha256(checkpoint_path),
        "tokenizer_sha256": _sha256(tokenizer_path / "tokenizer.json"),
        "statistics_sha256": _sha256(handoff.statistics),
        "dataset_adapter": str(adapter_manifest.resolve()),
        "prompt_format": "pi05-task-only-plus-newline",
        "splits": splits,
    }
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    torch.save(payload, temporary)
    temporary.replace(output)
    summary = {
        "schema": payload["schema"],
        "embedding_dim": payload["embedding_dim"],
        "train_episodes": len(splits["train"]["record_ids"]),
        "validation_episodes": len(splits["validation"]["record_ids"]),
        "output": str(output.resolve()),
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(tyro.cli(Args))
