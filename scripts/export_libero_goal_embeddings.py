"""Export frozen task-only PI0.5 language embeddings for LIBERO B0."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

from safetensors import safe_open
import torch
from tokenizers import Tokenizer
import tyro

from openpi.zeva.libero_contract import LIBERO_CHECKPOINT_SHA256
from openpi.zeva.libero_contract import LiberoHandoff
from openpi.zeva.libero_contract import sha256
from openpi.zeva.libero_data import LiberoEpisodeTable


SCHEMA = "zeva-libero-pi05-goal-embeddings-v1"
EMBEDDING_KEY = "paligemma_with_expert.paligemma.lm_head.weight"


@dataclasses.dataclass
class Args:
    handoff_root: str = "/data1/dingxin/libero-memory-baseline-v1"
    dataset_root: str = "/data1/dingxin/libero-memory-baseline-v1/data"
    tokenizer_path: str = (
        "/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/"
        "robotwin-memory-baseline-v1/checkpoint/pretrained_model/tokenizer"
    )
    output_path: str = "/data1/dingxin/zeva-runs/libero-v1-h5-lang/pi05_goal_embeddings.pt"


def _clean(task: str) -> str:
    return task.strip().replace("_", " ").replace("\n", " ")


def _tokenize(tokenizer: Tokenizer, task: str, max_length: int = 200) -> tuple[torch.Tensor, torch.Tensor]:
    # Mirror SentencePiece: encode(cleaned, add_bos=True) + encode("\n").
    # The saved Rust tokenizer's postprocessor adds EOS rather than BOS, so we
    # disable it and prepend the checkpoint's BOS id explicitly.
    bos = tokenizer.token_to_id("<bos>")
    if bos is None:
        raise ValueError("PI0.5 tokenizer has no BOS token.")
    ids = ([bos] + tokenizer.encode(_clean(task), add_special_tokens=False).ids
           + tokenizer.encode("\n", add_special_tokens=False).ids)[:max_length]
    mask = [1] * len(ids)
    pad_id = tokenizer.token_to_id("<pad>")
    pad_id = int(0 if pad_id is None else pad_id)
    ids += [pad_id] * (max_length - len(ids))
    mask += [0] * (max_length - len(mask))
    return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.float32)


def main(args: Args) -> None:
    handoff = LiberoHandoff.from_root(args.handoff_root)
    checkpoint = handoff.checkpoint / "model.safetensors"
    tokenizer = Tokenizer.from_file(str(Path(args.tokenizer_path) / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        embedding_weight = handle.get_tensor(EMBEDDING_KEY).float()
    if tuple(embedding_weight.shape) != (257_152, 2048):
        raise ValueError(f"Unexpected PI0.5 language table: {tuple(embedding_weight.shape)}")

    split_payload = {}
    prompt_digest = hashlib.sha256()
    task_cache: dict[str, torch.Tensor] = {}
    for subset in ("train", "validation"):
        table = LiberoEpisodeTable(args.dataset_root, subset)
        record_ids = []
        embeddings = []
        for record_index in range(len(table.episodes)):
            task = table.task(record_index)
            if task not in task_cache:
                ids, mask = _tokenize(tokenizer, task)
                token_embeddings = embedding_weight[ids]
                task_cache[task] = (token_embeddings * mask[:, None]).sum(0) / mask.sum().clamp_min(1.0)
            record_ids.append(table.record_id(record_index))
            embeddings.append(task_cache[task].to(torch.float16))
            prompt_digest.update((subset + "\0" + table.record_id(record_index) + "\0" + _clean(task) + "\n").encode())
        split_payload[subset] = {
            "record_ids": tuple(record_ids),
            "embeddings": torch.stack(embeddings),
        }

    payload = {
        "schema": SCHEMA,
        "embedding_dim": 2048,
        "embedding_key": EMBEDDING_KEY,
        "prompt_format": "pi05-task-only-plus-newline",
        "discrete_state_input": False,
        "foundation_checkpoint_sha256": sha256(checkpoint),
        "selected_checkpoint_sha256": LIBERO_CHECKPOINT_SHA256,
        "tokenizer_path": str(Path(args.tokenizer_path).resolve()),
        "tokenizer_sha256": sha256(Path(args.tokenizer_path) / "tokenizer.json"),
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "prompt_table_sha256": prompt_digest.hexdigest(),
        "task_count": len(task_cache),
        "splits": split_payload,
    }
    if payload["foundation_checkpoint_sha256"] != LIBERO_CHECKPOINT_SHA256:
        raise ValueError("Goal export checkpoint is not the selected LIBERO PI0.5.")
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    torch.save(payload, temporary)
    temporary.replace(output)
    summary = {
        "schema": SCHEMA,
        "task_count": len(task_cache),
        "train_episodes": len(split_payload["train"]["record_ids"]),
        "validation_episodes": len(split_payload["validation"]["record_ids"]),
        "output": str(output.resolve()),
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(tyro.cli(Args))
