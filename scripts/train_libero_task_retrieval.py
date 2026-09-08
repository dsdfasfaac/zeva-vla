"""Stage 1.5: align frozen LIBERO PI0.5 task language to frozen ZTE task keys."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import tyro

from openpi.zeva.libero_bank import LiberoCausalBank
from openpi.zeva.libero_contract import sha256
from openpi.zeva.libero_data import LiberoEpisodeTable
from openpi.zeva.retrieval import CausalRetrievalHead


@dataclasses.dataclass
class Args:
    dataset_root: str = "/data1/dingxin/libero-memory-baseline-v1/data"
    goal_embeddings: str = "/data1/dingxin/zeva-runs/libero-v1-h5-lang/pi05_goal_embeddings.pt"
    causal_bank: str = "/data1/dingxin/zeva-runs/libero-v3-h5-lang/stage1-zte/train_causal_bank.pt"
    output: str = (
        "/data1/dingxin/zeva-runs/libero-v3-h5-lang/"
        "stage1.5-task-retrieval/task_retrieval.pth"
    )
    epochs: int = 100
    batch_size: int = 512
    hidden_dim: int = 512
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    temperature: float = 0.05
    seed: int = 1000


def _split(payload, dataset_root, subset, task_ids):
    table = LiberoEpisodeTable(dataset_root, subset)
    rows = payload["splits"][subset]
    expected = tuple(table.record_id(index) for index in range(len(table.episodes)))
    if tuple(rows["record_ids"]) != expected:
        raise ValueError(f"LIBERO {subset} goal table differs from the dataset.")
    labels = torch.tensor([task_ids[table.task(index)] for index in range(len(table.episodes))])
    prompts = [table.task(index).strip().replace("\n", " ") + "\n" for index in range(len(table.episodes))]
    digest = hashlib.sha256("\n".join(prompts).encode()).hexdigest()
    return torch.as_tensor(rows["embeddings"], dtype=torch.float32), labels, digest


def _canonical_table(payload, dataset_root, task_ids):
    table = LiberoEpisodeTable(dataset_root, "train")
    rows = payload["splits"]["train"]
    result = torch.empty((len(task_ids), payload["embedding_dim"]), dtype=torch.float32)
    filled = torch.zeros(len(task_ids), dtype=torch.bool)
    for index, embedding in enumerate(rows["embeddings"]):
        task_id = task_ids[table.task(index)]
        if not filled[task_id]:
            result[task_id] = embedding.float()
            filled[task_id] = True
    if not filled.all():
        raise RuntimeError("Canonical LIBERO goal table is incomplete.")
    return result


@torch.no_grad()
def _accuracy(model: nn.Module, features, labels, prototypes):
    model.eval()
    predictions = []
    for chunk in features.split(2048):
        predictions.append((model(chunk.cuda()) @ prototypes.T).argmax(1).cpu())
    return float((torch.cat(predictions) == labels).float().mean())


def main(args: Args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    payload = torch.load(args.goal_embeddings, map_location="cpu", weights_only=False)
    if payload.get("schema") != "zeva-libero-pi05-goal-embeddings-v1":
        raise ValueError("LIBERO task retrieval requires task-only goal embeddings v1.")
    bank = LiberoCausalBank.load(args.causal_bank)
    task_ids = {name: index for index, name in enumerate(bank.task_names)}
    train_x, train_y, train_digest = _split(payload, args.dataset_root, "train", task_ids)
    validation_x, validation_y, validation_digest = _split(
        payload, args.dataset_root, "validation", task_ids
    )
    canonical = _canonical_table(payload, args.dataset_root, task_ids)
    prototypes = bank.task_prototype.float().cuda()
    model = CausalRetrievalHead(
        input_dim=train_x.shape[1],
        hidden_dim=args.hidden_dim,
        output_dim=prototypes.shape[1],
        dropout=0.1,
    ).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loader = DataLoader(
        TensorDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True, pin_memory=True
    )
    best_accuracy, best_state = -1.0, None
    for epoch in range(args.epochs):
        model.train()
        for features, labels in loader:
            logits = model(features.cuda(non_blocking=True)) @ prototypes.T / args.temperature
            loss = F.cross_entropy(logits, labels.cuda(non_blocking=True))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        validation_accuracy = _accuracy(model, validation_x, validation_y, prototypes)
        if validation_accuracy >= best_accuracy:
            best_accuracy = validation_accuracy
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"epoch={epoch + 1} validation_task_accuracy={validation_accuracy:.6f}", flush=True)
    if best_state is None:
        raise RuntimeError("LIBERO task retrieval produced no checkpoint.")
    model.load_state_dict(best_state)
    train_accuracy = _accuracy(model, train_x, train_y, prototypes)
    validation_accuracy = _accuracy(model, validation_x, validation_y, prototypes)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "schema": "zeva-libero-task-language-retrieval-v1",
        "source_feature": "task_language",
        "model_state_dict": best_state,
        "task_prototypes": bank.task_prototype.cpu(),
        "task_names": bank.task_names,
        "canonical_goal_embeddings": canonical,
        "train_accuracy": train_accuracy,
        "validation_accuracy": validation_accuracy,
        "goal_embeddings_sha256": sha256(args.goal_embeddings),
        "causal_bank_sha256": sha256(args.causal_bank),
        "instruction_prompt_sha256": {"train": train_digest, "validation": validation_digest},
        "args": dataclasses.asdict(args),
    }
    torch.save(result, output)
    output.with_suffix(".json").write_text(json.dumps({
        "schema": result["schema"],
        "train_accuracy": train_accuracy,
        "validation_accuracy": validation_accuracy,
        "goal_embeddings_sha256": result["goal_embeddings_sha256"],
        "causal_bank_sha256": result["causal_bank_sha256"],
    }, indent=2) + "\n")
    print(f"saved={output} train={train_accuracy:.6f} validation={validation_accuracy:.6f}")


if __name__ == "__main__":
    main(tyro.cli(Args))
