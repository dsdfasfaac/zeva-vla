"""Stage 1.5: align frozen PI0.5 task-language embeddings to frozen ZTE task keys."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import tyro
from safetensors import safe_open
from transformers import AutoTokenizer

from openpi.zeva.causal_bank import RobotWinCausalBank
from openpi.zeva.retrieval import CausalRetrievalHead
from openpi.zeva.robotwin_contract import RobotWinHandoff
try:
    from scripts.train_robotwin_zte import FFmpegRoboTwinDataset
except ModuleNotFoundError:
    from train_robotwin_zte import FFmpegRoboTwinDataset


EMBEDDING_KEY = "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"


@dataclasses.dataclass
class Args:
    handoff_root: str = (
        "/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1"
    )
    dataset_root: str = "/data1/huangbingjia/robotwin-lerobot-sidney-eef16-v1/data"
    goal_embeddings: str = "/data1/dingxin/zeva-runs/robotwin-v5-h15-tasklang/pi05_task_embeddings.pt"
    causal_bank: str = "/data1/dingxin/zeva-runs/robotwin-v5-h15-tasklang/stage1-zte/train_causal_bank.pt"
    output: str = "/data1/dingxin/zeva-runs/robotwin-v5-h15-tasklang/stage1.5-task-retrieval/task_retrieval.pth"
    epochs: int = 100
    batch_size: int = 512
    hidden_dim: int = 512
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    temperature: float = 0.05
    seed: int = 1000


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _instruction_split(
    adapter_manifest: Path,
    split: str,
    task_ids: dict[str, int],
    tokenizer,
    embedding_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    dataset = FFmpegRoboTwinDataset(adapter_manifest, split).dataset
    prompts = []
    labels = []
    for record_index, record in enumerate(dataset._records):  # noqa: SLF001
        sample = dataset[int(dataset._cumulative[record_index])]  # noqa: SLF001
        prompts.append(sample["task"].strip().replace("\n", " ") + "\n")
        labels.append(task_ids[record["key"][1]])
    features = []
    for start in range(0, len(prompts), 256):
        encoded = tokenizer(
            prompts[start : start + 256],
            max_length=200,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tokens = embedding_weight[encoded["input_ids"]].float()
        mask = encoded["attention_mask"].float().unsqueeze(-1)
        features.append((tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0))
    digest = hashlib.sha256("\n".join(prompts).encode("utf-8")).hexdigest()
    return torch.cat(features), torch.tensor(labels), digest


def _canonical_goal_table(payload, task_ids: dict[str, int]) -> torch.Tensor:
    result = torch.empty((len(task_ids), payload["embedding_dim"]), dtype=torch.float32)
    filled = torch.zeros(len(task_ids), dtype=torch.bool)
    rows = payload["splits"]["train"]
    for record_id, embedding in zip(rows["record_ids"], rows["embeddings"], strict=True):
        task_id = task_ids[record_id.split(":", 2)[1]]
        if not filled[task_id]:
            result[task_id] = embedding.float()
            filled[task_id] = True
    if not filled.all():
        raise RuntimeError("Canonical goal table is missing a RoboTwin task.")
    return result


@torch.no_grad()
def _accuracy(model: nn.Module, features: torch.Tensor, labels: torch.Tensor, prototypes: torch.Tensor) -> float:
    model.eval()
    predictions = []
    for chunk in features.split(2048):
        predictions.append((model(chunk.cuda()) @ prototypes.T).argmax(dim=1).cpu())
    return float((torch.cat(predictions) == labels).float().mean())


def main(args: Args) -> None:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    payload = torch.load(args.goal_embeddings, map_location="cpu")
    if payload.get("schema") != "zeva-robotwin-pi05-goal-embeddings-v2":
        raise ValueError("Task retrieval requires task-only PI0.5 goal embeddings v2.")
    bank = RobotWinCausalBank.load(args.causal_bank)
    task_ids = {name: index for index, name in enumerate(bank.task_names)}
    handoff = RobotWinHandoff.from_root(args.handoff_root)
    tokenizer = AutoTokenizer.from_pretrained(str(handoff.checkpoint / "tokenizer"), local_files_only=True)
    with safe_open(handoff.checkpoint / "model.safetensors", framework="pt", device="cpu") as handle:
        embedding_weight = handle.get_tensor(EMBEDDING_KEY)
    adapter_manifest = Path(args.dataset_root) / "adapter.json"
    train_x, train_y, train_prompt_sha256 = _instruction_split(
        adapter_manifest, "train", task_ids, tokenizer, embedding_weight
    )
    validation_x, validation_y, validation_prompt_sha256 = _instruction_split(
        adapter_manifest, "validation", task_ids, tokenizer, embedding_weight
    )
    del embedding_weight
    canonical_goal_embeddings = _canonical_goal_table(payload, task_ids)
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
    best_accuracy = -1.0
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        for features, labels in loader:
            features = features.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            logits = model(features) @ prototypes.T / args.temperature
            loss = F.cross_entropy(logits, labels)
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
        raise RuntimeError("Task retrieval training produced no checkpoint.")
    model.load_state_dict(best_state)
    train_accuracy = _accuracy(model, train_x, train_y, prototypes)
    validation_accuracy = _accuracy(model, validation_x, validation_y, prototypes)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "schema": "zeva-robotwin-task-language-retrieval-v1",
        "source_feature": "task_language",
        "model_state_dict": best_state,
        "task_prototypes": bank.task_prototype.cpu(),
        "task_names": bank.task_names,
        "canonical_goal_embeddings": canonical_goal_embeddings,
        "train_accuracy": train_accuracy,
        "validation_accuracy": validation_accuracy,
        "goal_embeddings_sha256": _sha256(args.goal_embeddings),
        "causal_bank_sha256": _sha256(args.causal_bank),
        "instruction_prompt_sha256": {
            "train": train_prompt_sha256,
            "validation": validation_prompt_sha256,
        },
        "args": dataclasses.asdict(args),
    }
    torch.save(result, output)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "schema": result["schema"],
                "train_accuracy": train_accuracy,
                "validation_accuracy": validation_accuracy,
                "goal_embeddings_sha256": result["goal_embeddings_sha256"],
                "causal_bank_sha256": result["causal_bank_sha256"],
            },
            indent=2,
        )
        + "\n"
    )
    print(f"saved={output} train={train_accuracy:.6f} validation={validation_accuracy:.6f}")


if __name__ == "__main__":
    main(tyro.cli(Args))
