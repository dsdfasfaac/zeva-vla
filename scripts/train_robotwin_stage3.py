"""Formal Stage 3: align frozen Stage 2 PI0.5 VLM queries to frozen ZTE keys."""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import math
from pathlib import Path
from typing import Any

from accelerate import Accelerator
from accelerate.utils import send_to_device
import safetensors.torch
import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
import tqdm
import tyro

from openpi.zeva.causal_bank import RobotWinCausalBank
from openpi.zeva.retrieval import CausalRetrievalHead
from openpi.zeva.robotwin_contract import ROBOTWIN_CAMERA_KEYS
from openpi.zeva.robotwin_contract import RobotWinHandoff
from openpi.zeva.robotwin_policy import RobotWinZevaPolicy
try:
    from scripts.train_robotwin_zte import FFmpegRoboTwinDataset
except ModuleNotFoundError:  # Direct `python scripts/...py` execution.
    from train_robotwin_zte import FFmpegRoboTwinDataset


@dataclasses.dataclass
class Args:
    handoff_root: str = (
        "/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1"
    )
    dataset_root: str = "/data1/huangbingjia/robotwin-lerobot-sidney-eef16-v1/data"
    stage2_checkpoint: str = (
        "/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage2-full/027000"
    )
    zte_checkpoint: str = "/data1/dingxin/zeva-runs/robotwin-v5-h15-tasklang/stage1-zte/zte_best.pth"
    causal_bank: str = "/data1/dingxin/zeva-runs/robotwin-v5-h15-tasklang/stage1-zte/train_causal_bank.pt"
    save_dir: str = "/data1/dingxin/zeva-runs/robotwin-v5-h15-tasklang/stage3-retrieval"
    extract_batch_size: int = 4
    num_workers: int = 4
    epochs: int = 300
    train_batch_size: int = 256
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    temperature: float = 0.03
    hidden_dim: int = 1024
    dropout: float = 0.1
    log_freq: int = 5
    seed: int = 1000
    force_extract: bool = False
    max_train_episodes: int | None = None
    max_validation_episodes: int | None = None


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class RobotWinEpisodeDataset(Dataset):
    """Exactly one first-frame PI0.5 observation from each frozen-split episode."""

    def __init__(
        self,
        adapter_manifest: str | Path,
        *,
        subset: str,
        rank: int,
        world_size: int,
        max_episodes: int | None = None,
    ):
        self.source = FFmpegRoboTwinDataset(adapter_manifest, subset=subset)
        self.dataset = self.source.dataset
        task_names = sorted({record["key"][1] for record in self.dataset._records})  # noqa: SLF001
        self.task_names = tuple(task_names)
        self._task_ids = {name: index for index, name in enumerate(task_names)}
        record_count = len(self.dataset._records)  # noqa: SLF001
        if max_episodes is not None:
            record_count = min(record_count, max_episodes)
        # Rank slicing never pads, so no held-out or training episode is duplicated.
        self._record_indices = list(range(rank, record_count, world_size))

    def __len__(self) -> int:
        return len(self._record_indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record_index = self._record_indices[index]
        record = self.dataset._records[record_index]  # noqa: SLF001
        dataset_index = int(self.dataset._cumulative[record_index])  # noqa: SLF001
        result = dict(self.dataset[dataset_index])
        images = self.source.read_images(record, [0])
        for key in ROBOTWIN_CAMERA_KEYS:
            result[key] = images[key][0]
        result["zeva.record_index"] = torch.tensor(record_index, dtype=torch.long)
        result["zeva.task_id"] = torch.tensor(self._task_ids[record["key"][1]], dtype=torch.long)
        result["zeva.episode_index"] = torch.tensor(int(record["episode_index"]), dtype=torch.long)
        result["zeva.partition_id"] = torch.tensor(int(record["partition_id"]), dtype=torch.long)
        return result


def _stage2_manifest(args: Args, handoff: RobotWinHandoff, bank: RobotWinCausalBank) -> dict[str, Any]:
    checkpoint = Path(args.stage2_checkpoint).resolve()
    required = [
        checkpoint / "model.safetensors",
        checkpoint / "zeva_adapter.pth",
        checkpoint / "training_state.pt",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete Stage 2 checkpoint: {missing}")
    manifest_path = checkpoint.parent / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing Stage 2 manifest: {manifest_path}")
    stage2 = json.loads(manifest_path.read_text())
    if stage2.get("schema") != "zeva-robotwin-stage2-manifest-v1":
        raise ValueError("Stage 3 requires a formal Stage 2 checkpoint.")
    if stage2["causal_bank_sha256"] != _sha256(args.causal_bank):
        raise ValueError("Stage 2 and Stage 3 causal banks differ.")
    if stage2["zte_checkpoint_sha256"] != _sha256(args.zte_checkpoint):
        raise ValueError("Stage 2 and Stage 3 ZTE checkpoints differ.")
    if stage2["statistics_sha256"] != _sha256(handoff.statistics):
        raise ValueError("Stage 2 and Stage 3 normalization statistics differ.")
    if bank.manifest["stage1_checkpoint_sha256"] != stage2["zte_checkpoint_sha256"]:
        raise ValueError("The causal bank was not exported from the selected ZTE.")
    return stage2


def _manifest(
    args: Args,
    handoff: RobotWinHandoff,
    bank: RobotWinCausalBank,
    stage2: dict[str, Any],
) -> dict[str, Any]:
    checkpoint = Path(args.stage2_checkpoint).resolve()
    return {
        "schema": "zeva-robotwin-stage3-manifest-v1",
        "handoff_root": str(handoff.root),
        "statistics_sha256": _sha256(handoff.statistics),
        "dataset_adapter": str((Path(args.dataset_root) / "adapter.json").resolve()),
        "train_split": "train95",
        "validation_split": "validation5",
        "stage2_checkpoint": str(checkpoint),
        "stage2_step": int(Path(checkpoint).name),
        "stage2_model_sha256": _sha256(checkpoint / "model.safetensors"),
        "stage2_adapter_sha256": _sha256(checkpoint / "zeva_adapter.pth"),
        "stage2_manifest": stage2,
        "zte_checkpoint": str(Path(args.zte_checkpoint).resolve()),
        "zte_checkpoint_sha256": _sha256(args.zte_checkpoint),
        "causal_bank": str(Path(args.causal_bank).resolve()),
        "causal_bank_sha256": _sha256(args.causal_bank),
        "task_names": list(bank.task_names),
        "source_feature": "first-frame-pi05-vlm-eos-after-stage2",
        "source_dim": 2048,
        "target": "train95-count-weighted-zte-task-key",
        "target_dim": int(bank.task_prototype.shape[-1]),
        "frozen": ["stage2_full_policy", "zte", "causal_bank", "source_features"],
        "trainable": ["vlm_to_zte_retrieval_head"],
        "train_args": dataclasses.asdict(args),
        "source_sha256": {
            "trainer": _sha256(Path(__file__).resolve()),
            "retrieval_head": _sha256(Path(inspect.getfile(CausalRetrievalHead)).resolve()),
            "robotwin_policy": _sha256(Path(inspect.getfile(RobotWinZevaPolicy)).resolve()),
        },
    }


@torch.inference_mode()
def _extract_vlm_features(policy: RobotWinZevaPolicy, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Run the frozen Stage 2 PaliGemma prefix and pool its last valid text token."""
    from lerobot.policies.common.vla_utils import make_att_2d_masks  # noqa: PLC0415
    from lerobot.policies.common.vla_utils import prepare_attention_masks_4d  # noqa: PLC0415
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK  # noqa: PLC0415
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS  # noqa: PLC0415

    foundation = policy.foundation
    core = foundation.model
    images, image_masks = foundation._preprocess_images(batch)  # noqa: SLF001
    states, state_masks = foundation._prepare_memory_states(batch)  # noqa: SLF001
    tokens = batch[OBS_LANGUAGE_TOKENS]
    token_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
    embeddings, pad_masks, attention_masks = core.embed_prefix(
        images,
        image_masks,
        tokens,
        token_masks,
        states,
        state_masks,
    )
    q_proj = core.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj
    embeddings = embeddings.to(q_proj.weight.dtype)
    attention_2d = make_att_2d_masks(pad_masks, attention_masks)
    attention_4d = prepare_attention_masks_4d(attention_2d, dtype=embeddings.dtype)
    position_ids = torch.cumsum(pad_masks, dim=1) - 1
    outputs, _ = core.paligemma_with_expert.forward(
        attention_mask=attention_4d,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[embeddings, None],
        use_cache=False,
    )
    hidden = outputs[0].float()
    last_valid = pad_masks.long().sum(dim=1).sub(1).clamp_min(0)
    gather_index = last_valid[:, None, None].expand(-1, 1, hidden.shape[-1])
    return F.normalize(hidden.gather(1, gather_index).squeeze(1), dim=-1)


def _extract_subset(
    policy: RobotWinZevaPolicy,
    preprocessor,
    args: Args,
    accelerator: Accelerator,
    subset: str,
    max_episodes: int | None,
    shard_dir: Path,
) -> None:
    dataset = RobotWinEpisodeDataset(
        Path(args.dataset_root) / "adapter.json",
        subset=subset,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
        max_episodes=max_episodes,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.extract_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    rows: dict[str, list[torch.Tensor]] = {
        "features": [],
        "record_indices": [],
        "task_ids": [],
        "episode_indices": [],
        "partition_ids": [],
    }
    progress = tqdm.tqdm(loader, desc=f"extract-{subset}-rank{accelerator.process_index}")
    for raw_batch in progress:
        metadata = {
            "record_indices": raw_batch.pop("zeva.record_index"),
            "task_ids": raw_batch.pop("zeva.task_id"),
            "episode_indices": raw_batch.pop("zeva.episode_index"),
            "partition_ids": raw_batch.pop("zeva.partition_id"),
        }
        raw_batch = send_to_device(raw_batch, accelerator.device, non_blocking=True)
        processed = preprocessor(raw_batch)
        rows["features"].append(_extract_vlm_features(policy, processed).cpu().to(torch.float16))
        for key, value in metadata.items():
            rows[key].append(value.cpu())
    payload = {key: torch.cat(value) for key, value in rows.items()}
    payload["task_names"] = dataset.task_names
    torch.save(payload, shard_dir / f"{subset}-rank{accelerator.process_index:02d}.pt")


def _merge_shards(save_dir: Path, shard_dir: Path, world_size: int, manifest: dict[str, Any]) -> None:
    source: dict[str, Any] = {
        "schema": "zeva-robotwin-stage3-source-features-v1",
        "manifest": manifest,
    }
    for subset in ("train", "validation"):
        shards = [
            torch.load(shard_dir / f"{subset}-rank{rank:02d}.pt", map_location="cpu")
            for rank in range(world_size)
        ]
        task_names = {tuple(shard["task_names"]) for shard in shards}
        if task_names != {tuple(manifest["task_names"])}:
            raise ValueError(f"{subset} task ordering differs across feature shards or causal bank.")
        merged = {
            key: torch.cat([shard[key] for shard in shards])
            for key in ("features", "record_indices", "task_ids", "episode_indices", "partition_ids")
        }
        order = torch.argsort(merged["record_indices"])
        source[subset] = {key: value[order] for key, value in merged.items()}
        expected = torch.arange(len(order))
        if not torch.equal(source[subset]["record_indices"], expected):
            raise RuntimeError(f"{subset} source extraction missed or duplicated episodes.")
    torch.save(source, save_dir / "source_features.pt")


def _task_prototypes(bank: RobotWinCausalBank) -> torch.Tensor:
    return F.normalize(bank.task_prototype.float(), dim=-1)


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    features: torch.Tensor,
    task_ids: torch.Tensor,
    prototypes: torch.Tensor,
    temperature: float,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    loader = DataLoader(TensorDataset(features, task_ids), batch_size=batch_size, shuffle=False)
    total_loss = 0.0
    total = 0
    top1 = 0
    top5 = 0
    for source, labels in loader:
        source = source.to(device=device, dtype=torch.float32)
        labels = labels.to(device)
        logits = model(source) @ prototypes.T / temperature
        total_loss += float(F.cross_entropy(logits, labels, reduction="sum"))
        ranking = logits.topk(min(5, logits.shape[1]), dim=1).indices
        top1 += int(ranking[:, :1].eq(labels[:, None]).any(dim=1).sum())
        top5 += int(ranking.eq(labels[:, None]).any(dim=1).sum())
        total += len(labels)
    return {"loss": total_loss / total, "recall_at_1": top1 / total, "recall_at_5": top5 / total}


def _train_head(args: Args, save_dir: Path, manifest: dict[str, Any], bank: RobotWinCausalBank, device) -> None:
    source = torch.load(save_dir / "source_features.pt", map_location="cpu")
    if source.get("schema") != "zeva-robotwin-stage3-source-features-v1":
        raise ValueError("Unsupported Stage 3 source feature file.")
    if source["manifest"]["stage2_model_sha256"] != manifest["stage2_model_sha256"]:
        raise ValueError("Cached features came from a different Stage 2 model.")
    train = source["train"]
    validation = source["validation"]
    prototypes = _task_prototypes(bank).to(device)
    model = CausalRetrievalHead(
        input_dim=train["features"].shape[-1],
        output_dim=prototypes.shape[-1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        TensorDataset(train["features"], train["task_ids"]),
        batch_size=args.train_batch_size,
        shuffle=True,
        generator=generator,
        drop_last=False,
    )
    best_recall = -math.inf
    best_loss = math.inf
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for features, labels in loader:
            features = features.to(device=device, dtype=torch.float32)
            labels = labels.to(device)
            logits = model(features) @ prototypes.T / args.temperature
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(labels)
            total += len(labels)
        scheduler.step()
        metrics = _evaluate(
            model,
            validation["features"],
            validation["task_ids"],
            prototypes,
            args.temperature,
            args.train_batch_size,
            device,
        )
        metrics.update(
            {"epoch": epoch, "train_loss": total_loss / total, "learning_rate": scheduler.get_last_lr()[0]}
        )
        history.append(metrics)
        checkpoint = {
            "schema": "zeva-robotwin-stage3-retrieval-head-v1",
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "input_dim": model.input_dim,
            "output_dim": model.output_dim,
            "hidden_dim": model.hidden_dim,
            "dropout": model.dropout,
            "task_prototypes": prototypes.cpu(),
            "task_names": list(bank.task_names),
            "metrics": metrics,
            "manifest": manifest,
        }
        torch.save(checkpoint, save_dir / "retrieval_head_latest.pth")
        is_best = metrics["recall_at_1"] > best_recall or (
            metrics["recall_at_1"] == best_recall and metrics["loss"] < best_loss
        )
        if is_best:
            best_recall = metrics["recall_at_1"]
            best_loss = metrics["loss"]
            torch.save(checkpoint, save_dir / "retrieval_head_best.pth")
        if epoch == 1 or epoch % args.log_freq == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch}/{args.epochs} train={metrics['train_loss']:.6f} "
                f"val={metrics['loss']:.6f} R@1={metrics['recall_at_1']:.4f} "
                f"R@5={metrics['recall_at_5']:.4f}",
                flush=True,
            )
        (save_dir / "metrics.json").write_text(
            json.dumps({"best_recall_at_1": best_recall, "history": history}, indent=2) + "\n"
        )


def main(args: Args) -> None:
    accelerator = Accelerator()
    torch.manual_seed(args.seed + accelerator.process_index)
    handoff = RobotWinHandoff.from_root(args.handoff_root)
    bank = RobotWinCausalBank.load(args.causal_bank, device="cpu")
    save_dir = Path(args.save_dir)
    shard_dir = save_dir / ".feature-shards"
    source_path = save_dir / "source_features.pt"
    if accelerator.is_main_process:
        stage2 = _stage2_manifest(args, handoff, bank)
        manifest = _manifest(args, handoff, bank, stage2)
        save_dir.mkdir(parents=True, exist_ok=True)
        shard_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    accelerator.wait_for_everyone()
    manifest = json.loads((save_dir / "manifest.json").read_text())

    reuse_source = source_path.is_file() and not args.force_extract
    if reuse_source:
        cached = torch.load(source_path, map_location="cpu")
        reuse_source = (
            cached.get("schema") == "zeva-robotwin-stage3-source-features-v1"
            and cached.get("manifest", {}).get("stage2_model_sha256") == manifest["stage2_model_sha256"]
            and cached.get("manifest", {}).get("causal_bank_sha256") == manifest["causal_bank_sha256"]
        )
    if not reuse_source:
        policy = RobotWinZevaPolicy.from_handoff(
            args.handoff_root,
            device=str(accelerator.device),
            zte_checkpoint=args.zte_checkpoint,
            adapter_checkpoint=Path(args.stage2_checkpoint) / "zeva_adapter.pth",
        )
        safetensors.torch.load_model(
            policy.foundation,
            Path(args.stage2_checkpoint) / "model.safetensors",
            strict=True,
        )
        policy.requires_grad_(False).eval()
        if any(parameter.requires_grad for parameter in policy.parameters()):
            raise RuntimeError("Stage 3 invariant failed: Stage 2 policy or ZTE remains trainable.")
        _extract_subset(
            policy,
            policy.preprocessor,
            args,
            accelerator,
            "train",
            args.max_train_episodes,
            shard_dir,
        )
        _extract_subset(
            policy,
            policy.preprocessor,
            args,
            accelerator,
            "validation",
            args.max_validation_episodes,
            shard_dir,
        )
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            _merge_shards(save_dir, shard_dir, accelerator.num_processes, manifest)
        del policy
        torch.cuda.empty_cache()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        _train_head(args, save_dir, manifest, bank, accelerator.device)
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main(tyro.cli(Args))
