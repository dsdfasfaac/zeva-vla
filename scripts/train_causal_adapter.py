"""Train Zeva causal-prompt injection while keeping the foundation policy frozen."""

from __future__ import annotations

import dataclasses
import itertools
import logging
from pathlib import Path

import jax
import numpy as np
import safetensors.torch
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import tqdm
import tyro

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import pi0_config as _pi0_config
from openpi.models_pytorch import pi0_pytorch as _pi0_pytorch
from openpi.shared import normalize as _normalize
from openpi.training import config as _config


@dataclasses.dataclass
class Args:
    config_name: str
    base_checkpoint: str
    causal_encoder_ckpt: str
    rollout_dir: str
    norm_stats_dir: str
    save_dir: str = "checkpoints/zeva_causal_adapter"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    epochs: int = 20
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_workers: int = 0


class CausalAdapterDataset(Dataset):
    """Pair a completed transition with the next decision in the same attempt."""

    def __init__(self, rollout_dir: str):
        groups: dict[tuple[int, int, int], list[Path]] = {}
        for path in sorted(Path(rollout_dir).glob("transition_*.npz")):
            with np.load(path) as item:
                key = (int(item["task_id"]), int(item["episode_id"]), int(item["attempt_id"]))
            groups.setdefault(key, []).append(path)
        self.pairs = []
        for paths in groups.values():
            self.pairs.extend(itertools.pairwise(paths))
        if not self.pairs:
            raise FileNotFoundError("Causal adapter training needs at least two transitions per attempt.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        previous_path, current_path = self.pairs[index]
        with np.load(previous_path) as previous, np.load(current_path) as current:
            return {
                "previous_image": previous["image_before"].copy(),
                "effect_image": previous["image_after"].copy(),
                "executed_actions": previous["executed_actions"].copy(),
                "current_image": current["image_before"].copy(),
                "current_wrist": current["wrist_before"].copy(),
                "current_state": current["state_before"].copy(),
                "prompt": str(current["task_description"]),
                "target_actions": current["planned_actions"].copy(),
            }


def _to_torch(tree, device):
    return jax.tree.map(lambda value: torch.as_tensor(np.asarray(value)).to(device), tree)


def _checkpoint_file(path: str) -> Path:
    path = Path(path)
    return path / "model.safetensors" if path.is_dir() else path


def _build_model(args: Args, train_config: _config.TrainConfig, device: torch.device):
    if not isinstance(train_config.model, _pi0_config.Pi0Config):
        raise TypeError("Zeva causal adaptation currently targets pi0/pi0.5 PyTorch models.")
    model_config = dataclasses.replace(
        train_config.model,
        use_zeva=True,
        use_action_prior=True,
        is_training=True,
        schema_memory_path=None,
        schema_retrieval_ckpt=None,
        causal_encoder_ckpt=args.causal_encoder_ckpt,
        causal_adapter_ckpt=None,
        use_behavior=None,
        use_apn=None,
        memory_bank_path=None,
        retrieval_ckpt=None,
        behavior_encoder_ckpt=None,
    )
    model = _pi0_pytorch.PI0Pytorch(model_config).to(device)
    safetensors.torch.load_model(model, _checkpoint_file(args.base_checkpoint), strict=False)
    model.requires_grad_(requires_grad=False)
    trainable_modules = [
        model.task_token_projector,
        model.memory_context_encoder,
        model.causal_projector,
        model.action_prior_network,
        model.prior_emb_proj,
    ]
    for module in trainable_modules:
        module.requires_grad_(requires_grad=True)
    model.causal_transition_encoder.eval()
    return model, model_config


def _make_input_transform(train_config, model_config, norm_stats):
    data_config = train_config.data.create(train_config.assets_dirs, model_config)
    return _transforms.compose(
        [
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=True),
            *data_config.model_transforms.inputs,
        ]
    )


def _prepare_model_batch(batch, input_transform, device, action_horizon):
    transformed = []
    for index in range(len(batch["prompt"])):
        target_actions = np.asarray(batch["target_actions"][index])
        if len(target_actions) < action_horizon:
            padding = np.repeat(target_actions[-1:], action_horizon - len(target_actions), axis=0)
            target_actions = np.concatenate([target_actions, padding], axis=0)
        target_actions = target_actions[:action_horizon]
        transformed.append(
            input_transform(
                {
                    "observation/image": np.asarray(batch["current_image"][index]),
                    "observation/wrist_image": np.asarray(batch["current_wrist"][index]),
                    "observation/state": np.asarray(batch["current_state"][index]),
                    "prompt": batch["prompt"][index],
                    "actions": target_actions,
                }
            )
        )
    collated = jax.tree.map(lambda *values: np.stack([np.asarray(value) for value in values]), *transformed)
    torch_batch = _to_torch(collated, device)
    return _model.Observation.from_dict(torch_batch), torch_batch["actions"].to(torch.float32)


def main(args: Args):
    device = torch.device(args.device)
    train_config = _config.get_config(args.config_name)
    norm_stats = _normalize.load(args.norm_stats_dir)
    model, model_config = _build_model(args, train_config, device)
    input_transform = _make_input_transform(train_config, model_config, norm_stats)
    dataset = CausalAdapterDataset(args.rollout_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(args.epochs):
        # Keep the foundation and CTE in eval mode; only injection modules train.
        model.eval()
        model.task_token_projector.train()
        model.memory_context_encoder.train()
        model.causal_projector.train()
        model.action_prior_network.train()
        model.prior_emb_proj.train()
        running = 0.0
        progress = tqdm.tqdm(loader, desc=f"causal adapter epoch {epoch + 1}/{args.epochs}")
        for batch in progress:
            observation, target_actions = _prepare_model_batch(
                batch, input_transform, device, model_config.action_horizon
            )
            previous_image = torch.as_tensor(batch["previous_image"].numpy(), device=device).to(torch.float32)
            effect_image = torch.as_tensor(batch["effect_image"].numpy(), device=device).to(torch.float32)
            previous_image = previous_image / 127.5 - 1.0
            effect_image = effect_image / 127.5 - 1.0
            executed_actions = torch.as_tensor(batch["executed_actions"].numpy(), device=device).to(torch.float32)
            action_stats = norm_stats["actions"]
            q01 = torch.as_tensor(action_stats.q01[:7], device=device)
            q99 = torch.as_tensor(action_stats.q99[:7], device=device)
            executed_actions = (executed_actions[..., :7] - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0

            with torch.no_grad():
                causal = model.causal_transition_encoder(previous_image, executed_actions, effect_image)
                phase_token = causal.phase_token[:, -1]
                causal_signal = causal.causal_signal[:, -1].unsqueeze(1)
                vlm_features = model.extract_vlm_features(observation)
            task_schema = model.retrieve_task_schema(vlm_features)
            causal_context = model.build_causal_context(
                task_schema,
                phase_token,
                brief_signals=causal_signal,
                retrieved_signals=causal_signal,
            )

            optimizer.zero_grad(set_to_none=True)
            loss = model(
                observation,
                target_actions,
                task_schema=task_schema,
                phase_token=phase_token,
                causal_context=causal_context,
            ).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            running += float(loss)
            progress.set_postfix(loss=f"{float(loss):.4f}")

        average_loss = running / max(1, len(loader))
        payload = {
            "epoch": epoch + 1,
            "loss": average_loss,
            "task_token_projector": model.task_token_projector.state_dict(),
            "memory_context_encoder": model.memory_context_encoder.state_dict(),
            "causal_projector": model.causal_projector.state_dict(),
            "action_prior_network": model.action_prior_network.state_dict(),
            "prior_emb_proj": model.prior_emb_proj.state_dict(),
        }
        torch.save(payload, save_dir / f"causal_adapter_epoch_{epoch + 1:03d}.pth")
        if average_loss <= best_loss:
            best_loss = average_loss
            torch.save(payload, save_dir / "best_model.pth")
        logging.info("epoch=%d loss=%.6f best=%.6f", epoch + 1, average_loss, best_loss)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
