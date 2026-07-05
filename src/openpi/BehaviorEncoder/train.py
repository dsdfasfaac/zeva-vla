import argparse
import os
from dataclasses import asdict

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import BehaviorModel
from dataloader import LiberoDataset, collate_fn_libero
from config import ModelConfig

# from src.openpi.BehaviorEncoder.model import BehaviorModel
# from src.openpi.BehaviorEncoder.dataloader import LiberoDataset, collate_fn_libero
# from src.openpi.BehaviorEncoder.config import ModelConfig

def compute_loss(outputs, batch, device):
    gt_act = batch['actions'].to(device)
    mask = batch['masks'].to(device)
    task_ids = batch['task_ids'].to(device)
    
    pred_act = outputs['pred_act']
    pred_vis = outputs['pred_vis']
    gt_vis = outputs['gt_vis']
    
    z_global = outputs['z_global']
    z_local = outputs['z_local']
    scale = outputs['logit_scale']
    
    # 1. Action MSE
    l_act = (F.mse_loss(pred_act, gt_act, reduction='none').sum(-1) * mask.float()).sum() / (mask.sum() + 1e-6)
    
    # 2. Vision MSE (Pred t -> Target t+1)
    mask_slice = mask[:, 1:].float()
    l_vis = (F.mse_loss(pred_vis[:, :-1], gt_vis[:, 1:], reduction='none').sum(-1) * mask_slice).sum() / (mask_slice.sum() + 1e-6)
    
    # 3. Global Task Loss
    # Masked Mean Pooling
    mask_float = mask.float().unsqueeze(-1)
    z_traj_mean = (z_global * mask_float).sum(1) / (mask_float.sum(1) + 1e-6)
    z_traj_mean = F.normalize(z_traj_mean, dim=-1)
    
    B = task_ids.shape[0]
    
    labels_mask = torch.eq(task_ids.unsqueeze(0), task_ids.unsqueeze(1)).float()
    labels_mask.fill_diagonal_(0) 
    
    logits_glob = torch.matmul(z_traj_mean, z_traj_mean.T) * scale.exp()
    
    
    if labels_mask.sum() > 0:
        
        exp_logits = torch.exp(logits_glob)
        
        log_prob = logits_glob - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
        
        l_glob = -(labels_mask * log_prob).sum() / (labels_mask.sum() + 1e-6)
    else:
        l_glob = torch.tensor(0.0, device=logits_glob.device)

    # 4. Temporal Progress Loss (Distinctiveness)
    valid_indices = mask.view(-1).bool()
    flat_z = z_local.view(-1, 128)[valid_indices]
    
    
    logits_loc = torch.matmul(flat_z, flat_z.T) * scale.exp()
    labels_loc = torch.arange(logits_loc.shape[0], device=logits_loc.device)
    l_prog = F.cross_entropy(logits_loc, labels_loc)
    
    # Weights
    total = 0.1 * l_act + 0.2 * l_vis + 2.0 * l_glob + 1.0 * l_prog
    return total, l_act, l_vis, l_glob, l_prog


DEFAULT_DATASET_ROOT = "/path/to/libero/dataset"
DEFAULT_NORM_STATS_DIR = (
    "/path/to/pi05_base/assets/physical-intelligence/libero"
)
DEFAULT_SAVE_DIR = "/path/to/behavior_encoder/output"


def train(
    dataset_root=DEFAULT_DATASET_ROOT,
    norm_stats_dir=DEFAULT_NORM_STATS_DIR,
    save_dir=DEFAULT_SAVE_DIR,
    epochs=80,
    batch_size=8,
):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    
    # model config
    config = ModelConfig(
        action_dim=7,
        d_model=256, 
        n_layers=4,
        dropout=0.1,
        ema_decay=0.99,
        img_size=224
    )
    
    model = BehaviorModel(config).to(device)
    
    if not os.path.exists(dataset_root):
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_root}")
    if not os.path.exists(os.path.join(norm_stats_dir, "norm_stats.json")):
        raise FileNotFoundError(f"pi0.5 normalization stats do not exist: {norm_stats_dir}")
        
    print("⏳ Loading dataset (this may take a while to read metadata)...")
    dataset = LiberoDataset(
        root_dir=dataset_root,
        img_size=224,
        frame_skip=2,
        action_norm_stats=norm_stats_dir,
    )
    print(f"Loaded {len(dataset)} episodes.")
    print(f"Action normalization: pi0.5 quantile ({norm_stats_dir})")
    print(f"q01: {dataset.action_normalizer.q01.tolist()}")
    print(f"q99: {dataset.action_normalizer.q99.tolist()}")

    # DataLoader
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=8, 
        collate_fn=collate_fn_libero, 
        pin_memory=True
    )
    
    param_groups = [
        {
            'params': model.vision_encoder.parameters(), 
            'lr': 1e-5 
        },
        {   
            'params': [p for n, p in model.named_parameters() if 'vision_encoder' not in n], 
            'lr': 1e-4 
        }
    ]
    optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
    
    os.makedirs(save_dir, exist_ok=True)
    
    model.train()
    
    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        epoch_loss = 0
        
        for batch in pbar:
            imgs = batch['images'].to(device)
            acts = batch['actions'].to(device)
            masks = batch['masks'].to(device)
            optimizer.zero_grad()
            outputs = model(imgs, acts)
            loss, l_act, l_vis, l_glob, l_prog = compute_loss(outputs, batch, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            model.update_ema()
            
            epoch_loss += loss.item()
                
            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}", 
                "Act": f"{l_act.item():.3f}",
                "Vis": f"{l_vis.item():.3f}"
            })
            
        
        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}")
        
        ckpt_path = os.path.join(save_dir, f"behavior_model_ep{epoch+1:02d}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'model_config': asdict(config),
            'data_config': {
                'dataset_root': dataset_root,
                'frame_skip': dataset.frame_skip,
                'image_range': [-1.0, 1.0],
            },
            'action_normalization': dataset.action_normalizer.metadata(),
        }, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

    print("Training finished!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--norm_stats_dir", default=DEFAULT_NORM_STATS_DIR)
    parser.add_argument("--save_dir", default=DEFAULT_SAVE_DIR)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    train(
        dataset_root=args.dataset_root,
        norm_stats_dir=args.norm_stats_dir,
        save_dir=args.save_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


"""
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/openpi/BehaviorEncoder/train.py
"""
