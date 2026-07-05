import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from model import BehaviorModel
from config import ModelConfig
from dataloader import LiberoDataset, collate_fn_libero

def build_memory_bank(ckpt_path, dataset_root, save_path, device="cuda", norm_stats_dir=None):
    print(f"Building Memory Bank on {device}...")
    config = ModelConfig(
        action_dim=7,
        d_model=256, 
        n_layers=4,
        dropout=0.0, 
        ema_decay=0.99,
        vision_pretrained=False,
    )
    
    model = BehaviorModel(config).to(device)
    
    print(f"Loading checkpoint from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    model.eval()
    print("Model loaded successfully.")

    action_norm_stats = checkpoint.get('action_normalization') if isinstance(checkpoint, dict) else None
    if action_norm_stats is None:
        if norm_stats_dir is None:
            raise ValueError(
                "Checkpoint does not contain action_normalization metadata. "
                "Use a quantile-trained checkpoint or explicitly pass --norm_stats_dir."
            )
        action_norm_stats = norm_stats_dir
        print("Checkpoint has no normalization metadata; using --norm_stats_dir explicitly.")
    else:
        print("Using action quantiles stored in the BehaviorEncoder checkpoint.")

    data_config = checkpoint.get("data_config", {}) if isinstance(checkpoint, dict) else {}
    frame_skip = int(data_config.get("frame_skip", 2))
    
    dataset = LiberoDataset(
        root_dir=dataset_root, 
        img_size=224,
        frame_skip=frame_skip,
        action_norm_stats=action_norm_stats,
    )
    
    loader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=4, 
        collate_fn=collate_fn_libero, 
    )
    
    memory_bank = []
    
    
    task_id_to_desc = dataset.task_map if hasattr(dataset, 'task_map') else {}

    print(f"Processing {len(dataset)} episodes...")
    
    with torch.no_grad():
        for i, batch in tqdm(enumerate(loader), total=len(loader)):
            
            imgs = batch['images'].to(device)
            acts = batch['actions'].to(device)
            masks = batch['masks'].to(device)
            task_ids = batch['task_ids'].to(device)
            
            # Forward
            outputs = model(imgs, acts)
            
            # Retrieval keys emphasize task identity; behavior values retain the
            # richer raw behavior representation consumed by APN and the VLM prefix.
            retrieval_seq = outputs['z_global']  # [B, T, 128]
            behavior_seq = outputs['z_seq']  # [B, T, 256]

            mask_float = masks.float().unsqueeze(-1)  # [B, T, 1]
            valid_steps = mask_float.sum(dim=1).clamp_min(1.0)
            retrieval_key = (retrieval_seq * mask_float).sum(dim=1) / valid_steps
            behavior_value = (behavior_seq * mask_float).sum(dim=1) / valid_steps

            retrieval_key = F.normalize(retrieval_key, dim=-1)
            behavior_value = F.normalize(behavior_value, dim=-1)
            
               
            retrieval_key = retrieval_key.cpu()
            behavior_value = behavior_value.cpu()
            task_id = task_ids.cpu().item()
                
            task_desc = task_id_to_desc.get(task_id, "unknown_task")
            
            entry = {
                "episode_idx": i,
                "task_id": task_id,
                "task_desc": task_desc,
                "retrieval_key": retrieval_key,
                "behavior_value": behavior_value,  
                "length": int(masks.sum().item()),
            }
            memory_bank.append(entry)

    
    print(f"Saving memory bank to {save_path}...")
    torch.save(memory_bank, save_path)
    
    print(f"Saved {len(memory_bank)} entries.")
    print("Example entry structure:")
    print(memory_bank[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to trained .pth file")
    parser.add_argument("--data_root", type=str, default="./libero", help="Path to Libero dataset")
    parser.add_argument("--save_path", type=str, default="behavior_memory_bank.pt", help="Output path")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--norm_stats_dir",
        type=str,
        default=None,
        help="Fallback pi0.5 stats directory for legacy checkpoints without normalization metadata",
    )
    
    args = parser.parse_args()
    
    build_memory_bank(args.ckpt, args.data_root, args.save_path, args.device, args.norm_stats_dir)

'''
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/openpi/BehaviorEncoder/build_memory.py \
  --ckpt /path/to/behavior/encoder/checkpoint.pth \
  --data_root /path/to/libero/dataset \
  --save_path /path/to/libero/memory_bank.pt \
  --device cuda
'''
