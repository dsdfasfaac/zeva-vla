import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import logging
import argparse
import time
import os  
from typing import Literal, Dict, List, Tuple
import tqdm


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



class ProjectionHead(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 1024, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return x + self.block(x) 

class AdvancedProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=1024, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        self.res_blocks = nn.Sequential(
            ResidualBlock(hidden_dim, dropout),
            ResidualBlock(hidden_dim, dropout)
        )
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.res_blocks(x)
        return self.output_proj(x)



def supervised_contrastive_loss(query_feats: torch.Tensor, key_feats: torch.Tensor, task_ids: torch.Tensor, temperature: float) -> torch.Tensor:

    query_norm = F.normalize(query_feats, dim=-1)
    key_norm = F.normalize(key_feats, dim=-1)
    
    logits = torch.matmul(query_norm, key_norm.T) / temperature
    
    labels_mask = (task_ids.unsqueeze(1) == task_ids.unsqueeze(0)).float()
    
    log_denominator = torch.log(torch.exp(logits).sum(dim=1, keepdim=True))
    log_prob = logits - log_denominator
    
    pos_mask_sum_q = labels_mask.sum(dim=1)
    loss_vlm_to_beh = - (labels_mask * log_prob).sum(dim=1) / pos_mask_sum_q.clamp(min=1e-6)
    
    pos_mask_sum_k = labels_mask.sum(dim=0)
    loss_beh_to_vlm = - (labels_mask.T * log_prob.T).sum(dim=1) / pos_mask_sum_k.clamp(min=1e-6)

    return (loss_vlm_to_beh.mean() + loss_beh_to_vlm.mean()) / 2



class AlignedRetrievalDataset(Dataset):

    def __init__(self, source_path: str, target_memory_path: str):
        logging.info("Loading and aligning features...")
        source_data = torch.load(source_path)
        target_memory = torch.load(target_memory_path)
        
        target_map = {}
        for item in target_memory:
            target = item.get('retrieval_key')
            if target is None:
                target = item.get('embedding')
            if target is None:
                raise KeyError(f"Memory entry has no retrieval key: {item.keys()}")
            target_map[item['episode_idx']] = target.squeeze()
        
        aligned_vlm_feats = []
        aligned_beh_tokens = []
        aligned_task_ids = []
        
        source_vlm_feats = source_data['features']
        source_episode_ids = source_data['episode_indices']
        source_task_ids = np.array(source_data['task_indices']) 

        for i, episode_idx in enumerate(source_episode_ids):
            if episode_idx in target_map:
                aligned_vlm_feats.append(source_vlm_feats[i].cpu())
                aligned_beh_tokens.append(target_map[episode_idx].cpu())
                aligned_task_ids.append(source_task_ids[i])
            else:
                logging.warning(f"Skipping episode {episode_idx}: Target not found in memory bank.")

        if not aligned_vlm_feats:
            raise RuntimeError("No aligned data found. Check episode IDs in both files.")
        
        self.vlm_feats = torch.stack(aligned_vlm_feats)
        self.beh_tokens = torch.stack(aligned_beh_tokens)
        self.task_ids = torch.tensor(aligned_task_ids, dtype=torch.long)
        
        self.input_dim = self.vlm_feats.shape[-1]
        self.output_dim = self.beh_tokens.shape[-1]
        
        logging.info(f"✅ Data aligned: Total pairs = {len(self)}")

    def __len__(self):
        return len(self.vlm_feats)

    def __getitem__(self, idx):
        return self.vlm_feats[idx], self.beh_tokens[idx], self.task_ids[idx]

def evaluate_retrieval(model: nn.Module, val_loader: DataLoader, k_values: List[int], device: str) -> Dict[str, float]:

    model.eval()
    all_queries = []
    all_keys = []
    all_task_ids_list = []
    
    with torch.no_grad():
        for vlm_feats, beh_tokens, task_ids in val_loader:
            vlm_feats = vlm_feats.to(device)
            
            projected_queries = F.normalize(model(vlm_feats), dim=-1)
            keys = F.normalize(beh_tokens.to(device), dim=-1)
            
            all_queries.append(projected_queries)
            all_keys.append(keys)
            all_task_ids_list.append(task_ids.to(device))

    queries = torch.cat(all_queries, dim=0)
    keys = torch.cat(all_keys, dim=0)
    all_task_ids = torch.cat(all_task_ids_list).to(device)

    start_time = time.time()
    similarity_matrix = torch.matmul(queries, keys.T)
    top_k_values, top_k_indices = torch.topk(similarity_matrix, max(k_values), dim=1)

    num_queries = queries.shape[0]
    print(f"Retrieved top-{max(k_values)} for {num_queries} queries in {(time.time() - start_time):.4f} seconds.")
    avg_retrieval_time_ms = ((time.time() - start_time)/ num_queries) * 1000

    metrics = {}
    
    retrieved_task_ids = all_task_ids[top_k_indices] 
    query_task_ids = all_task_ids.unsqueeze(1)
    
    for k in k_values:
        is_match = (retrieved_task_ids[:, :k] == query_task_ids).any(dim=1) 
        recall = is_match.float().mean().item()
        metrics[f'Recall@{k}'] = recall

    metrics['AvgTime_ms'] = avg_retrieval_time_ms

    return metrics



def train_retrieval_head(args):
    device = torch.device(args.device)

    
    os.makedirs(args.save_dir, exist_ok=True)
    logging.info(f"Checkpoints will be saved to: {args.save_dir}")

    dataset = AlignedRetrievalDataset(args.source_path, args.memory_bank_path)
    
    
    input_dim = dataset.input_dim
    output_dim = dataset.output_dim
    
    model = ProjectionHead(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    ).to(device)

    # model = AdvancedProjectionHead(
    #     input_dim=input_dim,
    #     output_dim=output_dim,
    #     hidden_dim=args.hidden_dim,
    #     dropout=args.dropout
    # ).to(device)

    
    train_size = int(args.train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    full_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))

    best_recall_1 = 0.0
    logging.info(f"Starting training for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for vlm_feats, beh_tokens, task_ids in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            vlm_feats = vlm_feats.to(device)
            beh_tokens = beh_tokens.to(device)
            task_ids = task_ids.to(device)

            optimizer.zero_grad()
            
            projected_vlm = model(vlm_feats)
            
            # Loss: Supervised Contrastive Loss (SupCon)
            loss = supervised_contrastive_loss(projected_vlm, beh_tokens, task_ids, args.temperature)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        logging.info(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}")
        
        
        if (epoch + 1) % args.eval_interval == 0:
            metrics = evaluate_retrieval(model, full_loader, [1, 5], args.device)
            
            logging.info(f"=== Validation Results (Epoch {epoch+1}) ===")
            logging.info(f"Loss: {avg_loss:.4f} | Recall@1: {metrics['Recall@1']:.4f} | Recall@5: {metrics['Recall@5']:.4f} | AvgTime_ms: {metrics['AvgTime_ms']:.5f}")

            
            if metrics['Recall@1'] >= best_recall_1:
                best_recall_1 = metrics['Recall@1']
                best_save_path = os.path.join(args.save_dir, f"best_model_R1_{best_recall_1:.4f}.pth")
                torch.save(model.state_dict(), best_save_path)
                torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pth"))
                logging.info(f"*** NEW BEST MODEL SAVED to {best_save_path} ***")

        if args.save_interval > 0 and (epoch + 1) % args.save_interval == 0:
            epoch_save_path = os.path.join(args.save_dir, f"checkpoint_epoch_{epoch+1:03d}.pth")
            torch.save(model.state_dict(), epoch_save_path)

    logging.info("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the VLM Retrieval Head using SupCon Loss.")
    parser.add_argument("--source-path", type=str, default="vlm_libero_source_features.pt",
                        help="Path to the extracted VLM source features file.")
    parser.add_argument("--memory-bank-path", type=str, default="memory_bank.pt",
                        help="Path to the Global Behavior Memory Bank file.")
    parser.add_argument("--device", type=str, default="cuda", help="Training device.")
    
    
    parser.add_argument("--save-dir", type=str, default="/path/to/retrieval/head/checkpoints",
                        help="Directory to save model checkpoints.")

    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for SupCon.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.03, help="Temperature for SupCon Loss.")
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--train-ratio", type=float, default=0.99, help="Ratio for training split.")
    parser.add_argument("--eval-interval", type=int, default=5, help="Evaluate every N epochs.")
    parser.add_argument("--save-interval", type=int, default=25,
                        help="Save an epoch checkpoint every N epochs; use 0 to disable.")
    
    train_retrieval_head(parser.parse_args())
