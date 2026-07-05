import os
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import io
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image

from openpi.shared import normalize as openpi_normalize


@dataclass(frozen=True)
class QuantileActionNormalizer:
    q01: torch.Tensor
    q99: torch.Tensor
    source: str | None = None

    @classmethod
    def from_source(cls, source, action_dim=7):
        """Load pi0.5 action quantiles from an OpenPI asset directory or checkpoint metadata."""
        if isinstance(source, (str, os.PathLike)):
            source_path = Path(source).expanduser().resolve()
            action_stats = openpi_normalize.load(source_path)["actions"]
            q01 = action_stats.q01
            q99 = action_stats.q99
            source_name = str(source_path)
        elif isinstance(source, Mapping):
            norm_type = source.get("type", "quantile")
            if norm_type != "quantile":
                raise ValueError(f"Expected quantile action normalization, got {norm_type!r}.")
            q01 = source.get("q01")
            q99 = source.get("q99")
            source_name = source.get("source")
        else:
            raise TypeError("action_norm_stats must be an OpenPI stats directory or checkpoint metadata mapping.")

        if q01 is None or q99 is None:
            raise ValueError("Action normalization requires both q01 and q99.")

        q01 = torch.as_tensor(q01, dtype=torch.float32).detach().cpu().flatten()
        q99 = torch.as_tensor(q99, dtype=torch.float32).detach().cpu().flatten()
        if q01.numel() < action_dim or q99.numel() < action_dim:
            raise ValueError(
                f"Action stats have {min(q01.numel(), q99.numel())} dimensions, expected at least {action_dim}."
            )

        q01 = q01[:action_dim].clone()
        q99 = q99[:action_dim].clone()
        if torch.any(q99 <= q01):
            raise ValueError("Each action q99 value must be greater than q01.")
        return cls(q01=q01, q99=q99, source=source_name)

    def normalize(self, actions):
        if actions.shape[-1] != self.q01.numel():
            raise ValueError(
                f"Action tensor has dimension {actions.shape[-1]}, expected {self.q01.numel()}."
            )
        q01 = self.q01.to(device=actions.device, dtype=actions.dtype)
        q99 = self.q99.to(device=actions.device, dtype=actions.dtype)
        # Match openpi.transforms.Normalize(use_quantiles=True) exactly; intentionally no clamp.
        return (actions - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0

    def metadata(self):
        return {
            "type": "quantile",
            "q01": self.q01.cpu().clone(),
            "q99": self.q99.cpu().clone(),
            "source": self.source,
        }

class LiberoDataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=224, frame_skip=1, *, action_norm_stats):
        self.root_dir = root_dir
        self.data_dir = os.path.join(root_dir, "data")
        self.meta_dir = os.path.join(root_dir, "meta")
        self.img_size = img_size
        self.frame_skip = frame_skip
        self.action_normalizer = QuantileActionNormalizer.from_source(action_norm_stats, action_dim=7)

        self._load_metadata()
        self._prepare_stats()
        
        self.transform = T.Compose([
            T.Resize((img_size, img_size), antialias=True),
            T.ToTensor(),
            # Match OpenPI's image contract used during online inference.
            T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

    def _load_metadata(self):
        
        info_path = os.path.join(self.meta_dir, "info.json")
        with open(info_path, 'r') as f:
            self.info = json.load(f)
        
        self.chunk_size = self.info.get("chunks_size", 1000)
        self.total_episodes = self.info.get("total_episodes", 0)
        
        self.task_map = {}
        tasks_path = os.path.join(self.meta_dir, "tasks.jsonl")
        if os.path.exists(tasks_path):
            with open(tasks_path, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    self.task_map[item['task_index']] = item['task']

    def _prepare_stats(self):

        stats_path = os.path.join(self.meta_dir, "stats.json")
        self.stats = {}
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                raw_stats = json.load(f)
            
            for key in ['state']:
                if key in raw_stats:
                    self.stats[key] = {
                        'mean': torch.tensor(raw_stats[key]['mean'], dtype=torch.float32),
                        'std': torch.tensor(raw_stats[key]['std'], dtype=torch.float32)
                    }
                    
                    self.stats[key]['std'][self.stats[key]['std'] == 0] = 1.0

    def _normalize(self, tensor, key):

        if key not in self.stats:
            return tensor
        mean = self.stats[key]['mean']
        std = self.stats[key]['std']
        return (tensor - mean) / std

    def _process_image_bytes(self, image_bytes_list):
        processed_imgs = []
        for i, img_data in enumerate(image_bytes_list):
            
            if isinstance(img_data, dict):
                if 'bytes' in img_data and img_data['bytes'] is not None:
                    img_data = img_data['bytes']
                else:
                    if i == 0: print(f"[Debug] Image dict keys at index 0: {img_data.keys()}")
                    continue

            
            if isinstance(img_data, bytes):
                try:
                    
                    img = Image.open(io.BytesIO(img_data))
                    processed_imgs.append(self.transform(img))
                except Exception as e1:
                    try:
                        arr = np.frombuffer(img_data, dtype=np.uint8).reshape(256, 256, 3)
                        img = Image.fromarray(arr)
                        processed_imgs.append(self.transform(img))
                    except Exception as e2:
                        
                        print(f"[Error] Decode failed at index {i}. Type: bytes. Len: {len(img_data)}")
                        print(f"  - PIL Error: {e1}")
                        print(f"  - Numpy Error: {e2}")
                        
                        # processed_imgs.append(torch.zeros(3, self.img_size, self.img_size)) 
                        continue
            elif isinstance(img_data, np.ndarray):
                try:
                    img = Image.fromarray(img_data.astype(np.uint8))
                    processed_imgs.append(self.transform(img))
                except Exception as e:
                    print(f"[Error] Numpy array error at index {i}: {e}")
                    continue
            else:
                if i == 0: 
                    print(f"[Error] Unknown image data type: {type(img_data)}")
                    print(f"Sample data: {img_data}")

        
        if not processed_imgs:
            raise RuntimeError(f"No images successfully processed in this episode! Input list len: {len(image_bytes_list)}")

        return torch.stack(processed_imgs)

    def __len__(self):
        return self.total_episodes

    def __getitem__(self, idx):
        
        chunk_idx = idx // self.chunk_size
        file_path = os.path.join(
            self.data_dir, 
            f"chunk-{chunk_idx:03d}", 
            f"episode_{idx:06d}.parquet"
        )

        columns = ['image', 'actions', 'state', 'task_index']
        try:
            df = pd.read_parquet(file_path, columns=columns, engine='pyarrow')
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            raise e

        raw_images = df['image'].values
        raw_actions = df['actions'].values
        raw_states = df['state'].values

        # Frame Skipping
        images_processed = raw_images[::self.frame_skip]
        actions_processed = raw_actions[::self.frame_skip]
        states_processed = raw_states[::self.frame_skip]

        images = self._process_image_bytes(images_processed)

        actions = torch.from_numpy(np.vstack(actions_processed)).float()
        states = torch.from_numpy(np.vstack(states_processed)).float()

        # Match the pi0.5 action domain used by OpenPI training and online inference.
        actions = self.action_normalizer.normalize(actions)
        states = self._normalize(states, 'state')
        
        task_idx = df['task_index'].iloc[0] if 'task_index' in df else 0
        
        if isinstance(task_idx, torch.Tensor) or isinstance(task_idx, np.integer):
            task_idx = int(task_idx)
            
        task_desc = self.task_map.get(task_idx, "")

        return {
            "images": images,          # [T, 3, 224, 224], float32 in [-1, 1]
            "actions": actions,        # [T, 7]
            "states": states,          # [T, 8]
            "task_id": torch.tensor(task_idx, dtype=torch.long),
            "task_desc": task_desc,    # String
            "length": len(images)
        }

def collate_fn_libero(batch):
    max_len = max([b['length'] for b in batch])
    
    batch_imgs = []
    batch_acts = []
    batch_masks = []
    batch_tasks = []
    batch_descs = []
    
    for b in batch:
        T = b['length']
        img = b['images'] # [T, 3, H, W]
        act = b['actions'] # [T, D]
        
        pad_size = max_len - T
        if pad_size > 0:
            # Image padding
            pad_img = torch.zeros(pad_size, *img.shape[1:], dtype=img.dtype)
            img_padded = torch.cat([img, pad_img], dim=0)
            
            # Action padding
            pad_act = torch.zeros(pad_size, *act.shape[1:], dtype=act.dtype)
            act_padded = torch.cat([act, pad_act], dim=0)
            
            # Mask (1=Valid, 0=Pad)
            mask = torch.cat([torch.ones(T), torch.zeros(pad_size)], dim=0)
        else:
            img_padded = img
            act_padded = act
            mask = torch.ones(T)
        
        batch_imgs.append(img_padded)
        batch_acts.append(act_padded)
        batch_masks.append(mask)
        batch_tasks.append(b['task_id'])
        batch_descs.append(b['task_desc'])
        
    return {
        "images": torch.stack(batch_imgs),       # [B, MaxT, 3, H, W]
        "actions": torch.stack(batch_acts),      # [B, MaxT, 7]
        "masks": torch.stack(batch_masks).bool(),# [B, MaxT]
        "task_ids": torch.stack(batch_tasks),    # [B]
        "task_descs": batch_descs                # [B] list of strings
    }


