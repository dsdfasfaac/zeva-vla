import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import copy
from dataclasses import dataclass
# from .config import ModelConfig
# from .vision import LightweightCNN
# from config import ModelConfig
# from vision import LightweightCNN
from openpi.BehaviorEncoder.config import ModelConfig
from openpi.BehaviorEncoder.vision import LightweightCNN

try:
    from mamba_ssm import Mamba
except ImportError:
    print("Warning: mamba_ssm not found.")
    Mamba = None

from mamba_ssm.utils.generation import InferenceParams


class TriStreamBlock(nn.Module):
    def __init__(self, config: ModelConfig, block_idx: int = None):
        super().__init__()
        d_model = config.d_model

        if block_idx is not None:
            id_v = block_idx * 3 + 0
            id_a = block_idx * 3 + 1
            id_b = block_idx * 3 + 2
        else:
            id_v = id_a = id_b = None

        self.mamba_v = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2, layer_idx=id_v)
        self.mamba_a = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2, layer_idx=id_a)
        self.mamba_b = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2, layer_idx=id_b)
        
        self.norm_v = nn.LayerNorm(d_model)
        self.norm_a = nn.LayerNorm(d_model)
        self.norm_b = nn.LayerNorm(d_model)
        
        self.attn_va = nn.MultiheadAttention(d_model, 4, batch_first=True, dropout=config.dropout)
        self.attn_av = nn.MultiheadAttention(d_model, 4, batch_first=True, dropout=config.dropout)
        self.attn_b  = nn.MultiheadAttention(d_model, 4, batch_first=True, dropout=config.dropout)
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(3)])

    def forward(self, h_v, h_a, h_b, inference_params=None):
        B, T, D = h_v.shape

        # Temporal
        if inference_params is not None:
            # Inference-time SSM
            h_v = h_v + self.mamba_v(self.norm_v(h_v), inference_params=inference_params)
            h_a = h_a + self.mamba_a(self.norm_a(h_a), inference_params=inference_params)
            h_b = h_b + self.mamba_b(self.norm_b(h_b), inference_params=inference_params)
        else:
            # Training-time SSM
            h_v = h_v + self.mamba_v(self.norm_v(h_v))
            h_a = h_a + self.mamba_a(self.norm_a(h_a))
            h_b = h_b + self.mamba_b(self.norm_b(h_b))
        
        # Interaction
        flat_v, flat_a, flat_b = h_v.reshape(B*T, 1, D), h_a.reshape(B*T, 1, D), h_b.reshape(B*T, 1, D)
        
        out_v, _ = self.attn_va(self.norms[0](flat_v), flat_a, flat_a)
        flat_v = flat_v + out_v
        out_a, _ = self.attn_av(self.norms[1](flat_a), flat_v, flat_v)
        flat_a = flat_a + out_a
        
        kv = torch.cat([flat_v, flat_a], dim=1)
        out_b, _ = self.attn_b(self.norms[2](flat_b), kv, kv)
        flat_b = flat_b + out_b
        
        return flat_v.reshape(B,T,D), flat_a.reshape(B,T,D), flat_b.reshape(B,T,D)


class BehaviorModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Encoders
        self.vision_encoder = LightweightCNN(
            output_dim=config.d_model,
            pretrained=config.vision_pretrained,
        )
        self.target_vision_encoder = copy.deepcopy(self.vision_encoder) # EMA Target
        for p in self.target_vision_encoder.parameters(): 
            p.requires_grad = False

        self.action_proj = nn.Linear(config.action_dim, config.d_model)
        self.action_sos = nn.Parameter(torch.randn(1, 1, config.d_model))
        self.behav_token = nn.Parameter(torch.randn(1, 1, config.d_model))
        self.norm_input = nn.LayerNorm(config.d_model)
        
        # Backbone
        self.blocks = nn.ModuleList([TriStreamBlock(config, block_idx=i) for i in range(config.n_layers)])
        self.norm_final = nn.LayerNorm(config.d_model)
        
        # --- Heads for Training ---
        # 1. Action Reconstruction (Progress & Dynamics)
        self.action_predictor = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.action_dim)
        )
        
        # 2. Vision Prediction (JEPA Style)
        self.vision_predictor = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model)
        )
        
        # 3. Task Identity Head (Global Consistency)
        
        self.task_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model), 
            nn.ReLU(),
            nn.Linear(config.d_model, 128)
        )
        
        # 4. Temporal Progress Head (Local Distinctiveness)
        
        self.progress_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model), 
            nn.ReLU(),
            nn.Linear(config.d_model, 128)
        )
        
        self.logit_scale = nn.Parameter(torch.ones([]) * 4.0)

    def update_ema(self):
        decay = self.config.ema_decay
        with torch.no_grad():
            for po, pt in zip(self.vision_encoder.parameters(), self.target_vision_encoder.parameters()):
                pt.data = pt.data * decay + po.data * (1. - decay)

    def forward(self, images, actions):
        """
        Args:
            images: [B, T, 3, H, W], float values in [-1, 1]
            actions: [B, T, 7]
        """
        B, T, C, H, W = images.shape
        flat_imgs = images.view(B*T, C, H, W)
        
        # Vision Streams
        online_vis = self.vision_encoder(flat_imgs).view(B, T, -1)
        with torch.no_grad():
            target_vis = self.target_vision_encoder(flat_imgs).view(B, T, -1).detach()
            
        # Action Shift
        act_emb = self.action_proj(actions)
        sos = self.action_sos.expand(B, 1, -1)
        shifted_act = torch.cat([sos, act_emb[:, :-1, :]], dim=1)
        
        # Backbone
        h_v = self.norm_input(online_vis)
        h_a = self.norm_input(shifted_act)
        h_b = self.behav_token.expand(B, T, -1)
        
        for block in self.blocks:
            h_v, h_a, h_b = block(h_v, h_a, h_b)
        
        z_seq = self.norm_final(h_b) 
        
        # --- Outputs ---
        # 1. Predictions
        pred_act = self.action_predictor(h_a)
        pred_vis = self.vision_predictor(h_v)
        
        # 2. Latents for Contrastive Learning
        # Per-timestep retrieval features. Training and memory construction
        # perform masked trajectory pooling outside the model.
        z_global_feat = self.task_head(z_seq) # [B, T, 128]
        
        
        z_local_feat = F.normalize(self.progress_head(z_seq), dim=-1) # [B, T, 128]
        
        return {
            "pred_act": pred_act,
            "pred_vis": pred_vis,
            "gt_vis": target_vis,
            "z_seq": z_seq,             
            "z_global": z_global_feat, # retrieval-key sequence / Global Task Loss
            "z_local": z_local_feat,   
            "logit_scale": self.logit_scale
        }

    def forward_features(self, images, actions):
        """
        [Pi0 Training Interface]
        Inputs: 
            images: [B, T, 3, H, W], float values in [-1, 1]
            actions: [B, T, 7]
        Outputs :
            z_seq: [B, T, D]
        """
        B, T, C, H, W = images.shape
        flat_imgs = images.view(B*T, C, H, W)
        
        # 1. Vision Stream
        online_vis = self.vision_encoder(flat_imgs).view(B, T, -1)
        
        # 2. Action Shift
        act_emb = self.action_proj(actions)
        sos = self.action_sos.expand(B, 1, -1)
        shifted_act = torch.cat([sos, act_emb[:, :-1, :]], dim=1)
        
        # 3. Backbone
        h_v = self.norm_input(online_vis)
        h_a = self.norm_input(shifted_act)
        h_b = self.behav_token.expand(B, T, -1)
        
        for block in self.blocks:
            h_v, h_a, h_b = block(h_v, h_a, h_b)
            
        z_seq = self.norm_final(h_b) # [B, T, D]
        print("BehaviorModel.forward_features output z_seq shape:", z_seq.shape)
        return z_seq

    @torch.no_grad()
    def encode_stateless(self, image, prev_action, has_prev_action):
        """Encode one observation without carrying Mamba state across calls.

        This is the shared local-token path for pi0.5 training and inference.
        ``prev_action`` is action[t-1] in the same quantile-normalized domain
        used to train this BehaviorEncoder.
        """
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"Expected image shape [B, 3, H, W], got {tuple(image.shape)}.")
        if prev_action.ndim != 2 or prev_action.shape[-1] != self.config.action_dim:
            raise ValueError(
                f"Expected prev_action shape [B, {self.config.action_dim}], got {tuple(prev_action.shape)}."
            )

        batch_size = image.shape[0]
        if prev_action.shape[0] != batch_size:
            raise ValueError("image and prev_action batch sizes must match.")

        has_prev_action = has_prev_action.to(device=image.device, dtype=torch.bool).reshape(batch_size)
        vision_feat = self.vision_encoder(image).unsqueeze(1)
        projected_action = self.action_proj(prev_action).unsqueeze(1)
        action_sos = self.action_sos.expand(batch_size, 1, -1)
        action_feat = torch.where(has_prev_action[:, None, None], projected_action, action_sos)

        h_v = self.norm_input(vision_feat)
        h_a = self.norm_input(action_feat)
        h_b = self.behav_token.expand(batch_size, 1, -1)

        for block in self.blocks:
            h_v, h_a, h_b = block(h_v, h_a, h_b)

        return self.norm_final(h_b).squeeze(1)

    @torch.no_grad()
    def step(self, image, prev_action, inference_params=None):
        """
        Inputs:
            image: [B, 3, H, W] (current frame), float values in [-1, 1]
            prev_action: [B, 7] or None (previous frame action. If first frame, pass None)
            inference_params: Mamba Cache Dict
        Outputs:
            z_step: [B, D]
            inference_params: updated Cache
        """
        # Initialize Cache
        if inference_params is None:
            inference_params = InferenceParams(max_seqlen=2048, max_batch_size=image.shape[0]) # Mamba uses dict to store key_value_memory

        # Add time dimension T=1
        if image.ndim == 4:
            image = image.unsqueeze(1) # [B, 1, 3, H, W]
        B = image.shape[0]
        T = 1

        if inference_params.seqlen_offset + T > inference_params.max_seqlen:
            raise ValueError(
                f"Behavior history exceeds max_seqlen={inference_params.max_seqlen}. "
                "Reset the inference state before starting a new episode."
            )

        # 1. Vision
        # LightweightCNN usually accepts [N, C, H, W]
        flat_img = image.view(B, -1, *image.shape[-2:])
        online_vis = self.vision_encoder(flat_img).view(B, T, -1)
        
        # 2. Action (no Shift needed because input is already prev_action)
        if prev_action is None:
            # Use SOS for the first frame
            act_feat = self.action_sos.expand(B, T, -1)
        else:
            if prev_action.ndim == 2:
                prev_action = prev_action.unsqueeze(1) # [B, 1, 7]
            act_feat = self.action_proj(prev_action)
            
        # 3. Backbone Step
        h_v = self.norm_input(online_vis)
        h_a = self.norm_input(act_feat)
        h_b = self.behav_token.expand(B, T, -1)
        
        for block in self.blocks:
            h_v, h_a, h_b = block(h_v, h_a, h_b, inference_params=inference_params)
            
        z_step = self.norm_final(h_b) # [B, 1, D]

        # Mamba only uses its cached incremental path when seqlen_offset > 0.
        # Advance once per behavior timestep, after all streams and layers have
        # consumed the same position.
        inference_params.seqlen_offset += T
        
        return z_step.squeeze(1), inference_params

    @torch.no_grad()
    def get_global_memory_token(self, images, actions):
        """
        [Offline Tool] Used to build Memory Bank
        Input a full expert trajectory, output an aggregated Global Token
        """
        out = self.forward(images, actions)
        z_seq = out['z_seq'] # [1, T, D]
        # Mean Pooling as Global Representation
        return z_seq.mean(dim=1) # [1, D]
