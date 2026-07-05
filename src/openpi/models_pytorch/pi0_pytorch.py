import logging
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
from openpi.BehaviorEncoder.model import BehaviorModel
from openpi.BehaviorEncoder.config import ModelConfig
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing

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
    
class ActionPriorNetwork(nn.Module):    
    def __init__(
            self, 
            behavior_dim: int, 
            action_dim: int, 
            hidden_dim: int = 512, 
            num_waypoints: int = 8,
            action_horizon: int = 10,   
        ):
        super().__init__()

        self.num_waypoints = num_waypoints
        self.action_horizon = action_horizon
        self.action_dim = action_dim

        self.temporal_waypoint = nn.Sequential(
            nn.Linear(behavior_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, behavior_dim * num_waypoints),
        )
        self.pos_emb = nn.Parameter(torch.randn(1, num_waypoints, behavior_dim) * 0.02)
        
        self.attention = nn.MultiheadAttention(embed_dim=behavior_dim, num_heads=4, batch_first=True)
        self.norm_global = nn.LayerNorm(behavior_dim)
        self.norm_local = nn.LayerNorm(behavior_dim)

        self.dist_head = nn.Sequential(
            nn.Linear(behavior_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_horizon * action_dim * 2),  # Predict mean and log-std
        )

        nn.init.normal_(self.dist_head[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.dist_head[-1].bias)


    def forward(self, z_global: torch.Tensor, z_local: torch.Tensor):
        B = z_global.size(0)

        waypoints = self.temporal_waypoint(z_global)  # (B, behavior_dim * num_waypoints)
        waypoints = waypoints.view(B, self.num_waypoints, -1)  # (B, num_waypoints, behavior_dim)
        waypoints = waypoints + self.pos_emb  # Add positional embedding

        q = self.norm_local(z_local).unsqueeze(1)  # (B, 1, behavior_dim)

        k = v = self.norm_global(waypoints)  # (B, num_waypoints, behavior_dim)
        context, _ = self.attention(q, k, v)  # (B, 1, behavior_dim)
        context = context.squeeze(1)  # (B, behavior_dim)

        out = self.dist_head(context).view(B, self.action_horizon, self.action_dim, 2) 
        mu = out[..., 0]  # (B, action_horizon, action_dim)
        log_std = out[..., 1]  # (B, action_horizon, action_dim)

        log_std = torch.clamp(log_std, min=-5, max=2)  # Prevent extreme std values
        std = torch.exp(log_std)

        return torch.distributions.Normal(mu, std)

def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype

def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)

def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))

def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks

class PI0Pytorch(nn.Module):
    def __init__(
        self, 
        config,
    ):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )

        self.action_in_proj = nn.Linear(32, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, 32)

        self.use_behavior = config.use_behavior
        self.use_apn = config.use_apn
        self.behavior_dim = config.behavior_dim
        self.action_horizon = config.action_horizon

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(32, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")
        self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None


        # self.is_training = True
        # ---- Behavior ----
        # Project the behavior global token to time embedding space
        if self.use_behavior:
            memory_bank_path = config.memory_bank_path

            retrieval_ckpt = config.retrieval_ckpt
            behavior_encoder_ckpt = config.behavior_encoder_ckpt    

            # Global -> VLM Prefix(Prompt)
            self.global_projector = nn.Linear(self.behavior_dim, 2048)

            # 12.19 Action Prior Network
            if self.use_apn:
                self.real_action_dim = 7 # 7 for libero
                self.action_prior_network = ActionPriorNetwork(
                    behavior_dim=self.behavior_dim,
                    action_dim=self.real_action_dim,
                    hidden_dim=512,
                    num_waypoints=8,
                    action_horizon=config.action_horizon,
                )
                self.prior_emb_proj = nn.Linear(self.real_action_dim, action_expert_config.width)
                nn.init.zeros_(self.prior_emb_proj.weight)  
                nn.init.zeros_(self.prior_emb_proj.bias)

                self.prior_loss_scale = 0.01

            # Load memory bank for fintuning
            if memory_bank_path is not None:
                print(f"Loading Memory Bank from {memory_bank_path}...")
                bank_data = torch.load(memory_bank_path, map_location="cpu")

                if not bank_data:
                    raise ValueError(f"Memory bank is empty: {memory_bank_path}")

                def get_memory_field(item, field):
                    if field in item:
                        return item[field].squeeze()
                    if "embedding" in item:
                        # Legacy banks used one tensor as both retrieval key and value.
                        return item["embedding"].squeeze()
                    raise KeyError(f"Memory entry is missing '{field}': {item.keys()}")

                max_id = max(item['episode_idx'] for item in bank_data)
                retrieval_keys = torch.stack(
                    [get_memory_field(item, "retrieval_key") for item in bank_data]
                )
                behavior_values = torch.stack(
                    [get_memory_field(item, "behavior_value") for item in bank_data]
                )

                value_dim = behavior_values.shape[-1]
                if value_dim != self.behavior_dim:
                    raise ValueError(
                        f"Memory behavior_value dim ({value_dim}) does not match "
                        f"behavior_dim ({self.behavior_dim})"
                    )

                self.register_buffer(
                    "global_token_storage",
                    torch.zeros((max_id + 1, value_dim)),
                    persistent=False,
                )

                for item, behavior_value in zip(bank_data, behavior_values, strict=True):
                    episode_idx = item['episode_idx']
                    self.global_token_storage[episode_idx] = behavior_value

                self.register_buffer("memory_keys", retrieval_keys, persistent=False)
                self.register_buffer("memory_values", behavior_values, persistent=False)

                print(
                    f"Memory Bank Tensors: keys={self.memory_keys.shape}, "
                    f"values={self.memory_values.shape}, storage={self.global_token_storage.shape}"
                )
                print(f"Loaded Memory Bank with {len(bank_data)} entries.")

            # Load pretrained Projection Head for retrieval
            if retrieval_ckpt is not None and config.is_training == False:
                print(f"Loading Projection Head from {retrieval_ckpt}...")
                if not hasattr(self, "memory_keys"):
                    raise ValueError("A memory bank is required when loading a retrieval head.")
                self.projector = ProjectionHead(
                    input_dim=2048,
                    output_dim=self.memory_keys.shape[-1],
                    hidden_dim=1024,
                )
                checkpoint = torch.load(retrieval_ckpt, map_location="cpu")
                self.projector.load_state_dict(checkpoint)
                for param in self.projector.parameters():
                    param.requires_grad = False
                self.projector.eval()
                print("Loaded Projection Head.")
            
            # The same frozen BehaviorEncoder is used to compute stateless local
            # tokens during both pi0.5 training and inference.
            if behavior_encoder_ckpt is not None:
                print(f"Loading Behavior Encoder from {behavior_encoder_ckpt}...")
                behavior_config = ModelConfig(
                    action_dim=7, 
                    d_model=256, 
                    n_layers=4,
                    dropout=0.0, 
                    ema_decay=0.99,
                    vision_pretrained=False,
                )
                self.behavior_encoder = BehaviorModel(config=behavior_config)
                checkpoint = torch.load(behavior_encoder_ckpt, map_location="cpu")
                action_normalization = checkpoint.get("action_normalization", {})
                if action_normalization.get("type") != "quantile":
                    raise ValueError(
                        "Stateless local tokens require a BehaviorEncoder trained with quantile action normalization."
                    )
                self.behavior_encoder.load_state_dict(checkpoint['model_state_dict'])
                for param in self.behavior_encoder.parameters():
                    param.requires_grad = False
                self.behavior_encoder.eval()
                print("Loaded Behavior Encoder.")

    @torch.no_grad()
    def encode_local_token(self, image, prev_action=None, has_prev_action=None):
        if not hasattr(self, "behavior_encoder"):
            raise RuntimeError("A BehaviorEncoder checkpoint is required to compute local tokens.")

        if image.ndim != 4:
            raise ValueError(f"Expected a batched behavior image, got shape {tuple(image.shape)}.")
        if image.shape[-1] == 3:
            image = image.permute(0, 3, 1, 2)
        elif image.shape[1] != 3:
            raise ValueError(f"Behavior image must be BHWC or BCHW, got shape {tuple(image.shape)}.")

        image = image.contiguous().to(dtype=torch.float32)
        batch_size = image.shape[0]
        action_dim = self.behavior_encoder.config.action_dim

        if prev_action is None:
            prev_action = torch.zeros((batch_size, action_dim), dtype=torch.float32, device=image.device)
            has_prev_action = torch.zeros(batch_size, dtype=torch.bool, device=image.device)
        else:
            prev_action = prev_action[..., :action_dim].to(device=image.device, dtype=torch.float32)
            if has_prev_action is None:
                has_prev_action = torch.ones(batch_size, dtype=torch.bool, device=image.device)
            else:
                has_prev_action = has_prev_action.to(device=image.device, dtype=torch.bool)

        # Parent model.train() would otherwise update the frozen ResNet BatchNorm statistics.
        self.behavior_encoder.eval()
        return self.behavior_encoder.encode_stateless(image, prev_action, has_prev_action)

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        if train == True:
            return (
                list(observation.images.values()),
                list(observation.image_masks.values()),
                observation.tokenized_prompt,
                observation.tokenized_prompt_mask,
                observation.state,
                observation.episode_index,
                observation.frame_index,
            )
        else:
            return (
                list(observation.images.values()),
                list(observation.image_masks.values()),
                observation.tokenized_prompt,
                observation.tokenized_prompt_mask,
                observation.state,
            )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks, z_global=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Process behavior global token
        if z_global is not None and self.use_behavior:
            logging.info("Using behavior global token for prefix embedding.")
            global_embed = self.global_projector(z_global)
            global_embed = global_embed.unsqueeze(1)  # (B, 1, D)
            embs.append(global_embed)
            bs = global_embed.shape[0]
            valid_mask = torch.ones((bs, 1), dtype=torch.bool, device=global_embed.device)
            pad_masks.append(valid_mask)
            att_masks += [0]

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)

            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep, prior_action=None):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if prior_action is not None and self.use_behavior:
            def prior_emb_proj_func(prior_action):
                return self.prior_emb_proj(prior_action)

            prior_emb = self._apply_checkpoint(prior_emb_proj_func, prior_action)
            if self.training:
                # === Dropout ===
                keep_prob = 0.6
                mask = torch.bernoulli(torch.full((prior_emb.shape[0], 1, 1), keep_prob, device=prior_emb.device))
                prior_emb = prior_emb * mask
                logging.info(f"Applied prior action dropout with keep_prob={keep_prob} for training")
            else:
                # === Guidance Scale ===
                guidance_scale = 0.5 
                prior_emb = prior_emb * guidance_scale
                logging.info(f"Applied prior action guidance scaling with scale={guidance_scale} for inference")

            action_emb = action_emb + prior_emb

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)
            
            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        z_local = None
        if self.use_behavior and self.use_apn and observation.previous_action is not None:
            z_local = self.encode_local_token(
                observation.images["base_0_rgb"],
                observation.previous_action,
                observation.has_previous_action,
            )
            z_local = z_local + 0.01 * torch.randn_like(z_local)
        elif self.use_behavior and self.use_apn and self.config.is_training:
            raise ValueError(
                "Stateless local-token training requires previous_action and has_previous_action in Observation."
            )

        images, img_masks, lang_tokens, lang_masks, state, episode_index, frame_index = self._preprocess_observation(observation, train=True)
        
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # Retrieve behavior global token from memory bank
        z_global = None
        if episode_index is not None and self.use_behavior:
            z_global = self.global_token_storage[episode_index].detach()

        # Compute action prior
        prior_loss = torch.tensor(0.0, device=actions.device)
        prior_action_mean = None
        if self.use_behavior and self.use_apn and z_global is not None and z_local is not None:
            # prior_dist = self.action_prior_network(z_global, z_local, state.to(torch.float32))
            prior_dist = self.action_prior_network(z_global, z_local)
            gt_actions = actions[:, :self.action_horizon, :self.real_action_dim]  # Use only the real action dimensions
            logp = prior_dist.log_prob(gt_actions)
            nll = -logp.sum(dim=-1)  # Sum over action dimensions
            prior_loss = nll.mean() * self.prior_loss_scale
            logging.info(f"Prior loss: {prior_loss.item()}")
            prior_action_mean = prior_dist.loc

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks, z_global=z_global)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time, prior_action=prior_action_mean)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Apply gradient checkpointing if enabled
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        # Apply gradient checkpointing to final action projection if enabled
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        return F.mse_loss(u_t, v_t, reduction="none") + prior_loss
        # return F.mse_loss(u_t, v_t, reduction="none")

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10, z_global=None, z_local=None) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state  = self._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks, z_global=z_global)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        prior = None
        if self.use_behavior and self.use_apn and z_global is not None and z_local is not None:
            prior_dist = self.action_prior_network(z_global, z_local)
            prior = prior_dist.loc

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
                prior=prior,
            )

            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
        prior,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep, prior_action=prior)
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)

    @torch.no_grad()
    def extract_vlm_features(self, observation):
        """
        Source Feature Extraction for Retrieval Memory Bank
        """
        # 1. Preprocess
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        # 2. Embedding Prefix (VLM Encoder)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )

        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):  
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        # 3. Construct Attention Masks
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks).to(dtype=prefix_embs.dtype)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # 4. VLM Transformer Forward
        model_output, _ = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None], 
            use_cache=False
        )
        
        prefix_output = model_output[0].to(dtype=torch.float32) # [Batch, Seq_Len, Hidden_Dim]

        # 5. EOS Token Pooling
        text_lens = lang_masks.sum(dim=1).long() 
        img_len = prefix_embs.shape[1] - lang_tokens.shape[1]
        last_token_indices = img_len + text_lens - 1
        
        indices = last_token_indices.view(-1, 1, 1).expand(-1, -1, prefix_output.size(-1))
        
        vlm_feature = torch.gather(prefix_output, 1, indices).squeeze(1)
        
        return vlm_feature
    
    def _retrieve_global_token(self, vlm_feat, k=5):
        """Retrieve behavior values using a VLM query in retrieval-key space."""
        if self.use_behavior:
            query = F.normalize(self.projector(vlm_feat), dim=-1)
            keys = F.normalize(self.memory_keys, dim=-1)
            scores = torch.matmul(query, keys.T)

            k = min(k, scores.shape[1])
            topk_scores, best_indices = torch.topk(scores, k=k, dim=1)

            # Retrieve rich behavior values using indices selected in key space.
            retrieved_values = self.memory_values[best_indices]
            weights = F.softmax(topk_scores, dim=1)

            z_global = torch.sum(weights.unsqueeze(-1) * retrieved_values, dim=1)
            return z_global
        else:
            return None
