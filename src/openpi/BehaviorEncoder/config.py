from dataclasses import dataclass

@dataclass
class ModelConfig:
    action_dim: int = 7
    d_model: int = 256
    n_layers: int = 4
    dropout: float = 0.1
    ema_decay: float = 0.99
    img_size: int = 224
    vision_pretrained: bool = True
    use_action_reconstruction: bool = True
    use_jepa: bool = True
