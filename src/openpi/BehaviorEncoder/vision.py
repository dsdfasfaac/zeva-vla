import torch
import torch.nn as nn
import torchvision.models as models

class LightweightCNN(nn.Module):
    def __init__(self, output_dim=256, pretrained=True):
        super().__init__()
        
        
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)
        
        self.backbone = nn.Sequential(*list(resnet.children())[:-1]) 
        
        self.proj = nn.Linear(512, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.init_weights()

    def init_weights(self):
        nn.init.orthogonal_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0)

    def forward(self, images):
        """
        Args:
            images: [N, 3, 224, 224], float values in [-1, 1]
        Returns:
            features: [N, D]
        """
        feat = self.backbone(images)
        feat = feat.view(feat.size(0), -1) # Flatten -> [N, 512]
        return self.norm(self.proj(feat))
