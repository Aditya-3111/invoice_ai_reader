import torch
import torch.nn as nn
from torchvision import models

class CNNEncoder(nn.Module):
    def __init__(self, output_dim=512):
        super(CNNEncoder, self).__init__()

        base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])

        self.fc = nn.Linear(2048, output_dim)

    def forward(self, images):
        """
        images: Tensor [B, 3, H, W]
        """
        features = self.feature_extractor(images)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        return features
