"""Extrator convolucional sem camada classificadora."""

import torch
from torch import nn


class FeatureCNN(nn.Module):
    def __init__(self, features: int = 16):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(128, features, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)
