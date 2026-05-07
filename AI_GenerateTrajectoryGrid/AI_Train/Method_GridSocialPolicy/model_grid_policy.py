from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GridSocialPolicyNet(nn.Module):
    def __init__(
        self,
        num_actions: int,
        map_channels: int = 3,
        feature_dim: int = 8,
        base_channels: int = 32,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.map_encoder = nn.Sequential(
            ConvBlock(map_channels, base_channels),
            nn.MaxPool2d(2),
            ConvBlock(base_channels, base_channels * 2),
            nn.MaxPool2d(2),
            ConvBlock(base_channels * 2, base_channels * 4),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        map_dim = base_channels * 4
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Linear(map_dim + hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.stop_head = nn.Linear(hidden_dim, 1)

    def forward(self, grid_map: torch.Tensor, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        map_feat = self.map_encoder(grid_map)
        agent_feat = self.feature_encoder(features)
        fused = self.fusion(torch.cat([map_feat, agent_feat], dim=1))
        action_logits = self.policy_head(fused)
        stop_logits = self.stop_head(fused).squeeze(1)
        return action_logits, stop_logits
