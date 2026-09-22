"""Exact generator architecture copied from the active original Pix2PixHD method."""

from __future__ import annotations

import torch
import torch.nn as nn


class ResNetBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, 1, 0),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, 1, 0),
            nn.InstanceNorm2d(channels, affine=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.block(value)


class GeneratorNetwork(nn.Module):
    """Pix2PixHD generator only: 3 down, 9 residual, 3 up, RGB Tanh output."""

    def __init__(self, in_channels: int = 3, out_channels: int = 3, n_blocks: int = 9):
        super().__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 64, 7, 1, 0),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.InstanceNorm2d(256, affine=True),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.InstanceNorm2d(512, affine=True),
            nn.ReLU(True),
        ]
        for _ in range(n_blocks):
            model.append(ResNetBlock(512))
        model.extend([
            nn.ConvTranspose2d(512, 256, 3, 2, 1, output_padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 3, 2, 1, output_padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, out_channels, 7, 1, 0),
            nn.Tanh(),
        ])
        self.model = nn.Sequential(*model)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.model(value)
