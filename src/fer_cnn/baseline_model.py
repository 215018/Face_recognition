# Proposed_Model_2 baseline CNN using only 48x48 grayscale image input.

import torch
from torch import nn


class ProposedModel2Baseline(nn.Module):
    # Same Proposed_Model_2 backbone, but input channel is 1 instead of 2.
    def __init__(self, num_classes: int = 7):
        super().__init__()

        self.features = nn.Sequential(
            self._conv_block(in_channels=1, out_channels=64, kernel_size=3),
            self._conv_block(in_channels=64, out_channels=128, kernel_size=5),
            self._conv_block(in_channels=128, out_channels=512, kernel_size=3),
            self._conv_block(in_channels=512, out_channels=512, kernel_size=3),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 3 * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.25),
            nn.Linear(512, num_classes),
        )

    def _conv_block(self, in_channels: int, out_channels: int, kernel_size: int):
        padding = kernel_size // 2

        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = self.classifier(x)
        return x