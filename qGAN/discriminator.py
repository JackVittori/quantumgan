import torch
from torch import nn
from typing import Tuple


class Discriminator(nn.Module):
    """Fully connected classical discriminator pytorch module."""

    def __init__(self, image_size: Tuple[int, int] = (28,28)):
        """
        Build Discriminator architecture.
         Args:
             image_size (Tuple[int,int]): image size. Example (28,28) for a 64x64 image
        """
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(image_size[0] * image_size[1], 64),
            nn.ReLU(),
            # First hidden layer (64 -> 16)
            nn.Linear(64, 16),
            nn.ReLU(),
            # Second hidden layer (16 -> output)
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
