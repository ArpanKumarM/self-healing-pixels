"""Growing Neural Cellular Automata (Mordvintsev et al., Distill 2020) in PyTorch.

Every pixel ("cell") holds a 16-channel state: RGBA plus 12 hidden channels. At each step, every
cell looks only at its 3x3 neighbourhood, runs the same tiny network, and updates its state.
A cell is alive if it or any neighbour has alpha > 0.1. Nothing else coordinates the cells, yet
they learn to grow a whole image from one seed pixel and to repair it after damage.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

CHANNELS = 16
HIDDEN = 128
ALIVE_THRESHOLD = 0.1

_IDENTITY = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
_SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0


def perception_kernel(channels: int, angle: float = 0.0) -> torch.Tensor:
    """Depthwise kernel of shape (3*channels, 1, 3, 3): identity, d/dx, d/dy for every channel.

    Rotating the gradient filters by `angle` makes a trained model grow a rotated image, even
    though it never saw rotation during training: cells only know "up" through these filters.
    """
    c, s = math.cos(angle), math.sin(angle)
    dx, dy = _SOBEL_X, _SOBEL_X.T
    kx, ky = c * dx - s * dy, s * dx + c * dy
    kernel = torch.stack([_IDENTITY, kx, ky])            # (3, 3, 3)
    return kernel.repeat(channels, 1, 1).unsqueeze(1)     # (3C, 1, 3, 3), channel-major


class NCA(nn.Module):
    def __init__(self, channels: int = CHANNELS, hidden: int = HIDDEN, fire_rate: float = 0.5):
        super().__init__()
        self.channels, self.fire_rate = channels, fire_rate
        self.w1 = nn.Conv2d(channels * 3, hidden, 1)
        self.w2 = nn.Conv2d(hidden, channels, 1, bias=False)
        nn.init.zeros_(self.w2.weight)  # start as "do nothing" for stable early training
        self.register_buffer("kernel", perception_kernel(channels))

    @staticmethod
    def alive(x: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(x[:, 3:4], 3, stride=1, padding=1) > ALIVE_THRESHOLD

    def perceive(self, x: torch.Tensor, angle: float = 0.0) -> torch.Tensor:
        kernel = self.kernel if angle == 0.0 else perception_kernel(self.channels, angle).to(x.device)
        return F.conv2d(x, kernel, padding=1, groups=self.channels)

    def forward(self, x: torch.Tensor, fire_rate: float | None = None, angle: float = 0.0) -> torch.Tensor:
        pre_alive = self.alive(x)
        dx = self.w2(F.relu(self.w1(self.perceive(x, angle))))
        # Stochastic update: each cell fires independently, so there is no global clock.
        rate = self.fire_rate if fire_rate is None else fire_rate
        mask = (torch.rand_like(x[:, :1]) <= rate).to(x.dtype)
        x = x + dx * mask
        life = (pre_alive & self.alive(x)).to(x.dtype)
        return x * life


def seed(size: int, channels: int = CHANNELS, batch: int = 1, device: str | torch.device = "cpu") -> torch.Tensor:
    """A single live cell in the centre: every non-RGB channel set to 1."""
    x = torch.zeros(batch, channels, size, size, device=device)
    x[:, 3:, size // 2, size // 2] = 1.0
    return x
