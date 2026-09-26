"""Render a GIF of a trained NCA growing from one pixel, getting cut in half, and healing."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch

from .nca import NCA, seed
from .train import CHECKPOINTS, GRID, ROOT

ASSETS = ROOT / "assets"


def to_rgb(x: torch.Tensor, scale: int) -> np.ndarray:
    """Premultiplied RGBA state composited onto white, upscaled with nearest-neighbour."""
    rgb, a = x[0, :3], x[0, 3:4].clamp(0, 1)
    img = (1 - a + rgb).clamp(0, 1).permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    return img.repeat(scale, 0).repeat(scale, 1)


@torch.no_grad()
def render(name: str, scale: int = 4, every: int = 2) -> Path:
    torch.manual_seed(0)
    model = NCA()
    model.load_state_dict(torch.load(CHECKPOINTS / f"{name}.pt", map_location="cpu"))
    x = seed(GRID)
    frames = []
    plan = [("grow", 200), ("cut", 0), ("heal", 200), ("slash", 0), ("heal", 200)]
    for phase, steps in plan:
        if phase == "cut":      # delete the right half
            x[..., GRID // 2:] = 0
            frames += [to_rgb(x, scale)] * 8
        elif phase == "slash":  # a diagonal stripe through the middle
            yy, xx = torch.meshgrid(torch.arange(GRID), torch.arange(GRID), indexing="ij")
            x *= ((yy - xx).abs() > 5).float()
            frames += [to_rgb(x, scale)] * 8
        for i in range(steps):
            x = model(x)
            if i % every == 0:
                frames.append(to_rgb(x, scale))
    ASSETS.mkdir(exist_ok=True)
    path = ASSETS / f"{name}.gif"
    iio.imwrite(path, frames, duration=40, loop=0)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render grow-and-heal GIFs for trained models.")
    parser.add_argument("names", nargs="*", help="model names (default: every checkpoint)")
    args = parser.parse_args()
    names = args.names or sorted(p.stem for p in CHECKPOINTS.glob("*.pt"))
    for n in names:
        print(render(n))


if __name__ == "__main__":
    main()
