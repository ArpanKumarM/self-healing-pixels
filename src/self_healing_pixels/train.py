"""Train an NCA to grow an emoji and to regrow it after damage.

Uses the "sample pool" + damage training from the paper: the model is repeatedly asked to continue
from its own earlier outputs (so the pattern stays stable instead of exploding), and some samples
get holes punched in them (so it learns to heal).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .nca import CHANNELS, NCA, seed

TARGET_SIZE = 40
PADDING = 16
GRID = TARGET_SIZE + 2 * PADDING  # 72x72 cells

ROOT = Path(__file__).resolve().parents[2]
EMOJI_DIR = ROOT / "emoji"
CHECKPOINTS = ROOT / "checkpoints"
MODELS_OUT = ROOT / "docs" / "models"


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_target(name: str) -> torch.Tensor:
    """Emoji as premultiplied RGBA, (4, GRID, GRID), in [0, 1]."""
    img = Image.open(EMOJI_DIR / f"{name}.png").convert("RGBA")
    img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.Resampling.LANCZOS)
    a = np.asarray(img, dtype=np.float32) / 255.0
    a[..., :3] *= a[..., 3:4]
    a = np.pad(a, ((PADDING, PADDING), (PADDING, PADDING), (0, 0)))
    return torch.from_numpy(a).permute(2, 0, 1).contiguous()


def damage_mask(n: int, size: int, dev: torch.device) -> torch.Tensor:
    """1 everywhere except a random circle per sample (0 inside)."""
    coords = torch.linspace(-1.0, 1.0, size, device=dev)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    centre = torch.rand(n, 2, 1, 1, device=dev) - 0.5
    radius = 0.1 + 0.3 * torch.rand(n, 1, 1, device=dev)
    inside = ((xx - centre[:, 0]) ** 2 + (yy - centre[:, 1]) ** 2) < radius**2
    return (~inside).float().unsqueeze(1)


def train(name: str, iterations: int, pool_size: int = 1024, batch: int = 8, log_every: int = 250) -> dict:
    dev = device()
    torch.manual_seed(0)
    target = load_target(name).to(dev)
    model = NCA().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[2000], gamma=0.1)
    pool = seed(GRID, batch=pool_size, device=dev)

    history = []
    start = time.time()
    for it in range(1, iterations + 1):
        idx = torch.randint(0, pool_size, (batch,), device=dev)
        x = pool[idx]
        with torch.no_grad():
            # Replace the worst sample with a fresh seed so the model never forgets how to grow.
            per_sample = ((x[:, :4] - target) ** 2).mean(dim=(1, 2, 3))
            order = per_sample.argsort(descending=True)
            idx, x = idx[order], x[order]
            x[:1] = seed(GRID, device=dev)
            # Damage the best-looking samples so the model learns to heal.
            x[-3:] *= damage_mask(3, GRID, dev)

        steps = int(torch.randint(64, 97, (1,)))
        for _ in range(steps):
            x = model(x)
        loss = ((x[:, :4] - target) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        for p in model.parameters():  # per-tensor gradient normalisation, as in the paper
            p.grad /= p.grad.norm() + 1e-8
        opt.step()
        sched.step()
        pool[idx] = x.detach()

        if it % log_every == 0 or it == 1:
            history.append((it, loss.item()))
            print(f"[{name}] iter {it:5d}  loss {loss.item():.5f}  ({time.time() - start:.0f}s)", flush=True)

    CHECKPOINTS.mkdir(exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINTS / f"{name}.pt")
    export_json(model, name)
    return {"name": name, "final_loss": history[-1][1], "seconds": round(time.time() - start), "history": history}


def export_json(model: NCA, name: str) -> Path:
    """Write weights in the layout the browser expects: dense layers over the 48-d perception vector."""
    w1 = model.w1.weight.detach().cpu()[:, :, 0, 0].T  # (3C, hidden)
    b1 = model.w1.bias.detach().cpu()                  # (hidden,)
    w2 = model.w2.weight.detach().cpu()[:, :, 0, 0].T  # (hidden, C)
    rnd = lambda t: [round(v, 6) for v in t.flatten().tolist()]  # noqa: E731
    data = {
        "name": name, "channels": CHANNELS, "hidden": w1.shape[1], "grid": GRID,
        "w1": rnd(w1), "b1": rnd(b1), "w2": rnd(w2),
    }
    MODELS_OUT.mkdir(parents=True, exist_ok=True)
    path = MODELS_OUT / f"{name}.json"
    path.write_text(json.dumps(data, separators=(",", ":")))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train self-healing emoji NCAs.")
    parser.add_argument("names", nargs="*", help="emoji names in emoji/ (default: all)")
    parser.add_argument("--iterations", type=int, default=8000)
    args = parser.parse_args()
    names = args.names or sorted(p.stem for p in EMOJI_DIR.glob("*.png"))
    results = [train(n, args.iterations) for n in names]
    for r in results:
        print(f"{r['name']:>10}: final loss {r['final_loss']:.5f} in {r['seconds']}s")


if __name__ == "__main__":
    main()
