import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from self_healing_pixels.nca import NCA, perception_kernel, seed
from self_healing_pixels.train import GRID, damage_mask, export_json, load_target

ROOT = Path(__file__).resolve().parents[1]


def random_model() -> NCA:
    torch.manual_seed(0)
    model = NCA()
    torch.nn.init.normal_(model.w2.weight, std=0.05)  # zero-init would make every step a no-op
    return model


def random_state(size: int = 24) -> torch.Tensor:
    torch.manual_seed(1)
    x = torch.randn(1, 16, size, size) * 0.5
    x[:, 3] = torch.rand(1, size, size) * 0.3  # mix of alive (>0.1) and dead cells
    return x


def test_seed_is_single_live_cell():
    x = seed(9)
    assert NCA.alive(x).sum() == 9  # the seed and its 8 neighbours count as alive
    assert x[0, :3].abs().sum() == 0 and x[0, 3:, 4, 4].eq(1).all()


def test_dead_cells_stay_empty():
    model = random_model()
    y = model(seed(GRID))
    assert y[0, :, :30, :30].abs().sum() == 0  # far from the seed nothing can appear


def test_rotation_by_90_degrees_swaps_gradients():
    k0, k90 = perception_kernel(1, 0.0), perception_kernel(1, math.pi / 2)
    assert torch.allclose(k90[1], -k0[2], atol=1e-6)  # new d/dx is the old -d/dy
    assert torch.allclose(k90[2], k0[1], atol=1e-6)


def test_target_is_premultiplied_and_padded():
    t = load_target("lizard")
    assert t.shape == (4, GRID, GRID)
    assert (t[:3] <= t[3:4] + 1e-6).all()          # premultiplied: rgb <= alpha
    assert t[:, :16].abs().sum() == 0               # padding is empty


def test_damage_mask_cuts_a_hole():
    m = damage_mask(4, GRID, torch.device("cpu"))
    assert m.shape == (4, 1, GRID, GRID)
    assert ((m == 0).flatten(1).sum(1) > 0).all() and ((m == 1).flatten(1).sum(1) > 0).all()


@pytest.mark.parametrize("angle", [0.0, 0.7])
def test_browser_step_matches_pytorch(tmp_path, monkeypatch, angle):
    """docs/nca.js (TensorFlow.js) must produce the same step as the PyTorch model it was trained as."""
    if not shutil.which("node") or not (ROOT / "node_modules" / "@tensorflow" / "tfjs").exists():
        pytest.skip("needs node and `npm install`")
    model, x = random_model(), random_state()
    monkeypatch.setattr("self_healing_pixels.train.MODELS_OUT", tmp_path)
    weights = json.loads(export_json(model, "test").read_text())
    (tmp_path / "in.json").write_text(json.dumps({
        "model": weights, "angle": angle, "size": x.shape[-1],
        "state": x.permute(0, 2, 3, 1).flatten().tolist(),  # NCHW -> NHWC
    }))
    subprocess.run(["node", str(ROOT / "tests" / "parity.js"), tmp_path / "in.json", tmp_path / "out.json"],
                   check=True, cwd=ROOT)
    js = torch.tensor(json.loads((tmp_path / "out.json").read_text())).reshape(1, x.shape[-1], x.shape[-1], 16)
    with torch.no_grad():
        expected = model(x, fire_rate=1.0, angle=angle).permute(0, 2, 3, 1)
    assert torch.allclose(js, expected, atol=1e-4), (js - expected).abs().max()
