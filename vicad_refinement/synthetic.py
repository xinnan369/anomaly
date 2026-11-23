"""Fixed photometric control and structure perturbations from held-out normal images."""

import numpy as np
import torch
from torch.nn import functional as F


def perturb(images, scenario, seed, protocol):
    """Return transformed normal images, changed light indices, and a synthetic mask."""
    result = images.clone()
    size = images.shape[-1]
    mask = torch.zeros((size, size), dtype=torch.bool)
    rng = np.random.default_rng(seed)
    start = int(rng.integers(0, 10))
    if scenario == "clean":
        return result, [], mask
    if scenario == "light_only":
        result[start] += 0.28
        return result, [start], mask
    # One weak quarter-patch-scale change and a larger approximately patch-scale change.
    weak = scenario.endswith("weak")
    width = int(rng.integers(8, 15) if weak else rng.integers(20, 31))
    y, x = (int(v) for v in rng.integers(size // 4, size * 3 // 4 - width, size=2))
    mask[y:y + width, x:x + width] = True
    n = {"one_weak": 1, "four_strong": 4, "all_weak": 10, "all_strong": 10}[scenario]
    lights = [(start + index * 3) % 10 for index in range(n)]
    alpha = protocol["alpha_weak" if weak else "alpha_strong"]
    # Same local structural transform at the same coordinates across affected lights.
    crop = images[lights, :, y:y + width, x:x + width]
    rows, cols = torch.meshgrid(torch.arange(width), torch.arange(width), indexing="ij")
    pattern = ((rows // 3 + cols // 3) % 2).to(crop).mul(2).sub(1)
    texture = crop.flip((-1,)) + pattern * 0.85
    result[lights, :, y:y + width, x:x + width] = crop.lerp(texture, alpha)
    return result, lights, mask


def patch_mask(mask):
    return F.max_pool2d(mask.float()[None, None], 14, 14).flatten().bool()
