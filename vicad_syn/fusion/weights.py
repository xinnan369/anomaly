"""Group-local reliability and finite same-view multi-light fusion."""

import torch


def light_weights(residual, signals, epsilon=1e-8, temperature=1.0):
    valid = residual.isfinite().flatten(2).all(-1)
    for signal in signals:
        valid = valid & signal.isfinite().flatten(2).all(-1)
    logits = torch.zeros_like(valid, dtype=residual.dtype)
    for signal in signals:
        clean = torch.where(valid.unsqueeze(-1), signal.flatten(2), 0).clamp_min(0)
        maximum = clean.amax((1, 2), keepdim=True)
        clean = clean / torch.where(maximum > 0, maximum, 1)
        values = clean.mean(-1)
        typical = values.sum(1, keepdim=True) / valid.sum(1, keepdim=True).clamp_min(1)
        logits -= values / (typical + epsilon) / temperature
    # An all-invalid group has zero reliability and produces a finite zero map.
    centered = logits - torch.where(valid.any(1, keepdim=True), logits.masked_fill(~valid, -torch.inf).amax(1, keepdim=True), 0)
    positive = torch.where(valid, centered, -torch.inf).exp()
    return positive / positive.sum(1, keepdim=True).clamp_min(epsilon)


def fuse_lights(residual, weights):
    usable = torch.where(residual.isfinite(), residual, 0)
    expanded = weights[(...,) + (None,) * (residual.ndim - 2)]
    scale = usable.abs().amax(1, keepdim=True)
    normalized = usable / torch.where(scale > 0, scale, 1)
    return (normalized * expanded).sum(1).clamp(-1, 1) * scale.squeeze(1)
