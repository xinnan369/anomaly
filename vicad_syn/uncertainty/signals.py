"""Distinct variation over context subsets and over independently trained heads."""

import torch


def squared_distance(first, second):
    return (first - second).square().sum(-1)


def context_instability(first, last):
    """Two deterministic, disjoint contexts, compared within each fixed head."""
    return squared_distance(first, last).mean(0) / 4


def epistemic_uncertainty(full):
    """Population dispersion across independent heads given identical full context."""
    return squared_distance(full, full.mean(0)).mean(0)


def inference_signals(predictor, features):
    full = predictor(features)
    stability = context_instability(predictor(features, "first"), predictor(features, "last"))
    epistemic = epistemic_uncertainty(full)
    residual = squared_distance(features, torch.nn.functional.normalize(full.mean(0), dim=-1))
    return residual, stability, epistemic
