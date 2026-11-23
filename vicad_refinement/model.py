"""Layer-local normal models and same-view, leave-one-light-out scoring."""

import torch
from torch.nn import functional as F

from vicad_syn.experiment import check
from vicad_syn.fusion.weights import light_weights
from vicad_syn.models.backbone import FrozenBackbone
from vicad_syn.uncertainty.signals import context_instability, epistemic_uncertainty, squared_distance


class MultiLayerBackbone(FrozenBackbone):
    def __init__(self, weights, config, blocks):
        super().__init__(weights, config)
        self.blocks = tuple(block - 1 for block in blocks)

    @torch.no_grad()
    def forward(self, images):
        layers = self.encoder.get_intermediate_layers(images, n=self.blocks)
        check(len(layers) == len(self.blocks) and self.encoder.num_register_tokens == 4, "incorrect block output")
        check(all(layer.shape[1:] == (261, 768) for layer in layers), "incorrect token layout")
        return torch.stack([F.normalize(layer[:, 5:].float(), dim=-1) for layer in layers], dim=1)


def normalize(x):
    return F.normalize(x.float(), dim=-1)


def sources(x):
    lights = x.shape[1]
    check(lights == 10, "expected all ten ordered illumination features")
    indices = (torch.arange(lights, device=x.device)[:, None] +
               torch.arange(1, lights, device=x.device)[None]) % lights
    return x[:, indices]


def estimate_prototypes(features, views, view_count=12):
    """Mean of normal fit instances only, independently at each view, light and patch."""
    result = []
    for view in range(view_count):
        x = features[views == view]
        check(len(x) > 0, f"no normal fit features at view {view}")
        # Chunk over objects to avoid a multi-GiB FP32 temporary.
        centroid = sum((normalize(part).sum(0) for part in x.split(4)), start=0)
        result.append(normalize(centroid))
    return torch.stack(result)


def residuals(features, model, reference):
    """All candidates share the official final-layer predictor and coordinate system."""
    x = normalize(features)
    leaf = x[:, :, -1]
    heads = model(leaf)
    primary = squared_distance(leaf, normalize(heads.mean(0)))
    intermediate = [squared_distance(layer, normalize(sources(layer).mean(2)))
                    for layer in x.unbind(dim=2)[:-1]]
    multi = torch.stack([*intermediate, primary]).mean(0)
    per_layer_normal = squared_distance(x, reference)
    normal = per_layer_normal.mean(2)
    return (
        {"A_final": primary, "B_multilayer": multi, "C_normal": normal,
         "D_combined": (multi + normal) / 2},
        context_instability(model(leaf, "first"), model(leaf, "last")),
        epistemic_uncertainty(heads), per_layer_normal,
    )


def context_predictions(leaf, model, prior):
    nine = sources(leaf)
    return {
        "nine_direct": normalize(model(leaf).mean(0)),
        "normal_conditional": prior,
        "nine_median": normalize(nine.median(dim=2).values),
        "four_subset": normalize(model(leaf, "first").mean(0)),
    }


def patch_weights(maps, signals):
    valid = maps.isfinite()
    for signal in signals:
        valid = valid & signal.isfinite()
    logits = torch.zeros_like(maps)
    count = valid.sum(1, keepdim=True).clamp_min(1)
    for signal in signals:
        clean = signal.masked_fill(~valid, 0).clamp_min(0)
        maximum = clean.amax(1, keepdim=True)
        clean = clean / maximum.masked_fill(maximum == 0, 1)
        typical = clean.sum(1, keepdim=True) / count
        logits -= clean / typical.clamp_min(1e-8)
    maximum = logits.masked_fill(~valid, -torch.inf).amax(1, keepdim=True)
    positive = (logits - maximum.masked_fill(~valid.any(1, keepdim=True), 0)).masked_fill(~valid, -torch.inf).exp()
    return positive / positive.sum(1, keepdim=True).clamp_min(1e-8)


def weighted_lights(maps, stability, epistemic, mode, protection):
    """Return contributions scaled by ten for the unmodified official mean fusion."""
    clean = maps.masked_fill(~maps.isfinite(), 0)
    if mode == "equal":
        return 10 * clean * patch_weights(maps, ())
    if mode == "v1_global":
        return (10 * light_weights(maps, (stability, epistemic))[:, :, None]) * clean
    signals = {"patch_both": (stability, epistemic), "protected_both": (stability, epistemic),
               "patch_stability": (stability,), "patch_epistemic": (epistemic,)}[mode]
    weights = patch_weights(maps, signals)
    if mode == "protected_both":
        # A deterministic equal-share blend over valid lights bounds suppression.
        valid_maps = maps.masked_fill(~weights.gt(0), torch.nan)
        weights = (1 - protection) * weights + protection * patch_weights(valid_maps, ())
    return (10 * weights) * clean


def upsample(maps):
    return F.interpolate(maps.reshape(-1, 1, 16, 16), (224, 224), mode="bilinear",
                         align_corners=False)[:, 0]
