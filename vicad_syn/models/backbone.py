"""Frozen final-normalized ViT-B/14 reg4 patch tokens from the fixed model."""

import collections
from collections.abc import Iterable

import torch
from torch import nn
from torch.nn import functional as F

from vicad_syn.experiment import check, digest


class FrozenBackbone(nn.Module):
    def __init__(self, weights, config):
        super().__init__()
        if not hasattr(collections, "Iterable"):
            collections.Iterable = Iterable
        from model.dinomaly_utils.dinov2.models.vision_transformer import vit_base

        check(weights.name == config["backbone"] and digest(weights) == config["backbone_sha256"],
              "pretrained DINOv2 checkpoint hash mismatch")
        self.encoder = vit_base(patch_size=14, img_size=518, block_chunks=0, init_values=1e-8,
                                num_register_tokens=4, interpolate_antialias=False, interpolate_offset=0.1)
        self.encoder.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, images):
        tokens, = self.encoder.get_intermediate_layers(images, n=1)
        check(tuple(tokens.shape[1:]) == (261, 768) and self.encoder.num_register_tokens == 4,
              "expected CLS, four registers, and a 16x16 patch grid")
        return F.normalize(tokens[:, 5:].float(), p=2, dim=-1)
