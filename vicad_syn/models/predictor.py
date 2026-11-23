"""Independent conditioned heads predicted from explicitly gathered source lights."""

import torch
from torch import nn
from torch.nn import functional as F

from vicad_syn.experiment import check


class Head(nn.Module):
    def __init__(self, channels, hidden, conditioned):
        super().__init__()
        self.input = nn.Linear(channels, hidden)
        self.output = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, channels))
        self.source = nn.Embedding(10, hidden) if conditioned else None
        self.target = nn.Embedding(10, hidden) if conditioned else None
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, contexts, indices, reference):
        projected = self.input(contexts)
        if self.source is not None:
            projected = projected * (1 + self.source(indices).tanh()[None, :, :, None, :])
        pooled = F.gelu(projected).mean(2)
        if self.target is not None:
            pooled = pooled + self.target.weight[None, :, None, :]
        return F.normalize(reference + self.output(pooled) / 4, dim=-1)


class Predictor(nn.Module):
    def __init__(self, channels=768, hidden=96, heads=3, conditioned=True):
        super().__init__()
        self.heads = nn.ModuleList(Head(channels, hidden, conditioned) for _ in range(heads))
        self.register_buffer("sources", (torch.arange(10)[:, None] + torch.arange(1, 10)[None, :]) % 10,
                             persistent=False)

    def indices(self, context):
        choices = {"full": self.sources, "first": self.sources[:, :4], "last": self.sources[:, 4:]}
        return choices[context]

    def forward(self, features, context="full"):
        check(features.ndim == 4 and features.shape[1] == 10, "expected [group, light, patch, channel]")
        indices = self.indices(context)
        contexts = features.index_select(1, indices.flatten()).unflatten(1, indices.shape)
        reference = contexts.mean(2)
        return torch.stack([head(contexts, indices, reference) for head in self.heads])
