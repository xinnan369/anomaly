
"""Runtime-only denominator guard and accounting for the pinned Dinomaly attention."""

from hashlib import sha256
import inspect
from pathlib import Path
import textwrap

import torch


EPSILON = 1e-6
IDENTITY = "Dinomaly-stabilized"
UPSTREAM_ATTENTION = "model/dinomaly_utils/vision_transformer.py"
ORIGINAL = "z = 1.0 / torch.einsum('...sd,...d->...s', q, k.sum(dim=-2))"
PATCH = ("denom = torch.einsum('...sd,...d->...s', q, k.sum(dim=-2))\n"
         "    z = _vicad_stabilized_reciprocal(self, denom)")


def source_sha256():
    return sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def metadata(mode):
    if mode == "official":
        return {"attention_mode": mode}
    if mode != "stabilized":
        raise ValueError(f"Unknown attention mode: {mode}")
    return {"attention_mode": mode, "model_identity": IDENTITY, "epsilon": EPSILON,
            "stability_patch_sha256": source_sha256()}


def stabilized_reciprocal(denominator):
    return 1.0 / denominator.clamp_min(EPSILON)


class GuardCounts:
    def __init__(self, batches, device):
        self.counts = torch.zeros((batches, 8, 4), dtype=torch.int64, device=device)
        self.calls = 0
        self.layers = {}

    def bind(self, net, attention_class):
        self.layers = {layer: (index, name) for index, (name, layer) in enumerate(
            (item for item in net.named_modules() if type(item[1]) is attention_class))}
        if len(self.layers) != 8:
            raise ValueError(f"Unexpected Dinomaly decoder layout: {len(self.layers)} layers")

    def reciprocal(self, module, denominator):
        batch, layer = divmod(self.calls, len(self.layers))
        if self.layers[module][0] != layer:
            raise ValueError("Pinned decoder attention call order changed")
        with torch.no_grad():
            value = denominator.detach()
            self.counts[batch, layer, 0] = torch.count_nonzero(value < EPSILON)
            self.counts[batch, layer, 1] = torch.count_nonzero(value == 0)
            self.counts[batch, layer, 2] = torch.count_nonzero(value < 0)
            self.counts[batch, layer, 3] = torch.count_nonzero(~value.isfinite())
        self.calls += 1
        return stabilized_reciprocal(denominator)

    def assert_valid_batch(self):
        if self.calls % 8 or not self.calls:
            raise ValueError("Incomplete decoder attention batch")
        batch = self.calls // 8 - 1
        if bool(self.counts[batch, :, 2:].any()):
            raise ValueError(f"non-finite or negative original attention denominator at batch {batch + 1}")

    def record(self):
        batches = (self.calls + 7) // 8
        counts = self.counts[:batches].cpu()
        layers = [name for _, name in self.layers.values()]
        totals = counts.sum(dim=0).tolist()
        return {
            "model_identity": IDENTITY, "epsilon": EPSILON, "patch_sha256": source_sha256(),
            "attention_calls": self.calls, "batches_observed": batches,
            "protected_positions": sum(row[0] for row in totals),
            "zero_positions": sum(row[1] for row in totals),
            "negative_positions": sum(row[2] for row in totals),
            "nonfinite_denominator_positions": sum(row[3] for row in totals),
            "by_layer": [{"layer": name, "protected": row[0], "zero": row[1],
                          "negative": row[2], "nonfinite": row[3]}
                         for name, row in zip(layers, totals, strict=True)],
            "batches_with_protection": [
                {"batch": batch + 1, "epoch": batch // 300 + 1,
                 "layers": [{"layer": name, "protected": row[0], "zero": row[1]}
                            for name, row in zip(layers, counts[batch].tolist(), strict=True) if row[0]]}
                for batch in counts[:, :, 0].any(dim=1).nonzero().flatten().tolist()
            ],
        }


def install(attention_class, counter, resources):
    """Derive the alternate forward from the actual upstream source using one guarded edit."""
    original_forward = attention_class.forward
    original_source = textwrap.dedent(inspect.getsource(original_forward))
    if original_source.count(ORIGINAL) != 1:
        raise ValueError("Expected one unmodified official reciprocal in LinearAttention2")
    amended = original_source.replace(ORIGINAL, PATCH)
    scope = {**original_forward.__globals__, "_vicad_stabilized_reciprocal": counter.reciprocal}
    exec(compile(amended, "<Dinomaly-stabilized: LinearAttention2.forward>", "exec"), scope)
    attention_class.forward = scope["forward"]
    resources.callback(setattr, attention_class, "forward", original_forward)
    return amended
