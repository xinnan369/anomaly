"""Smoke test real frozen multi-layer features and the local Synergy path on CUDA."""

import argparse
from pathlib import Path

import torch

from vicad_refinement.data import load_groups
from vicad_refinement.model import MultiLayerBackbone, patch_weights
from vicad_syn.data.cache import Images
from vicad_syn.experiment import check, pinned_upstream, protocol, read_json, write_json
from vicad_syn.models.backbone import FrozenBackbone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=lambda x: Path(x).resolve(), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    base = protocol()["config"]
    registry = read_json(Path(__file__).resolve().parents[2] / "vicad_syn/refinement/protocol.json")
    group = load_groups(args.inputs, "Car", base, (30, 70), "train")[0]
    input_source = Images(args.inputs / "dataset", group.paths)
    sample = torch.stack([input_source[i] for i in range(2)])
    results = {}
    with pinned_upstream(args.inputs / "inputs/upstream", base["upstream_revision"]):
        weights = args.inputs / "inputs/weights" / base["backbone"]
        parent = FrozenBackbone(weights, base)
        encoder = MultiLayerBackbone(weights, base, registry["feature_blocks_one_based"])
        for name in ("cpu", "cuda"):
            device = torch.device(name)
            pixels = sample.to(device)
            official = parent.to(device)(pixels)
            multi = encoder.to(device).train()(pixels)
            check(not multi.requires_grad and not encoder.training and
                  not any(p.requires_grad or p.grad is not None for p in encoder.parameters()), "unfrozen DINOv2")
            check(multi.shape == (2, 3, 256, 768) and bool(multi.isfinite().all()), "wrong extracted grid")
            torch.testing.assert_close(multi[:, 2], official, atol=0, rtol=0)
            check(not torch.equal(multi[:, 1], multi[:, 2]), "intermediate and final layers coincide")
            patch_score = torch.tensor([[[0., 3.], [float("nan"), 0.]]], device=device)
            check(bool(patch_weights(patch_score, ()).isfinite().all()), "nonfinite CUDA fusion weights")
            results[name] = {"shape": list(multi.shape), "frozen": True, "exact_final_layer_equivalence": True}
    write_json(args.output, {"status": "PASS", "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
                            "encoder_parameters": sum(p.numel() for p in encoder.parameters()), "results": results})


if __name__ == "__main__":
    main()
