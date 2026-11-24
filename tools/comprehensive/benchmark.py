"""Matched local scoring cost for the prototype and frozen refinement on one object."""
import argparse
from pathlib import Path
from statistics import median
from time import perf_counter

import torch

from tools.comprehensive.ablation import model_from_frozen
from tools.comprehensive.offline import enable_offline
from tools.comprehensive.resources import FREEZE, REPO, UPSTREAM
from vicad_refinement.data import as_tensor, load_features, load_groups
from vicad_refinement.model import MultiLayerBackbone, normalize, residuals, upsample, weighted_lights
from vicad_syn.data.cache import Images
from vicad_syn.evaluation.synergy import VIEWS
from vicad_syn.experiment import check, digest, pinned_upstream, read_json, write_json
from vicad_syn.uncertainty.signals import squared_distance


def scoring(feature, reference, model, name):
    if name == "B2":
        response = squared_distance(normalize(feature), reference).mean(2)
        return weighted_lights(response, response, response, "equal", .25)
    scores, stability, epistemic, _ = residuals(feature, model, reference)
    return weighted_lights(scores["D_combined"], stability, epistemic, "patch_stability", .25)


def repetitions(operation, count):
    operation()
    torch.cuda.synchronize()
    seconds, peaks = [], []
    for _ in range(count):
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        start = perf_counter()
        output = operation()
        torch.cuda.synchronize()
        seconds.append(perf_counter() - start)
        peaks.append(torch.cuda.max_memory_allocated() - baseline)
        check(bool(output.isfinite().all()), "non-finite cost probe")
    return {"seconds": seconds, "median_seconds": median(seconds), "peak_additional_allocated_bytes": max(peaks)}


@torch.inference_mode()
def run(args):
    ticket = args.frozen / "freeze.json"
    check(digest(ticket) == FREEZE, "TASK-004 freeze changed")
    base = read_json(ticket)["binding"]["inherited_v1"]["config"]
    folder = args.frozen / "confirmatory/Bird"
    groups = load_groups(args.bird, "Bird", base, (30, 70), "test")
    indices = [i for i, group in enumerate(groups) if group.identity == groups[0].identity]
    check(len(indices) == 12, "expected one full physical sample")
    checkpoint = folder / "seed-42"
    receipt = read_json(checkpoint / "training.json")
    reference_device = torch.device("cuda:0")
    model, prototype = model_from_frozen(checkpoint, receipt, "Bird", base, reference_device)
    cached = load_features(folder / "test-features")
    weight = args.initial / "inputs/weights" / base["backbone"]
    encoder = MultiLayerBackbone(weight, base, (4, 8, 12)).to(reference_device)
    result = {"sample_selection": "first physical hash in canonical test order; no performance filtering",
              "category": "Bird", "seed": 42, "checkpoint_sha256": receipt["checkpoint_sha256"],
              "scope": "raw end-to-end includes serial PNG reads and frozen 4/8/12 DINO features; both paths end at fused score map",
              "conditions": "same process and GPU; workstation can be shared; this is a latency probe, not an isolated hardware benchmark",
              "frozen_encoder_parameters": sum(p.numel() for p in encoder.parameters()),
              "encoder_weight_file_bytes": weight.stat().st_size,
              "predictor_parameters": sum(p.numel() for p in model.parameters()),
              "predictor_tensor_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
              "bn_or_other_parameters": 0,
              "prototype_bytes": prototype.numel() * prototype.element_size(),
              "stored_checkpoint_bytes": (checkpoint / "checkpoint.pt").stat().st_size,
              "stored_train_seconds_B6": receipt["seconds"], "stored_train_peak_cuda_bytes_B6": receipt["peak_cuda_bytes"],
              "probes": []}
    for size in (1, 12):
        selected = indices[:size]
        views = [VIEWS.index(groups[index].view) for index in selected]
        dataset = Images(args.bird / "dataset", [path for i in selected for path in groups[i].paths])
        prior = prototype[views]

        def score(features, method):
            maps = [scoring(part, prior[offset:offset + len(part)], model, method)
                    for offset, part in enumerate(features.split(1), start=0)]
            return upsample(torch.cat(maps).mean(1))

        def image_features():
            values = []
            for start in range(0, len(dataset), 20):
                batch = torch.stack([dataset[i] for i in range(start, min(len(dataset), start + 20))])
                values.append(encoder(batch.to(reference_device)).half())
            return torch.cat(values).reshape(size, 10, 3, 256, 768)

        for method in ("B2", "B6"):
            for scope, operation, count in (
                ("cache_read_and_score", lambda: score(as_tensor(cached, selected, reference_device), method), 7),
                ("png_read_encode_and_score", lambda: score(image_features(), method), 3),
            ):
                probe = repetitions(operation, count)
                result["probes"].append({"setting": method, "scope": scope, "images": size * 10, **probe})
                print(method, scope, size * 10, probe["median_seconds"], flush=True)
    write_json(args.output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("initial", "bird", "frozen", "output"):
        parser.add_argument("--" + key, type=lambda p: Path(p).resolve(), required=True)
    args = parser.parse_args()
    check(not args.output.is_relative_to(REPO) and not args.output.exists(), "use a new external result path")
    enable_offline()
    torch.set_num_threads(2)
    with pinned_upstream(args.initial / "inputs/upstream", UPSTREAM):
        run(args)


if __name__ == "__main__":
    main()
