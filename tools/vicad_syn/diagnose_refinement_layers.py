"""Read-only normal-data diagnostics on feature depth and directional contamination."""

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score
import torch

from vicad_refinement.data import as_tensor, load_features, load_groups
from vicad_refinement.development import csv_write, fixed_seed, split_development
from vicad_refinement.model import MultiLayerBackbone, context_predictions, normalize, upsample
from vicad_refinement.run import REGISTRY, fixed_protocol
from vicad_refinement.synthetic import patch_mask, perturb
from vicad_refinement.training import fit
from vicad_syn.data.cache import Images
from vicad_syn.data.manifest import VIEWS
from vicad_syn.experiment import check, digest, pinned_upstream, read_json, write_json
from vicad_syn.uncertainty.signals import squared_distance


@torch.inference_mode()
def compare(folder, root, category, encoder, registry, fixed, device):
    base = fixed["inherited_v1"]["config"]
    groups = load_groups(root, category, base, (30,70), "train")
    train, valid = split_development(groups, registry)
    cache_folder = folder / "train-features"
    cache = read_json(cache_folder / "features.json")
    check(digest(cache_folder / "features.npy") == cache["sha256"], "normal cache changed")
    features = load_features(cache_folder)
    model, proto, receipt = fit(folder / "fit", cache, groups, features, train,
                               registry["seed"], base, fixed, device)
    rows, direction = [], []
    for index, group in enumerate(groups):
        if group.identity not in valid or group.view not in registry["development_views"]:
            continue
        v = VIEWS.index(group.view)
        reference = proto[v:v + 1]
        x0 = as_tensor(features, slice(index, index + 1), device)
        leaf0 = normalize(x0[:, :, -1])
        pred0 = context_predictions(leaf0, model, reference[:, :, -1])
        image_source = Images(root / "dataset", group.paths)
        images = torch.stack([image_source[i] for i in range(10)])
        for scenario in registry["synthetic"][1:-1]:
            changed, lights, mask = perturb(images, scenario, fixed_seed(registry, category, group.identity, group.view), registry)
            x = x0.clone()
            x[0, lights] = encoder(changed[lights].to(device)).half()
            x = normalize(x)
            maps = squared_distance(x, reference).mean(1)
            candidates = upsample(maps)
            for block, value in zip(registry["feature_blocks_one_based"], candidates, strict=True):
                rows.append({"category": category, "object": group.identity, "view": group.view,
                             "scenario": scenario, "block": block,
                             "pixel_ap": float(average_precision_score(mask.flatten().numpy(), value.cpu().flatten().numpy()))})
            region = patch_mask(mask).to(device)
            delta = (x[:, :, -1] - leaf0)[0, lights][:, region].flatten()
            energy = delta.square().sum().item()
            for context, current in context_predictions(x[:, :, -1], model, reference[:, :, -1]).items():
                response = (current - pred0[context])[0, lights][:, region].flatten()
                projection = (response * delta).sum().item() / energy if energy else None
                direction.append({"category": category, "object": group.identity, "view": group.view,
                                  "scenario": scenario, "context": context, "direction_projection": projection,
                                  "test_role": "normal-only synthetic", "checkpoint_sha256": receipt["checkpoint_sha256"]})
    return rows, direction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("inputs", "run", "output"):
        parser.add_argument(f"--{key}", type=lambda p: Path(p).resolve(), required=True)
    args = parser.parse_args()
    registry = read_json(REGISTRY)
    fixed = fixed_protocol(registry)
    base = fixed["inherited_v1"]["config"]
    check(read_json(args.run / "development/development.json")["binding"] == fixed, "normal development receipt differs")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = torch.device("cuda")
    rows, direction = [], []
    with pinned_upstream(args.inputs / "inputs/upstream", base["upstream_revision"]):
        encoder = MultiLayerBackbone(args.inputs / "inputs/weights" / base["backbone"],
                                     base, registry["feature_blocks_one_based"]).to(device)
        for category in registry["development_categories"]:
            row, cone = compare(args.run / "development" / category, args.inputs, category, encoder, registry, fixed, device)
            rows.extend(row)
            direction.extend(cone)
            print(category, "layer and directional checks completed", flush=True)
    csv_write(args.output / "normal-layer-diagnosis.csv", rows)
    csv_write(args.output / "normal-direction-diagnosis.csv", direction)
    layer_summary = []
    for category in registry["development_categories"]:
        for layer in registry["feature_blocks_one_based"]:
            entries = [row["pixel_ap"] for row in rows if row["category"] == category and row["block"] == layer]
            layer_summary.append({"category": category, "block": layer, "views": len(entries), "mean_pixel_ap": float(np.mean(entries))})
    csv_write(args.output / "normal-layer-summary.csv", layer_summary)
    cone_summary = []
    for scenario in registry["synthetic"][1:-1]:
        for context in registry["contexts"]:
            estimates = [row["direction_projection"] for row in direction
                         if row["scenario"] == scenario and row["context"] == context and row["direction_projection"] is not None]
            cone_summary.append({"scenario": scenario, "context": context, "views": len(estimates),
                                 "mean_signed_projection": float(np.mean(estimates)),
                                 "positive_fraction": float(np.mean(np.asarray(estimates) > 0))})
    csv_write(args.output / "normal-direction-summary.csv", cone_summary)
    write_json(args.output / "layers.json", {"status": "PASS", "outputs": {
        p.name: digest(p) for p in sorted(args.output.glob("normal-*.csv"))}})


if __name__ == "__main__":
    main()
