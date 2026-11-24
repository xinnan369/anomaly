"""Normal-only candidate selection and controlled cross-light perturbation study."""

import csv
from hashlib import sha256

import numpy as np
from sklearn.metrics import average_precision_score
import torch

from vicad_refinement.data import as_tensor
from vicad_refinement.model import context_predictions, normalize, residuals, upsample, weighted_lights
from vicad_refinement.synthetic import patch_mask, perturb
from vicad_syn.data.cache import Images
from vicad_syn.data.manifest import VIEWS, split_ids
from vicad_syn.experiment import check, digest, read_json, write_json
from vicad_syn.uncertainty.signals import squared_distance


def csv_write(path, rows):
    check(bool(rows), f"empty table {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fixed_seed(protocol, category, identity, view):
    token = f"{protocol['seed']}:{category}:{identity}:{view}"
    return int(sha256(token.encode()).hexdigest()[:8], 16)


def split_development(groups, protocol):
    train_ids, val_ids = split_ids(groups, protocol["seed"], protocol["normal_fit_objects"])
    check(train_ids.isdisjoint(val_ids) and len(train_ids) == 24 and len(val_ids) == 6,
          "development expects 24 mutually exclusive fit objects and six validation objects")
    return train_ids, val_ids


@torch.inference_mode()
def study(category, groups, val_ids, features, root, folder, encoder, model, proto, protocol, device):
    receipt = folder / "proxy.json"
    if result := read_json(receipt):
        check(all(digest(folder / key) == value for key, value in result["sha256"].items()), "changed proxy evidence")
        return result
    scores, mechanisms, arrays = [], [], {}
    examples = []
    one_view = [i for i, g in enumerate(groups) if g.identity in val_ids and g.view in protocol["development_views"]]
    for number, index in enumerate(one_view):
        group = groups[index]
        reference = proto[VIEWS.index(group.view):VIEWS.index(group.view) + 1]
        source = Images(root / "dataset", group.paths)
        images = torch.stack([source[i] for i in range(10)])
        clean = as_tensor(features, slice(index, index + 1), device)
        clean_predictions = context_predictions(normalize(clean[:, :, -1]), model, reference[:, :, -1])
        group_seed = fixed_seed(protocol, category, group.identity, group.view)
        normal_thresholds = {}
        for scenario in protocol["synthetic"]:
            changed, lights, mask = perturb(images, scenario, group_seed, protocol)
            data = clean.clone()
            if lights:
                data[0, lights] = encoder(changed[lights].to(device)).half()
            residual, stability, epistemic, per_layer = residuals(data, model, reference)
            clean_leaf = normalize(clean[:, :, -1])
            leaf = normalize(data[:, :, -1])
            preds = context_predictions(leaf, model, reference[:, :, -1])
            positive = bool(mask.any())
            target = mask.numpy().reshape(-1)
            patch = patch_mask(mask).to(device) if positive else torch.ones(256, dtype=torch.bool, device=device)
            relevant = lights or list(range(10))
            shifts = squared_distance(leaf, clean_leaf)[0, relevant][:, patch].mean().item()
            for context, pred in preds.items():
                maps = squared_distance(leaf, pred)
                avg = upsample(maps.mean(1)).reshape(-1).cpu().numpy()
                mechanisms.append({"category": category, "object": group.identity, "view": group.view,
                                   "scenario": scenario, "context": context, "changed_lights": len(lights),
                                   "patches": int(patch.sum()), "target_feature_shift": shifts,
                                   "prediction_feature_shift": squared_distance(pred, clean_predictions[context])[0, relevant][:, patch].mean().item(),
                                   "target_residual": maps[0, relevant][:, patch].mean().item(),
                                   "stability": stability[0, relevant][:, patch].mean().item(),
                                   "epistemic": epistemic[0, relevant][:, patch].mean().item(),
                                   "pixel_ap": average_precision_score(target, avg) if positive else None})
                arrays[f"{number}_{scenario}_{context}"] = maps[0].cpu().numpy()
            for candidate, maps in residual.items():
                for fusion in protocol["fusion_modes"]:
                    contributions = weighted_lights(maps, stability, epistemic, fusion, protocol["protection"])
                    merged = upsample(contributions.mean(1)).reshape(-1).cpu().numpy()
                    key = candidate, fusion
                    if scenario == "clean":
                        normal_thresholds[key] = np.quantile(merged, 0.99)
                    scores.append({"category": category, "object": group.identity, "view": group.view,
                                   "scenario": scenario, "candidate": candidate, "fusion": fusion,
                                   "pixel_ap": average_precision_score(target, merged) if positive else None,
                                   "above_clean_p99": float((merged > normal_thresholds[key]).mean()),
                                   "mean_score": float(merged.mean())})
            arrays[f"{number}_{scenario}_mask"] = mask.numpy()
            arrays[f"{number}_{scenario}_stability"] = stability[0].cpu().numpy()
            arrays[f"{number}_{scenario}_epistemic"] = epistemic[0].cpu().numpy()
            if number == 0:
                arrays[f"figure_{scenario}_image"] = changed[relevant[0]].cpu().numpy()
                arrays[f"figure_{scenario}_layers"] = per_layer[0, relevant[0]].cpu().numpy()
                arrays[f"figure_{scenario}_weights"] = (
                    weighted_lights(residual["D_combined"], stability, epistemic, "protected_both", protocol["protection"])
                    / residual["D_combined"].clamp_min(1e-8) / 10
                )[0, relevant[0]].cpu().numpy()
        examples.append({"object": group.identity, "view": group.view, "perturbation_seed": group_seed})
        print(f"{category}: completed normal-only synthetic view {number + 1}/{len(one_view)}", flush=True)
    outputs = {"proxy-rows.csv": scores, "mechanism-rows.csv": mechanisms}
    for filename, rows in outputs.items():
        csv_write(folder / filename, rows)
    np.savez_compressed(folder / "synthetic-maps.npz", **arrays)
    result = {"status": "PASS", "normal_validation_objects": sorted(val_ids), "views": examples,
              "sha256": {name: digest(folder / name) for name in (*outputs, "synthetic-maps.npz")}}
    write_json(receipt, result)
    return result


def select(rows, protocol):
    category_scenario_buckets = {}
    for row in rows:
        if row["pixel_ap"] in ("", None):
            continue
        key = row["candidate"], row["fusion"], row["category"], row["scenario"]
        category_scenario_buckets.setdefault(key, []).append(float(row["pixel_ap"]))
    ranked = []
    best = None
    for candidate in protocol["candidates"]:
        for fusion in protocol["fusion_modes"]:
            cells = [values for key, values in category_scenario_buckets.items() if key[:2] == (candidate, fusion)]
            check(len(cells) == len(protocol["development_categories"]) * (len(protocol["synthetic"]) - 2), "incomplete proxy matrix")
            score = float(np.mean([np.mean(items) for items in cells]))
            row = {"candidate": candidate, "fusion": fusion, "macro_pixel_ap": score, "views": sum(map(len, cells))}
            ranked.append(row)
            if best is None or score - best["macro_pixel_ap"] > 1e-12:
                best = row
    return best, ranked
