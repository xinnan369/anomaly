"""Frozen network predictions, verified Synergy fusion, and official CUDA metrics."""

import csv
import importlib.util
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torchvision.transforms import InterpolationMode, Resize

from vicad_syn.data.manifest import LIGHTS, VIEWS
from vicad_syn.experiment import check, digest, read_json, write_json
from vicad_syn.fusion.weights import light_weights
from vicad_syn.uncertainty.signals import context_instability, epistemic_uncertainty, squared_distance


SETTINGS = {"A0": (), "A1": (), "A2": ("stability",), "A3": ("epistemic",), "A4": ("stability", "epistemic")}
METRICS = {"O-AUROC": "AUROC_obj_sp", "I-AUROC": "AUROC_view_sp", "AUPRO": "AUPRO_view"}
MASK_RESIZE = Resize((224, 224), InterpolationMode.BILINEAR)


def baseline_helpers():
    file = Path(__file__).resolve().parents[2] / "tools/environment/baseline_metrics.py"
    spec = importlib.util.spec_from_file_location("audited_baseline_metrics", file)
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    return helpers


def record_files(directory):
    return {path.name: {"sha256": digest(path), "bytes": path.stat().st_size}
            for path in sorted(directory.glob("*.npy"))}


@torch.inference_mode()
def predict(model, features, config, need_uncertainty):
    def sync():
        if features.is_cuda:
            torch.cuda.synchronize()
    if features.is_cuda:
        torch.cuda.reset_peak_memory_stats(features.device)
    rows = {"residual": [], "stability": [], "epistemic": [], "maps": []}
    base_seconds, extra_seconds = 0., 0.
    base_peak, total_peak = 0, 0
    for start in range(0, len(features), config["batch_groups"]):
        sync()
        start_base = perf_counter()
        x = F.normalize(features[start:start + config["batch_groups"]].float(), dim=-1)
        full = model(x)
        residual = squared_distance(x, F.normalize(full.mean(0), dim=-1))
        maps = F.interpolate(residual.reshape(-1, 1, 16, 16), (224, 224), mode="bilinear", align_corners=False)
        rows["residual"].append(residual.cpu())
        rows["maps"].append(maps[:, 0].cpu())
        sync()
        base_seconds += perf_counter() - start_base
        if features.is_cuda:
            base_peak = max(base_peak, torch.cuda.max_memory_allocated(features.device))
        if need_uncertainty:
            start_extra = perf_counter()
            rows["stability"].append(context_instability(model(x, "first"), model(x, "last")).cpu())
            rows["epistemic"].append(epistemic_uncertainty(full).cpu())
            sync()
            extra_seconds += perf_counter() - start_extra
        if features.is_cuda:
            total_peak = max(total_peak, torch.cuda.max_memory_allocated(features.device))
            torch.cuda.reset_peak_memory_stats(features.device)
    result = {name: torch.cat(values).numpy() for name, values in rows.items() if values}
    for name, values in result.items():
        check(np.isfinite(values).all(), f"nonfinite inference signal {name}")
    return result, {"prediction_seconds": base_seconds, "uncertainty_seconds": extra_seconds,
                    "peak_cuda_base_bytes": base_peak, "peak_cuda_bytes": total_peak}


def official_inputs(groups, dataset, maps):
    rows = [record for group in groups for record in group.records]
    masks = np.zeros((len(rows), 1, 224, 224), dtype=np.uint8)
    for i, row in enumerate(rows):
        if row["image_anomaly"]:
            with Image.open(dataset / row["mask_path"]) as source:
                pixels = (np.asarray(source.convert("L")) > 0).astype(np.uint8) * 255
                # Official test casts the resized floating [0,1] tensor to an integer.
                masks[i, 0] = np.asarray(MASK_RESIZE(Image.fromarray(pixels))) == 255
    return {
        **{key: np.array([row[field] for row in rows]) for key, field in (
            ("obj_names", "object_name"), ("cls_names", "cls_name"), ("illumination_list", "illumination"),
            ("view_list", "view"), ("image_anomalys", "image_anomaly"), ("object_anomalys", "object_anomaly"))},
        "imgs_masks": masks, "anomaly_scores": maps.max(axis=(1, 2)), "anomaly_maps": maps,
    }


def diagnostics(groups, signals, path):
    if "stability" not in signals:
        return {}
    keys = ("residual", "stability", "epistemic")
    pairs = np.corrcoef(np.stack([signals[key].reshape(-1).astype(np.float64) for key in keys]))
    rows = []
    subsets = [("light", light, np.arange(len(groups)), i) for i, light in enumerate(LIGHTS)]
    subsets += [("view", view, np.array([i for i, group in enumerate(groups) if group.view == view]), slice(None))
                for view in VIEWS]
    for scope, item, selected, light_idx in subsets:
        row = {"scope": scope, "id": item}
        for key in keys:
            values = signals[key][selected, light_idx]
            row[key + "_mean"] = float(values.mean())
            row[key + "_std"] = float(values.std())
        rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return {f"{first}_vs_{second}": float(pairs[i, j]) if np.isfinite(pairs[i, j]) else None
            for i, first in enumerate(keys) for j, second in enumerate(keys) if i < j}


def evaluate(folder, category, groups, dataset, model, features, cfg, train, test_cache, modes):
    from util.m2ad_utils import m2ad_fuse
    from util.metric import Evaluator

    check(len(groups) == len(features) == 840, "official test requires all 840 views")
    folder.mkdir(parents=True, exist_ok=True)
    binding = {"checkpoint_sha256": train["sha256"], "config_sha256": train["binding"]["frozen"]["config_sha256"],
               "test_cache_sha256": test_cache["sha256"], "settings": list(modes)}
    prior = read_json(folder / "evaluation.json")
    if prior:
        check(prior["binding"] == binding and prior["status"] == "PASS", "existing evaluation identity differs")
        return prior
    helpers = baseline_helpers()
    signals, performance = predict(model, features, cfg, any(SETTINGS[mode] for mode in modes))
    diag = diagnostics(groups, signals, folder / "diagnostics.csv")
    raw = official_inputs(groups, dataset, signals["maps"])
    raw_home = folder / "raw"
    raw_home.mkdir(exist_ok=True)
    for key, array in raw.items():
        np.save(raw_home / (key + ".npy"), array, allow_pickle=False)
    raw_records = record_files(raw_home)
    feature_signals = folder / "patch-signals.npz"
    np.savez(feature_signals, **{key: array for key, array in signals.items() if key != "maps"})
    del signals["maps"]
    metrics = {}
    residual = torch.from_numpy(signals["residual"])
    for mode in modes:
        mode_home = folder / mode
        mode_home.mkdir(exist_ok=True)
        started = perf_counter()
        weights = light_weights(residual, [torch.from_numpy(signals[key]) for key in SETTINGS[mode]],
                                cfg["epsilon"], cfg["temperature"]).numpy()
        check(np.isfinite(weights).all() and (weights >= 0).all() and np.allclose(weights.sum(1), 1),
              f"invalid evaluation weights {mode}")
        scale = (10 * weights).reshape(-1)
        weighted = raw if not SETTINGS[mode] else {**raw, "anomaly_scores": raw["anomaly_scores"] * scale,
                                                   "anomaly_maps": raw["anomaly_maps"] * scale[:, None, None]}
        with np.errstate(all="raise"):
            fused = m2ad_fuse(**weighted, fuse="add")
            fusion = helpers.check_fusion(weighted, fused)
        check((fusion["objects"], fusion["views"]) == (70, 840), "incomplete official evaluation groups")
        np.save(mode_home / "weights.npy", weights, allow_pickle=False)
        fused_home = mode_home / "fused"
        fused_home.mkdir(exist_ok=True)
        for key, array in fused.items():
            np.save(fused_home / (key + ".npy"), array, allow_pickle=False)
        evaluator = Evaluator(metrics=list(METRICS.values()), use_adeval=True)
        measured = evaluator.run(fused, category, "M2AD-Synergy")
        scores = {key: float(measured[value]) * 100 for key, value in METRICS.items()}
        check(all(np.isfinite(value) and 0 <= value <= 100 for value in scores.values()), "invalid official metrics")
        metrics[mode] = {**scores, "fusion": fusion, "evaluation_seconds": perf_counter() - started,
                         "fused": record_files(fused_home), "weights_sha256": digest(mode_home / "weights.npy")}
        write_json(mode_home / "metrics.json", {"binding": binding, **metrics[mode]})
        print(f"{category}/{mode}: {scores}", flush=True)
    result = {"status": "PASS", "binding": binding, "test_images": len(raw["cls_names"]),
              "metrics": metrics, "performance": performance, "diagnostic_correlations": diag,
              "raw": raw_records, "patch_signals_sha256": digest(feature_signals)}
    write_json(folder / "evaluation.json", result)
    return result
