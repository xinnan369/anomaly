"""Read-only v1 Synergy replay and localization diagnostics over all four seen classes."""

import argparse
from contextlib import nullcontext
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from vicad_refinement.development import csv_write
from vicad_syn.evaluation.synergy import METRICS
from vicad_syn.experiment import check, digest, pinned_upstream, protocol, read_json, write_json


def validated_arrays(folder, references):
    values = {}
    for name, receipt in references.items():
        path = folder / name
        check(path.stat().st_size == receipt["bytes"] and digest(path) == receipt["sha256"],
              f"v1 artifact hash changed: {path.name}")
        values[path.stem] = np.load(path, allow_pickle=False, mmap_mode="r")
    return values


def means(image, mask):
    return float(image[mask].mean()), float(image[~mask].mean())


def auc(scores, labels, rng):
    y = labels.reshape(-1)
    x = scores.reshape(-1)
    positives, negatives = np.flatnonzero(y), np.flatnonzero(~y)
    n = min(len(positives), len(negatives), 100_000)
    indices = np.r_[rng.choice(positives, n, replace=False), rng.choice(negatives, n, replace=False)]
    return float(roc_auc_score(y[indices], x[indices]))


def analyze(category, first, combined, source, binding, output):
    masks = first["view_instance_masks"].reshape(-1, 224, 224).astype(bool)
    stable_keys = ("view_instance_masks", "view_instance_names", "view_instance_anomalys", "view_instance_cls_names")
    check(all(np.array_equal(first[k], combined[k]) for k in stable_keys), "fused prediction rows not aligned")
    joined = masks.any(axis=(1, 2))
    masks = masks[joined]
    area = masks.sum(axis=(1, 2))
    tile = masks.reshape(-1, 16, 14, 16, 14).any(axis=(2, 4))
    tile_area = tile.sum(axis=(1, 2)) * 196
    before = first["view_instance_anomaly_maps"][joined].reshape(masks.shape)
    after = combined["view_instance_anomaly_maps"][joined].reshape(masks.shape)
    before_fg = np.sum(before, axis=(1, 2), where=masks, dtype=np.float64) / area
    after_fg = np.sum(after, axis=(1, 2), where=masks, dtype=np.float64) / area
    before_bg = np.sum(before, axis=(1, 2), where=~masks, dtype=np.float64) / ((224**2) - area)
    after_bg = np.sum(after, axis=(1, 2), where=~masks, dtype=np.float64) / ((224**2) - area)
    p = source / "patch-signals.npz"
    check(digest(p) == binding["patch_signals_sha256"], "v1 predictor signal hash moved")
    raw_mask = source / "raw/imgs_masks.npy"
    check(digest(raw_mask) == binding["raw"][raw_mask.name]["sha256"], "v1 raw mask hash changed")
    raw = np.load(raw_mask, allow_pickle=False, mmap_mode="r")
    positive = np.array([i for i, v in enumerate(raw) if v.any()])
    visible = np.bincount(positive // 10, minlength=len(raw)//10)
    cross_light_iou = []
    for i in np.flatnonzero(visible == 10):
        same_view = raw[i * 10:(i + 1) * 10].reshape(10, 224, 224).astype(bool)
        intersection = same_view.all(axis=0).sum()
        union = same_view.any(axis=0).sum()
        cross_light_iou.append(intersection / union)
    tile_labels = raw[positive].reshape(-1, 16, 14, 16, 14).any(axis=(2, 4)).reshape(-1, 256)
    rng = np.random.default_rng(42)
    with np.load(p, allow_pickle=False) as patch:
        local = {}
        for score in ("residual", "stability", "epistemic"):
            values = patch[score].reshape(-1, 256)[positive]
            foreground, background = means(values, tile_labels)
            local[f"{score}_patch_fg"] = foreground
            local[f"{score}_patch_bg"] = background
            local[f"{score}_patch_auc_subsampled"] = auc(values, tile_labels, rng)
    exemplar = int(np.flatnonzero(joined)[0])
    np.savez_compressed(output / f"{category}-v1-example.npz", mask=masks[0],
                        A1=before[0], A4=after[0], tile=tile[0], view_index=exemplar)
    return {"category": category, "positive_views": len(masks), "all_10_lights_positive": int((visible == 10).sum()),
            "all_lights_intersection_over_union_median": float(np.median(cross_light_iou)),
            "all_lights_identical_masks": int(np.count_nonzero(np.array(cross_light_iou) == 1)),
            "median_mask_pixels": float(np.median(area)), "mean_mask_pixels": float(area.mean()),
            "occupied_patch_area_precision": float(area.sum()/tile_area.sum()),
            "A1_fg_pixel_mean": means(before, masks)[0], "A1_bg_pixel_mean": means(before, masks)[1],
            "A4_fg_pixel_mean": means(after, masks)[0], "A4_bg_pixel_mean": means(after, masks)[1],
            "A1_pixel_auc_subsampled": auc(before, masks, rng), "A4_pixel_auc_subsampled": auc(after, masks, rng),
            "A4_lower_foreground_view_pct": float((after_fg < before_fg).mean()*100),
            "A4_lower_fg_bg_contrast_view_pct": float(((after_fg-after_bg) < (before_fg-before_bg)).mean()*100),
            **local}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "upstream", "output"):
        parser.add_argument(f"--{key}", type=lambda p: Path(p).resolve(), required=True)
    parser.add_argument("--mode", choices=("localization", "replay", "both"), default="both")
    args = parser.parse_args()
    cfg = protocol()["config"]
    repo = Path(__file__).resolve().parents[2]
    check(not args.output.is_relative_to(args.source), "write diagnostics away from accepted evidence")
    args.output.mkdir(parents=True, exist_ok=True)
    evidence = read_json(repo / "results/vicad_syn/evidence/verification.json")
    receipt = []
    extended = []
    with (repo / "results/vicad_syn/metrics/summary.csv").open(encoding="utf-8") as handle:
        old = {(row["class"], row["setting"]): row for row in csv.DictReader(handle)}
    environment = nullcontext() if args.mode == "localization" else pinned_upstream(args.upstream, cfg["upstream_revision"])
    with environment:
        if args.mode != "localization":
            from util.metric import Evaluator
        for category in ("Car", "Cube", "Doll", "Ring"):
            root = args.source / category / "conditioned/evaluation"
            stored = read_json(root / "evaluation.json")
            versions = {}
            for mode in ("A1", "A4"):
                folder = root / mode / "fused"
                verified = stored["metrics"][mode]["fused"]
                # The accepted verification is rooted in the original aggregate; independently check its scalar receipt.
                arrays = validated_arrays(folder, verified)
                if args.mode != "localization":
                    result = Evaluator(metrics=list(METRICS.values()), use_adeval=True).run(arrays, category, "M2AD-Synergy")
                    metrics = {key: 100 * float(result[alias]) for key, alias in METRICS.items()}
                    check(all(abs(value - float(old[(category, mode)][k])) < 1e-7 for k, value in metrics.items()),
                          f"v1 official scores differ from accepted master ledger: {category}/{mode}")
                    receipt.append({"category": category, "setting": mode, **metrics})
                versions[mode] = arrays
            if args.mode != "replay":
                extended.append(analyze(category, versions["A1"], versions["A4"], root, stored, args.output))
            print(f"{category}: v1 {args.mode} passed", flush=True)
    outputs = {}
    for name, rows in (("v1-replay.csv", receipt), ("v1-localization.csv", extended)):
        if rows:
            csv_write(args.output / name, rows)
            outputs[name] = digest(args.output / name)
    write_json(args.output / f"{args.mode}.json", {"status": "PASS", "accepted_status": evidence["status"], "outputs": outputs})


if __name__ == "__main__":
    main()
