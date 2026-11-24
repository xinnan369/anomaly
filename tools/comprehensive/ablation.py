"""Read-only, paired TASK-005 scores from frozen local features and checkpoints."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image
import torch

from tools.comprehensive.offline import enable_offline
from tools.comprehensive.resources import FREEZE, REPO, UPSTREAM, roots, scope_for
from vicad_refinement.data import as_tensor, load_groups, view_indices
from vicad_refinement.formal import official_evaluation
from vicad_refinement.model import residuals, upsample, weighted_lights
from vicad_syn.evaluation.synergy import MASK_RESIZE, METRICS
from vicad_syn.experiment import check, digest, pinned_upstream, read_json, write_json
from vicad_syn.training.train import build_predictor

PROTOCOL = Path(__file__).with_name("protocol.json")
ABLATION_SOURCES = {"B0": "v1_A4", "B6": "selected"}


def ablation_fields(scores, normal_layers):
    return {**scores, "C_leaf": normal_layers[:, :, -1]}


@torch.inference_mode()
def predict(features, views, reference, model, settings, protection, device):
    started = perf_counter()
    torch.cuda.reset_peak_memory_stats()
    contributions = {key: [] for key in settings}
    for offset in range(0, len(features), 4):
        batch = as_tensor(features, slice(offset, offset + 4), device)
        prior = reference[views[offset:offset + len(batch)]]
        scores, stability, epistemic, layers = residuals(batch, model, prior)
        fields = ablation_fields(scores, layers)
        for key, (candidate, fusion) in settings.items():
            result = weighted_lights(fields[candidate], stability, epistemic, fusion, protection)
            check(bool(result.isfinite().all()), f"nonfinite {key} at {offset}")
            contributions[key].append(result.cpu().numpy())
        if offset and offset % 120 == 0:
            print(f"cached scoring: {offset}/{len(features)} groups", flush=True)
    torch.cuda.synchronize()
    return {key: np.concatenate(value) for key, value in contributions.items()}, {
        "shared_seconds": perf_counter() - started,
        "shared_peak_cuda_bytes": torch.cuda.max_memory_allocated(device),
        "shared_settings": list(settings),
    }


def buffer_scores(contributions):
    """The pinned evaluator consumes the mean of ten interpolated per-image maxima."""
    chunks = []
    for batch in np.array_split(contributions, (len(contributions) + 7) // 8):
        maps = upsample(torch.from_numpy(batch)).numpy()
        chunks.append(maps.max(axis=(1, 2)).reshape(-1, 10))
    return np.concatenate(chunks)


def canonical_scores(groups, mask_labels, scores):
    """Reconstruct identity-aligned official means from the same ten image scores."""
    items = defaultdict(list)
    views = []
    for group, label, lights in zip(groups, mask_labels, scores, strict=True):
        first = group.records[0]
        identity = (first["cls_name"], first["object_name"], int(first["object_anomaly"]))
        items[identity].append(lights)
        views.append({"identity": list(identity), "view": group.view, "label": int(label),
                      "score": float(lights.mean())})
    objects = [{"identity": list(identity), "label": identity[2], "score": float(np.concatenate(scores).mean())}
               for identity, scores in sorted(items.items())]
    return {"objects": objects, "views": views}


def evaluation_metadata(groups, dataset):
    rows = [row for group in groups for row in group.records]
    masks = np.zeros((len(rows), 1, 224, 224), dtype=np.uint8)
    by_path = {}
    for i, row in enumerate(rows):
        if row["image_anomaly"]:
            path = row["mask_path"]
            if path not in by_path:
                with Image.open(dataset / path) as image:
                    pixels = (np.asarray(image.convert("L")) > 0).astype(np.uint8) * 255
                    by_path[path] = np.asarray(MASK_RESIZE(Image.fromarray(pixels))) == 255
            masks[i, 0] = by_path[path]
    fields = (("obj_names", "object_name"), ("cls_names", "cls_name"), ("illumination_list", "illumination"),
              ("view_list", "view"), ("image_anomalys", "image_anomaly"), ("object_anomalys", "object_anomaly"))
    return {**{key: np.array([row[field] for row in rows]) for key, field in fields}, "imgs_masks": masks}


def mask_binding(metadata):
    masks = metadata["imgs_masks"].reshape(-1, 10, 224, 224).max(axis=1)
    labels = masks.any(axis=(1, 2))
    return labels, sha256(masks.tobytes()).hexdigest()


def assert_auc(scores, reported):
    from sklearn.metrics import roc_auc_score

    for metric, name in (("O-AUROC", "objects"), ("I-AUROC", "views")):
        vals = scores[name]
        computed = 100 * roc_auc_score([x["label"] for x in vals], [x["score"] for x in vals])
        check(abs(computed - reported[metric]) <= 1e-6, f"{metric} identity-aligned score differs from official")


def model_from_frozen(folder, receipt, category, base, device):
    path = folder / "checkpoint.pt"
    check(digest(path) == receipt["checkpoint_sha256"], "checkpoint digest differs")
    saved = torch.load(path, map_location=device, weights_only=True)
    check(saved["epoch"] == base["epochs"] and saved["binding"] == receipt["binding"], f"{category}: changed checkpoint")
    model = build_predictor(base, True, device).eval().requires_grad_(False)
    model.load_state_dict(saved["predictor"])
    return model, saved["prototypes"]


def map_source(historical, name, registered):
    file = historical / name / "patch-contributions.npy"
    check(digest(file) == registered["fused_and_patch_inputs"][file.name]["sha256"], "historical map content differs")
    return np.load(file, allow_pickle=False)


def one_seed(root, directory, category, seed, settings, groups, features, views, base, registry, cache_sha, args):
    destination = args.output / category / f"seed-{seed}"
    if destination.exists():
        prior = read_json(destination / "ablation.json")
        check(prior and prior["protocol_sha256"] == digest(PROTOCOL), f"preserve unfinished attempt: {destination}")
        return
    historical = directory / f"seed-{seed}"
    prior = read_json(historical / "evaluation.json")
    binding = read_json(directory / "pretest.json")["binding"]
    checkpoint = prior["training"]["checkpoint_sha256"]
    check(binding["input_freeze_sha256"] == FREEZE and binding["checkpoints"][str(seed)] == checkpoint,
          "evaluation checkpoint differs from pre-test freeze")
    model, reference = model_from_frozen(historical, prior["training"], category, base, views.device)
    maps, costs = predict(features, views, reference, model,
                          {key: settings[key] for key in settings if key not in ABLATION_SOURCES},
                          registry["protection"], views.device)
    destination.mkdir(parents=True, exist_ok=False)
    score_rows = {}
    metadata = evaluation_metadata(groups, root / "dataset")
    labels, masks_hash = mask_binding(metadata)
    for key in settings:
        if key in ABLATION_SOURCES:
            source = ABLATION_SOURCES[key]
            original = prior["scores"][source]
            contributions = map_source(historical, source, original)
            score = {metric: original[metric] for metric in METRICS}
            stored_masks = historical / source / "view_instance_masks.npy"
            check(digest(stored_masks) == original["fused_and_patch_inputs"][stored_masks.name]["sha256"],
                  "historical masks differ")
            origin = "TASK-004 frozen contribution and official score"
        else:
            contributions = maps.pop(key)
            score, _ = official_evaluation(destination / key, category, groups, root / "dataset", contributions, metadata)
            origin = "TASK-005 official Synergy"
        canonical = canonical_scores(groups, labels, buffer_scores(contributions))
        assert_auc(canonical, score)
        canonical_path = destination / f"{key}-identity.json"
        write_json(canonical_path, canonical)
        score_rows[key] = {**score, "identity_sha256": digest(canonical_path), "origin": origin}
        print(f"{category}/{seed}/{key} {dict((m, score[m]) for m in METRICS)}", flush=True)
    write_json(destination / "ablation.json", {
        "status": "PASS", "category": category, "seed": seed, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": digest(PROTOCOL), "freeze_sha256": FREEZE, "checkpoint_sha256": checkpoint,
        "cache_sha256": cache_sha, "canonical_view_masks_sha256": masks_hash, "prediction": costs, "settings": score_rows,
    })


def execute(args):
    protocol = read_json(PROTOCOL)
    check(protocol["task004_freeze"] == digest(args.frozen / "freeze.json"), "frozen protocol mismatch")
    base = read_json(args.frozen / "freeze.json")["binding"]["inherited_v1"]["config"]
    registry = read_json(REPO / "vicad_syn/refinement/protocol.json")
    check(protocol["ablation"]["B6"] == ["D_combined", "patch_stability"], "registered B6 has drifted")
    check(torch.cuda.is_available() and torch.cuda.device_count() == 1, "one local GPU required for official ADEval")
    torch.set_num_threads(4)
    gpu = torch.device("cuda:0")
    audit = read_json(args.audit)
    with pinned_upstream(args.initial / "inputs/upstream", UPSTREAM):
        for category in ([args.category] if args.category else protocol["ablation_order"]):
            if audit["categories"][category]["status"] != "PASS":
                print(f"blocked: {category} local assets", flush=True)
                continue
            root = roots(category, args.initial, args.bird, args.holdout)
            directory = args.frozen / scope_for(category) / category
            expected = registry["confirmatory_groups"].get(category, (30, 70))
            groups = load_groups(root, category, base, expected, "test")
            cache = directory / "test-features"
            cache_link = read_json(directory / "category.json")["test_features"]
            order = "\n".join(path for group in groups for path in group.paths) + "\n"
            check(sha256(order.encode()).hexdigest() == cache_link["binding"]["order_sha256"],
                  "cached feature identities differ")
            feature_file = cache / "features.npy"
            print(f"{category}: checking cache bytes", flush=True)
            check(digest(feature_file) == cache_link["sha256"], "feature cache content differs")
            # Sequential reads avoid repeated small DriveFS faults over the Windows data volume.
            with Path("/proc/meminfo").open() as memory:
                available = next(int(line.split()[1]) * 1024 for line in memory if line.startswith("MemAvailable:"))
            check(available > feature_file.stat().st_size + 4 * 1024**3,
                  "insufficient WSL memory for one category; preserve existing cache and retry after memory frees")
            features = np.load(feature_file, allow_pickle=False).reshape(-1, 10, 3, 256, 768)
            print(f"{category}: cached feature array ready, {len(features)} groups", flush=True)
            views = view_indices(groups, gpu)
            if args.repeat and category in protocol["repeated_categories"]:
                for seed in protocol["repeat_seeds"]:
                    one_seed(root, directory, category, seed, protocol["repeated_settings"], groups, features, views,
                             base, registry, cache_link["sha256"], args)
            elif not args.repeat:
                one_seed(root, directory, category, protocol["primary_seed"], protocol["ablation"], groups, features,
                         views, base, registry, cache_link["sha256"], args)
            del features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("initial", "bird", "holdout", "frozen", "audit", "output"):
        parser.add_argument("--" + key, type=lambda p: Path(p).resolve(), required=True)
    parser.add_argument("--category", choices=read_json(PROTOCOL)["ablation_order"])
    parser.add_argument("--repeat", action="store_true")
    args = parser.parse_args()
    check(not args.output.is_relative_to(REPO), "external output required")
    enable_offline()
    execute(args)


if __name__ == "__main__":
    main()
