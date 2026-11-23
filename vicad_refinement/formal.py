"""One frozen Synergy comparison per registered category and seed."""

from time import perf_counter

import numpy as np
import torch

from vicad_refinement.data import as_tensor, encode, load_features, load_groups, view_indices
from vicad_refinement.model import residuals, upsample, weighted_lights
from vicad_refinement.training import fit
from vicad_syn.evaluation.synergy import METRICS, baseline_helpers, official_inputs, record_files
from vicad_syn.experiment import check, digest, read_json, write_json


@torch.inference_mode()
def predict(features, views, reference, model, settings, registry, device):
    started = perf_counter()
    torch.cuda.reset_peak_memory_stats()
    contributions = {name: [] for name in settings}
    for offset in range(0, len(features), 4):
        batch = as_tensor(features, slice(offset, offset + 4), device)
        prior = reference[views[offset:offset + len(batch)]]
        scores, stability, epistemic, _ = residuals(batch, model, prior)
        for name, (candidate, fusion) in settings.items():
            maps = weighted_lights(scores[candidate], stability, epistemic, fusion, registry["protection"])
            check(bool(maps.isfinite().all()), f"nonfinite predictions at {name}/{offset}")
            contributions[name].append(maps.cpu().numpy())
    torch.cuda.synchronize()
    return {name: np.concatenate(items) for name, items in contributions.items()}, {
        "seconds": perf_counter() - started, "peak_cuda_bytes": torch.cuda.max_memory_allocated()}


def official_evaluation(folder, category, groups, dataset, contributions, metadata=None):
    from util.m2ad_utils import m2ad_fuse
    from util.metric import Evaluator

    helpers = baseline_helpers()
    folder.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    score_maps = upsample(torch.from_numpy(contributions)).numpy()
    if metadata is None:
        raw = official_inputs(groups, dataset, score_maps)
        metadata = {key: val for key, val in raw.items() if key not in ("anomaly_scores", "anomaly_maps")}
    else:
        raw = {**metadata, "anomaly_maps": score_maps, "anomaly_scores": score_maps.max(axis=(1, 2))}
    patch = folder / "patch-contributions.npy"
    np.save(patch, contributions, allow_pickle=False)
    with np.errstate(all="raise"):
        fused = m2ad_fuse(**raw, fuse="add")
        verified = helpers.check_fusion(raw, fused)
    check((verified["objects"], verified["views"]) == (len(groups) // 12, len(groups)), "unexpected fusion counts")
    for key, array in fused.items():
        np.save(folder / f"{key}.npy", array, allow_pickle=False)
    evaluator = Evaluator(metrics=list(METRICS.values()), use_adeval=True)
    result = evaluator.run(fused, category, "M2AD-Synergy")
    scores = {name: float(result[key]) * 100 for name, key in METRICS.items()}
    check(all(np.isfinite(x) and 0 <= x <= 100 for x in scores.values()), "nonfinite evaluator result")
    return {**scores, **verified, "images": len(groups)*10, "evaluation_seconds": perf_counter()-started,
            "fused_and_patch_inputs": record_files(folder)}, metadata


def evaluate_category(args, registry, frozen, base, encoder, device, ticket):
    output = args.output
    explorer = args.phase == "explore"
    category = args.category
    allowed = registry["development_categories" if explorer else "confirmatory_order"]
    check(category in allowed, "category not registered for this split")
    data_root = args.inputs if explorer else args.holdout
    check(data_root is not None, "missing separate verification data root")
    destination = output / ("exploratory" if explorer else "confirmatory")
    if not explorer:
        position = allowed.index(category)
        check(all(read_json(destination / before / "category.json") for before in allowed[:position]),
              "run confirmation categories in registry order")
    folder = destination / category
    folder.mkdir(parents=True, exist_ok=True)
    receipt_path = folder / "category.json"
    if previous := read_json(receipt_path):
        check(previous["freeze_sha256"] == digest(output / "freeze.json"), "frozen candidate changed")
        return
    expected = (30, 70) if explorer else registry["confirmatory_groups"][category]
    fixed = {"revision": base["data_revision"], "encoder": frozen["versions"]["vicad_refinement/model.py"],
             "weights": base["backbone_sha256"]}
    train = load_groups(data_root, category, base, expected, "train")
    cache_path = output / "development" / category / "train-features" if explorer else folder / "train-features"
    train_cache = encode(data_root, train, cache_path, encoder, fixed, device)
    train_features = load_features(cache_path)
    seeds = [registry["seed"]] if explorer else ticket["seeds"]
    train_ids = {item.identity for item in train}
    # All configurations and per-seed final checkpoints are committed before test encoding.
    models = {}
    for seed in seeds:
        models[seed] = fit(folder / f"seed-{seed}", train_cache, train, train_features, train_ids, seed, base, frozen, device)
    checkpoint_ids = {str(seed): models[seed][2]["checkpoint_sha256"] for seed in seeds}
    from vicad_refinement.run import now
    pretest = folder / "pretest.json"
    binding = {"input_freeze_sha256": digest(output / "freeze.json"), "checkpoints": checkpoint_ids}
    if previous := read_json(pretest):
        check(previous["binding"] == binding, "checkpoint edited since pre-test freeze")
    else:
        write_json(pretest, {"binding": binding, "utc": now()})
    test = load_groups(data_root, category, base, expected, "test")
    test_folder = folder / "test-features"
    test_cache = encode(data_root, test, test_folder, encoder, fixed, device)
    features = load_features(test_folder)
    views = view_indices(test, device)
    results = {}
    metadata = None
    selected = ticket["winner"]
    settings = {"v1_A4": ("A_final", "v1_global"),
                "selected": (selected["candidate"], selected["fusion"])}
    for seed, (model, proto, fitted) in models.items():
        seed_folder = folder / f"seed-{seed}"
        check(digest(seed_folder / "checkpoint.pt") == fitted["checkpoint_sha256"], "checkpoint edited after freeze")
        maps, costs = predict(features, views, proto, model, settings, registry, device)
        scores = {}
        for name, contribution in maps.items():
            scores[name], metadata = official_evaluation(seed_folder / name, category, test, data_root / "dataset",
                                                         contribution, metadata)
            print(f"{category}/{seed}/{name}: {[scores[name][k] for k in METRICS]}", flush=True)
        results[str(seed)] = {"training": fitted, "prediction": costs, "scores": scores}
        write_json(seed_folder / "evaluation.json", results[str(seed)])
    result = {"status": "PASS", "split": args.phase, "category": category, "freeze_sha256": digest(output / "freeze.json"),
              "pretest_sha256": digest(pretest), "train_features": train_cache, "test_features": test_cache,
              "train_objects": len(train_ids), "test_objects": len(test)//12, "seeds": results}
    write_json(receipt_path, result)
