"""Rehash external TASK-003 artifacts and cross-check approved test identities and masks."""

import argparse
import csv
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import platform

import numpy as np
import torch

from vicad_syn.data.manifest import load_groups
from vicad_syn.experiment import check, digest, protocol, read_json, write_json


KEYS = ("cls_names", "obj_names", "object_anomalys", "view_list", "illumination_list")
METRICS = ("O-AUROC", "I-AUROC", "AUPRO")


def write_csv(path, records, fields=None):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields or list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def ref(root, path, sha=None):
    return {"path": path.relative_to(root).as_posix(), "sha256": sha or digest(path), "bytes": path.stat().st_size}


def arrays(folder, evidence):
    files = {item.name: item for item in folder.glob("*.npy")}
    check(files.keys() == evidence.keys(), f"unexpected array inventory {folder}")
    for name, path in files.items():
        check(path.stat().st_size == evidence[name]["bytes"] and digest(path) == evidence[name]["sha256"],
              f"array hash mismatch: {path}")
    return {path.stem: np.load(path, mmap_mode="r", allow_pickle=False) for path in files.values()}


def compare_approved(raw, root, baseline):
    folder = root / baseline["evaluation_dir"] / "raw"
    names = (*KEYS, "image_anomalys", "imgs_masks")
    reference = {}
    for name in names:
        path = folder / (name + ".npy")
        expected = baseline["predictions"]["raw"][path.name]
        check(path.stat().st_size == expected["bytes"] and digest(path) == expected["sha256"],
              f"TASK-002 source array differs: {path}")
        reference[name] = np.load(path, mmap_mode="r", allow_pickle=False)
    indices = {key: i for i, key in enumerate(zip(*(reference[name] for name in KEYS), strict=True))}
    ours = list(zip(*(raw[name] for name in KEYS), strict=True))
    check(len(ours) == len(set(ours)) == len(indices) == 8400 and set(ours) == indices.keys(),
          "TASK-003 test identities differ from approved INP-Former")
    order = np.array([indices[key] for key in ours])
    check(np.array_equal(raw["image_anomalys"], reference["image_anomalys"][order]), "image labels differ")
    for start in range(0, len(order), 64):
        check(np.array_equal(raw["imgs_masks"][start:start + 64], reference["imgs_masks"][order[start:start + 64]]),
              f"resized test masks differ near row {start}")
    return {"images": 8400, "labels_equal": True, "masks_equal": True,
            "baseline_identity": "OFFICIAL_CLEAN / INP-Former"}


def fusion_check():
    file = Path(__file__).resolve().parents[2] / "tools/environment/baseline_metrics.py"
    spec = importlib.util.spec_from_file_location("baseline_metrics", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.check_fusion


def verify_category(root, home, category, freeze, baseline):
    cfg = freeze["config"]
    report = read_json(home / "result.json")
    check(report and report["status"] == "PASS" and report["config_sha256"] == freeze["config_sha256"],
          "category run is incomplete")
    train = load_groups(root, category, "train", cfg)
    test = load_groups(root, category, "test", cfg)
    train_ids, test_ids = {item.identity for item in train}, {item.identity for item in test}
    cache = {}
    for split, images in (("train", 3600), ("test", 8400)):
        folder = home / (split + "-cache")
        receipt = read_json(folder / "cache.json")
        feature = folder / "features.npy"
        check(receipt == report["cache"][split] and receipt["status"] == "PASS"
              and receipt["encoded_images"] == images and receipt["sha256"] == digest(feature), f"{split} cache incomplete")
        shaped = np.load(feature, mmap_mode="r", allow_pickle=False)
        check(shaped.shape == (images, 256, 768) and shaped.dtype == np.float16, f"{split} feature schema")
        cache[split] = {"receipt": receipt, "path": feature.relative_to(home.parent).as_posix()}
    train_evidence = {}
    for name in sorted({"development", *report["settings"].values()}):
        run = home / name
        receipt = read_json(run / "train.json")
        binding = receipt["binding"]
        check(receipt["status"] == "PASS" and binding["frozen"] == freeze and binding["cache_sha256"] == cache["train"]["receipt"]["sha256"],
              f"{name}: training receipt")
        ids, validation = set(binding["training_objects"]), set(binding["validation_objects"])
        check(ids.isdisjoint(validation | test_ids) and validation.isdisjoint(test_ids) and ids | validation == train_ids,
              "physical instance leakage")
        wanted = (24, 6, 3) if name == "development" else (30, 0, 12)
        check((len(ids), len(validation), binding["epochs"]) == wanted and binding["conditioned"] == (name != "unconditioned"),
              f"{name}: unexpected train plan")
        checkpoint = run / "checkpoint.pt"
        check(receipt["sha256"] == digest(checkpoint), f"{name}: checkpoint SHA mismatch")
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
        check(loaded["binding"] == binding and loaded["epoch"] == wanted[2], f"{name}: checkpoint binding")
        epoch_log = run / "epochs.json"
        epochs = read_json(epoch_log)
        if not epochs or len(epochs) != wanted[2]:
            # The optimizer checkpoint is written before the JSON journal at each epoch boundary.
            epoch_log = run / "latest.pt"
            saved = torch.load(epoch_log, map_location="cpu", weights_only=True)
            check(saved["binding"] == binding
                  and saved["model"].keys() == loaded["model"].keys()
                  and all(torch.equal(saved["model"][key], tensor) for key, tensor in loaded["model"].items()),
                  f"{name}: interrupted JSON journal is not recoverable from the final optimizer checkpoint")
            epochs = saved["epochs"]
        check([rec["epoch"] for rec in epochs] == list(range(1, wanted[2]+1))
              and sum(rec["steps"] for rec in epochs) == receipt["steps"], f"{name}: epoch ledger mismatch")
        train_evidence[name] = {
            "checkpoint": ref(home.parent, checkpoint, receipt["sha256"]),
            "receipt": ref(home.parent, run / "train.json"), "epoch_log": ref(home.parent, epoch_log),
            "epoch_count": wanted[2], "updates": receipt["steps"],
            "train_objects": len(ids), "validation_objects": len(validation),
        }
    compare_fusion = fusion_check()
    evidence = {"cache": cache, "development": report["development"], "training": train_evidence, "evaluations": {}}
    rows = []
    for name in sorted(set(report["settings"].values())):
        run = home / name
        receipt = read_json(run / "train.json")
        evaluation = run / "evaluation"
        run_eval = read_json(evaluation / "evaluation.json")
        binding = run_eval["binding"]
        planned = [mode for mode in cfg["settings"][category] if (mode == "A0") == (name == "unconditioned")]
        check(run_eval["status"] == "PASS" and run_eval["test_images"] == 8400 and binding["settings"] == planned
              and binding["checkpoint_sha256"] == receipt["sha256"]
              and binding["test_cache_sha256"] == cache["test"]["receipt"]["sha256"]
              and binding["config_sha256"] == freeze["config_sha256"], "evaluation is not bound to frozen artifacts")
        raw = arrays(evaluation / "raw", run_eval["raw"])
        compare = compare_approved(raw, root, baseline)
        signal = evaluation / "patch-signals.npz"
        check(digest(signal) == run_eval["patch_signals_sha256"], "signal evidence hash mismatch")
        evidence["evaluations"][name] = {"receipt": run_eval, "official_mask_check": compare,
                                         "checkpoint": ref(home.parent, run / "checkpoint.pt"),
                                         "patch_signals": ref(home.parent, signal, run_eval["patch_signals_sha256"])}
        for mode in planned:
            folder = evaluation / mode
            metrics = run_eval["metrics"][mode]
            check(read_json(folder / "metrics.json") == {"binding": binding, **metrics}
                  and all(np.isfinite(metrics[key]) and 0 <= metrics[key] <= 100 for key in METRICS),
                  f"{mode}: metric artifact mismatch")
            file = folder / "weights.npy"
            weights = np.load(file, allow_pickle=False)
            check(weights.shape == (840,10) and (weights >= 0).all() and np.isfinite(weights).all()
                  and np.allclose(weights.sum(1), 1) and digest(file) == metrics["weights_sha256"], f"{mode}: invalid weights")
            if mode in ("A0","A1"):
                check(np.array_equal(weights, np.full_like(weights, 0.1)), "unweighted control is not uniform")
                inputs = raw
            else:
                scale = (10 * weights).reshape(-1)
                inputs = {**raw, "anomaly_scores": raw["anomaly_scores"] * scale,
                          "anomaly_maps": raw["anomaly_maps"] * scale[:,None,None]}
            fused = arrays(folder / "fused", metrics["fused"])
            checked = compare_fusion(inputs, fused)
            check(checked == metrics["fusion"] and (checked["objects"], checked["views"]) == (70,840),
                  f"{mode}: object and view reductions differ")
            perf = run_eval["performance"]
            extra = bool(mode in ("A2","A3","A4"))
            rows.append({"class": category, "setting": mode, **{key: metrics[key] for key in METRICS},
                         "training_seconds": receipt["training_seconds"], "training_peak_cuda_bytes": receipt["peak_cuda_bytes"],
                         "test_encoder_seconds": cache["test"]["receipt"]["seconds"],
                         "predictor_seconds": perf["prediction_seconds"],
                         "uncertainty_seconds_shared": perf["uncertainty_seconds"] if extra else 0,
                         "prediction_peak_cuda_bytes": perf["peak_cuda_bytes"] if extra else perf["peak_cuda_base_bytes"],
                         "evaluation_seconds": metrics["evaluation_seconds"], "checkpoint_sha256": receipt["sha256"]})
            del inputs, fused
        del raw
    return rows, evidence


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("inputs", "run", "output"):
        p.add_argument(f"--{key}", type=lambda item: Path(item).resolve(), required=True)
    args = p.parse_args()
    check(not args.output.exists(), "write a new verification output directory")
    args.output.mkdir(parents=True)
    repo = Path(__file__).resolve().parents[2]
    index = read_json(repo / "results/baselines/metrics/run-index.json")
    approved = {value["class"]: value for value in index["runs"] if value["model"] == "inpformer"}
    expected = protocol()
    freeze = read_json(args.run / "frozen-protocol.json")
    check(freeze and freeze["frozen"] == expected, "current source differs from the frozen protocol")
    state = read_json(args.run / "pipeline.json")
    rows, evidence, faults = [], {}, {}
    for cls in expected["config"]["settings"]:
        try:
            verified, item = verify_category(args.inputs, args.run / cls, cls, expected, approved[cls])
            rows.extend(verified)
            evidence[cls] = item
            print(f"{cls}: verified {len(verified)} settings", flush=True)
        except (OSError, TypeError, ValueError, KeyError, AssertionError) as error:
            faults[cls] = f"{type(error).__name__}: {error}"
            print(f"{cls}: {faults[cls]}", flush=True)
    write_csv(args.output / "summary.csv", rows, list(rows[0]) if rows else ["class","setting",*METRICS])
    status = "PASS" if state["status"] == "PASS" and not faults else "PARTIAL"
    write_json(args.output / "verification.json", {"status": status, "pipeline": state, "faults": faults,
        "run_root": str(args.run), "utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {"platform": platform.platform(), "numpy": np.__version__, "torch": str(torch.__version__)},
        "summary_sha256": digest(args.output / "summary.csv"), "frozen": freeze, "categories": evidence})
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
