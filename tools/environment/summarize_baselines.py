"""Summarize the eight on-disk run records without inventing missing scores."""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean


MODELS = ("dinomaly", "inpformer")
CLASSES = ("Car", "Cube", "Doll", "Ring")
DOLL = ("dinomaly", "Doll")
CLEAN_CONTINUOUS = "CLEAN_CONTINUOUS"
RECOVERED_DEVIATION = "RECOVERED_DEVIATION"
FAILED = "FAILED"


def run_paths(runs, evaluations, model, cls, doll_attempt=None):
    if doll_attempt is not None and (model, cls) == DOLL:
        return doll_attempt / "train", doll_attempt / "evaluation"
    return runs / model / cls, evaluations / model / cls


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def row(root, evaluations, model, cls, doll_attempt=None):
    run, evaluation = run_paths(root, evaluations, model, cls, doll_attempt)
    record = read_json(run / "status.json")
    tested = read_json(evaluation / "status.json")
    if record.get("mode") == "preflight":
        raise ValueError(f"Preflight record found in formal run directory: {run}")
    cfg = read_json(run / "config.json")
    epochs = read_json(run / "epochs.json")
    synergy = read_json(evaluation / "synergy.json")
    final_sha = record.get("final_checkpoint", {}).get("sha256")
    metrics = {}
    if synergy:
        if not final_sha or synergy["checkpoint_sha256"] != final_sha or synergy["epoch"] != 100:
            raise ValueError(f"Metrics do not match the training checkpoint: {evaluation}")
        if (synergy["model"], synergy["class"]) != (model, cls):
            raise ValueError(f"Unexpected model or category in {evaluation}")
    if tested.get("status") == "PASS":
        for file in ("synergy.json", "predictions.json", "diagnostics.csv", "gpu.csv"):
            if not (evaluation / file).is_file():
                raise ValueError(f"Missing result evidence: {evaluation / file}")
        if tested.get("evaluation_checkpoint_sha256") != final_sha:
            raise ValueError(f"Unexpected evaluated checkpoint: {evaluation}")
        metrics = synergy.get("metrics_pct", {})
        for name in ("AUROC_obj_sp", "AUROC_view_sp", "AUPRO_view"):
            if metrics.get(name) is None or not math.isfinite(metrics[name]):
                raise ValueError(f"Missing or nonfinite official {name}: {evaluation}")
    trained = max(record.get("epochs_completed", 0), epochs[-1]["epoch"] if epochs else 0)
    samples = []
    if (run / "gpu.csv").is_file():
        with (run / "gpu.csv").open(encoding="utf-8", newline="") as csv_file:
            samples = list(csv.DictReader(csv_file))
    memory = [float(x["device_mib_used"]) for x in samples if (x.get("device_mib_used") or "").isdigit()]
    usage = [float(x["gpu_util_pct"]) for x in samples if (x.get("gpu_util_pct") or "").isdigit()]
    seconds = record.get("train_seconds")
    train_status = record.get("status", "NOT_STARTED")
    evaluation_status = tested.get("status", "NOT_STARTED")
    status = train_status
    if train_status == "PASS":
        status = {"NOT_STARTED": "TRAINED", "RUNNING": "EVALUATING"}.get(evaluation_status, evaluation_status)
    provenance = None
    if train_status == "PASS":
        provenance = RECOVERED_DEVIATION if (run / "recovery.json").is_file() else CLEAN_CONTINUOUS
    elif train_status == "FAILED":
        provenance = FAILED
    return {
        "model": model, "class": cls, "status": status, "training_provenance": provenance,
        "train_status": train_status, "evaluation_status": evaluation_status,
        "train_started_utc": record.get("train_started_utc"), "train_finished_utc": record.get("train_finished_utc"),
        "epochs": trained, "iterations": record.get("optimization_steps", epochs[-1]["iteration"] if epochs else None),
        "last_observed_iteration": record.get("last_observed_iteration"),
        "seed": cfg.get("seed"), "release_revision": cfg.get("release_revision"),
        "upstream_revision": cfg.get("upstream_revision"), "config_sha256": cfg.get("cfg_sha256"),
        "backbone_sha256": cfg.get("weight_sha256"), "checkpoint_sha256": final_sha,
        "mean_epoch_seconds": seconds / trained if seconds and trained else None,
        "train_seconds": seconds, "peak_allocated_bytes": record.get("peak_allocated_bytes"),
        "peak_reserved_bytes": record.get("peak_reserved_bytes"),
        "peak_device_mib": max(memory, default=None), "mean_device_util_pct": mean(usage) if usage else None,
        "inference_seconds": tested.get("inference_through_concatenation_seconds"),
        "evaluation_peak_allocated_bytes": tested.get("peak_allocated_bytes"),
        "evaluation_peak_reserved_bytes": tested.get("peak_reserved_bytes"),
        "parameters": record.get("parameters"),
        "weight_bytes": sum(x["bytes"] for x in record["weights"]) if record.get("weights") else None,
        "oom": bool(record.get("oom") or tested.get("oom")) if record or tested else None,
        "nonfinite": bool(record.get("nonfinite") or tested.get("nonfinite")) if record or tested else None,
        "O-AUROC_pct": metrics.get("AUROC_obj_sp"), "I-AUROC_pct": metrics.get("AUROC_view_sp"),
        "AUPRO_pct": metrics.get("AUPRO_view"), "error": record.get("error") or tested.get("error"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--evaluations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--doll-attempt", type=Path, help="use this clean attempt, including if it failed")
    args = parser.parse_args()
    rows = [row(args.runs, args.evaluations, model, cls, args.doll_attempt)
            for model in MODELS for cls in CLASSES]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
