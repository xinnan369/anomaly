"""Summarize the eight on-disk run records without inventing missing scores."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


MODELS = ("dinomaly", "inpformer")
CLASSES = ("Car", "Cube", "Doll", "Ring")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def row(root, evaluations, model, cls):
    run = root / model / cls
    evaluation = evaluations / model / cls
    record = read_json(run / "status.json")
    tested = read_json(evaluation / "status.json")
    if record.get("mode") == "preflight":
        raise ValueError(f"Preflight record found in formal run directory: {run}")
    epochs = read_json(run / "epochs.json")
    metrics = read_json(evaluation / "synergy.json").get("metrics_pct", {})
    trained = max(record.get("epochs_completed", 0), epochs[-1]["epoch"] if epochs else 0)
    samples = []
    if (run / "gpu.csv").is_file():
        with (run / "gpu.csv").open(encoding="utf-8", newline="") as csv_file:
            samples = list(csv.DictReader(csv_file))
    memory = [float(x["device_mib_used"]) for x in samples if x.get("device_mib_used", "").isdigit()]
    usage = [float(x["gpu_util_pct"]) for x in samples if x.get("gpu_util_pct", "").isdigit()]
    seconds = record.get("train_seconds")
    train_status = record.get("status", "NOT_STARTED")
    evaluation_status = tested.get("status", "NOT_STARTED")
    status = train_status
    if train_status == "PASS":
        status = {"NOT_STARTED": "TRAINED", "RUNNING": "EVALUATING"}.get(evaluation_status, evaluation_status)
    return {
        "model": model, "class": cls, "status": status,
        "train_status": train_status, "evaluation_status": evaluation_status,
        "train_started_utc": record.get("train_started_utc"), "train_finished_utc": record.get("train_finished_utc"),
        "epochs": trained, "mean_epoch_seconds": seconds / trained if seconds and trained else None,
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
    args = parser.parse_args()
    rows = [row(args.runs, args.evaluations, model, cls) for model in MODELS for cls in CLASSES]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
