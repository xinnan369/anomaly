"""Verify the eight TASK-002 run records and their local, external artifacts."""

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path, PurePosixPath
import platform
import re
import sys
import zipfile

import numpy as np
import torch

from baseline_metrics import check_fusion, read_arrays
from prepare_release import ARCHIVE_SHA256, RELEASE_REVISION
from run_baseline import CONFIGS, REVISION, WEIGHT_NAME, WEIGHT_SHA
from run_baseline import check_upstream, digest, read_record, require, validate_checkpoint, write_json
from sources import MANIFEST
from stabilized_attention import UPSTREAM_ATTENTION, metadata
from summarize_baselines import (
    CLEAN_CONTINUOUS, CLASSES, DOLL, FAILED, MODELS, RECOVERED_DEVIATION, row, run_paths,
)


METRICS = {"O-AUROC_pct": "AUROC_obj_sp", "I-AUROC_pct": "AUROC_view_sp", "AUPRO_pct": "AUPRO_view"}
IDENTITY = {
    "obj_names": "object_name", "cls_names": "cls_name", "object_anomalys": "object_anomaly",
    "view_list": "view", "illumination_list": "illumination", "image_anomalys": "image_anomaly",
}
RAW_SHAPES = {**dict.fromkeys(IDENTITY, (8400,)), "anomaly_scores": (8400,),
              "anomaly_maps": (8400, 224, 224), "imgs_masks": (8400, 1, 224, 224)}
FUSED_SHAPES = {
    f"{scope}_instance_{suffix}": (size,)
    for scope, size in (("object", 70), ("view", 840))
    for suffix in ("names", "cls_names", "anomalys", "anomaly_scores")
}
FUSED_SHAPES.update(view_instance_anomaly_maps=(840, 224, 224), view_instance_masks=(840, 224, 224))
VIEWS = tuple(f"{angle:03d}" for angle in range(0, 360, 30))
LIGHTS = tuple(f"{light:02d}" for light in range(1, 11))
DIAGNOSTIC_KEYS = (
    {("view_fused", view, "") for view in VIEWS}
    | {("raw_light", "", light) for light in LIGHTS}
    | {("raw_configuration", view, light) for view in VIEWS for light in LIGHTS}
)
ERROR_PATTERN = re.compile(r"NaN/Inf loss|non-finite (?:loss|gradient|adam)|CUDA out of memory|OutOfMemoryError", re.I)
EXPECTED_PROBES = {"preprocessed_input", "encoder_features", "decoder_features", "loss",
                   "gradients_before_clip", "gradients_after_clip", "adam_moments", "updated_parameters"}


def utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def csv_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def file_ref(root, path, sha=None):
    return {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
            "sha256": digest(path) if sha is None else sha}


def recorded_path(value):
    """Resolve the recorded WSL drive path when performing the CPU audit on Windows."""
    parts = PurePosixPath(value).parts
    if platform.system() == "Windows" and parts[:2] == ("/", "mnt") and len(parts) >= 3 and len(parts[2]) == 1:
        return Path(f"{parts[2]}:/", *parts[3:])
    return Path(value)


def canonical_source(path):
    return path.read_bytes().replace(b"\r\n", b"\n").rstrip(b"\n")


def instance(row):
    return row["object_name"], row["cls_name"], row["object_anomaly"]


def check_split(manifest, category, marker, archive):
    train, test = (manifest[split][category] for split in ("train", "test"))
    train_ids, test_ids = ({instance(item) for item in records} for records in (train, test))
    train_paths, test_paths = ({item["img_path"] for item in records} for records in (train, test))
    require(len(train_ids) == 30 and len(test_ids) == 70, "manifest object counts")
    require(len(train_paths) == len(train) == 3600 and len(test_paths) == len(test) == 8400, "manifest image counts")
    require(train_ids.isdisjoint(test_ids) and train_paths.isdisjoint(test_paths), "train/test overlap")
    require(all(item["object_anomaly"] == item["image_anomaly"] == 0 for item in train), "non-normal train rows")
    require(marker["release_revision"] == RELEASE_REVISION and marker["category"] == category
            and marker["archive_sha256"] == ARCHIVE_SHA256[category]
            and marker["archive_bytes"] == archive.stat().st_size and marker["images"] == 12000
            and marker["crc_verified_files"] == sum(marker[k] for k in ("images", "masks", "segments"))
            and marker["split_crc_size_overlap"] == 0, "verified archive receipt")
    return {"train_images": len(train), "test_images": len(test), "train_instances": len(train_ids),
            "test_instances": len(test_ids), "instance_overlap": 0, "path_overlap": 0,
            "archive_receipt": marker}


def audited_files(directory, records, shapes):
    arrays = read_arrays(directory)
    require(set(arrays) == set(shapes) and set(records) == {key + ".npy" for key in shapes},
            f"missing or extra prediction arrays: {directory}")
    for name, shape in shapes.items():
        path = directory / (name + ".npy")
        evidence = records[path.name]
        array = arrays[name]
        require(array.shape == tuple(evidence["shape"]) == shape and str(array.dtype) == evidence["dtype"]
                and path.stat().st_size == evidence["bytes"] and digest(path) == evidence["sha256"],
                f"prediction mismatch: {path}")
    return arrays


def check_predictions(directory, records, test_rows):
    require(set(records) == {"raw", "fused"}, "unexpected prediction scopes")
    raw = audited_files(directory / "raw", records["raw"], RAW_SHAPES)
    fused = audited_files(directory / "fused", records["fused"], FUSED_SHAPES)
    expected = {tuple(row[key] for key in IDENTITY.values()) for row in test_rows}
    observed = set(zip(*(raw[key] for key in IDENTITY), strict=True))
    require(len(observed) == len(expected) == 8400 and observed == expected, "prediction identities differ from official manifest")
    check = check_fusion(raw, fused)
    require((check["objects"], check["views"]) == (70, 840), "fusion group coverage")
    return check


def check_diagnostics(path):
    rows = csv_rows(path)
    ids = {(item["scope"], item["view"], item["illumination"]) for item in rows}
    require(ids == DIAGNOSTIC_KEYS and len(rows) == len(ids), "diagnostics missing, duplicated, or out of scope")
    for item in rows:
        count = {"view_fused": 70, "raw_light": 840, "raw_configuration": 70}[item["scope"]]
        require(int(item["images"]) == count, "diagnostic group coverage")
        values = [item[name] for name in ("AUROC_pct", "AUPRO_pct")]
        if 0 < int(item["positive"]) < count:
            require(item["status"] == "OK" and all(value and 0 <= float(value) <= 100 for value in values),
                    "missing or invalid diagnostic metric")
        else:
            require(not any(values), "undefined diagnostic metric must be blank")
    return len(rows)


def check_metrics(directory, status, final_sha, model, category, attention_mode="official"):
    metrics = read_record(directory / "synergy.json")
    require(status["status"] == "PASS" and status["mode"] == "test" and status["epochs_completed"] == 100
            and (status["model"], status["class"]) == (model, category), "evaluation status")
    require(status["evaluation_checkpoint_sha256"] == metrics["checkpoint_sha256"] == final_sha
            and metrics["epoch"] == 100 and (metrics["model"], metrics["class"]) == (model, category),
            "metric evidence is not bound to the final checkpoint")
    binding = metadata(attention_mode)
    for record in (status, metrics):
        require(all(record.get(key, "official" if key == "attention_mode" else None) == value
                    for key, value in binding.items()), "metric attention mode or patch mismatch")
    scores = metrics["metrics_pct"]
    require(not metrics["nonfinite"] and len(scores) == 10 and set(METRICS.values()) <= scores.keys()
            and all(math.isfinite(value) and 0 <= value <= 100 for value in scores.values()), "official score coverage")
    return metrics


def check_repeat(primary, repeat):
    require(primary["checkpoint_sha256"] == repeat["checkpoint_sha256"], "repeat changed checkpoint")
    before, after = primary["metrics_pct"], repeat["metrics_pct"]
    require(before.keys() == after.keys(), "repeat reported different metrics")
    delta = {key: after[key] - value for key, value in before.items()}
    require(all(abs(value) <= 1e-10 for value in delta.values()), f"official metric repeat disagrees: {delta}")
    return delta


def check_batch_order(path, expected):
    """Verify each observed epoch is one full permutation of the official training set."""
    seen = set()
    count = 0
    with path.open(encoding="utf-8") as stream:
        for count, line in enumerate(stream, 1):
            batch = json.loads(line)
            paths = set(batch["paths"])
            require(batch["iteration"] == count and len(batch["paths"]) == len(paths) == 12
                    and paths <= expected and seen.isdisjoint(paths), f"invalid sampled batch at iteration {count}")
            seen.update(paths)
            if count % 300 == 0:
                require(seen == expected, f"incomplete sampled epoch ending at iteration {count}")
                seen.clear()
    require(count == 30_000 and not seen, "batch-order log did not cover 100 epochs")
    return count


class Audit:
    def __init__(self, name):
        self.name = name
        self.checks = []

    def check(self, name, action):
        try:
            value = action()
        except (AssertionError, EOFError, OSError, RuntimeError, ValueError, TypeError, KeyError) as error:
            self.checks.append({"check": name, "checked_utc": utc(), "status": "FAIL",
                                "error": f"{type(error).__name__}: {error}"})
            print(f"FAIL {self.name}: {name}: {error}", file=sys.stderr, flush=True)
            return None
        self.checks.append({"check": name, "checked_utc": utc(), "status": "PASS"})
        return value

    def result(self):
        return {"name": self.name, "status": "PASS" if all(c["status"] == "PASS" for c in self.checks) else "FAIL",
                "checks": self.checks}


def run_audit(root, model, category, doll_attempt, manifest, summary, split, attention_mode="official"):
    train, test = run_paths(root / "runs", root / "evaluations", model, category, doll_attempt)
    upstream = root / "inputs/upstream"
    clean_doll = (model, category) == DOLL
    binding = metadata(attention_mode)
    stabilized = attention_mode == "stabilized"
    audit = Audit(f"{binding.get('model_identity', model)}/{category}")
    index = {**binding, "model": model, "class": category, "train_dir": train.relative_to(root).as_posix(),
             "evaluation_dir": test.relative_to(root).as_posix()}
    training = audit.check("training_status_record", lambda: read_record(train / "status.json"))
    if training is not None:
        index["training"] = training
        if training["status"] == "FAILED":
            index["training_provenance"] = FAILED
    logs = ([doll_attempt / "train.log", doll_attempt / "evaluation.log"] if clean_doll else
            [root / "logs" / f"{model}-{category}-{mode}.log" for mode in ("train", "test")])

    def check_training():
        require(training["status"] == "PASS" and training["mode"] == "train"
                and (training["epochs_completed"], training["optimization_steps"]) == (100, 30_000)
                and (training["model"], training["class"]) == (model, category),
                f"training failed or incomplete: {training.get('status')} {training.get('error', '')}")
        config = read_record(train / "config.json")
        require(config["cfg_path"] == CONFIGS[model] and config["cfg_sha256"] == digest(upstream / CONFIGS[model])
                and config["upstream_revision"] == REVISION and config["release_revision"] == RELEASE_REVISION
                and config["weight_sha256"] == WEIGHT_SHA, "run source or configuration mismatch")
        index["configuration"] = config
        evidence = read_record(train / "epochs.json")
        require([(item["epoch"], item["iteration"]) for item in evidence]
                == [(epoch, epoch * 300) for epoch in range(1, 101)], "incomplete or interrupted epoch sequence")
        argv = training["argv"]
        require("--checkpoint" not in argv and ("--diagnose-doll" in argv) == clean_doll,
                "unexpected initialization arguments")
        require(not training["oom"] and not training["nonfinite"] and not (train / "recovery.json").exists()
                and not ERROR_PATTERN.search(logs[0].read_text(encoding="utf-8")), "training exception or recovery recorded")
        index["training_provenance"] = CLEAN_CONTINUOUS
        index["epoch_log"] = file_ref(root, train / "epochs.json")
        index["training_logs"] = [file_ref(root, logs[0])]
        index["train_gpu_csv"] = file_ref(root, train / "gpu.csv")

    audit.check("epochs_30000_steps_no_recorded_recovery", check_training)

    def check_final():
        require(training and "final_checkpoint" in training, "no final epoch-100 checkpoint from training")
        record = training["final_checkpoint"]
        checkpoint = train / record["path"]
        sha = validate_checkpoint(checkpoint, model, category, upstream, attention_mode)
        index["checkpoint"] = file_ref(root, checkpoint, sha)
        index["weights"] = [file_ref(root, train / weight["path"]) for weight in training["weights"]]
        require(index["weights"] and all(
            got["sha256"] == expected["sha256"] and got["bytes"] == expected["bytes"]
            for got, expected in zip(index["weights"], training["weights"], strict=True)), "weight export mismatch")
        return sha

    final_sha = audit.check("final_checkpoint_and_weights", check_final)
    status = audit.check("evaluation_status_record", lambda: read_record(test / "status.json"))
    if status is not None:
        index["evaluation"] = status
    scores = audit.check("official_metrics_frozen_checkpoint", lambda: check_metrics(test, status, final_sha, model, category, attention_mode))
    if scores is not None:
        index["official_metrics"] = scores

    def check_eval_argument(eval_status):
        require(eval_status is not None and final_sha is not None, "evaluation is blocked without a final checkpoint and evaluation status")
        require(not eval_status["oom"] and not eval_status["nonfinite"], "evaluation exception recorded")
        argv = eval_status["argv"]
        require("--checkpoint" in argv and recorded_path(argv[argv.index("--checkpoint") + 1])
                == train / training["final_checkpoint"]["path"], "evaluation invoked with a different checkpoint")

    audit.check("evaluation_checkpoint_argument", lambda: check_eval_argument(status))

    def check_summary():
        provenance = row(root / "runs", root / "evaluations", model, category, doll_attempt)
        require(str(provenance["training_provenance"] or "") == summary["training_provenance"], "summary provenance disagrees")
        for field, value in provenance.items():
            if field in METRICS and value is not None:
                require(float(summary[field]) == scores["metrics_pct"][METRICS[field]] == value,
                        f"official metric and summary differ: {field}")
            else:
                require(summary[field] == ("" if value is None else str(value)), f"CSV differs from run record: {field}")

    audit.check("official_json_and_summary_csv", check_summary)

    def check_outputs():
        require(split is not None, "fixed dataset manifest failed the split audit")
        predictions = read_record(test / "predictions.json")
        audit_result = check_predictions(test, predictions, manifest["test"][category])
        require(audit_result == status["fusion_crosscheck"], "live and recorded fusion checks disagree")
        test_log = logs[1].read_text(encoding="utf-8")
        require("700/700" in test_log and "AUPRO_view" in test_log, "official evaluation log incomplete")
        index.update(predictions=predictions, evaluation_log=file_ref(root, logs[1]),
                     evaluation_gpu_csv=file_ref(root, test / "gpu.csv"),
                     diagnostics_csv=file_ref(root, test / "diagnostics.csv"),
                     diagnostics_rows=check_diagnostics(test / "diagnostics.csv"))
        return predictions["raw"]

    raw = audit.check("predictions_and_fusion_and_142_diagnostics", check_outputs)

    if clean_doll:
        def check_launcher():
            receipt = (doll_attempt / "launcher.sha256").read_text(encoding="utf-8").splitlines()
            executed = {PurePosixPath(name).name: sha for sha, name in (line.split(maxsplit=1) for line in receipt)}
            expected_files = {"run_baseline.py", "training_diagnostics.py", "baseline_metrics.py"}
            if stabilized:
                expected_files.update({"stabilized_attention.py", "diagnose_doll_failure.py", "check_doll_stability.py"})
            require(set(executed) == expected_files, "launcher source receipt has unexpected files")
            copies = []
            for name, sha in executed.items():
                current = Path(__file__).with_name(name)
                snapshot = doll_attempt / "source" / name
                if snapshot.is_file():
                    require(digest(snapshot) == sha, f"executed source snapshot hash mismatch: {name}")
                    if stabilized:
                        require(canonical_source(snapshot) == canonical_source(current), f"source differs from executed code: {name}")
                    copies.append(file_ref(root, snapshot, sha))
                else:
                    require(digest(current) == sha, f"current source differs from launcher receipt: {name}")
            index["source_snapshots"] = copies

        def check_probes():
            probes = read_record(train / "finite-diagnostics.json")
            require(probes["status"] == "PASS" and probes["successful_updates"] == 30_000
                    and probes["stage_check_counts"] == {"initial_parameters": 1, **dict.fromkeys(EXPECTED_PROBES, 30_000)}
                    and not (train / "first-nonfinite.json").exists(), "continuous finite probes incomplete")
            expected_rng = {"initial.pt"} | {f"epoch-{n:03d}.pt" for n in range(1, 101)}
            actual_rng = {p.name for p in (train / "rng").glob("*.pt")}
            require(actual_rng == expected_rng, "RNG boundary snapshots incomplete")
            check_batch_order(train / "batch-order.jsonl", {item["img_path"] for item in manifest["train"][category]})
            index["diagnostic_artifacts"] = [file_ref(root, train / name)
                for name in ("batch-order.jsonl", "diagnostic-environment.json", "dependency-freeze.txt", "finite-diagnostics.json")]
            index["rng_states"] = [file_ref(root, train / "rng" / name) for name in sorted(expected_rng)]
            index["launcher_receipts"] = [file_ref(root, doll_attempt / name)
                for name in ("launcher.sha256", "parent-commit", "gpu-environment.csv", "pipeline.log", "exit-code")]

        def check_re_evaluation():
            require(final_sha is not None and scores is not None, "repeat is blocked without a successful formal evaluation")
            folder = doll_attempt / "re-evaluation"
            repeated_status = read_record(folder / "status.json")
            repeat = check_metrics(folder, repeated_status, final_sha, model, category, attention_mode)
            check_eval_argument(repeated_status)
            delta = check_repeat(scores, repeat)
            predictions = read_record(folder / "predictions.json")
            require(check_predictions(folder, predictions, manifest["test"][category])
                    == repeated_status["fusion_crosscheck"], "repeat fusion mismatch")
            compare_labels(raw, predictions["raw"])
            after_sha = digest(train / training["final_checkpoint"]["path"])
            require(after_sha == final_sha, "checkpoint differs after repeat")
            index["repeat_evaluation"] = {
                "official_metrics": repeat, "metric_deltas_pct": delta, "predictions": predictions,
                "checkpoint_sha256_after_repeat": after_sha, "log": file_ref(root, doll_attempt / "re-evaluation.log"),
                "diagnostics_rows": check_diagnostics(folder / "diagnostics.csv"),
                "diagnostics_csv": file_ref(root, folder / "diagnostics.csv"),
                "status": file_ref(root, folder / "status.json"),
            }

        audit.check("launch_source_receipt", check_launcher)
        audit.check("finite_probes_rng_and_full_batch_order", check_probes)
        audit.check("read_only_repeat_same_frozen_checkpoint", check_re_evaluation)

    if stabilized:
        def check_guards():
            preflight = read_record(doll_attempt / "preflight.json")
            require(preflight["status"] == "PASS" and preflight["epsilon"] == binding["epsilon"]
                    and preflight["patch_sha256"] == binding["stability_patch_sha256"],
                    "pre-training numerical gate does not match")
            forward_hash = training["patched_forward_sha256"]
            source_hash = digest(upstream / UPSTREAM_ATTENTION)
            audit_data = {}
            for label, folder, batches in (("train", train, 30_000), ("evaluation", test, 700),
                                            ("repeat", doll_attempt / "re-evaluation", 700)):
                evaluated = read_record(folder / "status.json")
                counts, ref = check_guard_record(root, folder, evaluated, batches, binding)
                require(evaluated["patched_forward_sha256"] == digest(folder / "patched-forward.py.txt") == forward_hash
                        and evaluated["official_attention_source_sha256"] == source_hash,
                        f"{label}: executed forward or upstream source hash mismatch")
                audit_data[label] = {"protected_positions": counts["protected_positions"], "zero_positions": counts["zero_positions"],
                                     "batches_with_protection": len(counts["batches_with_protection"]),
                                     "by_layer": counts["by_layer"], "record": ref,
                                     "patched_forward": file_ref(root, folder / "patched-forward.py.txt", forward_hash)}
            require(audit_data["evaluation"]["by_layer"] == audit_data["repeat"]["by_layer"],
                    "evaluation guard counts changed on repeat")
            index["stability_evidence"] = audit_data

        audit.check("full_guard_counts_and_forward_sha256", check_guards)

    return audit.result(), index, raw


def check_guard_record(root, folder, status, batches, binding):
    guard_file = folder / "attention-guards.json"
    counts = read_record(guard_file)
    require(counts["model_identity"] == binding["model_identity"] and counts["epsilon"] == binding["epsilon"]
            and counts["patch_sha256"] == binding["stability_patch_sha256"]
            and counts["attention_calls"] == batches * 8 and counts["batches_observed"] == batches
            and counts["nonfinite_denominator_positions"] == counts["negative_positions"] == 0
            and counts["protected_positions"] == status["attention_guard_positions"],
            "guard coverage or recorded counts differ")
    layers = {item["layer"]: item for item in counts["by_layer"]}
    observed = {name: {"protected": 0, "zero": 0} for name in layers}
    prior = 0
    for event in counts["batches_with_protection"]:
        batch = event["batch"]
        require(prior < batch <= batches and event["epoch"] == (batch - 1) // 300 + 1,
                "guard event batch order or epoch differs")
        prior = batch
        require(len({row["layer"] for row in event["layers"]}) == len(event["layers"]) > 0,
                "guard event repeats a layer or has no triggers")
        for row in event["layers"]:
            require(0 <= row["zero"] <= row["protected"] and row["protected"] > 0, "invalid guard counts")
            for name in ("protected", "zero"):
                observed[row["layer"]][name] += row[name]
    require(len(layers) == len(counts["by_layer"]) == 8
            and all(layers[layer][name] == value
                    for layer, values in observed.items() for name, value in values.items())
            and sum(row["protected"] for row in layers.values()) == counts["protected_positions"]
            and sum(row["zero"] for row in layers.values()) == counts["zero_positions"],
            "guard event sums disagree with layer totals")
    return counts, file_ref(root, guard_file)


def compare_labels(first, second):
    require(first is not None and second is not None, "cross-model comparison requires both formal evaluations")
    require(all(first[name + ".npy"] == second[name + ".npy"] for name in (*IDENTITY, "imgs_masks")),
            "original inputs, labels, or masks differ between matching runs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "doll-attempt", "summary", "output"):
        parser.add_argument(f"--{name}", required=True, type=lambda value: Path(value).resolve())
    args = parser.parse_args()
    root, attempt = args.root, args.doll_attempt
    require(attempt.is_relative_to(root), "the isolated attempt must be beneath the external data root")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest_file = root / "dataset/jsons/meta_unsupervised.json"
    manifest = read_record(manifest_file)
    rows = csv_rows(args.summary)
    summary = {(item["model"], item["class"]): item for item in rows}
    require(set(summary) == {(model, cls) for model in MODELS for cls in CLASSES}
            and len(rows) == len(summary), "summary must have eight unique groups")
    audit = Audit("shared_inputs")
    upstream = root / "inputs/upstream"
    audit.check("pinned_official_source", lambda: check_upstream(upstream))
    weight_path = root / "inputs/weights" / WEIGHT_NAME

    def check_weight():
        weight = file_ref(root, weight_path)
        require(weight["sha256"] == WEIGHT_SHA, "pretrained backbone mismatch")
        return weight

    weight = audit.check("pretrained_backbone_sha256", check_weight)

    def check_manifest_pin():
        archive = root / "cache" / MANIFEST.name
        require(digest(archive) == MANIFEST.sha256, "cached release manifest ZIP SHA-256 mismatch")
        with zipfile.ZipFile(archive) as zipped, zipped.open(manifest_file.name) as source:
            require(json.load(source) == manifest, "staged manifest differs from the pinned release")
        return file_ref(root, archive, MANIFEST.sha256)

    manifest_zip = audit.check("staged_manifest_matches_pinned_release", check_manifest_pin)
    splits = {}
    for cls in CLASSES:
        splits[cls] = audit.check(f"{cls}_instance_isolation_and_verified_archive_receipt",
                                  lambda: check_split(manifest, cls, read_record(root / "data_audit" / f"{cls}.json"),
                                                      root / "archives" / f"{cls}.zip"))
    results, indices, raw_refs = [], [], {}
    for model in MODELS:
        for cls in CLASSES:
            result, index, refs = run_audit(root, model, cls, attempt, manifest, summary[(model, cls)], splits[cls])
            results.append(result)
            indices.append(index)
            raw_refs[(model, cls)] = refs
            print(f"{result['status']} {result['name']}", flush=True)

    pairs = Audit("paired_inputs")
    for cls in CLASSES:
        pairs.check(f"{cls}_same_raw_identities_labels_masks",
                    lambda: compare_labels(*(raw_refs[(model, cls)] for model in MODELS)))

    historical_train, historical_test = run_paths(root / "runs", root / "evaluations", *DOLL)
    historical = {}

    def check_recovery_history():
        status = read_record(historical_train / "status.json")
        recovery = read_record(historical_train / "recovery.json")
        scores = check_metrics(historical_test, read_record(historical_test / "status.json"),
                               status["final_checkpoint"]["sha256"], *DOLL)
        require(recovery["rng_and_loader_state_available"] is False, "unexpected historical recovery record")
        failed = root / "attempts/dinomaly/Doll-original-nonfinite-20261120"
        failed_status = read_record(failed / "status.json")
        require(failed_status["status"] == "FAILED" and failed_status["nonfinite"] and "11241" in failed_status["error"],
                "original NaN record missing")
        early_checkpoint = failed / failed_status["intermediate_checkpoint"]["path"] if (
            "intermediate_checkpoint" in failed_status) else next((failed / "checkpoints").glob("*/*_ckpt.pth"), None)
        require(early_checkpoint is not None and digest(early_checkpoint) == recovery["training_checkpoint_sha256"],
                "original recovery input checkpoint mismatch")
        require(digest(historical_train / status["final_checkpoint"]["path"]) == scores["checkpoint_sha256"],
                "historical recovery checkpoint mismatch")
        historical.update(training_provenance=RECOVERED_DEVIATION, official_metrics=scores,
                          original_failed_status=file_ref(root, failed / "status.json"),
                          original_checkpoint=file_ref(root, early_checkpoint, recovery["training_checkpoint_sha256"]),
                          original_log=file_ref(root, root / "logs/dinomaly-Doll-train.log"),
                          training_log=file_ref(root, root / "logs/dinomaly-Doll-resume-20261121.log"),
                          recovery=file_ref(root, historical_train / "recovery.json"),
                          checkpoint=file_ref(root, historical_train / status["final_checkpoint"]["path"],
                                              scores["checkpoint_sha256"]),
                          evaluation=file_ref(root, historical_test / "synergy.json"),
                          prediction_manifest=file_ref(root, historical_test / "predictions.json"),
                          evaluation_log=file_ref(root, root / "logs/dinomaly-Doll-test.log"),
                          evaluation_status=file_ref(root, historical_test / "status.json"),
                          training_status=file_ref(root, historical_train / "status.json"))

    audit.check("historical_recovery_preserved", check_recovery_history)
    checks = [audit.result(), *results, pairs.result()]
    passed = all(item["status"] == "PASS" for item in checks)
    runtime = {"platform": platform.platform(), "python": platform.python_version(),
               "torch": str(torch.__version__), "numpy": np.__version__}
    write_json(args.output / "verification.json", {"status": "PASS" if passed else "PARTIAL",
        "checked_utc": utc(), "runtime": runtime, "summary_sha256": digest(args.summary), "checks": checks})
    write_json(args.output / "run-index.json", {"verified_utc": utc(), "status": "PASS" if passed else "PARTIAL",
        "official_source": REVISION, "release_revision": RELEASE_REVISION,
        "storage_root": str(root), "manifest": file_ref(root, manifest_file),
        "release_manifest_archive": manifest_zip, "pretrained_backbone": weight,
        "archive_note": "Previously verified SHA-256 and CRC receipts and current file sizes were checked.",
        "archive_and_split_checks": splits, "historical_recovered_doll": historical, "runs": indices})
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
