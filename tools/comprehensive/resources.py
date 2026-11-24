"""Audit all local M2AD files and frozen TASK-004 artifact bindings."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
from datetime import datetime, timezone

from vicad_syn.experiment import check, digest, read_json, write_json

REPO = Path(__file__).resolve().parents[2]
MANIFEST = "7ac578bf7bef5148edccd318a55a73694eb9d216c87b7f9186a45e112ea27e43"
WEIGHT = "73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71"
UPSTREAM = "a606aa0e1614f23dad70da26430dd66095192141"
RELEASE = "05abc5d03167354a7ef98541236b5c0bd00abadf"
PROTOCOL = REPO / "vicad_syn/refinement/protocol.json"
FREEZE = "41fb52634e61a5023cd8172c83d9982f075eedfc65a6cf807bdca8286b4e503f"


def roots(category, initial, bird, holdout):
    if category in ("Car", "Cube", "Doll", "Ring"):
        return initial
    return bird if category == "Bird" else holdout


def scope_for(category):
    return "exploratory" if category in ("Car", "Cube", "Doll", "Ring") else "confirmatory"


def klass_files(dataset, category, manifest):
    split = [manifest[part][category] for part in ("train", "test")]
    ids = [{(row["cls_name"], row["object_name"], row["object_anomaly"]) for row in rows} for rows in split]
    check(ids[0].isdisjoint(ids[1]), "official product identity overlaps")
    check(all(row["object_anomaly"] == row["image_anomaly"] == 0 for row in split[0]), "not normal-only training")
    expected = {row[key] for rows in split for row in rows for key in ("img_path", "mask_path", "seg_path") if row[key]}
    found = {path.relative_to(dataset).as_posix(): path.stat().st_size
             for path in (dataset / category).rglob("*") if path.is_file()}
    check(set(found) == expected and min(found.values()) > 0, f"{category}: missing, empty or unexpected release files")
    return {"train_objects": len(ids[0]), "test_objects": len(ids[1]), "train_images": len(split[0]),
            "test_images": len(split[1]), "exact_member_paths": len(found), "extracted_bytes": sum(found.values())}


def class_receipt(args, category, registry):
    source = roots(category, args.initial, args.bird, args.holdout)
    dataset = source / "dataset"
    path = dataset / "jsons/meta_unsupervised.json"
    check(digest(path) == MANIFEST, "local release manifest differs")
    item = klass_files(dataset, category, read_json(path))
    audit_path = source / "data_audit" / f"{category}.json"
    audit = read_json(audit_path)
    expected = registry["confirmatory_groups"].get(category, (30, 70))
    check([item["train_objects"], item["test_objects"]] == list(expected)
          and audit["release_revision"] == RELEASE and audit["category"] == category
          and audit["images"] == item["train_images"] + item["test_images"]
          and audit["crc_verified_files"] == item["exact_member_paths"] and audit["split_crc_size_overlap"] == 0,
          "local data differs from archival receipt")
    published = registry["archive_sha256"].get(category)
    if published:
        check(audit["archive_sha256"] == published, "confirmation archive checksum differs")
    directory = args.frozen / scope_for(category) / category
    receipt = read_json(directory / "category.json")
    cached = {}
    for name in ("train", "test"):
        parent = args.frozen / "development" / category if name == "train" and scope_for(category) == "exploratory" else directory
        path = parent / f"{name}-features/features.npy"
        evidence = receipt[f"{name}_features"]
        check(path.stat().st_size == evidence["bytes"], "feature cache size differs")
        cached[name] = {"bytes": evidence["bytes"], "prior_sha256": evidence["sha256"]}
    pretest = read_json(directory / "pretest.json")
    check(digest(directory / "pretest.json") == receipt["pretest_sha256"]
          and pretest["binding"]["input_freeze_sha256"] == receipt["freeze_sha256"] == FREEZE,
          "frozen result receipt differs")
    checkpoints = {}
    for seed, checksum in pretest["binding"]["checkpoints"].items():
        folder = directory / f"seed-{seed}"
        check(digest(folder / "checkpoint.pt") == checksum, "frozen checkpoint digest differs")
        frozen = read_json(folder / "evaluation.json")
        predictions = {}
        for setting, output in frozen["scores"].items():
            files = output["fused_and_patch_inputs"]
            check(all((folder / setting / name).stat().st_size == metadata["bytes"] for name, metadata in files.items()),
                  "frozen prediction files differ in size")
            predictions[setting] = sum(value["bytes"] for value in files.values())
        checkpoints[seed] = {"sha256": checksum, "bytes": (folder / "checkpoint.pt").stat().st_size,
                             "prediction_bytes": predictions}
    return {**item, "root": str(dataset.resolve()), "historical_archive_sha256": audit["archive_sha256"],
            "historical_archive_bytes": audit["archive_bytes"], "crc_receipt_sha256": digest(audit_path),
            "feature_caches": cached, "checkpoints": checkpoints, "status": "PASS"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("initial", "bird", "holdout", "frozen", "output"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    args = parser.parse_args()
    check(not args.output.exists(), "retain existing audit")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    check(digest(args.frozen / "freeze.json") == FREEZE, "TASK-004 freeze changed")
    weight = args.initial / "inputs/weights/dinov2_vitb14_reg4_pretrain.pth"
    check(digest(weight) == WEIGHT, "frozen encoder weight differs")
    registry = read_json(PROTOCOL)
    report = {"checked_utc": datetime.now(timezone.utc).isoformat(), "release": RELEASE,
              "manifest_sha256": MANIFEST, "freeze_sha256": FREEZE, "weight_sha256": WEIGHT,
              "weight_bytes": weight.stat().st_size, "output_free_bytes": shutil.disk_usage(args.output.parent).free,
              "data_files_checked": "exact paths and nonzero lengths; archived CRC from existing receipts",
              "categories": {}}
    categories = registry["confirmatory_order"] + registry["development_categories"]
    for category in categories:
        try:
            result = class_receipt(args, category, registry)
        except (OSError, ValueError, TypeError, KeyError) as error:
            result = {"status": "BLOCKED", "error": str(error)}
        report["categories"][category] = result
        print(category, result["status"], flush=True)
    write_json(args.output, report)
    return any(item["status"] != "PASS" for item in report["categories"].values())


if __name__ == "__main__":
    raise SystemExit(main())
