"""Audit a stabilized Doll attempt while keeping the official and historical identities separate."""

import argparse
import csv
from pathlib import Path
import platform

from run_baseline import digest, read_record, require, write_json
from stabilized_attention import IDENTITY, metadata
from summarize_baselines import CLASSES, DOLL, MODELS, row
from verify_baselines import Audit, METRICS, check_split, compare_labels, csv_rows, file_ref, run_audit, utc


OFFICIAL_CLEAN = "OFFICIAL_CLEAN"
OFFICIAL_FAILED = "OFFICIAL_FAILED"
NUMERICALLY_STABILIZED = "NUMERICALLY_STABILIZED"
RECOVERED_DEVIATION = "RECOVERED_DEVIATION"


def check_official(audit, root, summary, directory):
    index = read_record(directory / "run-index.json")
    verification = read_record(directory / "verification.json")
    require(verification["summary_sha256"] == digest(summary), "official audit was for another CSV")
    by_name = {group["name"]: group for group in verification["checks"]}
    expected = {f"{model}/{cls}" for model in MODELS for cls in CLASSES}
    require(set(by_name) == expected | {"shared_inputs", "paired_inputs"}, "unexpected official audit group names")
    failed = "dinomaly/Doll"
    for name in expected - {failed} | {"shared_inputs"}:
        require(by_name[name]["status"] == "PASS", f"unexpected official baseline failure: {name}")
    pairs = {check["check"]: check["status"] for check in by_name["paired_inputs"]["checks"]}
    require(pairs == {f"{cls}_same_raw_identities_labels_masks": "FAIL" if cls == "Doll" else "PASS"
                      for cls in CLASSES}, "unexpected scope of official pairing")
    checks = {item["check"]: item["status"] for item in by_name[failed]["checks"]}
    require(all(checks.get(name) == "PASS" for name in (
        "training_status_record", "official_json_and_summary_csv", "launch_source_receipt")),
        "official failure is missing its source or status evidence")
    evidence = root / "doll-clean-20261122-run"
    original = read_record(evidence / "train/status.json")
    stop = read_record(evidence / "train/first-nonfinite.json")
    require(original["status"] == "FAILED" and original["nonfinite"] and not original["oom"]
            and "final_checkpoint" not in original and not (evidence / "evaluation").exists()
            and stop["iteration"] == 11241 and stop["stage"] == "decoder_features",
            "official failure does not match archived evidence")
    table = {(item["model"], item["class"]): item for item in csv_rows(summary)}
    require(set(table) == {(m, c) for m in MODELS for c in CLASSES}, "incorrect official summary")
    for key, item in table.items():
        identity = OFFICIAL_FAILED if key == DOLL else OFFICIAL_CLEAN
        require(item["experiment_identity"] == identity, "official row identity mismatch")
        if identity == OFFICIAL_FAILED:
            require(not any(item[name] for name in METRICS), "failed official Doll cannot have formal metrics")
    for item in index["runs"]:
        item["experiment_identity"] = table[(item["model"], item["class"])]["experiment_identity"]
    index["historical_recovered_doll"]["experiment_identity"] = RECOVERED_DEVIATION
    audit.check("official_failed_evidence", lambda: file_ref(root, evidence / "train/first-nonfinite.json"))
    return index, verification


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "attempt", "official-summary", "official-audit", "output"):
        parser.add_argument(f"--{key}", required=True, type=lambda p: Path(p).resolve())
    args = parser.parse_args()
    root, attempt = args.root, args.attempt
    require(attempt.is_relative_to(root), "variant attempt must be outside the repository in the experiment root")
    require((attempt / "exit-code").is_file(), "wait for the formal pipeline before auditing")
    args.output.mkdir(parents=True, exist_ok=False)
    audit = Audit("official_and_variant_separation")
    official = audit.check("seven_originals_and_one_verified_official_failure", lambda: check_official(
        audit, root, args.official_summary, args.official_audit))
    require(official is not None, "official verification failed")
    index, original_verification = official
    model, category = DOLL
    manifest = read_record(root / "dataset/jsons/meta_unsupervised.json")
    split = check_split(manifest, category, read_record(root / "data_audit/Doll.json"), root / "archives/Doll.zip")
    variant = row(root / "runs", root / "evaluations", model, category, attempt)
    as_csv = {key: "" if value is None else str(value) for key, value in variant.items()}
    verified, experiment, raw = run_audit(root, model, category, attempt, manifest, as_csv, split, "stabilized")
    if raw:
        inpformer = next(item for item in index["runs"] if (item["model"], item["class"]) == ("inpformer", "Doll"))
        audit.check("variant_and_inpformer_share_doll_test_samples_masks_and_labels",
                    lambda: compare_labels(raw, inpformer["predictions"]["raw"]))
    else:
        audit.check("variant_and_inpformer_share_doll_test_samples_masks_and_labels",
                    lambda: require(False, "comparison is blocked without complete variant predictions"))
    require((root / "doll-clean-20261122-run/train/first-nonfinite.json").is_file(), "official failure evidence disappeared")
    readiness = "EVIDENCE_COMPLETE" if verified["status"] == audit.result()["status"] == "PASS" else "PARTIAL"
    binding = metadata("stabilized")
    experiment["experiment_identity"] = NUMERICALLY_STABILIZED
    experiment["model"] = IDENTITY
    experiment["verification_status"] = readiness
    public = {"experiment_identity": NUMERICALLY_STABILIZED, **variant, "model": IDENTITY,
              "verification_status": readiness, "epsilon": binding["epsilon"],
              "stability_patch_sha256": binding["stability_patch_sha256"]}
    if readiness != "EVIDENCE_COMPLETE":
        public.update(dict.fromkeys(METRICS))
    table = args.output / "stabilized-doll.csv"
    write_csv(table, [public])
    proof = root / "doll-clean-20261122-run/forensic-qk/forensic-replay.json"
    sources = [file_ref(root, attempt / name) for name in
               ("preflight.json", "launcher.sha256", "parent-commit", "gpu-environment.csv", "pipeline.log", "exit-code")]
    train = attempt / "train"
    failure_artifacts = [file_ref(root, name) for name in (
        attempt / "train.log", train / "gpu.csv", train / "batch-order.jsonl", train / "first-nonfinite.json",
        train / "attention-guards.json", train / "epochs.json",
    ) if name.is_file()] if readiness == "PARTIAL" else []
    index.update(status=readiness, official_reproduction_status="PARTIAL", verified_utc=utc(),
                 bounded_qk_replay=file_ref(root, proof), stabilized_doll=experiment,
                 stabilized_eligibility_and_launch=sources, stabilized_partial_artifacts=failure_artifacts)
    record = {"status": readiness, "official_reproduction_status": "PARTIAL", "checked_utc": utc(),
              "runtime": {"platform": platform.platform(), "python": platform.python_version()},
              "summary_sha256": digest(args.official_summary), "stabilized_csv_sha256": digest(table),
              "checks": original_verification["checks"], "separation_checks": audit.result(),
              "stabilized_checks": verified}
    write_json(args.output / "verification.json", record)
    write_json(args.output / "run-index.json", index)
    print(readiness, verified["name"], flush=True)
    return 0 if readiness == "EVIDENCE_COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
