"""Cross-check existing TASK-004 receipts and export only actual measured rows."""

import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np

from vicad_refinement.development import csv_write, select
from vicad_syn.evaluation.synergy import METRICS
from vicad_syn.experiment import check, digest, read_json, write_json


def load_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def check_files(folder, hashes):
    for filename, value in hashes.items():
        path = folder / filename
        check(digest(path) == (value["sha256"] if isinstance(value, dict) else value), f"changed artifact {path}")
        if isinstance(value, dict):
            check(path.stat().st_size == value["bytes"], f"wrong artifact size {path}")


def summarize_mechanism(rows):
    index = {}
    for row in rows:
        index.setdefault((row["scenario"], row["context"]), []).append(row)
    outputs = []
    keys = ("prediction_feature_shift", "target_feature_shift", "target_residual", "stability", "epistemic", "pixel_ap")
    for (scenario, context), selected in index.items():
        stats = {}
        for key in keys:
            items = [float(row[key]) for row in selected if row[key]]
            stats[key] = float(np.mean(items)) if items else None
        outputs.append({"scenario": scenario, "context": context, "views": len(selected), **stats})
    return outputs


def development_cells(rows):
    cells = {}
    for row in rows:
        key = tuple(row[field] for field in ("category", "scenario", "candidate", "fusion"))
        cells.setdefault(key, []).append(row)
    output = []
    for (category, scenario, feature, fusion), items in cells.items():
        aps = [float(row["pixel_ap"]) for row in items if row["pixel_ap"]]
        output.append({"category": category, "scenario": scenario, "candidate": feature, "fusion": fusion,
                       "views": len(items), "mean_pixel_ap": float(np.mean(aps)) if aps else None,
                       "mean_fraction_above_clean_p99": float(np.mean([float(x["above_clean_p99"]) for x in items]))})
    return output

def verify_formal(run, folder, ticket, registry, allowed, written):
    results = []
    for category in allowed:
        category_dir = folder / category
        receipt = read_json(category_dir / "category.json")
        if not receipt:
            continue
        scope = folder.name
        expected = registry["confirmatory_groups"][category] if scope == "confirmatory" else [30, 70]
        check(receipt["freeze_sha256"] == digest(run / "freeze.json"), "formal result freeze binding differs")
        check((receipt["train_objects"], receipt["test_objects"]) == tuple(expected), "formal result has wrong split sizes")
        check(receipt["pretest_sha256"] == digest(category_dir / "pretest.json"), "changed timestamp gate")
        gate = read_json(category_dir / "pretest.json")
        check(gate["binding"]["input_freeze_sha256"] == digest(run / "freeze.json") and
              datetime.fromisoformat(gate["utc"]) >= datetime.fromisoformat(ticket["sealed_utc"]), "pretest gate chronology differs")
        ticket_checkpoint = gate["binding"]["checkpoints"]
        for cache_name in ("train", "test"):
            directory = (run / "development" / category / "train-features" if scope == "exploratory"
                         and cache_name == "train" else category_dir / f"{cache_name}-features")
            check(digest(directory / "features.npy") == receipt[f"{cache_name}_features"]["sha256"], "feature cache changed")
        seen = {"confirmatory": ticket["seeds"], "exploratory": [registry["seed"]]}[scope]
        check(sorted(map(int, receipt["seeds"])) == sorted(seen), "missing or unexpected seeds")
        for seed, payload in receipt["seeds"].items():
            train = payload["training"]
            directory = category_dir / f"seed-{seed}"
            check(train["checkpoint_sha256"] == ticket_checkpoint[seed] == digest(directory / "checkpoint.pt"),
                  "prediction checkpoint and pretest differ")
            check(len(train["binding"]["training_objects"]) == expected[0], "incorrect normal-only training coverage")
            check(train["steps"] == 12 * ((expected[0] * 12 + 3) // 4), "incorrect optimization steps")
            check(payload == read_json(directory / "evaluation.json"), "seed and aggregate receipts differ")
            check(set(payload["scores"]) == {"v1_A4", "selected"}, "missing paired v1/selected setting")
            for name, scores in payload["scores"].items():
                check_files(directory / name, scores["fused_and_patch_inputs"])
                check((scores["objects"], scores["views"], scores["images"]) == (expected[1], 12*expected[1], 120*expected[1]),
                      "formal fusion had incomplete groups")
                results.append({"scope": scope, "category": category, "seed": seed, "setting": name,
                                **{k: scores[k] for k in METRICS}, "fit_seconds": train["seconds"],
                                "train_peak_bytes": train["peak_cuda_bytes"], "prediction_shared_seconds": payload["prediction"]["seconds"],
                                "prediction_peak_bytes": payload["prediction"]["peak_cuda_bytes"],
                                "train_feature_seconds": receipt["train_features"]["seconds"],
                                "test_feature_seconds": receipt["test_features"]["seconds"],
                                "test_feature_peak_bytes": receipt["test_features"]["peak_cuda_bytes"],
                                "evaluation_seconds": scores["evaluation_seconds"], "parameters": train["trainable_parameters"],
                                "checkpoint_sha256": train["checkpoint_sha256"]})
        written[f"{scope}/{category}"] = digest(category_dir / "category.json")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("run", "diagnostics", "output"):
        parser.add_argument(f"--{argument}", type=lambda p: Path(p).resolve(), required=True)
    args = parser.parse_args()
    run = args.run
    folder = args.output
    folder.mkdir(parents=True, exist_ok=True)
    registry_file = Path(__file__).resolve().parents[2] / "vicad_syn/refinement/protocol.json"
    registry = read_json(registry_file)
    development = read_json(run / "development/development.json")
    ticket = read_json(run / "freeze.json")
    output = {"diagnostic_artifacts": {}, "formal_receipts": {}, "stages": {}}
    for mode in ("localization", "replay"):
        if info := read_json(args.diagnostics / f"{mode}.json"):
            check_files(args.diagnostics, info["outputs"])
            output["diagnostic_artifacts"].update(info["outputs"])
    if development:
        codes = []
        all_mechanisms = []
        for category, record in development["categories"].items():
            path = run / "development" / category
            check_files(path, record["sha256"])
            check(set(record["normal_validation_objects"]).isdisjoint(record["training"]["binding"]["training_objects"]),
                  "normal-only development leak")
            check(digest(path / "fit/checkpoint.pt") == record["training"]["checkpoint_sha256"], "development checkpoint mismatch")
            check(digest(path / "train-features/features.npy") == record["features"]["sha256"], "development features differ")
            codes.extend(load_rows(path / "proxy-rows.csv"))
            all_mechanisms.extend(load_rows(path / "mechanism-rows.csv"))
        choice, rank = select(codes, registry)
        check(development["winner"] == choice, "normal-only winner mismatch")
        csv_write(folder / "development-matrix.csv", rank)
        csv_write(folder / "contamination-summary.csv", summarize_mechanism(all_mechanisms))
        csv_write(folder / "development-cells.csv", development_cells(codes))
        csv_write(folder / "development-rows.csv", codes)
        csv_write(folder / "contamination-rows.csv", all_mechanisms)
        output["stages"]["development"] = "PASS"
    formal = []
    if ticket:
        check(digest(run / "development/development.json") == ticket["development_sha256"] and
              digest(run / "development/selection.csv") == ticket["selection_sha256"] and
              digest(registry_file) == ticket["binding"]["registered_sha256"], "preconfirm frozen protocol changed")
        output["stages"]["freeze"] = ticket["sealed_utc"]
        output["freeze_sha256"] = digest(run / "freeze.json")
        for scope, groups in (("exploratory", registry["development_categories"]),
                              ("confirmatory", registry["confirmatory_order"])):
            formal.extend(verify_formal(run, run / scope, ticket, registry, groups, output["formal_receipts"]))
    if formal:
        csv_write(folder / "formal.csv", formal)
        partitions = sorted({(r["scope"], r["category"], r["setting"]) for r in formal})
        means = []
        for scope, cat, setting in partitions:
            part = [r for r in formal if (r["scope"], r["category"], r["setting"]) == (scope, cat, setting)]
            means.append({"scope": scope, "category": cat, "setting": setting, "n": len(part),
                          **{f"{key}_{metric}": method([r[key] for r in part])
                             if len(part)>1 or metric=="mean" else None
                             for key in METRICS for metric,method in (("mean",np.mean),("sample_sd",lambda x: np.std(x,ddof=1)))}})
        csv_write(folder / "formal-summary.csv", means)
        accepted = load_rows(registry_file.parents[2] / "results/vicad_syn/metrics/baseline-comparison.csv")
        paired = []
        for row in formal:
            if row["scope"] != "exploratory" or row["setting"] != "selected":
                continue
            for base in accepted:
                if base["class"] == row["category"]:
                    paired.append({"category": row["category"], "reference": base["method"], "identity": base["identity"],
                                   **{k: base[k] for k in METRICS},
                                   **{"selected_minus_" + k: row[k] - float(base[k]) if base[k] else None for k in METRICS}})
        if paired:
            csv_write(folder / "accepted-baseline-comparison.csv", paired)
    output["sha256"] = {p.name: digest(p) for p in sorted(folder.glob("*.csv"))}
    write_json(folder / "verification.json", output)


if __name__ == "__main__":
    main()
