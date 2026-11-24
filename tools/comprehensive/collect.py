"""Export publication tables from verified artifacts; missing results stay empty."""
import argparse
from collections import defaultdict
import csv
from pathlib import Path
from statistics import mean, stdev

from tools.comprehensive.resources import FREEZE, REPO, RELEASE
from vicad_syn.experiment import check, digest, read_json

METRICS = ("O-AUROC", "I-AUROC", "AUPRO")
OFFICIAL_KEYS = ("AUROC_obj_sp", "AUROC_view_sp", "AUPRO_view")
MODELS = {"dinomaly": "Dinomaly", "inpformer": "INP-Former"}
PROTOCOL = Path(__file__).with_name("protocol.json")


def csv_rows(path):
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def store(path, rows):
    if rows:
        with path.open("x", newline="", encoding="utf-8") as target:
            out = csv.DictWriter(target, list(rows[0]), lineterminator="\n")
            out.writeheader()
            out.writerows(rows)


def official_rows(work, config, audit):
    official = work / "official"
    progress = read_json(official / "official-progress.json") or {}
    table = []
    costs = []
    for category in config["baseline_order"]:
        for key, method in MODELS.items():
            train_path = official / "train" / key / category
            eval_path = official / "evaluation" / key / category
            train = read_json(train_path / "status.json") or {}
            evaluation = read_json(eval_path / "status.json") or {}
            row = progress.get(f"{category}/{key}", {})
            phase = row.get("status", train.get("status", "NOT_STARTED"))
            if phase == "PASS" and not row:
                phase = "AWAITING_EVALUATION"
            if audit["categories"][category]["status"] != "PASS":
                phase = "BLOCKED_INPUT"
            metrics = dict.fromkeys(METRICS)
            checkpoint = train.get("final_checkpoint", {}).get("sha256", "")
            if phase == "PASS":
                scores = read_json(eval_path / "synergy.json")
                check(checkpoint == evaluation["evaluation_checkpoint_sha256"] == scores["checkpoint_sha256"],
                      f"{category}/{method}: unbound baseline score")
                check(train["epochs_completed"] == 100 and train["optimization_steps"]
                      == (29_000 if category == "Dice" else 30_000), "wrong final official baseline")
                check(digest(train_path / train["final_checkpoint"]["path"]) == checkpoint, "baseline bytes changed")
                metrics = dict(zip(METRICS, (scores["metrics_pct"][name] for name in OFFICIAL_KEYS), strict=True))
            elif phase == "FAILED":
                phase = "TRAIN_FAILED"
            identity = "OFFICIAL_CLEAN" if phase == "PASS" else "OFFICIAL_FAILED" if phase == "TRAIN_FAILED" else "OFFICIAL_PENDING"
            table.append({"category": category, "method": method, "source": "TASK-005 local",
                          "identity": identity, "status": phase, "seed": 42,
                          "release": RELEASE, "epochs": train.get("epochs_completed", ""),
                          "iterations": train.get("optimization_steps", ""),
                          "last_observed_iteration": train.get("last_observed_iteration", ""),
                          "checkpoint_sha256": checkpoint, **metrics})
            if phase == "PASS":
                counts = audit["categories"][category]
                infer = evaluation["inference_through_concatenation_seconds"]
                weight = next(iter(train.get("weights", [])), {})
                costs.append({"category": category, "method": method, "seed": 42,
                              "frozen_parameters": train["parameters"] - train["trainable_parameters"],
                              "trainable_parameters": train["trainable_parameters"],
                              "weight_file_bytes": weight.get("bytes", ""),
                              "train_seconds": train["train_seconds"], "raw_image_inference_seconds": infer,
                              "inference_peak_allocated_bytes": evaluation["peak_allocated_bytes"],
                              "train_peak_allocated_bytes": train["peak_allocated_bytes"],
                              "derived_seconds_per_120_images": infer / counts["test_objects"],
                              "derived_seconds_per_10_images": infer / counts["test_objects"] / 12})
    return table, costs


def ablation_rows(work, config):
    present = {}
    receipts = []
    for category in config["ablation_order"]:
        for seed in (config["primary_seed"], *config["repeat_seeds"]):
            run = work / "ablation" / category / f"seed-{seed}"
            if receipt := read_json(run / "ablation.json"):
                check(receipt["protocol_sha256"] == digest(PROTOCOL) and receipt["freeze_sha256"] == FREEZE,
                      "ablation receipt has changed frozen inputs")
                receipts.append(receipt)
                for method, item in receipt["settings"].items():
                    check(digest(run / f"{method}-identity.json") == item["identity_sha256"],
                          "ablation identity receipt changed")
                    present[category, int(seed), method] = {
                        "source": item["origin"], "checkpoint_sha256": receipt["checkpoint_sha256"],
                        **{metric: item[metric] for metric in METRICS},
                    }
    prior = REPO / "results/vicad_refinement/metrics/formal.csv"
    for row in csv_rows(prior):
        key = row["category"], int(row["seed"]), {"v1_A4": "B0", "selected": "B6"}[row["setting"]]
        previous = {"source": "TASK-004 frozen official score", "checkpoint_sha256": row["checkpoint_sha256"],
                    **{metric: float(row[metric]) for metric in METRICS}}
        if key in present:
            check(all(present[key][metric] == previous[metric] for metric in (*METRICS, "checkpoint_sha256")),
                  "TASK-004 reported comparator changed")
        else:
            present[key] = previous
    table = []
    for category in config["ablation_order"]:
        primary = int(config["primary_seed"])
        for seed in [primary, *config["repeat_seeds"]]:
            settings = list(config["ablation"]) if seed == primary else config["repeated_settings"]
            if seed != primary and category not in config["repeated_categories"]:
                continue
            for method in settings:
                data = present.get((category, seed, method), {})
                table.append({"category": category, "seed": seed, "setting": method,
                              "status": "PASS" if data else "PENDING", "source": data.get("source", ""),
                              "checkpoint_sha256": data.get("checkpoint_sha256", ""),
                              **{metric: data.get(metric) for metric in METRICS}})
    return table, receipts


def main_table(audit, official, ablations):
    table = []

    def add(category, method, identity, status, seed, source, metrics):
        counts = audit["categories"][category]
        table.append({"category": category, "method": method, "identity": identity, "status": status, "seed": seed,
                      "source": source, "train_objects": counts.get("train_objects", ""),
                      "test_objects": counts.get("test_objects", ""), **dict(zip(METRICS, metrics, strict=True))})

    historical = REPO / "results/baselines/metrics/baseline-summary.csv"
    stable = historical.with_name("stabilized-doll.csv")
    for row in [*csv_rows(historical), *csv_rows(stable)]:
        if row["experiment_identity"] == "RECOVERED_DEVIATION":
            continue
        method = "Dinomaly-stabilized" if row["experiment_identity"] == "NUMERICALLY_STABILIZED" else MODELS[row["model"]]
        add(row["class"], method, row["experiment_identity"], row["status"], row["seed"],
            "TASK-002 local", [row[m + "_pct"] for m in METRICS])
    for row in official:
        add(row["category"], row["method"], row["identity"], row["status"], row["seed"], row["source"],
            [row[m] for m in METRICS])
    for row in ablations:
        if row["setting"] not in ("B0", "B2", "B6") or row["seed"] != 42:
            continue
        method = {"B0": "VICAD-Syn v1 (paired)", "B2": "Multilayer normal prototype", "B6": "TASK-004 frozen"}[row["setting"]]
        add(row["category"], method, row["setting"], row["status"], row["seed"], row["source"], [row[m] for m in METRICS])
    return table


def coverage_means(table):
    known = defaultdict(list)
    for row in table:
        if row["status"] == "PASS" and all(row[m] not in ("", None) for m in METRICS):
            known[row["method"]].append(row)
    return [{"method": method, "category_count": len(rows), "categories": ";".join(sorted(row["category"] for row in rows)),
             **{m: mean(float(row[m]) for row in rows) for m in METRICS}} for method, rows in sorted(known.items())]


def paired_baselines(table):
    selected = {row["category"]: row for row in table if row["method"] == "TASK-004 frozen"}
    result = []
    for method in ("Dinomaly", "INP-Former", "Dinomaly-stabilized", "Multilayer normal prototype"):
        subset = [r for r in table if r["method"] == method and r["status"] == "PASS"]
        for metric in METRICS:
            pairs = [(float(selected[r["category"]][metric]), float(r[metric])) for r in subset
                     if selected[r["category"]][metric] not in (None, "") and r[metric] not in (None, "")]
            if pairs:
                result.append({"reference": method, "metric": metric, "category_count": len(pairs),
                               "categories": ";".join(sorted(r["category"] for r in subset)),
                               "frozen_mean": mean(x for x, _ in pairs), "reference_mean": mean(y for _, y in pairs),
                               "higher": sum(x > y for x, y in pairs), "lower": sum(x < y for x, y in pairs),
                               "equal": sum(x == y for x, y in pairs)})
    return result


def change_counts(pairs):
    contrasts = defaultdict(list)
    for row in pairs:
        contrasts[(row["left"], row["right"], row["metric"], row["seeds"])].append(row)
    return [{"left": left, "right": right, "metric": metric, "seeds": seeds, "category_count": len(rows),
             "improved": sum(row["mean_delta_pp"] > 0 for row in rows),
             "decreased": sum(row["mean_delta_pp"] < 0 for row in rows),
             "unchanged": sum(row["mean_delta_pp"] == 0 for row in rows)}
            for (left, right, metric, seeds), rows in contrasts.items()]


def paired_seed_summary(ablations, config):
    keys = {(row["category"], row["seed"], row["setting"]): row for row in ablations if row["status"] == "PASS"}
    pairs = []
    for category in config["ablation_order"]:
        planned = config["bootstrap"]["comparisons"] + [["B6", "B0"]]
        for left, right in planned:
            for metric in METRICS:
                deltas = [(seed, keys[category, seed, left][metric] - keys[category, seed, right][metric])
                          for seed in (42, 43, 44) if (category, seed, left) in keys and (category, seed, right) in keys]
                if deltas:
                    nums = [d for _, d in deltas]
                    pairs.append({"category": category, "left": left, "right": right, "metric": metric,
                                  "seeds": ";".join(str(seed) for seed, _ in deltas),
                                  **{f"delta_{seed}_pp": dict(deltas).get(seed) for seed in (42, 43, 44)},
                                  "mean_delta_pp": mean(nums), "sample_sd_pp": stdev(nums) if len(nums) > 1 else None})
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = read_json(PROTOCOL)
    audit = read_json(args.work / "evidence/resource-audit.json")
    official, costs = official_rows(args.work, config, audit)
    ablations, receipts = ablation_rows(args.work, config)
    all_results = main_table(audit, official, ablations)
    store(args.output / "official-progress.csv", official)
    store(args.output / "official-costs.csv", costs)
    store(args.output / "ablation.csv", ablations)
    pairs = paired_seed_summary(ablations, config)
    store(args.output / "paired-seed-summary.csv", pairs)
    store(args.output / "paired-change-counts.csv", change_counts(pairs))
    store(args.output / "same-coverage-baseline.csv", paired_baselines(all_results))
    store(args.output / "local-results.csv", all_results)
    store(args.output / "coverage-macro.csv", coverage_means(all_results))
    shared = [{"category": r["category"], "seed": r["seed"],
               **{k: v for k, v in r["prediction"].items() if k != "shared_settings"},
               "shared_settings": ";".join(r["prediction"]["shared_settings"]),
               "feature_cache_bytes": audit["categories"][r["category"]]["feature_caches"]["test"]["bytes"]}
              for r in receipts]
    store(args.output / "shared-cached-costs.csv", shared)


if __name__ == "__main__":
    main()
