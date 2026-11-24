"""Produce independent descriptive tables from immutable TASK-004 evidence."""
import argparse
from collections import defaultdict
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev

METRICS = ("O-AUROC", "I-AUROC", "AUPRO")
SEEDS = ("42", "43", "44")
CONFIRM = ("Bird", "Motor", "Teapot", "Tube", "Dice", "Holder")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def check(condition, detail):
    if not condition:
        raise ValueError(detail)


def verify_receipts(evidence, formal):
    verification = read_json(evidence / "verification.json")
    freeze = evidence / "freeze.json"
    frozen_at = datetime.fromisoformat(read_json(freeze)["sealed_utc"])
    freeze_hash = verification["freeze_sha256"]
    check(sha256(freeze) == freeze_hash, "frozen protocol digest changed")
    records = {}
    for category in CONFIRM:
        pretest = evidence / f"confirmatory-{category}-pretest.json"
        receipt = evidence / f"confirmatory-{category}-category.json"
        data = read_json(pretest)
        category_data = read_json(receipt)
        check(sha256(receipt) == verification["formal_receipts"][f"confirmatory/{category}"],
              f"category receipt digest changed: {category}")
        check(sha256(pretest) == category_data["pretest_sha256"]
              and data["binding"]["input_freeze_sha256"] == category_data["freeze_sha256"] == freeze_hash,
              f"pretest receipt not bound to freeze: {category}")
        pretest_at = datetime.fromisoformat(data["utc"])
        check(frozen_at <= pretest_at, f"pretest predates freeze: {category}")
        seeds = data["binding"]["checkpoints"]
        check(set(seeds) == set(SEEDS), f"unexpected seeds: {category}")
        for seed, checkpoint in seeds.items():
            for setting in ("v1_A4", "selected"):
                row = formal["confirmatory", category, seed, setting]
                check(row["checkpoint_sha256"] == checkpoint, f"checkpoint mismatch: {category}/{seed}")
        records[category] = {"pretest_utc": data["utc"], "pretest_sha256": sha256(pretest),
                             "checkpoints": seeds}
    return freeze_hash, records


def summarize(formal, original):
    groups = defaultdict(list)
    for (scope, category, _, setting), row in formal.items():
        groups[scope, category, setting].append(row)
    check(set(original) == set(groups), "group set changed")
    summaries = []
    for key, rows in sorted(groups.items()):
        record = dict(zip(("scope", "category", "setting"), key))
        record["n"] = len(rows)
        for metric in METRICS:
            values = [float(row[metric]) for row in rows]
            record[f"{metric}_mean"] = mean(values)
            record[f"{metric}_sample_sd"] = stdev(values) if len(values) > 1 else ""
        for name, value in record.items():
            old = original[key][name]
            check(math.isclose(float(old), value, rel_tol=0, abs_tol=1e-10) if isinstance(value, (int, float))
                  else old == value, f"summary mismatch: {key}/{name}")
        summaries.append(record)
    return summaries


def paired_tables(formal):
    pairs = []
    for category in CONFIRM:
        for seed in SEEDS:
            baseline = formal["confirmatory", category, seed, "v1_A4"]
            selected = formal["confirmatory", category, seed, "selected"]
            row = {"category": category, "seed": seed, "checkpoint_sha256": selected["checkpoint_sha256"]}
            for metric in METRICS:
                left, right = float(baseline[metric]), float(selected[metric])
                row[f"v1_{metric}"] = left
                row[f"selected_{metric}"] = right
                row[f"delta_{metric}_pp"] = right - left
            pairs.append(row)
    columns = [name for name in pairs[0] if name not in ("category", "seed", "checkpoint_sha256")]
    macros = [{"aggregation": seed, "categories": len(CONFIRM),
               **{name: mean(row[name] for row in pairs if row["seed"] == seed) for name in columns}}
              for seed in SEEDS]
    aggregates = [{"aggregation": label, "categories": len(CONFIRM),
                   **{name: fn(row[name] for row in macros) for name in columns}}
                  for label, fn in (("seed_mean", mean), ("seed_sample_sd", stdev))]
    return pairs, macros + aggregates


def rank_counts(summaries):
    indexed = {(row["scope"], row["category"], row["setting"]): row for row in summaries}
    counts = {}
    for scope in ("exploratory", "confirmatory"):
        categories = sorted({category for s, category, _ in indexed if s == scope})
        counts[scope] = {}
        for metric in METRICS:
            changes = {category: indexed[scope, category, "selected"][f"{metric}_mean"]
                       - indexed[scope, category, "v1_A4"][f"{metric}_mean"] for category in categories}
            counts[scope][metric] = {
                "improved": sum(value > 0 for value in changes.values()),
                "decreased": sum(value < 0 for value in changes.values()),
                "unchanged": sum(value == 0 for value in changes.values()),
                "category_deltas_pp": changes,
            }
    return counts


def run(root, replay, output):
    metrics, evidence = root / "metrics", root / "evidence"
    key = lambda row: tuple(row[name] for name in ("scope", "category", "seed", "setting"))
    formal_rows = read_csv(metrics / "formal.csv")
    formal = {key(row): row for row in formal_rows}
    replay_rows = read_csv(replay)
    replay_index = {key(row): row for row in replay_rows}
    check(len(formal_rows) == len(formal) == len(replay_rows) == len(replay_index) == 44
          and formal.keys() == replay_index.keys(), "formal/replay row set changed")
    max_difference = {}
    for metric in METRICS:
        max_difference[metric] = max(abs(float(formal[k][metric]) - float(replay_index[k][metric])) for k in formal)
        check(max_difference[metric] <= 0.001, f"official replay differs: {metric}")
    for k, row in formal.items():
        check(row["checkpoint_sha256"] == replay_index[k]["checkpoint_sha256"], f"replay checkpoint changed: {k}")
    freeze_hash, checkpoints = verify_receipts(evidence, formal)
    summary_key = lambda row: (row["scope"], row["category"], row["setting"])
    summaries = summarize(formal, {summary_key(row): row for row in read_csv(metrics / "formal-summary.csv")})
    pairs, macros = paired_tables(formal)
    coverage = {row["category"]: row for row in read_csv(metrics / "confirmatory-coverage.csv")}
    check(coverage["Dice"]["train_objects"] == "29" and coverage["Holder"]["test_objects"] == "69",
          "unexpected count in release reconciliation")
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "formal-recomputed-summary.csv", summaries)
    write_csv(output / "confirmatory-paired-seeds.csv", pairs)
    write_csv(output / "confirmatory-macro.csv", macros)
    report = {"formal_rows": len(formal), "grouped_summary_rows": len(summaries), "paired_rows": len(pairs),
              "max_official_replay_difference_pp": max_difference, "source_formal_sha256": sha256(metrics / "formal.csv"),
              "replay_sha256": sha256(replay), "freeze_sha256": freeze_hash,
              "pretest_checkpoints": checkpoints, "category_changes": rank_counts(summaries)}
    with (output / "formal-audit.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/vicad_refinement"))
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.replay, args.output)


if __name__ == "__main__":
    main()
