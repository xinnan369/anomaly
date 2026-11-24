"""Replay all TASK-004 Synergy metrics from immutable, locally saved fused arrays."""
import argparse
import csv
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tools.vicad_syn.export_refinement import check_files, load_rows
from vicad_syn.evaluation.synergy import METRICS
from vicad_syn.experiment import check, pinned_upstream, read_json


def evaluate(rows, run):
    from util.metric import Evaluator

    for row in rows:
        folder = run / row["scope"] / row["category"] / ("seed-" + row["seed"])
        receipt = read_json(folder / "evaluation.json")
        recorded = receipt["scores"][row["setting"]]
        check(receipt["training"]["checkpoint_sha256"] == row["checkpoint_sha256"], "checkpoint mismatch")
        expected = {name: proof for name, proof in recorded["fused_and_patch_inputs"].items()
                    if name != "patch-contributions.npy"}
        base = folder / row["setting"]
        check_files(base, expected)
        arrays = {Path(name).stem: np.load(base / name, allow_pickle=False) for name in expected}
        results = Evaluator(metrics=list(METRICS.values()), use_adeval=True).run(arrays, row["category"], "M2AD-Synergy")
        outcome = {key: row[key] for key in ("scope", "category", "seed", "setting", "checkpoint_sha256")}
        for metric, key in METRICS.items():
            value = float(results[key]) * 100
            delta = value - float(row[metric])
            check(np.isfinite(value) and abs(delta) <= 0.001, f"{outcome}: {metric} divergence {delta}")
            outcome.update({metric: value, metric + "_delta_pp": delta})
        print(outcome, flush=True)
        yield outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "upstream", "formal", "output"):
        parser.add_argument("--" + name, type=lambda x: Path(x).resolve(), required=True)
    args = parser.parse_args()
    rows = load_rows(args.formal)
    check(len(rows) == 44, "expected all 44 frozen results")
    frozen = read_json(args.run / "freeze.json")
    revision = frozen["binding"]["inherited_v1"]["config"]["upstream_revision"]
    with patch("socket.socket.connect", side_effect=RuntimeError("offline replay")), pinned_upstream(args.upstream, revision):
        executed = list(evaluate(rows, args.run))
    with args.output.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(executed[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(executed)


if __name__ == "__main__":
    main()
