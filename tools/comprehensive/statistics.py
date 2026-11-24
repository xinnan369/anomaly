"""Stratified paired bootstrap with physical products as the resampling units."""
import argparse
import csv
from pathlib import Path

import numpy as np

from tools.comprehensive.ablation import PROTOCOL
from vicad_syn.experiment import check, digest, read_json


def auc_components(rows, order):
    """Aggregate positive-vs-negative win and trial counts by a pair of products."""
    labels = np.array([row["label"] for row in rows], dtype=int)
    scores = np.array([row["score"] for row in rows], dtype=float)
    check(set(labels) == {0, 1} and np.isfinite(scores).all(), "AUC requires finite scores and both labels")
    position = {identity: index for index, identity in enumerate(order)}
    product = np.array([position[tuple(row["identity"])] for row in rows])
    left, right = scores[labels == 1, None], scores[labels == 0]
    wins = (left > right) + 0.5 * (left == right)
    numerator = np.zeros((len(order), len(order)))
    np.add.at(numerator, (product[labels == 1, None], product[labels == 0]), wins)
    positive = np.bincount(product[labels == 1], minlength=len(order))
    negative = np.bincount(product[labels == 0], minlength=len(order))
    return numerator, positive, negative


def auc_by_draw(rows, order, weights):
    numerator, positive, negative = auc_components(rows, order)
    hits = ((weights @ numerator) * weights).sum(1)
    trials = (weights @ positive) * (weights @ negative)
    return np.divide(hits, trials, out=np.full(len(weights), np.nan), where=trials > 0)


def sampling_weights(order, replicates, seed):
    prng = np.random.default_rng(seed)
    weights = np.zeros((replicates, len(order)))
    for stratum in (0, 1):
        column = [i for i, identity in enumerate(order) if identity[2] == stratum]
        check(bool(column), "empty product stratum")
        weights[:, column] = prng.multinomial(len(column), np.full(len(column), 1 / len(column)), size=replicates)
    return weights


def index_rows(rows, scope):
    keys = {(*row["identity"], row.get("view", "")): row["label"] for row in rows}
    check(len(keys) == len(rows), f"duplicate evaluation identity: {scope}")
    return keys


def paired_intervals(left, right, replicates, seed, family_size, confidence):
    order = sorted(tuple(row["identity"]) for row in left["objects"])
    weights = np.vstack((np.ones((1, len(order))), sampling_weights(order, replicates, seed)))
    result = {}
    for metric, scope in (("O-AUROC", "objects"), ("I-AUROC", "views")):
        check(index_rows(left[scope], scope) == index_rows(right[scope], scope), "method labels or identities differ")
        delta = 100 * (auc_by_draw(left[scope], order, weights) - auc_by_draw(right[scope], order, weights))
        valid = delta[1:][np.isfinite(delta[1:])]
        check(np.isfinite(delta[0]) and len(valid), "no valid stratified comparison")
        alpha = 1 - confidence
        marginal = np.quantile(valid, [alpha / 2, 1 - alpha / 2], method="linear")
        simultaneous = np.quantile(valid, [alpha / (2 * family_size), 1 - alpha / (2 * family_size)], method="linear")
        result[metric] = {"delta_pp": float(delta[0]), "lower_95_pp": float(marginal[0]),
                          "upper_95_pp": float(marginal[1]), "simultaneous_lower_pp": float(simultaneous[0]),
                          "simultaneous_upper_pp": float(simultaneous[1]), "valid_draws": len(valid)}
    return result


def generate(root, output):
    plan = read_json(PROTOCOL)
    boot = plan["bootstrap"]
    family_size = len(plan["ablation_order"]) * len(boot["comparisons"]) * len(boot["primary_metrics"])
    results = []
    for i, category in enumerate(plan["ablation_order"]):
        path = root / category / f'seed-{plan["primary_seed"]}'
        receipt = read_json(path / "ablation.json")
        if not receipt:
            continue
        check(receipt["protocol_sha256"] == digest(PROTOCOL), "ablation protocol changed")
        for left, right in boot["comparisons"]:
            inputs = []
            for key in (left, right):
                file = path / f"{key}-identity.json"
                check(digest(file) == receipt["settings"][key]["identity_sha256"], "bootstrap input hash differs")
                inputs.append(read_json(file))
            seed = np.random.SeedSequence([boot["seed"], plan["primary_seed"], i])
            bounds = paired_intervals(*inputs, boot["replicates"], seed, family_size, boot["confidence"])
            for metric, row in bounds.items():
                expected = receipt["settings"][left][metric] - receipt["settings"][right][metric]
                check(abs(row["delta_pp"] - expected) <= 1e-6, "paired AUC differs from official scores")
                results.append({"category": category, "seed": plan["primary_seed"], "left": left, "right": right,
                                "metric": metric, "family_size": family_size, **row})
    if results:
        with output.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, list(results[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(results)
    return len(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(f"paired CI rows: {generate(args.ablation, args.output)}")


if __name__ == "__main__":
    main()
