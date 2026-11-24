"""Extend the verified historical B6/B0 comparison with product-level intervals."""
import argparse
from collections import defaultdict
import csv
from pathlib import Path

import numpy as np
import torch

from tools.comprehensive.ablation import PROTOCOL, assert_auc, buffer_scores, canonical_scores, map_source
from tools.comprehensive.offline import enable_offline
from tools.comprehensive.resources import FREEZE, REPO, roots, scope_for
from tools.comprehensive.statistics import paired_intervals
from vicad_refinement.data import load_groups
from vicad_syn.experiment import check, digest, read_json


def frozen_labels(groups, seed_folder, previous):
    """Map frozen masks to product identities; a reused normal identifier cannot be positive."""
    scope = defaultdict(list)
    for index, group in enumerate(groups):
        row = group.records[0]
        scope[f'{row["object_name"]}_{group.view}'].append((index, row["object_anomaly"]))
    records = {}
    for mode in ("selected", "v1_A4"):
        source = seed_folder / mode
        files = previous["scores"][mode]["fused_and_patch_inputs"]
        names, masks_old = [source / f"view_instance_{key}.npy" for key in ("names", "masks")]
        for path in (names, masks_old):
            check(digest(path) == files[path.name]["sha256"], "frozen mask file changed")
        states = defaultdict(list)
        for name, mask in zip(np.load(names), np.load(masks_old, mmap_mode="r"), strict=True):
            states[str(name)].append(bool(mask.any()))
        records[mode] = {name: sorted(labels) for name, labels in states.items()}
    check(records["selected"] == records["v1_A4"] and scope.keys() == records["selected"].keys(),
          "frozen comparison masks or manifest identities differ")
    labels = np.zeros(len(groups), dtype=bool)
    for name, group in scope.items():
        values = records["selected"][name]
        check(len(values) == len(group), "frozen physical group multiplicity differs")
        members = sorted(group, key=lambda item: item[1])
        check(len({anomaly for _, anomaly in members}) == len(members), "ambiguous physical identity")
        for (index, anomaly), value in zip(members, values, strict=True):
            check(anomaly or not value, "normal object has a positive fused mask")
            labels[index] = value
    return labels


def execute(args):
    ticket = args.frozen / "freeze.json"
    check(digest(ticket) == FREEZE, "frozen TASK-004 ticket changed")
    base = read_json(ticket)["binding"]["inherited_v1"]["config"]
    registry = read_json(REPO / "vicad_syn/refinement/protocol.json")
    plan = read_json(PROTOCOL)
    primary = plan["bootstrap"]
    count = 2 * (4 + len(registry["confirmatory_order"]) * 3)
    with args.output.open("x", newline="", encoding="utf-8") as stream:
        writer = None
        for i, category in enumerate(plan["ablation_order"]):
            source = roots(category, args.initial, args.bird, args.holdout)
            expect = registry["confirmatory_groups"].get(category, (30, 70))
            groups = load_groups(source, category, base, expect, "test")
            directory = args.frozen / scope_for(category) / category
            pretest = read_json(directory / "pretest.json")["binding"]
            check(pretest["input_freeze_sha256"] == FREEZE, "historical evaluation changed")
            for seed, checkpoint in pretest["checkpoints"].items():
                folder = directory / f"seed-{seed}"
                historical = read_json(folder / "evaluation.json")
                check(historical["training"]["checkpoint_sha256"] == checkpoint, "historical checkpoint mismatch")
                masks = frozen_labels(groups, folder, historical)
                values = {}
                for setting in ("selected", "v1_A4"):
                    official = historical["scores"][setting]
                    patch = map_source(folder, setting, official)
                    values[setting] = canonical_scores(groups, masks, buffer_scores(patch))
                    assert_auc(values[setting], official)
                random_seed = np.random.SeedSequence([primary["seed"], int(seed), i])
                intervals = paired_intervals(values["selected"], values["v1_A4"], primary["replicates"],
                                             random_seed, count, primary["confidence"])
                for metric, result in intervals.items():
                    row = {"role": "historical_supplement", "category": category, "seed": seed,
                           "left": "B6", "right": "B0", "metric": metric,
                           "family_size": count, "checkpoint_sha256": checkpoint, **result}
                    if writer is None:
                        writer = csv.DictWriter(stream, list(row), lineterminator="\n")
                        writer.writeheader()
                    writer.writerow(row)
                stream.flush()
                print(category, seed, "PASS", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("initial", "bird", "holdout", "frozen", "output"):
        parser.add_argument("--" + key, type=lambda p: Path(p).resolve(), required=True)
    args = parser.parse_args()
    check(not args.output.is_relative_to(REPO), "external output required")
    enable_offline()
    torch.set_num_threads(3)
    execute(args)


if __name__ == "__main__":
    main()
