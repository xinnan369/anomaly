"""Evaluate predeclared optional A2/A3 using existing frozen predictions only."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from tools.vicad_syn.verify import METRICS, write_csv
from vicad_syn.evaluation.synergy import METRICS as OFFICIAL, baseline_helpers, record_files
from vicad_syn.experiment import check, digest, pinned_upstream, protocol, read_json, write_json
from vicad_syn.fusion.weights import light_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("inputs", "run", "verified", "output"):
        parser.add_argument(f"--{key}", type=lambda path: Path(path).resolve(), required=True)
    args = parser.parse_args()
    proof = read_json(args.verified)
    freeze = protocol()
    check(proof["status"] == "PASS" and proof["frozen"]["frozen"] == freeze,
          "complete and verify the four-category core first")
    plan_path = args.run / "ablation-extension-plan.json"
    plan = read_json(plan_path)
    check(plan["classes"] == ["Cube", "Doll", "Ring"] and plan["settings"] == ["A2", "A3"]
          and plan["uses_frozen_patch_signals"] and not plan["extra_training"], "unexpected extension scope")
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=False)
    config = freeze["config"]
    rows, index = [], {}
    with pinned_upstream(args.inputs / "inputs/upstream", config["upstream_revision"]):
        from util.m2ad_utils import m2ad_fuse
        from util.metric import Evaluator

        verify_fusion = baseline_helpers().check_fusion
        for category in plan["classes"]:
            source = args.run / category / "conditioned/evaluation"
            evaluation = proof["categories"][category]["evaluations"]["conditioned"]["receipt"]
            signals_file = source / "patch-signals.npz"
            check(digest(signals_file) == evaluation["patch_signals_sha256"], "frozen signal hash mismatch")
            raw = {file.stem: np.load(file, allow_pickle=False)
                   for file in sorted((source / "raw").glob("*.npy"))}
            check(set(evaluation["raw"]) == {key + ".npy" for key in raw}, "raw prediction file set changed")
            index[category] = {}
            with np.load(signals_file, allow_pickle=False) as signals:
                for mode, signal_key in zip(plan["settings"], ("stability", "epistemic"), strict=True):
                    started = perf_counter()
                    weights = light_weights(torch.from_numpy(signals["residual"]), [torch.from_numpy(signals[signal_key])],
                                            config["epsilon"], config["temperature"]).numpy()
                    check(np.isfinite(weights).all() and (weights >= 0).all() and np.allclose(weights.sum(1), 1),
                          "optional evaluation weights are invalid")
                    scale = (10 * weights).reshape(-1)
                    weighted = {**raw, "anomaly_scores": raw["anomaly_scores"] * scale,
                                "anomaly_maps": raw["anomaly_maps"] * scale[:, None, None]}
                    fused = m2ad_fuse(**weighted, fuse="add")
                    groups = verify_fusion(weighted, fused)
                    check((groups["objects"], groups["views"]) == (70,840), "optional evaluation group mismatch")
                    got = Evaluator(metrics=list(OFFICIAL.values()), use_adeval=True).run(fused, category, "M2AD-Synergy")
                    scores = {key: 100 * float(got[OFFICIAL[key]]) for key in METRICS}
                    check(all(np.isfinite(value) and 0 <= value <= 100 for value in scores.values()), "invalid metric")
                    target = args.output / category / mode
                    target.mkdir(parents=True)
                    np.save(target / "weights.npy", weights, allow_pickle=False)
                    fused_home = target / "fused"
                    fused_home.mkdir()
                    for key, result in fused.items():
                        np.save(fused_home / (key + ".npy"), result, allow_pickle=False)
                    rows.append({"class": category, "setting": mode, **scores,
                                 "evaluation_seconds": perf_counter() - started,
                                 "checkpoint_sha256": evaluation["binding"]["checkpoint_sha256"]})
                    index[category][mode] = {"groups": groups, "scores": scores, "fused": record_files(fused_home),
                        "weights_sha256": digest(target / "weights.npy"), "binding": evaluation["binding"],
                        "signal_sha256": evaluation["patch_signals_sha256"]}
                    print(f"{category}/{mode} {scores}", flush=True)
                    del weighted, fused
            del raw
    write_csv(args.output / "extension.csv", rows)
    write_json(args.output / "extension.json", {"status": "PASS", "plan": plan, "plan_sha256": digest(plan_path),
               "utc": datetime.now(timezone.utc).isoformat(), "summary_sha256": digest(args.output / "extension.csv"),
               "categories": index})


if __name__ == "__main__":
    main()
