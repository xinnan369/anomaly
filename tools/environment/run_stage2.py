"""Run the reviewed TASK-002 plan in order and preserve each attempt outside Git."""

import argparse
from datetime import datetime, timezone
import fcntl
from pathlib import Path
import subprocess
import sys

from run_baseline import read_record, validate_checkpoint
from summarize_baselines import CLASSES, MODELS


REPO = Path(__file__).resolve().parents[2]


def run_attempt(root, model, category, mode, checkpoint=None):
    parent = "runs" if mode == "train" else "evaluations"
    output = root / parent / model / category
    if output.exists():
        status = read_record(output / "status.json")
        if (status.get("mode"), status.get("model"), status.get("class"), status.get("status")) != (
            mode, model, category, "PASS"
        ):
            raise RuntimeError(f"Preserve and diagnose existing {mode} attempt before retrying: {output}")
        return status
    upstream = root / "inputs" / "upstream"
    weight = root / "inputs" / "weights" / "dinov2_vitb14_reg4_pretrain.pth"
    command = [sys.executable, "-B", "-u", str(REPO / "tools/environment/run_baseline.py"),
               "--upstream", str(upstream), "--weight", str(weight), "--release-root", str(root),
               "--model", model, "--category", category, "--output", str(output)]
    if checkpoint is not None:
        command.extend(("--checkpoint", str(checkpoint)))
    log = root / "logs" / f"{model}-{category}-{mode}.log"
    with log.open("a", encoding="utf-8", buffering=1) as stream:
        stamp = datetime.now(timezone.utc).isoformat()
        print(f"{stamp} {model}/{category} {mode}: {log}", flush=True)
        print(f"\n{stamp} {command!r}", file=stream)
        subprocess.run(command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT, check=True)
    return read_record(output / "status.json")


def run_plan(root, only_car):
    plan = [("dinomaly", "Car")] if only_car else [(m, cls) for m in MODELS for cls in CLASSES]
    for model, category in plan:
        if not (root / "data_audit" / f"{category}.json").is_file():
            raise RuntimeError(f"{category}: verified archive required before launching the baseline")
        training = run_attempt(root, model, category, "train")
        checkpoint_record = training["final_checkpoint"]
        checkpoint = root / "runs" / model / category / checkpoint_record["path"]
        if (root / "evaluations" / model / category).exists():
            validate_checkpoint(checkpoint, model, category, root / "inputs" / "upstream")
        evaluation = run_attempt(root, model, category, "test", checkpoint)
        if evaluation["evaluation_checkpoint_sha256"] != checkpoint_record["sha256"]:
            raise RuntimeError(f"Evaluation checkpoint does not match {model}/{category} training")
        print(f"{model}/{category}: training and evaluation PASS", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=lambda path: Path(path).resolve(), required=True)
    parser.add_argument("--only-car", action="store_true", help="run only the initial Dinomaly/Car closure")
    args = parser.parse_args()
    if args.root.is_relative_to(REPO):
        raise ValueError("Use a data and experiment root outside Git")
    (args.root / "logs").mkdir(parents=True, exist_ok=True)
    with (args.root / "stage2.lock").open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_plan(args.root, args.only_car)


if __name__ == "__main__":
    main()
