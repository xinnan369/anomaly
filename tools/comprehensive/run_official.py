"""Execute the six-category official baseline matrix with local-only inputs."""
import argparse
from datetime import datetime, timezone
import fcntl
from pathlib import Path
import subprocess
import sys

from tools.comprehensive.resources import FREEZE, REPO, roots
from vicad_syn.experiment import check, read_json, write_json

CATEGORIES = ("Bird", "Motor", "Teapot", "Tube", "Dice", "Holder")
MODELS = ("dinomaly", "inpformer")


def attempt(args, category, model, phase, checkpoint=None):
    folder = args.output / phase / model / category
    record_path = folder / "status.json"
    if folder.exists():
        previous = read_json(record_path)
        check(previous and previous.get("status") != "RUNNING", f"preserve incomplete external attempt: {folder}")
        return previous
    release = roots(category, args.initial, args.bird, args.holdout)
    cmd = [sys.executable, "-B", "-u", str(REPO / "tools/environment/run_baseline.py"), "--offline",
           "--upstream", str(args.initial / "inputs/upstream"), "--release-root", str(release),
           "--weight", str(args.initial / "inputs/weights/dinov2_vitb14_reg4_pretrain.pth"),
           "--model", model, "--category", category, "--output", str(folder)]
    if phase == "preflight":
        cmd.append("--preflight")
    if checkpoint:
        cmd.extend(("--checkpoint", str(checkpoint)))
    stamp = datetime.now(timezone.utc).isoformat()
    log = args.output / "logs" / f"{category}-{model}-{phase}.log"
    print(f"{stamp} {category}/{model}/{phase}", flush=True)
    with log.open("x", encoding="utf-8") as stream:
        print(cmd, file=stream, flush=True)
        process = subprocess.run(cmd, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT)
    if record := read_json(record_path):
        return record
    return {"status": "FAILED", "exit_code": process.returncode, "log": str(log)}


def execute(args, audit):
    summary = read_json(args.output / "official-progress.json") or {}
    for category in (CATEGORIES[:1] if args.preflight or args.bird_only else CATEGORIES):
        if category != "Bird" and not args.preflight:
            check(all(summary.get(f"Bird/{m}", {}).get("status") == "PASS" for m in MODELS),
                  "Bird must pass both frozen evaluation workflows before remaining classes")
        if audit["categories"].get(category, {}).get("status") != "PASS":
            print(f"{category}: local assets blocked", flush=True)
            continue
        for model in MODELS:
            key = f"{category}/{model}"
            if args.preflight:
                result = attempt(args, category, model, "preflight")
                print(key, result["status"], flush=True)
                continue
            train = attempt(args, category, model, "train")
            row = {"status": "TRAIN_FAILED", "train": train}
            if train["status"] == "PASS":
                frozen = train["final_checkpoint"]
                checkpoint = args.output / "train" / model / category / frozen["path"]
                evaluation = attempt(args, category, model, "evaluation", checkpoint)
                row.update(status="EVALUATION_FAILED", evaluation=evaluation)
                if evaluation["status"] == "PASS":
                    check(evaluation["evaluation_checkpoint_sha256"] == frozen["sha256"],
                          f"{key}: test checkpoint differs from the final training artifact")
                    row["status"] = "PASS"
            summary[key] = row
            write_json(args.output / "official-progress.json", summary)
            print(key, row["status"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("initial", "bird", "holdout", "audit", "output"):
        parser.add_argument("--" + option, type=lambda p: Path(p).resolve(), required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--bird-only", action="store_true")
    args = parser.parse_args()
    check(not args.output.is_relative_to(REPO), "external output required")
    audit = read_json(args.audit)
    check(audit and audit["freeze_sha256"] == FREEZE, "run the local asset audit first")
    (args.output / "logs").mkdir(parents=True, exist_ok=True)
    with (args.output / "official.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(args, audit)


if __name__ == "__main__":
    main()
