"""Plan or stage M²AD confirmation archives using only local files and ranges."""
import argparse
import json
from pathlib import Path
import sys

from filelock import FileLock

from tools.download_policy import describe, require
from vicad_syn.experiment import read_json

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "environment"))
from prepare_release import archive_item, stage_archive


REGISTRY = Path(__file__).resolve().parents[2] / "vicad_syn/refinement/protocol.json"


def main():
    registry = read_json(REGISTRY)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest-cache", type=Path, required=True)
    parser.add_argument("--category", choices=registry["confirmatory_order"], required=True)
    parser.add_argument("--task-id", default="TASK-004")
    parser.add_argument("--stage-local", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    require(not root.is_relative_to(REGISTRY.parents[2]), "stage outside Git")
    item = archive_item(root, args.category, registry["archive_sha256"][args.category])
    marker = root / "data_audit" / f"{args.category}.json"
    if args.stage_local:
        root.mkdir(parents=True, exist_ok=True)
        with FileLock(root / "staging.lock"):
            stage_archive(root, args.manifest_cache, args.category, item["sha256"])
    else:
        plan = {"task_id": args.task_id, "files": [] if marker.exists() else [item]}
        report = describe(plan)
        if marker.is_file():
            report["existing_receipt"] = str(marker)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
