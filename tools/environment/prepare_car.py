"""Stage three official-training Car images from a verified local dataset."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "audit"))
from manifest import load_manifest
from sources import RELEASE_REVISION, require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True, help="existing extracted dataset")
    parser.add_argument("--output", type=Path, required=True, help="new scratch directory outside this repository")
    args = parser.parse_args()
    output = args.output.resolve()
    require(not output.is_relative_to(Path(__file__).resolve().parents[2]), "data output must be outside the repository")
    require(not output.exists(), "choose a new output directory")
    data = load_manifest(args.cache)
    candidates = {
        (row["object_name"], row["illumination"]): row for row in data["train"]["Car"]
        if row["view"] == "000" and row["image_anomaly"] == row["object_anomaly"] == 0
    }
    fit_id, probe_id = sorted({obj for obj, _ in candidates})[:2]
    selections = {"fit": [(fit_id, "01"), (fit_id, "05")], "probe": [(probe_id, "01")]}
    summary = {"release_revision": RELEASE_REVISION, "official_split": "train", "groups": {}}
    for group, keys in selections.items():
        rows = [candidates[key] for key in keys]
        results = []
        for row in rows:
            name = row["img_path"]
            image = (args.dataset / name).read_bytes()
            target = output / group / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(image)
            results.append({"path": name, "size_bytes": len(image), "sha256": hashlib.sha256(image).hexdigest()})
        meta = output / group / "jsons" / "meta_unsupervised.json"
        meta.parent.mkdir()
        meta.write_text(json.dumps({"train": {"Car": rows}, "test": {"Car": []}}), encoding="utf-8")
        summary["groups"][group] = results
    summary["successful_range_bytes"] = 0
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
