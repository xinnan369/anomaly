"""Stage three official-training Car images by bounded ZIP Range reads."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "audit"))
from archives import RangeReader
from manifest import load_manifest
from sources import RELEASE_REVISION, require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
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
    reader = RangeReader("Car")
    reader.prefetch(reader.size - 65_557, 65_557)
    summary = {"release_revision": RELEASE_REVISION, "official_split": "train", "groups": {}}
    with zipfile.ZipFile(reader) as archive:
        for group, keys in selections.items():
            rows = [candidates[key] for key in keys]
            results = []
            for row in rows:
                name = row["img_path"]
                member = archive.getinfo(name)
                size = min(reader.size - member.header_offset, member.compress_size + 4096)
                reader.prefetch(member.header_offset, size)
                image = archive.read(member)
                target = output / group / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(image)
                results.append({"path": name, "size_bytes": len(image), "sha256": hashlib.sha256(image).hexdigest()})
            meta = output / group / "jsons" / "meta_unsupervised.json"
            meta.parent.mkdir()
            meta.write_text(json.dumps({"train": {"Car": rows}, "test": {"Car": []}}), encoding="utf-8")
            summary["groups"][group] = results
    summary["successful_range_bytes"] = reader.successful_bytes
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
