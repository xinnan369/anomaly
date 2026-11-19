"""Audit the pinned M²AD manifest without downloading image archives."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import re
import zipfile

from sources import ARCHIVES, MANIFEST, RELEASE_REVISION, require, verified_resource


VIEWS = tuple(f"{angle:03d}" for angle in range(0, 360, 30))
LIGHTS = tuple(f"{light:02d}" for light in range(1, 11))
CONFIGURATIONS = {(view, light) for view in VIEWS for light in LIGHTS}
IMAGE_NAME = re.compile(r"A([0-9]{3})_I([0-9]{2})\.png")
PAPER_FIGURE = "https://arxiv.org/html/2505.10996v1/dataset_statistics.png"
# Figure 3(a), #Object panel. These counts are transcribed from the preprint.
PAPER_OBJECTS = {
    "Bird": (50, 50), "Car": (50, 50), "Cube": (50, 50), "Dice": (49, 50),
    "Doll": (50, 50), "Holder": (50, 50), "Motor": (50, 50), "Ring": (50, 50),
    "Teapot": (50, 50), "Tube": (50, 50),
}


def load_manifest(cache):
    with zipfile.ZipFile(verified_resource(cache, MANIFEST)) as archive:
        name = "meta_unsupervised.json"
        require(archive.namelist() == [name], "unexpected manifest archive members")
        require(archive.getinfo(name).file_size <= 55_000_000, "manifest member exceeds byte limit")
        with archive.open(name) as source:
            return json.load(source)


def check_manifest(data):
    require(set(data) == {"train", "test"}, "unexpected split names")
    counts = {
        cls: {
            "train_good": {"objects": 0, "images": 0},
            "test_good": {"objects": 0, "images": 0},
            "test_ng": {"objects": 0, "images": 0, "positive_images": 0, "undetectable_positive_images": 0},
        }
        for cls in ARCHIVES
    }
    instances, paths = {}, {}
    for split in ("train", "test"):
        classes = data[split]
        require(set(classes) == set(ARCHIVES), f"{split}: unexpected classes")
        groups = defaultdict(set)
        paths[split] = set()
        for cls, rows in classes.items():
            for row in rows:
                path = row["img_path"]
                parts = path.split("/")
                require(len(parts) == 4 and ".." not in parts, f"{split}/{cls}: malformed image path")
                folder, kind, obj, filename = parts
                require(folder == row["cls_name"] == cls and obj == row["object_name"] and obj,
                        f"{split}/{cls}: inconsistent class or object ID")
                require(kind in {"Good", "NG"} and row["object_anomaly"] == int(kind == "NG"),
                        f"{split}/{cls}: inconsistent object label")
                require(split != "train" or (kind == "Good" and row["image_anomaly"] == 0),
                        f"{split}/{cls}: non-normal training record")
                match = IMAGE_NAME.fullmatch(filename)
                config = (row["view"], row["illumination"])
                require(match is not None and match.groups() == config and config in CONFIGURATIONS,
                        f"{split}/{cls}: inconsistent view/light")
                key = (cls, kind, obj)
                require(path not in paths[split] and config not in groups[key],
                        f"{split}/{cls}: duplicate path or configuration")
                paths[split].add(path)
                groups[key].add(config)
                count = counts[cls][f"{split}_{kind.lower()}"]
                count["images"] += 1
                if kind == "NG" and row["image_anomaly"] == 1:
                    count["positive_images"] += 1
                    count["undetectable_positive_images"] += int(row["detectable"] is False)
        for (cls, kind, _), configs in groups.items():
            require(configs == CONFIGURATIONS, f"{split}/{cls}: incomplete object configuration grid")
            counts[cls][f"{split}_{kind.lower()}"]["objects"] += 1
        instances[split] = set(groups)

    train, test = instances["train"], instances["test"]
    train_id = {(cls, obj) for cls, _, obj in train}
    test_id = {(cls, obj) for cls, _, obj in test}
    overlaps = {
        "full_keys": len(train & test),
        "image_paths": len(paths["train"] & paths["test"]),
        "class_and_id": len(train_id & test_id),
        "bare_id": len({obj for _, _, obj in train} & {obj for _, _, obj in test}),
    }
    require(not any(overlaps[key] for key in ("full_keys", "image_paths", "class_and_id")),
            f"cross-split overlap: {overlaps}")
    split_mismatches = {}
    for cls in ARCHIVES:
        train_good = {obj for category, kind, obj in train if category == cls and kind == "Good"}
        normals = sorted(obj for category, kind, obj in train | test if category == cls and kind == "Good")
        random.Random(42).shuffle(normals)
        expected_train = set(normals[:int(len(normals) * 0.6)])
        split_mismatches[cls] = len(expected_train ^ train_good)
    require(not any(split_mismatches.values()), f"published split differs from generator: {split_mismatches}")
    paper_objects = sum(sum(pair) for pair in PAPER_OBJECTS.values())
    difference = {}
    for cls, (paper_good, paper_ng) in PAPER_OBJECTS.items():
        row = counts[cls]
        good = row["train_good"]["objects"] + row["test_good"]["objects"]
        ng = row["test_ng"]["objects"]
        if (good, ng) != (paper_good, paper_ng):
            difference[cls] = {
                "paper_good": paper_good, "release_good": good,
                "paper_ng": paper_ng, "release_ng": ng,
            }
    return {
        "release_revision": RELEASE_REVISION,
        "manifest_sha256": MANIFEST.sha256,
        "classes": counts,
        "train": {"objects": len(train), "images": len(paths["train"])},
        "test": {"objects": len(test), "images": len(paths["test"])},
        "total": {"objects": len(train) + len(test), "images": sum(map(len, paths.values()))},
        "views": VIEWS,
        "illuminations": LIGHTS,
        "overlaps": overlaps,
        "official_good_split_replay": {"seed": 42, "train_ratio": 0.6, "train_id_symdiff": split_mismatches},
        "paper_comparison": {
            "source": PAPER_FIGURE,
            "paper_objects": paper_objects,
            "paper_images": 119_880,
            "paper_minus_release_objects": paper_objects - len(train) - len(test),
            "paper_minus_release_images": 119_880 - sum(map(len, paths.values())),
            "different_object_counts_by_class": difference,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True, help="cache directory outside the repository")
    args = parser.parse_args()
    print(json.dumps(check_manifest(load_manifest(args.cache)), indent=2))


if __name__ == "__main__":
    main()
