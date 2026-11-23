"""Identity and physical instance isolation for the fixed Synergy release."""

from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath

from vicad_syn.experiment import check, digest, read_json


LIGHTS = tuple(f"{i:02d}" for i in range(1, 11))
VIEWS = tuple(f"{i:03d}" for i in range(0, 360, 30))


@dataclass(frozen=True)
class Group:
    identity: str
    view: str
    records: tuple

    @property
    def paths(self):
        return tuple(record["img_path"] for record in self.records)


def groups(records, category):
    by_group = defaultdict(dict)
    for row in records:
        path = PurePosixPath(row["img_path"])
        view, light = row["view"], row["illumination"]
        check(path.parts[:1] == (category,) and len(path.parts) == 4
              and path.parts[2] == row["object_name"] and row["cls_name"] == category
              and view in VIEWS and light in LIGHTS and path.name == f"A{view}_I{light}.png",
              "manifest record disagrees with released path")
        instance = sha256(path.parent.as_posix().encode()).hexdigest()
        key = instance, view
        check(light not in by_group[key], f"duplicate illumination {key}/{light}")
        by_group[key][light] = row
    result = []
    views = defaultdict(set)
    for (instance, view), lights in sorted(by_group.items()):
        check(set(lights) == set(LIGHTS), f"incomplete view {instance}/{view}")
        result.append(Group(instance, view, tuple(lights[light] for light in LIGHTS)))
        views[instance].add(view)
    check(all(values == set(VIEWS) for values in views.values()), "incomplete physical instance")
    return result


def split_ids(groups, seed, dev_objects):
    identities = {group.identity for group in groups}
    ordered = sorted(identities, key=lambda key: sha256(f"{seed}:{key}".encode()).digest())
    check(0 < dev_objects < len(ordered), "empty developer fit or normal validation split")
    return set(ordered[:dev_objects]), set(ordered[dev_objects:])


def load_groups(root, category, split, config):
    dataset = root / "dataset"
    path = dataset / "jsons/meta_unsupervised.json"
    receipt = read_json(root / "data_audit" / (category + ".json"))
    check(receipt and receipt["release_revision"] == config["data_revision"] and receipt["images"] == 12000
          and receipt["split_crc_size_overlap"] == 0, "no verified full archive receipt")
    check(digest(path) == config["manifest_sha256"], "fixed manifest SHA-256 mismatch")
    rows = read_json(path)
    train = groups(rows["train"][category], category)
    check(len(train) == 360 and len({g.identity for g in train}) == 30, "unexpected official train shape")
    check(all(r["object_anomaly"] == r["image_anomaly"] == 0 for g in train for r in g.records), "training rows must be normal")
    if split == "train":
        return train
    check(split == "test", "unknown split")
    test = groups(rows["test"][category], category)
    check(len(test) == 840 and len({g.identity for g in test}) == 70, "unexpected official test shape")
    check({g.identity for g in train}.isdisjoint(g.identity for g in test), "physical instance crosses official splits")
    return test
