"""Exploratory localization contrasts for the strongest and weakest observed fusion changes."""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score

from tools.comprehensive.figures import save
from tools.comprehensive.resources import REPO
from vicad_syn.experiment import check, digest, read_json, write_json

METHODS = ("B1", "B2", "B3", "B4", "B6")


def positive_views(folder, receipt):
    fields = ("names", "masks", "anomaly_maps")
    files = {key: folder / f"view_instance_{key}.npy" for key in fields}
    for path in files.values():
        check(digest(path) == receipt["fused_and_patch_inputs"][path.name]["sha256"], "visual input hash changed")
    names, masks, maps = (np.load(files[key], mmap_mode="r", allow_pickle=False) for key in fields)
    indices = [i for i, mask in enumerate(masks) if mask.any()]
    match = {str(names[i]): (masks[i], maps[i]) for i in indices}
    check(len(match) == len(indices), "ambiguous positive view identity")
    return match


def cases(folders):
    check(all(value.keys() == folders["B4"].keys() for value in folders.values()), "case identities differ")
    differences = {}
    for name in folders["B4"]:
        mask, baseline = folders["B4"][name]
        for value in folders.values():
            check(np.array_equal(mask, value[name][0]), f"mask disagreement at {name}")
        refined = folders["B6"][name][1]
        ap = [float(average_precision_score(mask.ravel(), data.ravel())) for data in (baseline, refined)]
        differences[name] = {"equal_ap": ap[0], "stability_ap": ap[1], "delta_ap": ap[1] - ap[0]}
    ordered = sorted(differences, key=lambda key: (differences[key]["delta_ap"], key))
    return [(role, name, differences[name]) for role, name in (("Smallest change", ordered[0]), ("Largest change", ordered[-1]))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("ablation", "frozen", "bird", "output"):
        parser.add_argument("--" + key, type=lambda p: Path(p).resolve(), required=True)
    args = parser.parse_args()
    latest = args.ablation / "Bird/seed-42"
    recorded = read_json(latest / "ablation.json")
    check(recorded and recorded["status"] == "PASS", "finish Bird ablation before selecting cases")
    historical = args.frozen / "confirmatory/Bird/seed-42"
    frozen = read_json(historical / "evaluation.json")
    sources = {m: positive_views(latest / m, recorded["settings"][m]) for m in METHODS if m != "B6"}
    sources["B6"] = positive_views(historical / "selected", frozen["scores"]["selected"])
    examples = cases(sources)
    data = args.bird / "dataset"
    manifest = read_json(data / "jsons/meta_unsupervised.json")
    sample = {f'{r["object_name"]}_{r["view"]}': r["img_path"] for r in manifest["test"]["Bird"]
              if r["object_anomaly"] == 1 and r["illumination"] == "01"}
    fig, axes = plt.subplots(2, 7, figsize=(8.5, 3), layout="constrained")
    visual_title = ("Image (I01)", "Mask", "Leaf proto", "Multi proto", "Prediction", "Joint mean", "Joint local")
    for idx, (role, name, statistics) in enumerate(examples):
        image = Image.open(data / sample[name]).convert("RGB").resize((224, 224))
        mask = sources["B4"][name][0]
        maps = [sources[method][name][1] for method in METHODS]
        scale = float(np.percentile(np.concatenate([item.ravel() for item in maps]), 99))
        for column, (ax, item) in enumerate(zip(axes[idx], (image, mask, *maps), strict=True)):
            properties = {} if column < 2 else {"cmap": "magma", "vmin": 0, "vmax": max(scale, 1e-6)}
            if column == 1:
                properties.update(cmap="gray", vmin=0, vmax=1)
            ax.imshow(item, **properties)
            ax.set_xticks([])
            ax.set_yticks([])
            if idx == 0:
                ax.set_title(visual_title[column], fontsize=8)
            if column == 0:
                ax.set_ylabel(role, fontsize=8)
    fig.suptitle("Bird localization examples ranked by B6 minus B4 pixel AP", fontsize=10)
    args.output.mkdir(parents=True, exist_ok=True)
    save(fig, args.output, "bird-localization-cases")
    write_json(args.output / "bird-localization-cases.json", {
        "examples": [{"role": role, "instance_view": name, **stats} for role, name, stats in examples],
        "heatmap_scaling": "shared 99th percentile within each example; exploratory extremes",
    })


if __name__ == "__main__":
    main()
