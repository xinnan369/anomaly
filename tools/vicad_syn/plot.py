"""Render paper-facing results and pre-specified diagnostic examples from verified arrays."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
from torch.nn import functional as F

from tools.vicad_syn.verify import METRICS, read_csv, write_csv
from vicad_syn.data.manifest import load_groups
from vicad_syn.experiment import check, read_json, write_json
from vicad_syn.training.train import frozen_model


matplotlib.rcParams.update({"pdf.fonttype": 42, "font.size": 8, "axes.spines.top": False,
                           "axes.spines.right": False})
COLORS = {"Dinomaly": "#486B8B", "INP-Former": "#9C6336", "Dinomaly-stabilized": "#8172A5", "VICAD-Syn": "#287969"}


def save(fig, output, name):
    fig.savefig(output / (name + ".pdf"))
    fig.savefig(output / (name + ".png"), dpi=230)
    plt.close(fig)


def compare(rows, repo, output):
    official = read_csv(repo / "results/baselines/metrics/baseline-summary.csv")
    stable, = read_csv(repo / "results/baselines/metrics/stabilized-doll.csv")
    table = []
    for item in official + [stable]:
        model = {"dinomaly": "Dinomaly", "inpformer": "INP-Former"}.get(item["model"], item["model"])
        table.append({"class": item["class"], "method": model, "identity": item["experiment_identity"],
                      **{name: item[name + "_pct"] for name in METRICS}})
    for item in rows:
        if item["setting"] == "A4":
            table.append({"class": item["class"], "method": "VICAD-Syn", "identity": "TASK-003 A4",
                          **{name: item[name] for name in METRICS}})
    write_csv(output / "baseline-comparison.csv", table)
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.2), layout="constrained")
    classes = ("Car", "Cube", "Doll", "Ring")
    for axis, name in zip(axes, METRICS, strict=True):
        for shift, (method, color) in zip((-0.24,-0.08,0.08,0.24), COLORS.items(), strict=True):
            vals = {item["class"]: float(item[name]) for item in table if item["method"] == method and item[name]}
            axis.bar(np.arange(4) + shift, [vals.get(cls, np.nan) for cls in classes], width=0.15, color=color, label=method)
        axis.set_xticks(range(4), classes)
        axis.set_ylabel(f"{name} (%)")
        axis.set_ylim(0,100)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    axes[0].text(2-0.24, 1, "failed", rotation=90, fontsize=7, va="bottom", ha="center")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=4, frameon=False)
    save(fig, output, "baseline-comparison")


def ablation(rows, output):
    selected = sorted((item for item in rows if item["class"] == "Car"), key=lambda item: item["setting"])
    if len(selected) != 5:
        return
    fig, axis = plt.subplots(figsize=(5.8, 3.2), layout="constrained")
    x = np.arange(5)
    for offset, metric, color in zip((-0.24,0,0.24), METRICS, ("#486B8B","#9C6336","#287969"), strict=True):
        axis.bar(x+offset, [float(item[metric]) for item in selected], width=0.22, color=color, label=metric)
    axis.set_xticks(x, [item["setting"] for item in selected])
    axis.set_ylabel("Score (%)")
    axis.set_ylim(0,100)
    handles, labels = axis.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=3, frameon=False)
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    save(fig, output, "car-ablations")


def condition_profiles(input, output):
    rows = read_csv(input)
    fig, axes = plt.subplots(3, 2, figsize=(7.1, 6.0), layout="constrained")
    for i, (key, name) in enumerate((("residual","Residual"), ("stability","Context disagreement"), ("epistemic","Head dispersion"))):
        for j, scope in enumerate(("light","view")):
            axis = axes[i,j]
            sel = [row for row in rows if row["scope"] == scope]
            x = [int(item["id"]) for item in sel]
            axis.plot(x, [float(item[key+"_mean"]) for item in sel], color="#287969", marker="o", ms=2.7)
            axis.set_ylabel(name)
            axis.ticklabel_format(axis="y", style="sci", scilimits=(-2,2))
            axis.grid(alpha=0.23)
            if i == 2:
                axis.set_xlabel("Illumination ID" if j == 0 else "View angle (°)")
    save(fig, output, "car-condition-profiles")


def representative(upstream_root, run, frozen, output):
    home = run / "Car"
    evaluation = home / "conditioned/evaluation"
    groups = load_groups(upstream_root, "Car", "test", frozen["config"])
    masks = np.load(evaluation / "raw/imgs_masks.npy", mmap_mode="r", allow_pickle=False).reshape(840, 10,224,224)
    # Selection is by identity order and nonempty ground truth only, before consulting any scores.
    group, light = next((g, l) for g in range(len(groups)) for l in range(10) if masks[g,l].any())
    sample = groups[group].records[light]
    cache = np.load(home / "test-cache/features.npy", mmap_mode="r", allow_pickle=False)
    x = torch.from_numpy(cache[group*10:(group+1)*10].copy()).unsqueeze(0)
    x = F.normalize(x.float(), dim=-1)
    torch.set_num_threads(4)
    model, receipt = frozen_model(home / "conditioned", frozen, read_json(home / "train-cache/cache.json"),
                                  True, torch.device("cpu"))
    with torch.inference_mode():
        features = x[0,light]
        predicted = F.normalize(model(x).mean(0)[0,light], dim=-1)
        center = features.mean(0)
        _, _, basis = torch.linalg.svd(features - center, full_matrices=False)
        axis = basis[0]
        axis = torch.where(axis[axis.abs().argmax()] < 0, -axis, axis)
        a = ((features-center) @ axis).reshape(16,16).numpy()
        b = ((predicted-center) @ axis).reshape(16,16).numpy()
    with np.load(evaluation / "patch-signals.npz", allow_pickle=False) as signals:
        r, s, e = (signals[name][group,light].reshape(16,16) for name in ("residual","stability","epistemic"))

    def fused(mode):
        dir = evaluation / mode / "fused"
        names = np.load(dir / "view_instance_names.npy", allow_pickle=False)
        labels = np.load(dir / "view_instance_anomalys.npy", allow_pickle=False)
        selection = (names == f"{sample['object_name']}_{sample['view']}") & (labels == 1)
        check(selection.sum() == 1, "representative view is ambiguous")
        return np.load(dir / "view_instance_anomaly_maps.npy", allow_pickle=False, mmap_mode="r")[selection][0]
    plain, weighted = fused("A1"), fused("A4")
    union = masks[group].max(0)
    ys, xs = np.nonzero(union)
    x0, x1 = max(0, int(xs.min())-6), min(224, int(xs.max())+7)
    y0, y1 = max(0, int(ys.min())-6), min(224, int(ys.max())+7)
    vmax = max(float(r.max()), float(plain.max()), float(weighted.max()))
    pmax = max(abs(float(a.min())), abs(float(a.max())), abs(float(b.min())), abs(float(b.max())))
    fig, axes = plt.subplots(2,4,figsize=(8.0,4.35),layout="constrained")
    panels = (("Target feature PC1",a,"coolwarm",-pmax,pmax), ("Predicted feature PC1",b,"coolwarm",-pmax,pmax),
              ("Target residual",r,"magma",0,vmax), ("View mask union",union,"gray",0,1),
              ("Context variation",s,"viridis",0,float(s.max())), ("Head dispersion",e,"viridis",0,float(e.max())),
              ("Equal light mean",plain,"magma",0,vmax), ("Reliability fusion",weighted,"magma",0,vmax))
    for axis, (title, values, cmap, lower, upper) in zip(axes.flat,panels,strict=True):
        handle = axis.imshow(values, cmap=cmap, vmin=lower, vmax=upper, interpolation="nearest")
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        if title == "View mask union":
            axis.add_patch(Rectangle((x0-0.5,y0-0.5),x1-x0,y1-y0,fill=False,linewidth=0.8,edgecolor="#49E8CF"))
        else:
            fig.colorbar(handle, ax=axis, fraction=0.052, shrink=0.77)
    save(fig,output,"car-feature-and-reliability")
    write_json(output / "car-figure-source.json", {"sample": sample["img_path"],
        "criterion": "first view/light in fixed identity order with positive resized mask",
        "checkpoint_sha256": receipt["sha256"], "projection": "PC1 of target patch features, reused for prediction",
        "interpolation": "display only, preserve original feature or official evaluation grid",
        "mask_overlay": "six-pixel padding around visible union mask bounds", "mask_positive_pixels": int(union.sum())})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("inputs", "run", "verification", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = read_csv(args.verification / "summary.csv")
    if not rows:
        return
    compare(rows, Path(__file__).resolve().parents[2], args.output)
    ablation(rows, args.output)
    proof = read_json(args.verification / "verification.json")
    if "Car" in proof["categories"]:
        condition_profiles(args.run / "Car/conditioned/evaluation/diagnostics.csv", args.output)
        representative(args.inputs, args.run, proof["frozen"]["frozen"], args.output)


if __name__ == "__main__":
    main()
