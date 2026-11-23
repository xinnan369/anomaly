"""Render mechanism and Synergy figures from sealed TASK-004 outputs."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

from vicad_syn.experiment import read_json


plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.right": False,
                     "axes.spines.top": False, "pdf.fonttype": 42, "svg.fonttype": "none"})
INK = "#22364d"
PALETTE = ["#536f91", "#b36a58", "#528a80", "#8c709f"]


def save(fig, folder, name):
    for ext in ("pdf", "png"):
        fig.savefig(folder / f"{name}.{ext}", bbox_inches="tight", dpi=220)
    plt.close(fig)


def architecture(folder):
    fig, ax = plt.subplots(figsize=(8.3, 3.5))
    ax.set(xlim=(0, 10), ylim=(0, 6))
    ax.axis("off")
    nodes = {
        "input": (0.2, 2.5, 2, "One instance, one view\nTen aligned lights"),
        "frozen": (2.7, 2.5, 2.0, "Frozen ViT-B/14 reg4\nRemove CLS + registers"),
        "v1": (5.3, 4.0, 2.1, "v1: block 12\nLOLO feature prediction"),
        "multi": (5.3, 2.5, 2.1, "Blocks 4, 8, 12\nMulti-layer prediction"),
        "normal": (5.3, 1.0, 2.1, "Train-only normal objects\nView/light patch prior"),
        "global": (8.0, 4.0, 1.8, "v1 global\nlight weighting"),
        "local": (8.0, 1.75, 1.8, "Patch weights\nand score floor"),
    }
    for name, (x, y, w, label) in nodes.items():
        color = "#e7eef5" if name in ("input", "frozen", "multi", "normal", "local") else "#ece9f3"
        ax.add_patch(FancyBboxPatch((x, y), w, .85, boxstyle="round,pad=0.03,rounding_size=.08",
                                    facecolor=color, edgecolor="#bac6d3", linewidth=.9))
        ax.text(x + w/2, y + .43, label, va="center", ha="center", color=INK, fontsize=8)
    for a, b in (("input", "frozen"), ("frozen", "v1"), ("frozen", "multi"), ("frozen", "normal"),
                 ("v1", "global"), ("multi", "local"), ("normal", "local")):
        xa, ya, wa, _ = nodes[a]
        xb, yb, _, _ = nodes[b]
        ax.annotate("", (xb-.04, yb+.42), (xa+wa+.04, ya+.42),
                    arrowprops={"arrowstyle":"->", "color":"#64788e", "linewidth":1})
    ax.text(.2, 5.4, "A  |  Fixed same-view scoring paths", color=INK, fontsize=11, weight="bold")
    ax.text(5.3, .2, "Candidate layer combination and fusion are selected on normal-only proxy data.",
            color=INK, fontsize=7.5)
    save(fig, folder, "method-structure")


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def v1_diagnosis(source, folder):
    data = read_csv(source / "v1-localization.csv")
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.65), layout="constrained",
                             gridspec_kw={"width_ratios": [1, 1.2, 1.2]})
    x = np.arange(len(data))
    categories = [r["category"] for r in data]
    columns = [("median_mask_pixels", "A  |  Defect mask extent", "Median mask area (pixels)"),
               ("A4_lower_fg_bg_contrast_view_pct", "B  |  Contrast suppression", "Views reduced by A4 (%)")]
    for axis, (key, title, y) in zip(axes, columns, strict=False):
        axis.bar(x, [float(row[key]) for row in data], color=PALETTE[0], width=.65)
        axis.set(xticks=x, xticklabels=categories, ylabel=y, title=title)
    keys = ("residual_patch_auc_subsampled", "stability_patch_auc_subsampled", "epistemic_patch_auc_subsampled")
    for i, (key, label) in enumerate(zip(keys, ("Residual", "Stability", "Epistemic"))):
        axes[2].plot(x, [float(row[key]) for row in data], "o-", label=label, color=PALETTE[i], lw=1)
    axes[2].axhline(.5, color=".6", linestyle=":", lw=.8)
    axes[2].set(xticks=x, xticklabels=categories, ylabel="Patch AUROC (sampled)", title="C  |  Signal discrimination")
    axes[2].legend(loc="lower right", fontsize=6.5)
    save(fig, folder, "v1-localization-mechanisms")

    with np.load(source / "Car-v1-example.npz", allow_pickle=False) as run:
        fig, axes = plt.subplots(1, 4, figsize=(7, 2.05), layout="constrained")
        limits = np.quantile(np.stack([run["A1"], run["A4"]]), [.01, .99])
        panels = [(run["mask"], "A  |  Released mask"), (run["tile"], "B  |  Occupied patches"),
                  (run["A1"], "C  |  Equal fusion"), (run["A4"], "D  |  v1 reliability")]
        for i, (axis, (image, title)) in enumerate(zip(axes, panels, strict=True)):
            view = axis.imshow(image, cmap="gray_r", vmin=0, vmax=1) if i < 2 else axis.imshow(
                image, cmap="magma", vmin=limits[0], vmax=limits[1])
            if i >= 2:
                axis.contour(run["mask"], levels=[.5], colors=["#55f0e5"], linewidths=.65)
            axis.set_title(title, loc="left")
            axis.set(xticks=[], yticks=[])
            for spine in axis.spines.values():
                spine.set(visible=True, color="#d0d7e2", linewidth=.6)
        fig.colorbar(view, ax=list(axes[2:]), fraction=.055, pad=.03, label="Residual")
        save(fig, folder, "car-v1-patch-resolution")


def development(source, folder):
    matrix = read_csv(source / "selection.csv")
    cands = list(dict.fromkeys(x["candidate"] for x in matrix))
    fusions = list(dict.fromkeys(x["fusion"] for x in matrix))
    values = np.array([float(x["macro_pixel_ap"]) for x in matrix]).reshape(len(cands), len(fusions)) * 100
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.85), layout="constrained")
    plot = axes[0].imshow(values, cmap="Blues", aspect="auto", vmin=0, vmax=values.max()*1.2)
    for i in range(len(cands)):
        for j in range(len(fusions)):
            axes[0].text(j, i, f"{values[i,j]:.2f}", va="center", ha="center", fontsize=7)
    axes[0].set(yticks=range(len(cands)), yticklabels=["Final layer", "Multi-layer", "Normal reference", "Combined"],
                xticks=range(len(fusions)), xticklabels=["Equal", "Global", "Patch joint", "Protected", "Patch stab.", "Patch epis."],
                title="A  |  Normal-only model selection")
    axes[0].tick_params(axis="x", labelrotation=32)
    fig.colorbar(plot, ax=axes[0], shrink=.8, label="Macro pixel AP (%)")
    rows = [r for directory in source.iterdir() if (directory / "mechanism-rows.csv").exists()
            for r in read_csv(directory / "mechanism-rows.csv")]
    names = ["one_weak", "four_strong", "all_weak", "all_strong"]
    labels = {"nine_direct":"9-light learned", "normal_conditional":"Normal reference",
              "nine_median":"9-light median", "four_subset":"4-light subset"}
    for i, (context, label) in enumerate(labels.items()):
        y = [np.mean([float(row["prediction_feature_shift"]) for row in rows
                      if row["scenario"] == scenario and row["context"] == context]) for scenario in names]
        axes[1].plot(range(4), y, "o-", color=PALETTE[i], label=label, lw=1)
    axes[1].legend(loc="upper left", fontsize=7)
    axes[1].set(xticks=range(4), xticklabels=["1 weak", "4 strong", "10 weak", "10 strong"],
                ylabel="Mean squared prediction shift", title="B  |  Synthetic structural contamination")
    save(fig, folder, "normal-development-and-contamination")


def example(source, folder):
    with np.load(source / "Car/synthetic-maps.npz", allow_pickle=False) as records:
        fig, axes = plt.subplots(2, 4, figsize=(7.7, 4.5), layout="constrained")
        scenario = "all_strong"
        image = records[f"figure_{scenario}_image"]
        image = np.clip(image.transpose(1, 2, 0) * [.229, .224, .225] + [.485, .456, .406], 0, 1)
        layers = records[f"figure_{scenario}_layers"].reshape(3, 16, 16)
        mask = records[f"0_{scenario}_mask"]
        residual = records[f"0_{scenario}_nine_direct"].mean(0).reshape(16, 16)
        stability = records[f"0_{scenario}_stability"].mean(0).reshape(16, 16)
        epistemic = records[f"0_{scenario}_epistemic"].mean(0).reshape(16, 16)
        weights = records[f"figure_{scenario}_weights"].reshape(16, 16)
        panels = [(image, "A  |  Synthetic structure", None), (layers[0], "B  |  Block 4 reference", "magma"),
                  (layers[1], "C  |  Block 8 reference", "magma"), (layers[2], "D  |  Block 12 reference", "magma"),
                  (residual, "E  |  v1 prediction residual", "magma"), (stability, "F  |  Context instability", "magma"),
                  (epistemic, "G  |  Head disagreement", "magma"), (weights, "H  |  Protected weight", "viridis")]
        shared = (float(layers.min()), float(layers.max()))
        for index, (axis, (arr, name, cmap)) in enumerate(zip(axes.flat, panels, strict=True)):
            im = axis.imshow(arr, cmap=cmap, extent=[0, 224, 224, 0],
                             vmin=shared[0] if index in (1, 2, 3) else None,
                             vmax=shared[1] if index in (1, 2, 3) else None)
            axis.contour(mask, levels=[.5], colors=["#4eeeec"], linewidths=.65, origin="upper", extent=[0, 224, 224, 0])
            axis.set_title(name, loc="left")
            axis.axis("off")
            if index >= 1:
                fig.colorbar(im, ax=axis, fraction=.045, pad=.03)
        save(fig, folder, "car-synthetic-feature-and-reliability")


def feature_prediction(source, folder):
    with np.load(source / "normal-feature-example.npz", allow_pickle=False) as data:
        title = ("A  |  Target light tokens", "B  |  v1 predicted tokens", "C  |  Normal-reference tokens")
        entries = [data[name] for name in ("target", "v1_prediction", "normal_reference")]
        joined = np.concatenate(entries)
        joined = joined - joined.mean(0)
        _, _, basis = np.linalg.svd(joined, full_matrices=False)
        projected = joined @ basis[:3].T
        limits = np.quantile(projected, [.01, .99], axis=0)
        colors = ((projected-limits[0])/np.maximum(limits[1]-limits[0], 1e-6)).clip(0,1)
        mask = data["mask"]
        fig, axes = plt.subplots(1, 4, figsize=(7.7, 2.5), layout="constrained")
        for i, axis in enumerate(axes[:3]):
            axis.imshow(colors[i*256:(i+1)*256].reshape(16,16,3), extent=(0,224,224,0), interpolation="nearest")
            axis.contour(mask, colors=["white"], levels=[.5], linewidths=.75, origin="upper", extent=(0,224,224,0))
            axis.set_title(title[i],loc="left")
            axis.axis("off")
        residual = ((entries[0]-entries[1])**2).sum(-1).reshape(16,16)
        heat = axes[3].imshow(residual, cmap="magma", extent=(0,224,224,0), interpolation="nearest")
        axes[3].contour(mask, colors=["#53ede6"], levels=[.5], linewidths=.75, origin="upper", extent=(0,224,224,0))
        axes[3].set_title("D  |  v1 token residual",loc="left")
        axes[3].axis("off")
        fig.colorbar(heat,ax=axes[3],fraction=.045,pad=.03)
        fig.suptitle("Car, all-light synthetic perturbation: shared feature PCA")
        save(fig,folder,"car-feature-prediction")

def exploratory(source, folder):
    rows = [r for r in read_csv(source) if r["scope"] == "exploratory"]
    if not rows:
        return
    baselines = Path(__file__).resolve().parents[2] / "results/vicad_syn/metrics/baseline-comparison.csv"
    original = read_csv(baselines)
    cats = list(dict.fromkeys(r["category"] for r in rows))
    x_positions = np.arange(len(cats))
    markers = {"v1_A4": ("v1 A4, paired", "o", PALETTE[0]),
               "selected": ("Selected", "D", PALETTE[1]),
               "Dinomaly": ("Dinomaly, original", "^", PALETTE[2]),
               "INP-Former": ("INP-Former, original", "s", PALETTE[3]),
               "Dinomaly-stabilized": ("Dinomaly-stabilized", "x", "#6f7582")}
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.15), layout="constrained")
    for ax, metric in zip(axes, ("O-AUROC", "I-AUROC", "AUPRO"), strict=True):
        for i, (method, (label, marker, color)) in enumerate(markers.items()):
            if method in ("v1_A4", "selected"):
                values = {r["category"]: float(r[metric]) for r in rows if r["setting"] == method}
            else:
                values = {r["class"]: float(r[metric]) for r in original if r["method"] == method and r[metric]}
            ax.scatter(x_positions + (i-2)*.10, [values.get(cat, np.nan) for cat in cats],
                       label=label, marker=marker, color=color, linewidths=0.9, s=24)
        ax.set(xticks=x_positions, xticklabels=cats, ylabel=f"{metric} (%)", ylim=(0,105))
        ax.grid(axis="y", alpha=.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncols=3,fontsize=7.2)
    fig.suptitle("Frozen comparison on previously seen categories")
    save(fig,folder,"exploratory-official-comparison")

def confirmation(source, folder):
    rows = read_csv(source)
    cats = list(dict.fromkeys(row["category"] for row in rows if row["scope"] == "confirmatory"))
    if not cats:
        return
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.6), layout="constrained")
    for ax, metric in zip(axes, ("O-AUROC", "I-AUROC", "AUPRO"), strict=True):
        for i, category in enumerate(cats):
            seeds = sorted({r["seed"] for r in rows if r["category"] == category})
            for seed in seeds:
                paired = {r["setting"]: float(r[metric]) for r in rows
                          if r["category"] == category and r["seed"] == seed and r["scope"] == "confirmatory"}
                ax.plot([i-.14, i+.14], [paired["v1_A4"], paired["selected"]], "-", color=".65", lw=.7)
                ax.scatter([i-.14, i+.14], [paired["v1_A4"], paired["selected"]], c=PALETTE[:2], s=12, zorder=3)
        ax.set(xticks=range(len(cats)), xticklabels=cats, ylabel=f"{metric} (%)")
    fig.suptitle("Previously unused categories: v1 A4 (blue) and selected candidate (orange)")
    save(fig, folder, "confirmatory-by-seed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("diagnostics", "development", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--formal", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    architecture(args.output)
    if (args.diagnostics / "v1-localization.csv").exists():
        v1_diagnosis(args.diagnostics, args.output)
    if (args.diagnostics / "normal-feature-example.npz").exists():
        feature_prediction(args.diagnostics,args.output)
    if (args.development / "selection.csv").exists():
        development(args.development, args.output)
        example(args.development, args.output)
    if args.formal and args.formal.exists():
        exploratory(args.formal, args.output)
        confirmation(args.formal, args.output)


if __name__ == "__main__":
    main()
