"""Plot observed stability and product-bootstrap intervals from the export tables."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

METHOD_COLOR = {"B0": "#546475", "B6": "#2c7e6d"}
METRICS = ("O-AUROC", "I-AUROC", "AUPRO")


def rows(path):
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def save(fig, output, name):
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"{name}.{suffix}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def stability(source, output):
    data = [r for r in rows(source) if r["status"] == "PASS" and r["setting"] in METHOD_COLOR]
    categories = [c for c in dict.fromkeys(row["category"] for row in data)
                  if all(sum(row["category"] == c and row["setting"] == method for row in data) == 3
                         for method in METHOD_COLOR)]
    if not categories:
        return
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65), layout="constrained", sharey=True)
    for ax, metric in zip(axes, METRICS, strict=True):
        for method, color in METHOD_COLOR.items():
            offset = -.14 if method == "B0" else .14
            for idx, category in enumerate(categories):
                values = [float(r[metric]) for r in data if r["category"] == category and r["setting"] == method]
                ax.scatter(values, (idx + offset - .055, idx + offset, idx + offset + .055),
                           color=color, marker="o" if method == "B0" else "D", s=12)
        ax.set(title=metric, xlabel="Official score (%)", xlim=(0, 102))
        ax.set_yticks(range(len(categories)), categories)
        ax.grid(axis="x", color="#e6e8e9", zorder=-1)
    legends = [Line2D([], [], color=color, marker="o" if m == "B0" else "D", linestyle="None", label=label)
               for (m, color), label in zip(METHOD_COLOR.items(), ("Paired v1", "Frozen refinement"), strict=True)]
    axes[0].legend(handles=legends, loc="lower left", fontsize=6, bbox_to_anchor=(.05, -.45), ncol=2)
    fig.suptitle("Three fixed seeds on six previously held-out categories", x=.05, ha="left")
    save(fig, output, "historical-seed-stability")


def intervals(source, output):
    data = [row for row in rows(source) if row["seed"] == "42"]
    if not data:
        return
    labels = list(dict.fromkeys(row["category"] for row in data))
    fig, axes = plt.subplots(1, 2, figsize=(7.1, max(2.1, .38 * len(labels) + 1.55)), layout="constrained", sharey=True, sharex=True)
    for ax, metric in zip(axes, METRICS[:2], strict=True):
        selected = {row["category"]: row for row in data if row["metric"] == metric}
        for index, category in enumerate(labels):
            row = selected[category]
            score, low, high = [float(row[key]) for key in ("delta_pp", "lower_95_pp", "upper_95_pp")]
            outer_lo, outer_hi = [float(row[key]) for key in ("simultaneous_lower_pp", "simultaneous_upper_pp")]
            ax.hlines(index, outer_lo, outer_hi, color="#aeb9c2", linewidth=.85)
            ax.hlines(index, low, high, color="#306f62", linewidth=1.9)
            ax.scatter(score, index, color="#204e48", s=16, zorder=3)
        ax.axvline(0, color="#546475", linestyle="--", linewidth=.75)
        ax.set(title=metric, xlabel="B6 − B0 (percentage points)")
        ax.set_yticks(range(len(labels)), labels)
        ax.grid(axis="x", color="#e8e8e8", zorder=-1)
    axes[0].invert_yaxis()
    text = f"Paired product bootstrap · seed 42 · {len(labels)} categories"
    fig.suptitle(text, x=.05, ha="left")
    save(fig, output, "historical-paired-bootstrap")


def costs(source, output):
    record = json.loads(source.read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.65), layout="constrained")
    for ax, mode, title in zip(axes, ("cache_read_and_score", "png_read_encode_and_score"),
                               ("Cached features", "Read PNG, encode, score"), strict=True):
        for i, method in enumerate(("B2", "B6")):
            subset = [row for row in record["probes"] if row["setting"] == method and row["scope"] == mode]
            ax.bar([k + (i - .5) * .34 for k in range(len(subset))],
                   [r["median_seconds"] for r in subset], width=.34, color=("#687b91", "#2c7e6d")[i],
                   label=("Multilayer prototype", "Frozen refinement")[i])
        ax.set(title=title, ylabel="Median seconds", xticks=(0, 1), xticklabels=("10 images", "120 images"))
    axes[1].legend(loc="upper left", fontsize=6, handlelength=1)
    fig.suptitle("Bird latency probe · one product · shared RTX 3090", x=.05, ha="left")
    save(fig, output, "bird-runtime")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation", type=Path, required=True)
    parser.add_argument("--intervals", type=Path)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "axes.spines.right": False, "axes.spines.top": False,
                         "axes.linewidth": .75, "pdf.fonttype": 42, "legend.frameon": False})
    stability(args.ablation, args.output)
    if args.intervals and args.intervals.is_file():
        intervals(args.intervals, args.output)
    if args.runtime and args.runtime.is_file():
        costs(args.runtime, args.output)


if __name__ == "__main__":
    main()
