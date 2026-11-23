"""Generate paper figures with separate official and stabilized data panels."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from plot_baselines import COLORS, LABELS, METRICS, save_figure, write_csv
from summarize_baselines import CLASSES, MODELS
from verify_baselines import csv_rows
from verify_doll_variant import NUMERICALLY_STABILIZED, OFFICIAL_CLEAN, OFFICIAL_FAILED


def official_panels(table, output):
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.5), layout="constrained")
    locations = np.arange(len(CLASSES))
    for ax, (key, name) in zip(axes, METRICS, strict=True):
        for model, offset in zip(MODELS, (-0.18, 0.18), strict=True):
            values = [float(table[model, cls][key]) if table[model, cls][key] else np.nan for cls in CLASSES]
            ax.bar(locations + offset, values, width=0.33, color=COLORS[model], label=LABELS[model])
        ax.text(2 - 0.18, 2, "failed", rotation=90, va="bottom", ha="center", fontsize=8)
        ax.set_xticks(locations, CLASSES)
        ax.set_ylabel(name)
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.grid(axis="y", alpha=0.22)
        ax.set_axisbelow(True)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside upper center", ncols=2, frameon=False, fontsize=9)
    save_figure(fig, output / "official-only-metrics")


def doll_panels(official, variant, output):
    fig, ax = plt.subplots(figsize=(7.2, 3.3), layout="constrained")
    x = np.arange(len(METRICS))
    labels = [label.replace(" (%)", "") for _, label in METRICS]
    ax.bar(x - 0.18, [float(official[key]) for key, _ in METRICS], 0.33, color=COLORS["inpformer"],
           label="INP-Former (official)")
    ax.bar(x + 0.18, [float(variant[key]) for key, _ in METRICS], 0.33, color=COLORS["dinomaly"],
           hatch="///", edgecolor="white", label="Dinomaly-stabilized (variant)")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside upper center", ncols=2, frameon=False, fontsize=9)
    save_figure(fig, output / "doll-variant-metrics")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("official", "variant", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    rows = csv_rows(args.official)
    table = {(item["model"], item["class"]): item for item in rows}
    assert len(rows) == len(table) == 8
    assert all(item["experiment_identity"] == (OFFICIAL_FAILED if key == ("dinomaly", "Doll") else OFFICIAL_CLEAN)
               for key, item in table.items())
    assert not any(table["dinomaly", "Doll"][key] for key, _ in METRICS)
    variant, = csv_rows(args.variant)
    assert variant["experiment_identity"] == NUMERICALLY_STABILIZED
    args.output.mkdir(parents=True, exist_ok=False)
    fields = ("experiment_identity", *(key for key, _ in METRICS),
              "train_seconds", "peak_allocated_bytes", "inference_seconds")
    pairs = [{"class": category, **{f"{model}_{field}": table[model, category][field]
                                    for model in MODELS for field in fields}} for category in CLASSES]
    write_csv(args.output / "model-comparison.csv", pairs)
    official_panels(table, args.output)
    if variant["verification_status"] == "EVIDENCE_COMPLETE":
        doll_panels(table["inpformer", "Doll"], variant, args.output)


if __name__ == "__main__":
    main()
