"""Generate TASK-002 comparison artifacts from eight verified formal evaluations."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from baseline_metrics import read_arrays
from summarize_baselines import CLASSES, MODELS, row


LABELS = {"dinomaly": "Dinomaly", "inpformer": "INP-Former"}
COLORS = {"dinomaly": "#246596", "inpformer": "#AB461F"}
VIEWS = tuple(f"{angle:03d}" for angle in range(0, 360, 30))
LIGHTS = tuple(f"{light:02d}" for light in range(1, 11))
METRICS = (("O-AUROC_pct", "O-AUROC (%)"), ("I-AUROC_pct", "I-AUROC (%)"), ("AUPRO_pct", "AUPRO (%)"))


def save_figure(fig, target):
    fig.savefig(target.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(target.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def grouped_bars(table, fields, output, percent=False):
    fig, axes = plt.subplots(1, len(fields), figsize=(3.4 * len(fields), 3.35), squeeze=False,
                             layout="constrained")
    for ax, (key, label, scale) in zip(axes[0], fields):
        for offset, model in zip((-0.18, 0.18), MODELS):
            values = [table[(model, cls)][key] / scale for cls in CLASSES]
            ax.bar(np.arange(len(CLASSES)) + offset, values, width=0.35, color=COLORS[model],
                   label=LABELS[model])
        ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=15)
        ax.set_ylabel(label)
        if percent:
            ax.set_ylim(0, 100)
        ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.grid(axis="y", alpha=0.22)
        ax.set_axisbelow(True)
    axes[0, 0].legend(loc="lower left" if percent else "upper left", frameon=False)
    save_figure(fig, output)


def load_diagnostics(root):
    rows = {}
    for model in MODELS:
        for cls in CLASSES:
            path = root / "evaluations" / model / cls / "diagnostics.csv"
            with path.open(encoding="utf-8", newline="") as stream:
                records = list(csv.DictReader(stream))
            keys = {(r["scope"], r["view"], r["illumination"]) for r in records}
            expected = {("view_fused", v, "") for v in VIEWS} | {("raw_light", "", l) for l in LIGHTS}
            expected |= {("raw_configuration", v, l) for v in VIEWS for l in LIGHTS}
            if keys != expected or len(records) != len(expected):
                raise ValueError(f"Incomplete or duplicated diagnostics: {path}")
            rows[(model, cls)] = records
    return rows


def plot_conditions(rows, mode, output):
    scope, column, values, xlabel = (("view_fused", "view", VIEWS, "View angle (°)") if mode == "view"
                                     else ("raw_light", "illumination", LIGHTS, "Illumination index"))
    fig, axes = plt.subplots(2, len(CLASSES), figsize=(12, 5), sharey=True, layout="constrained")
    for col, cls in enumerate(CLASSES):
        axes[0, col].set_title(cls)
        for ax, (metric, label) in zip(axes[:, col], (("AUROC_pct", "AUROC (%)"), ("AUPRO_pct", "AUPRO (%)"))):
            for model in MODELS:
                by_id = {r[column]: r[metric] for r in rows[(model, cls)] if r["scope"] == scope}
                # A gap represents an undefined metric.
                series = [float(by_id[x]) if by_id[x] else np.nan for x in values]
                ax.plot(range(len(values)), series, ".-", color=COLORS[model], label=LABELS[model])
            step = 3 if mode == "view" else 2
            positions = range(0, len(values), step)
            ax.set_xticks(list(positions), [str(int(values[i])) for i in positions])
            ax.set_ylim(0, 102)
            ax.grid(axis="y", alpha=0.22)
            if col == 0:
                ax.set_ylabel(label)
        axes[-1, col].set_xlabel(xlabel)
    axes[0, 0].legend(loc="lower left", frameon=False, fontsize=8)
    save_figure(fig, output)


def row_keys(arrays):
    return list(zip(arrays["obj_names"], arrays["object_anomalys"], arrays["view_list"],
                    arrays["illumination_list"], strict=True))


def plot_heatmaps(root, output):
    fig, axes = plt.subplots(len(CLASSES), 3, figsize=(7.2, 9.2), layout="constrained")
    chosen, normal_scores = [], []
    for row_index, cls in enumerate(CLASSES):
        arrays = {model: read_arrays(root / "evaluations" / model / cls / "raw") for model in MODELS}
        first = arrays[MODELS[0]]
        keys = row_keys(first)
        candidates = sorted((key, i) for i, key in enumerate(keys)
                            if first["object_anomalys"][i] == first["image_anomalys"][i] == 1)
        key, index = next(((key, i) for key, i in candidates if first["imgs_masks"][i].any()), (None, None))
        if key is None:
            raise ValueError(f"No manifest-positive sample with a mask for {cls}")
        mask = first["imgs_masks"][index, 0]
        maps = []
        chosen.append({"class": cls, "object": str(key[0]), "view": str(key[2]), "illumination": str(key[3])})
        for model in MODELS:
            model_rows = arrays[model]
            own_keys = row_keys(model_rows)
            if len(own_keys) != len(keys) or set(own_keys) != set(keys):
                raise ValueError(f"Prediction identities differ between models for {cls}")
            row_id = own_keys.index(key)
            if not np.array_equal(mask, model_rows["imgs_masks"][row_id, 0]):
                raise ValueError(f"Ground-truth mask differs between models for {cls}")
            maps.append(model_rows["anomaly_maps"][row_id])
            for light in LIGHTS:
                selected = (model_rows["object_anomalys"] == 0) & (model_rows["illumination_list"] == light)
                scores = model_rows["anomaly_scores"][selected]
                normal_scores.append({"model": model, "class": cls, "illumination": light, "images": int(selected.sum()),
                                      "normal_score_median": float(np.median(scores)),
                                      "normal_score_p95": float(np.quantile(scores, 0.95))})
        axes[row_index, 0].imshow(mask, cmap="gray", vmin=0, vmax=1)
        low, high = min(float(m.min()) for m in maps), max(float(m.max()) for m in maps)
        for ax, anomaly_map in zip(axes[row_index, 1:], maps):
            image = ax.imshow(anomaly_map, cmap="magma", vmin=low, vmax=high, interpolation="nearest")
        fig.colorbar(image, ax=axes[row_index, 1:], shrink=0.68, label="Anomaly score")
        for ax in axes[row_index]:
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row_index, 0].set_ylabel(cls)
    for ax, title in zip(axes[0], ("Ground truth", *[LABELS[m] for m in MODELS])):
        ax.set_title(title)
    save_figure(fig, output / "reference-heatmaps")
    write_csv(output / "heatmap-inputs.csv", chosen)
    write_csv(output / "normal-scores-by-light.csv", normal_scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=lambda p: Path(p).resolve(), required=True)
    parser.add_argument("--output", type=lambda p: Path(p).resolve(), required=True)
    args = parser.parse_args()
    table = {(model, cls): row(args.root / "runs", args.root / "evaluations", model, cls)
             for model in MODELS for cls in CLASSES}
    if any(record["status"] != "PASS" for record in table.values()):
        raise ValueError("All eight formal training/evaluation records must pass before generating figures")
    diagnostics = load_diagnostics(args.root)
    args.output.mkdir(parents=True, exist_ok=False)
    compared_fields = (*[key for key, _ in METRICS], "train_seconds", "peak_allocated_bytes")
    write_csv(args.output / "model-comparison.csv", [
        {"class": cls, **{f"{model}_{key}": table[(model, cls)][key]
                          for model in MODELS for key in compared_fields}}
        for cls in CLASSES
    ])
    grouped_bars(table, [(key, label, 1) for key, label in METRICS], args.output / "official-metrics", percent=True)
    grouped_bars(table, (("train_seconds", "Training time (hours)", 3600),
                         ("peak_allocated_bytes", "Peak GPU memory allocated (GiB)", 2**30)),
                 args.output / "training-resources")
    for mode in ("view", "illumination"):
        plot_conditions(diagnostics, mode, args.output / f"{mode}-diagnostics")
    plot_heatmaps(args.root, args.output)
    write_csv(args.output / "diagnostics.csv", [{"model": model, "class": cls, **result}
                                               for (model, cls), results in diagnostics.items() for result in results])


if __name__ == "__main__":
    main()
