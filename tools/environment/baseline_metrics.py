"""Collect official evaluation inputs and run read-only diagnostics after final evaluation."""

import csv
import math
from collections import defaultdict

import numpy as np


def check_fusion(raw, fused):
    """Cross-check the official output against the group reductions audited in TASK-001."""
    objects, views = defaultdict(list), defaultdict(list)
    groups = zip(raw["obj_names"], raw["view_list"])
    for i, (obj, view) in enumerate(groups):
        objects[obj].append(i)
        views[f"{obj}_{view}"].append(i)
    for scope, groups, size, labels in (("object", objects, 120, "object_anomalys"),
                                        ("view", views, 10, "image_anomalys")):
        names = fused[f"{scope}_instance_names"]
        if len(names) != len(groups) or set(names) != set(groups):
            raise ValueError(f"Unexpected official {scope} fusion groups")
        for i, name in enumerate(names):
            indices = groups[name]
            if len(indices) != size:
                raise ValueError(f"Unexpected group size for {name}: {len(indices)}")
            np.testing.assert_allclose(fused[f"{scope}_instance_anomaly_scores"][i],
                                       raw["anomaly_scores"][indices].mean(), rtol=1e-6)
            np.testing.assert_equal(fused[f"{scope}_instance_anomalys"][i], raw[labels][indices].max())
            if scope == "view":
                mask = raw["imgs_masks"][indices, 0].max(axis=0)
                np.testing.assert_array_equal(fused["view_instance_masks"][i], mask)
                np.testing.assert_allclose(fused["view_instance_anomaly_maps"][i],
                                           raw["anomaly_maps"][indices].mean(axis=0), rtol=1e-6)


def save_arrays(directory, arrays):
    directory.mkdir(parents=True, exist_ok=False)
    for key, value in arrays.items():
        array = np.asarray(value)
        if array.dtype.kind == "O":
            array = array.astype(str)
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            raise ValueError(f"Non-finite official output in {key}")
        np.save(directory / f"{key}.npy", array, allow_pickle=False)


def read_arrays(directory):
    return {path.stem: np.load(path, mmap_mode="r", allow_pickle=False) for path in directory.glob("*.npy")}


def diagnostic_row(scope, view, light, selection, arrays, evaluator, keys, mask_key, cls):
    selected = {key: arrays[key][selection] for key in keys}
    present = selected[mask_key].reshape(selection.sum(), -1).max(axis=1)
    row = {"scope": scope, "view": view, "illumination": light, "images": len(present),
           "positive": int(np.count_nonzero(present)), "AUROC_pct": "", "AUPRO_pct": "", "status": "OK"}
    if len(np.unique(present)) < 2:
        row["status"] = "undefined_single_label"
        return row
    metrics, _ = evaluator(selected, cls)
    suffix = "view" if scope == "view_fused" else "img"
    auroc, pro = (float(metrics[name]) * 100 for name in (f"AUROC_{suffix}_sp", f"AUPRO_{suffix}"))
    row.update(AUROC_pct=auroc if math.isfinite(auroc) else "",
               AUPRO_pct=pro if math.isfinite(pro) else "",
               status="OK" if math.isfinite(auroc) and math.isfinite(pro) else "nonfinite_official_output")
    return row


def save_diagnostics(directory, cls, use_adeval):
    from util.metric import Evaluator

    raw = read_arrays(directory / "raw")
    fused = read_arrays(directory / "fused")
    img = Evaluator(metrics=["AUROC_img_sp", "AUPRO_img"], use_adeval=use_adeval)
    view_eval = Evaluator(metrics=["AUROC_view_sp", "AUPRO_view"], use_adeval=use_adeval)
    views, lights = sorted(set(raw["view_list"])), sorted(set(raw["illumination_list"]))
    img_keys = ("cls_names", "imgs_masks", "anomaly_maps", "image_anomalys", "anomaly_scores")
    view_keys = ("view_instance_cls_names", "view_instance_masks", "view_instance_anomaly_maps",
                 "view_instance_anomalys", "view_instance_anomaly_scores")
    view_ids = np.array([name.rsplit("_", 1)[-1] for name in fused["view_instance_names"]])
    rows = []
    for view in views:
        rows.append(diagnostic_row("view_fused", view, "", view_ids == view, fused,
                                   view_eval.execute_view_instance, view_keys, "view_instance_masks", cls))
        for light in lights:
            rows.append(diagnostic_row("raw_configuration", view, light,
                                       (raw["view_list"] == view) & (raw["illumination_list"] == light),
                                       raw, img.execute_img_instance, img_keys, "imgs_masks", cls))
    for light in lights:
        rows.append(diagnostic_row("raw_light", "", light, raw["illumination_list"] == light,
                                   raw, img.execute_img_instance, img_keys, "imgs_masks", cls))
    with (directory / "diagnostics.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {"rows": len(rows), "nonfinite_or_undefined": sum(row["status"] != "OK" for row in rows)}
