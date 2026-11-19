"""Collect official evaluation inputs and run read-only diagnostics after final evaluation."""

import csv
import hashlib
import math
from collections import defaultdict

import numpy as np


def complete_match(candidates):
    """Match each expected row to a distinct compatible observed row."""
    assigned = {}

    def assign(expected, seen):
        for index in candidates[expected]:
            if index in seen:
                continue
            seen.add(index)
            if index not in assigned or assign(assigned[index], seen):
                assigned[index] = expected
                return True
        return False

    return all(assign(i, set()) for i in range(len(candidates)))


def match_fusion_scope(raw, fused, scope, groups, group_size, label_key):
    prefix = f"{scope}_instance_"
    names, classes, labels, scores = (fused[prefix + key] for key in
                                     ("names", "cls_names", "anomalys", "anomaly_scores"))
    columns = [names, classes, labels, scores]
    if scope == "view":
        masks, maps = fused[prefix + "masks"], fused[prefix + "anomaly_maps"]
        columns.extend((masks, maps))
    if any(len(column) != len(groups) for column in columns):
        raise ValueError(f"Unexpected official {scope} fusion row count")

    expected, observed = defaultdict(list), defaultdict(list)
    for identity in groups:
        obj, cls, *_ = identity
        name = f"{obj}_{identity[2]}" if scope == "view" else obj
        expected[(cls, name)].append(identity)
    for i, key in enumerate(zip(classes, names)):
        observed[key].append(i)
    if expected.keys() != observed.keys():
        raise ValueError(f"Unexpected official {scope} fusion class or name")

    for key, identities in expected.items():
        indices = observed[key]
        if len(identities) != len(indices):
            raise ValueError(f"Unexpected official {scope} multiplicity for {key}")
        candidates = []
        for identity in identities:
            rows = groups[identity]
            if len(rows) != group_size:
                raise ValueError(f"Unexpected input group size for {identity}: {len(rows)}")
            label = raw[label_key][rows].max()
            score = raw["anomaly_scores"][rows].mean()
            compatible = [i for i in indices if labels[i] == label
                          and np.isclose(scores[i], score, rtol=1e-6, atol=0)]
            if scope == "view" and compatible:
                mask = raw["imgs_masks"][rows, 0].max(axis=0)
                unmasked = raw["anomaly_maps"][rows].mean(axis=0)
                compatible = [i for i in compatible if np.array_equal(masks[i], mask)
                              and maps[i].shape == unmasked.shape
                              and np.allclose(maps[i], unmasked, rtol=1e-6, atol=0)]
            candidates.append(compatible)
        if not complete_match(candidates):
            raise AssertionError(f"Official {scope} fusion values do not match the input groups for {key}")
    return sum(len(identities) > 1 for identities in expected.values())


def check_fusion(raw, fused):
    """Check official reductions even when their displayed names repeat."""
    objects, views = defaultdict(list), defaultdict(list)
    groups = zip(raw["obj_names"], raw["cls_names"], raw["view_list"], raw["object_anomalys"], strict=True)
    for i, (obj, cls, view, anomaly) in enumerate(groups):
        objects[(obj, cls, anomaly)].append(i)
        views[(obj, cls, view, anomaly)].append(i)
    repeated_objects = match_fusion_scope(raw, fused, "object", objects, 120, "object_anomalys")
    repeated_views = match_fusion_scope(raw, fused, "view", views, 10, "image_anomalys")
    return {"objects": len(objects), "views": len(views), "repeated_object_names": repeated_objects,
            "repeated_view_names": repeated_views}


def save_arrays(directory, arrays):
    directory.mkdir(parents=True, exist_ok=False)
    evidence = {}
    for key, value in arrays.items():
        array = np.asarray(value)
        if array.dtype.kind == "O":
            array = array.astype(str)
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            raise ValueError(f"Non-finite official output in {key}")
        path = directory / f"{key}.npy"
        np.save(path, array, allow_pickle=False)
        with path.open("rb") as source:
            evidence[path.name] = {"shape": list(array.shape), "dtype": str(array.dtype),
                                   "bytes": path.stat().st_size,
                                   "sha256": hashlib.file_digest(source, "sha256").hexdigest()}
    return evidence


def read_arrays(directory):
    return {path.stem: np.load(path, mmap_mode="r", allow_pickle=False) for path in directory.glob("*.npy")}


def diagnostic_row(scope, view, light, selection, arrays, evaluator, keys, mask_key, cls):
    selected = {key: arrays[key][selection] for key in keys}
    mask = selected[mask_key]
    present = mask.any(axis=tuple(range(1, mask.ndim)))
    label_key = "view_instance_anomalys" if scope == "view_fused" else "image_anomalys"
    metadata_labels = selected[label_key]
    # The pinned evaluator uses mask presence for image and view AUROC.
    row = {"scope": scope, "view": view, "illumination": light, "images": len(present),
           "positive": int(np.count_nonzero(present)), "metadata_positive": int(np.count_nonzero(metadata_labels)),
           "mask_label_disagreements": int(np.count_nonzero(present != metadata_labels)),
           "AUROC_pct": "", "AUPRO_pct": "", "status": "OK"}
    if not len(present):
        row["status"] = "undefined_no_rows"
        return row
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
    return {"rows": len(rows), "nonfinite_or_undefined": sum(row["status"] != "OK" for row in rows),
            "rows_with_label_disagreements": sum(row["mask_label_disagreements"] > 0 for row in rows)}
