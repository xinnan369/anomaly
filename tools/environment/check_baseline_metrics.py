"""Verify pinned manifest identity, official metrics and collision-safe fusion on CUDA."""

import argparse
from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import runpy
import sys
from tempfile import TemporaryDirectory
import warnings

import numpy as np

from baseline_metrics import check_fusion, save_arrays, save_diagnostics
from prepare_release import RELEASE_REVISION, check_manifest, load_manifest
from smoke_dinomaly import check_upstream


def manifest_identities(cache):
    data = load_manifest(cache)
    audit = check_manifest(data)
    counts = {}
    for cls, records in data["test"].items():
        good, ng = ({r["object_name"] for r in records if r["object_anomaly"] == status}
                    for status in (0, 1))
        names = good & ng
        counts[cls] = {"good_ids": len(good), "ng_ids": len(ng), "shared_bare_ids": sorted(names),
                       "official_objects": len(good) + len(ng),
                       "official_views": 12 * (len(good) + len(ng)), "rows": len(records)}
        checked = audit["classes"][cls]
        assert len(good) == checked["test_good"]["objects"]
        assert len(ng) == checked["test_ng"]["objects"]
    return {"release_revision": RELEASE_REVISION, "manifest_sha256": audit["manifest_sha256"], "classes": counts}


def fixture():
    object_labels = np.repeat([0, 1], 120)
    views = np.tile(np.repeat([f"{v:03d}" for v in range(0, 360, 30)], 10), 2)
    lights = np.tile([f"{light:02d}" for light in range(1, 11)], 24)
    manifest_positive = (object_labels == 1) & (views != "300") & (lights == "01")
    mask_positive = manifest_positive & (views != "330")
    mask_positive[121] = True  # a mask-positive image whose metadata label is zero
    masks = np.zeros((240, 1, 4, 4), dtype=np.uint8)
    maps = np.repeat(np.linspace(0.01, 0.017, 16, dtype=np.float32).reshape(1, 4, 4), 240, axis=0)
    masks[mask_positive, 0, 1, 1] = 1
    maps[mask_positive, 1, 1] = 0.9
    return dict(obj_names=np.repeat("000", 240), cls_names=np.repeat("Car", 240),
                view_list=views, illumination_list=lights, imgs_masks=masks,
                image_anomalys=manifest_positive.astype(int), object_anomalys=object_labels,
                anomaly_scores=maps.max(axis=(1, 2)), anomaly_maps=maps)


def verify_regression(fuse):
    from util.metric import Evaluator

    raw = fixture()
    notices = io.StringIO()
    with redirect_stdout(notices):
        fused = fuse(**raw, fuse="add")
    assert "Inconsistent object anomaly values" in notices.getvalue()
    assert check_fusion(raw, fused) == {
        "objects": 2, "views": 24, "repeated_object_names": 1, "repeated_view_names": 12,
    }
    reordered = {key: array[::-1] for key, array in fused.items()}
    check_fusion(raw, reordered)
    for key in ("object_instance_anomaly_scores", "view_instance_anomaly_scores"):
        bad = {**reordered, key: reordered[key].copy()}
        bad[key][0] += 1
        try:
            check_fusion(raw, bad)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"corrupted {key} was accepted")

    metric_names = ("AUROC_obj_sp", "AUROC_view_sp", "AUPRO_view")
    with TemporaryDirectory(prefix="vicad-test-fusion-") as temp, warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in divide",
                                category=RuntimeWarning, module=r"adeval\.mem_effic")
        path = Path(temp)
        save_arrays(path / "raw", raw)
        save_arrays(path / "fused", reordered)
        diagnostics = save_diagnostics(path, "Car", use_adeval=True)
        with (path / "diagnostics.csv").open(newline="", encoding="utf-8") as stream:
            rows = {(row["scope"], row["view"], row["illumination"]): row for row in csv.DictReader(stream)}
        no_mask = rows[("raw_configuration", "330", "01")]
        assert (no_mask["positive"], no_mask["metadata_positive"], no_mask["mask_label_disagreements"]) == ("0", "1", "1")
        assert not no_mask["AUROC_pct"] and not no_mask["AUPRO_pct"] and no_mask["status"] == "undefined_single_label"
        no_label = rows[("raw_configuration", "000", "02")]
        assert (no_label["positive"], no_label["metadata_positive"], no_label["mask_label_disagreements"]) == ("1", "0", "1")
        assert no_label["status"] == "OK"
        hidden_view = rows[("view_fused", "330", "")]
        assert hidden_view["metadata_positive"] == "1" and hidden_view["status"] == "undefined_single_label"
        official = Evaluator(metrics=list(metric_names), use_adeval=True).run(reordered, "Car", "M2AD-Synergy")
    assert diagnostics["rows"] == 142 and diagnostics["rows_with_label_disagreements"] == 5, diagnostics
    np.testing.assert_allclose([official[name] for name in metric_names], [1, 1, 1], rtol=0, atol=1e-6)
    return {"fixture": "synthetic_only", "raw_rows": len(raw["obj_names"]), **check_fusion(raw, reordered),
            "unordered_outputs_accepted": True, "corrupted_object_and_view_scores_rejected": True,
            "partly_visible_illumination_verified": True, "mask_metadata_disagreement_verified": True,
            "synthetic_official_metrics": {key: float(official[key]) for key in metric_names}, **diagnostics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=lambda path: Path(path).resolve(), required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    check_upstream(args.upstream)
    sys.path.insert(0, str(args.upstream))
    fuse = runpy.run_path(str(args.upstream / "util" / "m2ad_utils.py"))["m2ad_fuse"]
    audit = manifest_identities(args.cache)
    regression = verify_regression(fuse)
    print(json.dumps({"manifest": audit, "regression": regression}, indent=2))


if __name__ == "__main__":
    main()
