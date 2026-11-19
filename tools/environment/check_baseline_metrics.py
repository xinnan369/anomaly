"""Exercise the final-output fusion checks and diagnostics on small CUDA fixtures."""

import argparse
import json
from pathlib import Path
import runpy
import sys
from tempfile import TemporaryDirectory
import warnings

import numpy as np

from baseline_metrics import check_fusion, save_arrays, save_diagnostics
from smoke_dinomaly import check_upstream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=lambda path: Path(path).resolve(), required=True)
    args = parser.parse_args()
    check_upstream(args.upstream)
    sys.path.insert(0, str(args.upstream))
    fuse = runpy.run_path(str(args.upstream / "util" / "m2ad_utils.py"))["m2ad_fuse"]

    labels = np.repeat([0, 1], 120)
    masks = np.zeros((240, 1, 4, 4), dtype=np.uint8)
    maps = np.repeat(np.linspace(0.01, 0.2, 16, dtype=np.float32).reshape(1, 4, 4), 240, axis=0)
    masks[120:, 0, 1, 1] = 1
    maps[120:, 1, 1] = 0.9
    raw = {
        "obj_names": np.repeat(["normal", "abnormal"], 120),
        "cls_names": np.repeat("Car", 240),
        "view_list": np.tile(np.repeat([f"{angle:03d}" for angle in range(0, 360, 30)], 10), 2),
        "illumination_list": np.tile([f"{light:02d}" for light in range(1, 11)], 24),
        "imgs_masks": masks,
        "image_anomalys": labels,
        "object_anomalys": labels,
        "anomaly_scores": maps.max(axis=(1, 2)),
        "anomaly_maps": maps,
    }
    fused = fuse(**raw, fuse="add")
    check_fusion(raw, fused)
    bad = {**fused, "object_instance_anomaly_scores": fused["object_instance_anomaly_scores"] + 1}
    try:
        check_fusion(raw, bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted fused score was accepted")

    with TemporaryDirectory(prefix="vicad-test-fusion-") as temp:
        path = Path(temp)
        save_arrays(path / "raw", raw)
        save_arrays(path / "fused", fused)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered in divide",
                                    category=RuntimeWarning, module=r"adeval\.mem_effic")
            diagnostics = save_diagnostics(path, "Car", use_adeval=True)
    if diagnostics != {"rows": 142, "nonfinite_or_undefined": 0}:
        raise AssertionError(diagnostics)
    print(json.dumps({
        "fixture": "synthetic_only", "raw_rows": len(labels),
        "fused_objects": len(fused["object_instance_names"]), "fused_views": len(fused["view_instance_names"]),
        "fusion_crosscheck": "PASS", "corrupted_score_rejected": True, **diagnostics,
    }, indent=2))


if __name__ == "__main__":
    main()
