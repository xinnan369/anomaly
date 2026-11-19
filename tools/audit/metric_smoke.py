"""Test pinned upstream fusion and ADEval with synthetic arrays only."""

import argparse
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import platform
import sys

import cv2
import numpy as np
import torch

from sources import ADEVAL, CODE_REVISION, FUSION, require, verified_resource


def check_fusion(source_path):
    spec = importlib.util.spec_from_file_location("m2ad_fusion", source_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    maps = np.arange(24, dtype=np.float32).reshape(6, 1, 2, 2)
    masks = np.zeros_like(maps)
    masks[0, 0, 0, 0] = 1
    masks[1, 0, 1, 1] = 1
    notices = io.StringIO()
    with redirect_stdout(notices):
        fused = module.m2ad_fuse(
            obj_names=np.array(["a", "a", "a", "b", "b", "b"]),
            cls_names=np.array(["Car"] * 6),
            illumination_list=np.array(["01", "02", "01", "01", "02", "01"]),
            view_list=np.array(["000", "000", "030", "000", "000", "030"]),
            imgs_masks=masks,
            image_anomalys=np.array([0, 1, 1, 0, 0, 0]),
            object_anomalys=np.array([1, 1, 1, 0, 0, 0]),
            anomaly_scores=np.array([1., 3., 8., 2., 1., 2.]),
            anomaly_maps=maps,
            fuse="add",
        )
    objects = dict(zip(fused["object_instance_names"], fused["object_instance_anomaly_scores"]))
    views = dict(zip(fused["view_instance_names"], fused["view_instance_anomaly_scores"]))
    require(set(objects) == {"a", "b"}, "wrong object groups")
    np.testing.assert_allclose([objects["a"], objects["b"]], [4, 5 / 3])
    require(set(views) == {"a_000", "a_030", "b_000", "b_030"}, "wrong view groups")
    np.testing.assert_allclose([views[name] for name in sorted(views)], [2, 8, 1.5, 2])
    idx = np.flatnonzero(fused["view_instance_names"] == "a_000")[0]
    np.testing.assert_array_equal(fused["view_instance_masks"][idx], [[1, 0], [0, 1]])
    np.testing.assert_allclose(fused["view_instance_anomaly_maps"][idx], maps[:2].mean(axis=0))
    require(fused["view_instance_anomalys"][idx] == 1, "wrong fused image label")
    require("Inconsistent object anomaly values" in notices.getvalue(), "expected upstream mixed-label notice")
    return {
        "object_scores": {name: float(objects[name]) for name in sorted(objects)},
        "view_scores": {name: float(views[name]) for name in sorted(views)},
        "union_mask_assertion": "PASS", "mean_heatmap_assertion": "PASS", "mixed_label_notice": "PASS",
    }


def check_adeval(wheel_path, backend):
    sys.path.insert(0, str(wheel_path))
    import adeval

    require(str(adeval.__file__).startswith(str(wheel_path)), "ADEval did not load from the verified wheel")
    if backend == "cuda":
        require(torch.cuda.is_available(), "CUDA was requested but PyTorch cannot access a GPU")
        accumulator = adeval.EvalAccumulatorCuda
        device = torch.device("cuda:0")
    else:
        accumulator = adeval.EvalAccumulator

    masks = np.array([[[0, 0], [0, 0]], [[1, 0], [0, 0]], [[1, 0], [0, 0]]], dtype=np.uint8)
    perfect = np.array([[[.1, .2], [.3, .4]], [[.9, .05], [.15, .25]], [[.85, .06], [.16, .26]]],
                       dtype=np.float32)
    partial = perfect.copy()
    partial[2, 0, 0] = .01
    metrics = {}
    for case, predictions, expected in (
        ("perfect", perfect, (1.0, 1.0, 1.0)),
        ("one_of_two_regions", partial, (1.0, 0.5, 0.5)),
    ):
        acc = accumulator(0., 1., 0., 1., skip_pixel_aupro=False, nstrips=50)
        if backend == "cuda":
            acc.add_anomap_batch(torch.as_tensor(predictions, device=device), torch.as_tensor(masks, device=device))
        else:
            acc.add_anomap_batch(predictions, masks)
        for score, label in zip((.1, .9, .85), (0, 1, 1)):
            acc.add_image(score, label)
            acc.add_sample(score, label)
        values = acc.summary()
        metrics[case] = {key: float(values[key]) for key in ("i_auroc", "p_auroc", "p_aupro")}
        np.testing.assert_allclose(list(metrics[case].values()), expected, atol=1e-6, rtol=0)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True, help="cache directory outside the repository")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    fusion = check_fusion(verified_resource(args.cache, FUSION))
    cases = check_adeval(verified_resource(args.cache, ADEVAL).resolve(), args.device)
    print(json.dumps({
        "upstream_revision": CODE_REVISION, "fusion_sha256": FUSION.sha256,
        "adeval_version": "1.1.0", "adeval_wheel_sha256": ADEVAL.sha256, "nstrips": 50,
        "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "compute_capability": torch.cuda.get_device_capability(0) if args.device == "cuda" else None,
        "environment": {
            "system": platform.system(), "python": platform.python_version(), "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "numpy": np.__version__, "opencv": cv2.__version__,
        },
        "fusion": fusion, "adeval": cases, "result": "PASS",
    }, indent=2))


if __name__ == "__main__":
    main()
