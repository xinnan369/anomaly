"""Train a pinned official baseline, or evaluate its frozen epoch-100 checkpoint."""

import argparse
import collections
from collections.abc import Iterable
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
from time import perf_counter
import warnings

import numpy as np
import torch

from baseline_metrics import check_fusion, save_arrays, save_diagnostics
from prepare_release import ARCHIVE_SHA256, RELEASE_REVISION
from smoke_dinomaly import REVISION, WEIGHT_NAME, check_upstream, require


WEIGHT_SHA = "73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71"
CLASSES = ("Car", "Cube", "Doll", "Ring")
CONFIGS = {
    "dinomaly": "configs/benchmark/dinomaly/dinomaly_100e.py",
    "inpformer": "configs/benchmark/inpformer/inpformer_100e.py",
}
FINAL_EPOCH = 100
FINAL_ITERATION = 30_000


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def write_json(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def read_record(path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate_checkpoint(checkpoint, model, category, upstream):
    """Accept only the recorded final checkpoint produced by the matching upstream trainer."""
    require(checkpoint.parent.parent.name == "checkpoints"
            and checkpoint.name == f"{[category]}_ckpt.pth", "unexpected official checkpoint path")
    run_root = checkpoint.parents[2]
    source = read_record(run_root / "config.json")
    status = read_record(run_root / "status.json")
    require(source["class"] == category and source["cfg_path"] == CONFIGS[model]
            and source["cfg_sha256"] == digest(upstream / CONFIGS[model])
            and source["upstream_revision"] == REVISION and source["release_revision"] == RELEASE_REVISION
            and source["weight_sha256"] == WEIGHT_SHA and source["mode"] == "train"
            and source["protocol"] == "M2AD-Synergy" and source["seed"] == 42
            and source["epoch"] == FINAL_EPOCH and source["batch_train"] == 12
            and source["test_during_training"] is False, "checkpoint provenance does not match this run")
    completed = status.get("final_checkpoint", {})
    checksum = digest(checkpoint)
    require(status["status"] == "PASS" and status["mode"] == "train" and status["model"] == model
            and status["class"] == category and status["epochs_completed"] == FINAL_EPOCH
            and status["optimization_steps"] == FINAL_ITERATION
            and completed.get("path") == checkpoint.relative_to(run_root).as_posix()
            and completed.get("bytes") == checkpoint.stat().st_size
            and completed.get("sha256") == checksum, "checkpoint does not match the recorded final training artifact")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    require(state["epoch"] == FINAL_EPOCH and state["iter"] == FINAL_ITERATION,
            "evaluation requires the final checkpoint at epoch 100 and iteration 30000")
    return checksum


def sample_gpu(stop, output):
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("utc", "device_mib_used", "gpu_util_pct"))
        while not stop.is_set():
            try:
                reading = subprocess.run(
                    ["nvidia-smi", "--id=0", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10,
                )
            except (OSError, subprocess.TimeoutExpired) as error:
                print(f"GPU sampling stopped: {type(error).__name__}", file=sys.stderr)
                return
            if reading.returncode == 0:
                writer.writerow((now(), *[x.strip() for x in reading.stdout.strip().split(",")]))
                stream.flush()
            stop.wait(2)


def run(args, result):
    check_upstream(args.upstream)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "expose exactly one CUDA device")
    require(args.weight.name == WEIGHT_NAME and digest(args.weight) == WEIGHT_SHA, "backbone SHA-256 mismatch")
    if not args.preflight:
        marker_path = args.release_root / "data_audit" / f"{args.category}.json"
        require(marker_path.is_file(), f"{args.category}: no complete archive audit; run prepare_release.py first")
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        require(marker["release_revision"] == RELEASE_REVISION and marker["category"] == args.category
                and marker["archive_sha256"] == ARCHIVE_SHA256[args.category]
                and marker["images"] == 12_000 and marker["split_crc_size_overlap"] == 0,
                "verified, complete class is required")
    if args.checkpoint:
        result["evaluation_checkpoint_sha256"] = validate_checkpoint(args.checkpoint, args.model, args.category,
                                                                     args.upstream)
        result["epochs_completed"] = FINAL_EPOCH
    os.chdir(args.upstream)
    sys.path.insert(0, str(args.upstream))
    if not hasattr(collections, "Iterable"):
        collections.Iterable = Iterable
    import data.ad_dataset as dataset
    from model.dinomaly_utils import vit_encoder as dinomaly_encoder
    from model.inp_former_utils import vit_encoder as inpformer_encoder
    import run_dataset as entry
    from trainer import _base_trainer as upstream
    warnings.filterwarnings("default")

    dinomaly_encoder._WEIGHTS_DIR = inpformer_encoder._WEIGHTS_DIR = str(args.weight.parent)
    dataset.DATA_ROOT = str(args.release_root / "dataset")
    dataset.DATA_SUBDIR = {"m2ad": "."}
    entry.CLASS_NAMES = {"m2ad": [args.category]}
    original_setup, original_trainer = entry.setup_cfg, entry.get_trainer
    original_test, original_fuse = upstream.BaseTrainer.test, upstream.m2ad_fuse
    epochs = []
    training_started = None

    def setup(cfg, protocol, cls):
        cfg = original_setup(cfg, protocol, cls)
        require(cfg.trainer.epoch_full == FINAL_EPOCH and cfg.batch_train == cfg.batch_test_per == 12
                and cfg.seed == 42 and cfg.size == 224 and cfg.use_adeval,
                "unexpected pinned upstream configuration")
        cfg.trainer.checkpoint = str(args.output / "checkpoints")
        cfg.vis = False  # Raw predictions are retained; suppress optional bulk image rendering.
        if args.checkpoint:
            cfg.model.kwargs["checkpoint_path"] = str(args.checkpoint)
        write_json(args.output / "config.json", {
            "upstream_revision": REVISION, "cfg_path": CONFIGS[args.model],
            "cfg_sha256": digest(args.upstream / CONFIGS[args.model]),
            "release_revision": RELEASE_REVISION, "protocol": protocol, "class": cls,
            "mode": "preflight" if args.preflight else cfg.mode,
            "seed": cfg.seed, "epoch": cfg.trainer.epoch_full, "size": cfg.size,
            "batch_train": cfg.batch_train, "batch_test": cfg.batch_test_per,
            "optimizer": {"lr": cfg.optim.lr, **cfg.optim.kwargs},
            "scheduler": cfg.trainer.scheduler_kwargs, "loss": cfg.loss.loss_terms,
            "clip_grad": cfg.loss.clip_grad, "default_test_interval": cfg.trainer.test_per_epoch,
            "formal_test_epoch": FINAL_EPOCH, "test_during_training": False, "bulk_visualization": cfg.vis,
            "model_kwargs": {key: value for key, value in cfg.model.kwargs.items() if key != "checkpoint_path"},
            "weight_sha256": WEIGHT_SHA, "metrics": cfg.metrics, "use_adeval": cfg.use_adeval,
        })
        return cfg

    def build(cfg):
        nonlocal training_started
        trainer = original_trainer(cfg)
        require(len(trainer.train_set) == 3600 and len(trainer.test_set) == 8400, "official loader count mismatch")
        require(len(trainer.train_loader) == 300 and len(trainer.test_loader) == 700,
                "official batch count mismatch")
        result["parameters"] = sum(param.numel() for param in trainer.net.parameters())
        result["trainable_parameters"] = sum(param.numel() for param in trainer.net.parameters() if param.requires_grad)
        result["optimizer_parameters"] = sum(param.numel() for group in trainer.optim.param_groups
                                             for param in group["params"])
        if args.preflight:
            result.update(train_rows=len(trainer.train_set), test_rows=len(trainer.test_set),
                          train_batches=len(trainer.train_loader), test_batches=len(trainer.test_loader))
            trainer.run = trainer.writer.close  # No data iteration, training or evaluation in preflight.
            return trainer
        if not args.checkpoint:
            result["train_started_utc"] = now()
            training_started = perf_counter()
        original_compute = trainer.compute_loss
        original_save = trainer.save_checkpoint
        original_eval = trainer.evaluator.run

        def loss(*pos, **kw):
            value, logs = original_compute(*pos, **kw)
            require(bool(torch.isfinite(value)), f"NaN/Inf loss at iteration {trainer.iter}")
            if trainer.iter % 20 == 0:
                result["last_observed_iteration"] = trainer.iter
                write_json(args.output / "status.json", result)
            return value, logs

        def save():
            original_save()
            if not epochs or epochs[-1]["epoch"] < trainer.epoch:
                epochs.append({"epoch": trainer.epoch, "iteration": trainer.iter, "utc": now()})
                write_json(args.output / "epochs.json", epochs)
                result.update(epochs_completed=trainer.epoch, optimization_steps=trainer.iter)
                write_json(args.output / "status.json", result)

        def evaluate(*pos, **kw):
            metrics = original_eval(*pos, **kw)
            numbers = {name: float(value) * 100 for name, value in metrics.items()}
            invalid = [name for name, value in numbers.items() if not math.isfinite(value)]
            write_json(args.output / "synergy.json", {
                "epoch": FINAL_EPOCH, "class": args.category, "model": args.model,
                "checkpoint_sha256": result["evaluation_checkpoint_sha256"],
                "metrics_pct": {key: value if math.isfinite(value) else None for key, value in numbers.items()},
                "nonfinite": invalid,
            })
            require(not invalid, f"non-finite official metrics: {invalid}")
            return metrics

        trainer.compute_loss = loss
        trainer.save_checkpoint = save
        trainer.evaluator.run = evaluate
        return trainer

    def fuse(**results):
        option = results.pop("fuse")
        require(option == "add" and len(results["cls_names"]) == 8400, "unexpected official evaluation inputs")
        columns = ("obj_names", "object_anomalys", "view_list", "illumination_list", "image_anomalys")
        rows = list(zip(*(results[name] for name in columns)))
        manifest = args.release_root / "dataset" / "jsons" / "meta_unsupervised.json"
        records = json.loads(manifest.read_text(encoding="utf-8"))["test"][args.category]
        fields = ("object_name", "object_anomaly", "view", "illumination", "image_anomaly")
        expected = {tuple(row[name] for name in fields) for row in records}
        require(len(set(rows)) == len(expected) == 8400 and set(rows) == expected
                and set(results["cls_names"]) == {args.category}, "evaluation rows differ from the audited manifest")
        require(set(results["view_list"]) == {f"{v:03d}" for v in range(0, 360, 30)}
                and set(results["illumination_list"]) == {f"{l:02d}" for l in range(1, 11)},
                "official configuration coverage mismatch")
        masks = results["imgs_masks"]
        require(results["anomaly_scores"].shape == (8400,) and results["anomaly_maps"].shape == (8400, 224, 224),
                "unexpected official score or heatmap shape")
        require(masks.shape == (8400, 1, 224, 224) and np.issubdtype(masks.dtype, np.integer)
                and masks.min() == 0 and masks.max() == 1,
                "unexpected official mask shape or labels")
        result["inference_finished_utc"] = now()
        result["inference_through_concatenation_seconds"] = perf_counter() - infer_started
        evidence = {"raw": save_arrays(args.output / "raw", {**results, "imgs_masks": masks.astype(np.uint8)})}
        fused = original_fuse(**results, fuse=option)
        require(len(fused["object_instance_names"]) == 70 and len(fused["view_instance_names"]) == 840,
                "unexpected fused object or view count")
        require(fused["object_instance_anomalys"].shape == fused["object_instance_anomaly_scores"].shape == (70,)
                and fused["view_instance_anomalys"].shape == fused["view_instance_anomaly_scores"].shape == (840,)
                and fused["view_instance_masks"].shape == fused["view_instance_anomaly_maps"].shape == (840, 224, 224),
                "unexpected fused label, score, mask or heatmap shape")
        result["fusion_crosscheck"] = check_fusion(results, fused)
        evidence["fused"] = save_arrays(args.output / "fused", fused)
        write_json(args.output / "predictions.json", evidence)
        return fused

    infer_started = None

    def final_test(trainer, vis=True):
        nonlocal infer_started
        if not args.checkpoint:
            if trainer.epoch == FINAL_EPOCH:
                result["train_finished_utc"] = now()
                result["train_seconds"] = perf_counter() - training_started
                result["epochs_completed"] = trainer.epoch
                result["optimization_steps"] = trainer.iter
            return trainer.test_ghost()
        infer_started = perf_counter()
        original_test(trainer, vis=False)
        torch.cuda.synchronize()
        result["final_evaluation_seconds"] = perf_counter() - infer_started

    entry.setup_cfg, entry.get_trainer = setup, build
    upstream.BaseTrainer.test, upstream.m2ad_fuse = final_test, fuse
    sys.argv = ["run_dataset.py", "--cfg_path", CONFIGS[args.model],
                "-m", "test" if args.checkpoint else "train", "-s", "M2AD-Synergy"]
    torch.cuda.reset_peak_memory_stats()
    entry.main()
    result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    if args.checkpoint:
        result["diagnostics"] = save_diagnostics(args.output, args.category, use_adeval=True)
    elif not args.preflight:
        require(epochs and epochs[-1]["epoch"] == FINAL_EPOCH and epochs[-1]["iteration"] == FINAL_ITERATION,
                "training did not reach the frozen checkpoint")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("upstream", "release-root", "weight", "output"):
        parser.add_argument(f"--{option}", type=lambda path: Path(path).resolve(), required=True)
    parser.add_argument("--model", choices=CONFIGS, required=True)
    parser.add_argument("--category", choices=CLASSES, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--checkpoint", type=lambda path: Path(path).resolve())
    mode.add_argument("--preflight", action="store_true", help="build the official trainer from the manifest only")
    args = parser.parse_args()
    require(not args.output.is_relative_to(Path(__file__).resolve().parents[2]), "write artifacts outside the repository")
    args.output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    result = {"model": args.model, "class": args.category, "status": "RUNNING", "started_utc": now(),
              "mode": "preflight" if args.preflight else "test" if args.checkpoint else "train",
              "epochs_completed": 0, "oom": False, "nonfinite": False, "argv": sys.argv[1:]}
    write_json(args.output / "status.json", result)
    stop = threading.Event()
    monitor = threading.Thread(target=sample_gpu, args=(stop, args.output / "gpu.csv"), daemon=True)
    monitor.start()
    try:
        run(args, result)
        result["status"] = "PASS"
    except BaseException as error:
        result.update(status="FAILED", error=f"{type(error).__name__}: {error}",
                      oom=isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower(),
                      nonfinite=any(token in str(error).lower() for token in ("non-finite", "nan/inf")))
        raise
    finally:
        peaks = {}
        if torch.cuda.is_initialized():
            peaks = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                     "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
        stop.set()
        monitor.join(timeout=12)
        for name, value in peaks.items():
            result[name] = max(result.get(name, 0), value)
        weights = list((args.output / "checkpoints").glob("**/*_net_100.pth"))
        if weights:
            result["weights"] = [{"path": path.relative_to(args.output).as_posix(), "bytes": path.stat().st_size,
                                  "sha256": digest(path)} for path in weights]
            checkpoint = next(weights[0].parent.glob("*_ckpt.pth"), None)
            if checkpoint:
                result["final_checkpoint"] = {"path": checkpoint.relative_to(args.output).as_posix(),
                                              "bytes": checkpoint.stat().st_size, "sha256": digest(checkpoint)}
        result.update(finished_utc=now(), wall_seconds=perf_counter() - started)
        write_json(args.output / "status.json", result)


if __name__ == "__main__":
    main()
