"""Run one official baseline update plus disjoint-normal Car inference on CUDA."""

import argparse
import collections
from collections.abc import Iterable
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from time import perf_counter

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader


REVISION = "a606aa0e1614f23dad70da26430dd66095192141"
WEIGHT_NAME = "dinov2_vitb14_reg4_pretrain.pth"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fixture_ids(root, group, expected_count):
    path = root / group
    meta = json.loads((path / "jsons/meta_unsupervised.json").read_text(encoding="utf-8"))
    rows = meta["train"]["Car"]
    require(meta["test"] == {"Car": []} and len(rows) == expected_count, "wrong smoke fixture size or split")
    ids = set()
    for row in rows:
        relative = Path(row["img_path"])
        require(relative.parts[:2] == ("Car", "Good") and len(relative.parts) == 4,
                "fixture must have Car/Good image paths")
        require(row["image_anomaly"] == row["object_anomaly"] == 0 and relative.parts[2] == row["object_name"],
                "fixture must contain only matching normal object IDs")
        with Image.open(path / relative) as image:
            image.verify()
        ids.add(row["object_name"])
    require(len(ids) == 1, "expected one normal object per fixture group")
    return ids


def check_upstream(path):
    revision = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    require(revision == REVISION, f"wrong upstream revision: {revision}")
    subprocess.run(["git", "-C", str(path), "diff", "--quiet", "HEAD", "--"], check=True)


def run(args):
    started = perf_counter()
    check_upstream(args.upstream)
    require(torch.cuda.is_available(), "CUDA is required for this smoke")
    require(args.weight.name == WEIGHT_NAME, "upstream cache requires the original checkpoint basename")
    with args.weight.open("rb") as checkpoint:
        digest = hashlib.file_digest(checkpoint, "sha256").hexdigest()
    require(digest == args.weight_sha256.lower(), "weight SHA-256 mismatch")
    require(fixture_ids(args.data, "fit", 2).isdisjoint(fixture_ids(args.data, "probe", 1)),
            "fit and probe contain the same object")

    os.chdir(args.upstream)
    sys.path.insert(0, str(args.upstream))
    # util/net.py imports the legacy location, removed in Python 3.10.
    if not hasattr(collections, "Iterable"):
        collections.Iterable = Iterable
    from configs.benchmark.dinomaly.dinomaly_100e import cfg as dinomaly_cfg
    from configs.benchmark.inpformer.inpformer_100e import cfg as inpformer_cfg
    import data.ad_dataset as dataset_module
    from data.utils import get_transforms
    from loss import get_loss_terms
    from model import get_model
    from model.dinomaly_utils import vit_encoder as dinomaly_encoder
    from model.inp_former_utils import vit_encoder as inpformer_encoder
    from optim import get_optim
    from trainer.dinomaly_trainer import DinomalyTrainer
    from trainer.inpformer_trainer import INPFormerTrainer
    from util.net import get_autocast, get_loss_scaler
    import adeval
    import torchvision

    dinomaly_encoder._WEIGHTS_DIR = inpformer_encoder._WEIGHTS_DIR = str(args.weight.parent)
    if args.model == "dinomaly":
        cfg, trainer_class = dinomaly_cfg(), DinomalyTrainer
    else:
        cfg, trainer_class = inpformer_cfg(), INPFormerTrainer
    cfg.train_data.cls_names = ["Car"]
    inp_cfg = inpformer_cfg()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats()
    timings = {}

    def timed(name, operation):
        torch.cuda.synchronize()
        start = perf_counter()
        value = operation()
        torch.cuda.synchronize()
        timings[name] = perf_counter() - start
        return value

    def load_batch(group, size):
        dataset_module.DATA_ROOT = str(args.data / group)
        dataset_module.DATA_SUBDIR = {"m2ad": "."}
        spec = cfg.train_data
        images = get_transforms(cfg, train=True, cfg_transforms=spec.train_transforms)
        masks = get_transforms(cfg, train=False, cfg_transforms=spec.target_transforms)
        dataset = dataset_module.M2AD(spec, train=True, transform=images, target_transform=masks, setup="M2AD-Synergy")
        return next(iter(DataLoader(dataset, batch_size=size, shuffle=False, num_workers=0)))

    fit = timed("read_fit_batch", lambda: load_batch("fit", 2))
    probe = timed("read_probe_batch", lambda: load_batch("probe", 1))
    require(set(fit["object_name"]).isdisjoint(probe["object_name"]), "loaded objects overlap")
    runner = trainer_class.__new__(trainer_class)
    runner.cfg = cfg
    runner.net = timed("initialize_model_cuda", lambda: get_model(cfg.model).cuda())
    encoder = runner.net.model.encoder
    require(encoder.patch_size == 14 and encoder.num_register_tokens == 4, "unexpected DINOv2 variant")
    # Use the upstream optimizer, loss, autocast and update primitive. The epoch
    # runner is intentionally bypassed so it cannot load the official test split.
    runner.optim = get_optim(cfg.optim.kwargs, runner.net, lr=cfg.optim.lr)
    runner.loss_terms = get_loss_terms(cfg.loss.loss_terms, device="cuda:0")
    runner.amp_autocast = get_autocast(cfg.trainer.scaler)
    runner.loss_scaler = get_loss_scaler(cfg.trainer.scaler)
    runner.net.train()
    prototypes = None
    if args.model == "dinomaly":
        trainable = next(p for p in runner.net.model.bottleneck.parameters() if p.requires_grad)
    else:
        trainable = runner.net.INP[0]
        prototypes = list(trainable.shape)
        require(prototypes == [6, 768], "unexpected INP prototype dimensions")
    before = trainable.detach().clone()
    timed("transfer_fit_batch", lambda: runner.set_input(fit))

    def fit_forward():
        with runner.amp_autocast():
            runner.forward(train=True)
            return runner.compute_loss(train=True)[0]

    loss = timed("forward_and_loss", fit_forward)
    require(bool(torch.isfinite(loss)), "non-finite fit loss")
    timed("backward_and_optimizer_step", lambda: runner.backward_term(loss, runner.optim))
    require(not torch.equal(before, trainable), "optimizer step did not update the sampled trainable parameter")
    require(all(p.grad is None for p in encoder.parameters()), "encoder received gradients")

    def infer():
        runner.net.eval()
        runner.set_input(probe)
        with torch.inference_mode():
            runner.forward(train=False)
            return runner.compute_anomaly_scores()

    maps, scores = timed("probe_inference_and_scoring", infer)
    require(np.isfinite(maps).all() and np.isfinite(scores).all(), "non-finite inference output")
    return {
        "model": args.model, "prototype_shape": prototypes,
        "revision": REVISION, "weight_sha256": digest, "weight_bytes": args.weight.stat().st_size,
        "gpu": torch.cuda.get_device_name(0), "capability": torch.cuda.get_device_capability(0),
        "torch": torch.__version__, "torchvision": torchvision.__version__, "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "python": sys.version.split()[0],
        "encoder": {"patch_size": encoder.patch_size, "register_tokens": encoder.num_register_tokens},
        "adeval_cuda_import": callable(adeval.EvalAccumulatorCuda),
        "input_shape": list(fit["img"].shape), "probe_shape": list(probe["img"].shape),
        "fit_steps": 1, "seed": cfg.seed, "lr": cfg.optim.lr, "optim": cfg.optim.kwargs,
        "loss_config": cfg.loss.loss_terms, "precision_scaler": cfg.trainer.scaler,
        "fit_loss": float(loss.detach()), "parameter_changed": True, "encoder_gradients": False,
        "anomaly_map_shape": list(maps.shape), "score_shape": list(scores.shape), "finite_inference": True,
        "seconds": timings, "wall_seconds": perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "inp_former": {"trainer": INPFormerTrainer.__name__, "input_size": inp_cfg.size,
                       "model": inp_cfg.model.name, "config": inp_cfg.model.kwargs},
        "official_test_used": False, "result": "PASS",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=lambda path: Path(path).resolve())
    parser.add_argument("--data", required=True, type=lambda path: Path(path).resolve())
    parser.add_argument("--weight", required=True, type=lambda path: Path(path).resolve())
    parser.add_argument("--weight-sha256", required=True)
    parser.add_argument("--model", choices=("dinomaly", "inpformer"), default="dinomaly")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
