"""Run a gated Car-first VICAD-Syn workload on the fixed M2AD release."""

import argparse
from datetime import datetime, timezone
import fcntl
from pathlib import Path
import platform
from time import perf_counter
import traceback

import torch

from vicad_syn.data.cache import Images, encode, feature_tensor
from vicad_syn.data.manifest import load_groups
from vicad_syn.evaluation.synergy import evaluate
from vicad_syn.experiment import check, pinned_upstream, protocol, read_json, write_json
from vicad_syn.models.backbone import FrozenBackbone
from vicad_syn.training.train import build_predictor, fit, frozen_model, prediction_loss


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def smoke(root, output, frozen, encoder):
    config = frozen["config"]
    group = load_groups(root, "Car", "train", config)[0]
    inputs = Images(root / "dataset", group.paths)
    images = torch.stack([inputs[index] for index in range(len(inputs))])
    records = {}
    for device in (torch.device("cpu"), torch.device("cuda")):
        encoder.to(device).train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        now = perf_counter()
        features = encoder(images.to(device)).unsqueeze(0)
        check(features.shape == (1, 10, 256, 768) and not features.requires_grad, "invalid frozen patch tensor")
        torch.manual_seed(config["seed"])
        predictor = build_predictor(config, True, device)
        prediction_loss(predictor, features).backward()
        check(not encoder.training and not encoder.encoder.training and
              all(not parameter.requires_grad and parameter.grad is None for parameter in encoder.parameters()),
              "encoder was not frozen")
        check(all(p.grad is not None and bool(p.grad.isfinite().all()) for p in predictor.parameters()),
              "predictor smoke gradient failed")
        if device.type == "cuda":
            torch.cuda.synchronize()
        records[device.type] = {"seconds": perf_counter() - now, "patch_shape": list(features.shape),
                                "encoder_frozen": True, "classification_register_tokens_removed": 5,
                                "predictor_gradients_finite": True,
                                "allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None}
    artifact = {"status": "PASS", "frozen": frozen, "device": torch.cuda.get_device_name(),
                "torch": str(torch.__version__), "python": platform.python_version(),
                "platform": platform.platform(), "records": records, "utc": timestamp()}
    write_json(output / "smoke.json", artifact)
    print(f"CPU and {artifact['device']} smoke passed", flush=True)


def run_category(root, output, frozen, encoder, category):
    config = frozen["config"]
    device = torch.device("cuda")
    folder = output / category
    folder.mkdir(exist_ok=True)
    dataset = root / "dataset"
    train_groups = load_groups(root, category, "train", config)
    train_cache = encode(dataset, train_groups, folder / "train-cache", encoder, config, device)
    features = feature_tensor(folder / "train-cache", device)
    dev = fit(folder / "development", features, train_groups, train_cache, frozen, True, "development")
    freeze = output / "frozen-protocol.json"
    if not freeze.exists():
        check(category == "Car", "Car normal-only development must precede protocol freeze")
        write_json(freeze, {"frozen": frozen, "sealed_utc": timestamp(), "Car_development_sha256": dev["sha256"],
                            "selection": "pre-fixed final epoch; development does not select hyperparameters"})
    check(read_json(freeze)["frozen"] == frozen, "formal protocol changed after freezing")
    modes = config["settings"][category]
    fits = [("conditioned", True, [mode for mode in modes if mode != "A0"])]
    if "A0" in modes:
        fits += [("unconditioned", False, ["A0"])]
    for name, conditioned, _ in fits:
        fit(folder / name, features, train_groups, train_cache, frozen, conditioned, "formal")
    del features
    torch.cuda.empty_cache()
    loaded = [(name, *frozen_model(folder / name, frozen, train_cache, conditioned, device), settings)
              for name, conditioned, settings in fits]
    test_groups = load_groups(root, category, "test", config)
    test_cache = encode(dataset, test_groups, folder / "test-cache", encoder, config, device)
    test_features = feature_tensor(folder / "test-cache", device)
    receipts = {}
    for name, model, train, settings in loaded:
        receipts[name] = evaluate(folder / name / "evaluation", category, test_groups, dataset,
                                  model, test_features, config, train, test_cache, settings)
    check(set().union(*(run["metrics"].keys() for run in receipts.values())) == set(modes),
          "configured evaluations incomplete")
    result = {"status": "PASS", "config_sha256": frozen["config_sha256"], "category": category,
              "development": {"train_objects": len(dev["binding"]["training_objects"]),
                              "validation_objects": len(dev["binding"]["validation_objects"])},
              "cache": {"train": train_cache, "test": test_cache},
              "settings": {mode: name for name, receipt in receipts.items() for mode in receipt["metrics"]},
              "utc": timestamp()}
    write_json(folder / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("smoke", "run"))
    for name in ("inputs", "output"):
        parser.add_argument(f"--{name}", type=lambda p: Path(p).resolve(), required=True)
    args = parser.parse_args()
    root, output = args.inputs, args.output
    check(not output.is_relative_to(Path(__file__).resolve().parents[1]), "keep experiment outputs outside Git")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    check(torch.cuda.is_available() and torch.cuda.device_count() == 1, "expose exactly one CUDA device")
    frozen = protocol()
    config = frozen["config"]
    if args.phase == "run":
        gate = read_json(output / "smoke.json")
        check(gate and gate["status"] == "PASS" and gate["frozen"] == frozen, "run current CPU and CUDA smoke first")
    with (output / "pipeline.lock").open("a") as lock, pinned_upstream(root / "inputs/upstream", config["upstream_revision"]):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        encoder = FrozenBackbone(root / "inputs/weights" / config["backbone"], config)
        if args.phase == "smoke":
            smoke(root, output, frozen, encoder)
            return
        progress = {"status": "RUNNING", "categories": {}, "updated_utc": timestamp()}
        encoder.cuda()
        try:
            for category in config["settings"]:
                prior = read_json(output / category / "result.json")
                if prior and prior["status"] == "PASS":
                    check(prior["config_sha256"] == frozen["config_sha256"], "completed category config differs")
                    progress["categories"][category] = prior["status"]
                    continue
                progress["current"] = category
                write_json(output / "pipeline.json", progress)
                progress["categories"][category] = run_category(root, output, frozen, encoder, category)["status"]
            progress["status"] = "PASS"
        except Exception as error:
            progress.update(status="PARTIAL", error=f"{type(error).__name__}: {error}")
            traceback.print_exc()
            raise
        finally:
            progress["updated_utc"] = timestamp()
            write_json(output / "pipeline.json", progress)


if __name__ == "__main__":
    main()
