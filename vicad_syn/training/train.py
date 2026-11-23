"""Fixed normal-object training, normal-only development and epoch-boundary recovery."""

from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from vicad_syn.data.manifest import split_ids
from vicad_syn.experiment import check, digest, read_json, write_json
from vicad_syn.models.predictor import Predictor
from vicad_syn.uncertainty.signals import squared_distance


def save_checkpoint(path, state):
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def build_predictor(config, conditioned, device):
    return Predictor(hidden=config["hidden"], heads=config["heads"], conditioned=conditioned).to(device)


def prediction_loss(model, features):
    patched = torch.nn.functional.normalize(features.float(), dim=-1)
    return squared_distance(model(patched), patched).mean()


def fit(folder, features, groups, cache, frozen, conditioned, purpose):
    config = frozen["config"]
    seed = config["seed"]
    torch.manual_seed(seed)
    model = build_predictor(config, conditioned, features.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    epochs = config["epochs"] if purpose == "formal" else config["dev_epochs"]
    train_ids, validation_ids = (split_ids(groups, seed, config["dev_objects"]) if purpose == "development"
                                 else ({group.identity for group in groups}, set()))
    train_indices = np.array([i for i, group in enumerate(groups) if group.identity in train_ids])
    val_indices = [i for i, group in enumerate(groups) if group.identity in validation_ids]
    check(train_ids.isdisjoint(validation_ids) and len(train_ids | validation_ids) == 30,
          "invalid physical-instance training partition")
    binding = {"frozen": frozen, "conditioned": conditioned, "purpose": purpose,
               "cache_sha256": cache["sha256"], "epochs": epochs, "training_objects": sorted(train_ids),
               "validation_objects": sorted(validation_ids), "training_groups": len(train_indices),
               "validation_groups": len(val_indices), "seed": seed}
    folder.mkdir(parents=True, exist_ok=True)
    latest = folder / "latest.pt"
    final = folder / "checkpoint.pt"
    receipt = folder / "train.json"
    done = read_json(receipt)
    if done and done["status"] == "PASS":
        check(done["binding"] == binding and done["sha256"] == digest(final), "sealed training differs")
        return done
    records = []
    initial_time = datetime.now(timezone.utc).isoformat()
    recovered_peaks = 0
    if latest.exists():
        saved = torch.load(latest, weights_only=True, map_location="cpu")
        check(saved["binding"] == binding and len(saved["epochs"]) <= epochs, "resume configuration differs")
        initial_time, records = saved["started_utc"], saved["epochs"]
        recovered_peaks = max((item["peak_cuda_bytes"] or 0 for item in records), default=0)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_cpu_rng"])
        if features.is_cuda:
            torch.cuda.set_rng_state(saved["torch_cuda_rng"], features.device)
    if features.is_cuda:
        torch.cuda.reset_peak_memory_stats(features.device)
    for epoch in range(len(records) + 1, epochs + 1):
        start = perf_counter()
        order = np.random.default_rng(seed + epoch).permutation(train_indices)
        model.train()
        fitted = 0.0
        for offset in range(0, len(order), config["batch_groups"]):
            selected = order[offset:offset + config["batch_groups"]].tolist()
            loss = prediction_loss(model, features[selected])
            check(bool(loss.isfinite()), f"nonfinite loss at {folder.name}/{epoch}/{offset}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            fitted += loss.item() * len(selected)
        model.eval()
        validation = None
        if val_indices:
            with torch.no_grad():
                validation = sum(prediction_loss(model, features[val_indices[offset:offset + config["batch_groups"]]]).item()
                                 * len(val_indices[offset:offset + config["batch_groups"]])
                                 for offset in range(0, len(val_indices), config["batch_groups"])) / len(val_indices)
            check(np.isfinite(validation), "development validation produced nonfinite metrics")
        records.append({"epoch": epoch, "train_loss": fitted / len(order), "normal_validation_loss": validation,
                        "steps": (len(order) + config["batch_groups"] - 1) // config["batch_groups"],
                        "seconds": perf_counter() - start,
                        "final_grad_norm": float(grad_norm),
                        "peak_cuda_bytes": torch.cuda.max_memory_allocated(features.device) if features.is_cuda else None})
        save_checkpoint(latest, {"binding": binding, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                 "epochs": records, "started_utc": initial_time, "torch_cpu_rng": torch.get_rng_state(),
                                 "torch_cuda_rng": torch.cuda.get_rng_state(features.device) if features.is_cuda else None})
        write_json(folder / "epochs.json", records)
        print(f"{folder.parent.name}/{folder.name} epoch {epoch}/{epochs} normal loss {records[-1]['train_loss']:.6f}"
              + (f" val {validation:.6f}" if validation is not None else ""), flush=True)
    save_checkpoint(final, {"binding": binding, "model": model.state_dict(), "epoch": epochs})
    result = {"status": "PASS", "sha256": digest(final), "binding": binding,
              "training_seconds": sum(row["seconds"] for row in records),
              "peak_cuda_bytes": max(recovered_peaks, torch.cuda.max_memory_allocated(features.device)) if features.is_cuda else None,
              "steps": sum(row["steps"] for row in records), "started_utc": initial_time,
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    write_json(receipt, result)
    return result


def frozen_model(folder, frozen, cache, conditioned, device):
    config = frozen["config"]
    receipt = read_json(folder / "train.json")
    path = Path(folder) / "checkpoint.pt"
    check(receipt and receipt["status"] == "PASS" and receipt["sha256"] == digest(path),
          "formal evaluation requires a sealed checkpoint")
    state = torch.load(path, weights_only=True, map_location="cpu")
    binding = state["binding"]
    check(binding == receipt["binding"] and binding["frozen"] == frozen and binding["cache_sha256"] == cache["sha256"]
          and binding["conditioned"] == conditioned and binding["purpose"] == "formal"
          and binding["validation_groups"] == 0 and binding["training_groups"] == 360
          and state["epoch"] == config["epochs"] and receipt["steps"] == config["epochs"] * 90,
          "checkpoint is not the fixed final all-normal training run")
    model = build_predictor(config, conditioned, device)
    model.load_state_dict(state["model"])
    return model.eval().requires_grad_(False), receipt
