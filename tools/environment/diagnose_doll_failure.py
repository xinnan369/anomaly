"""Bounded forensic replay of a clean Doll failure, for diagnosis only."""

import argparse
import collections
from collections.abc import Iterable
from contextlib import ExitStack
import json
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/environment"))
import run_baseline as official
from training_diagnostics import NonFiniteTrainingError, TrainingDiagnostics


class ReplayFinished(Exception):
    pass


def restore(trainer, checkpoint, rng):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert (state["epoch"], state["iter"]) == (rng["epoch"], rng["iteration"])
    assert state["scaler"] is None
    expected = set(trainer.net.get_learnable_params())
    assert set(state["net"]) == expected
    missing, unexpected = trainer.net.load_state_dict(state["net"], strict=False)
    assert not unexpected and all("encoder" in name.split(".") for name in missing)
    trainer.optim.load_state_dict(state["optimizer"])
    trainer.scheduler.load_state_dict(state["scheduler"])
    trainer.epoch, trainer.iter = state["epoch"], state["iter"]
    trainer.reset(isTrain=True)
    trainer.pre_train()
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch_cpu"])
    torch.cuda.set_rng_state_all(rng["torch_cuda"])


def read_batches(log, start, end):
    batches = {}
    with log.open(encoding="utf-8") as source:
        for line in source:
            batch = json.loads(line)
            if start < batch["iteration"] <= end:
                batches[batch["iteration"]] = batch["paths"]
    assert list(batches) == list(range(start + 1, end + 1))
    return batches


def monitor_attention(net, stack):
    """Observe the actual official attention denominator and its batch/heads."""
    records = []
    original = torch.einsum
    original_div = torch.Tensor.__rtruediv__
    current = None

    def start(name):
        def pre(module, args):
            nonlocal current
            current = name
        return pre

    def clear(*args):
        nonlocal current
        current = None

    def einsum(equation, *args):
        value = original(equation, *args)
        if current is not None and equation == "...sd,...d->...s":
            with torch.no_grad():
                tensor = value.detach()
                zero = tensor == 0
                bad = ~tensor.isfinite()
                finite = tensor[~bad]
                records.append({
                    "attention": current, "equation": equation, "shape": list(tensor.shape),
                    "minimum_finite": finite.min().item() if finite.numel() else None,
                    "maximum_finite": finite.max().item() if finite.numel() else None,
                    "zeros": int(zero.sum()),
                    "nonfinite": int(bad.sum()),
                    "first_zero_coordinates_batch_head_token": zero.nonzero().tolist()[:32],
                    "first_nonfinite_coordinates_batch_head_token": bad.nonzero().tolist()[:32],
                })
        return value

    def right_divide(self, other):
        value = original_div(self, other)
        if current is not None and isinstance(other, float) and other == 1.0 and records:
            with torch.no_grad():
                nonfinite = ~value.detach().isfinite()
                records[-1]["reciprocal_nonfinite"] = int(nonfinite.sum())
        return value

    for name, module in net.named_modules():
        if type(module).__name__ == "LinearAttention2":
            for handle in (module.register_forward_pre_hook(start(name)), module.register_forward_hook(clear)):
                stack.callback(handle.remove)
    torch.einsum = einsum
    torch.Tensor.__rtruediv__ = right_divide
    stack.callback(setattr, torch, "einsum", original)
    stack.callback(setattr, torch.Tensor, "__rtruediv__", original_div)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "attempt", "output"):
        parser.add_argument(f"--{key}", required=True, type=lambda item: Path(item).resolve())
    args = parser.parse_args()
    train = args.attempt / "train"
    failure = official.read_record(train / "first-nonfinite.json")
    checkpoint, = (train / "checkpoints").glob("*/*_ckpt.pth")
    target = failure["iteration"]
    epoch = failure["epoch_completed"]
    rng_file = train / "rng" / f"epoch-{epoch:03d}.pt" if epoch else train / "rng/initial.pt"
    rng = torch.load(rng_file, map_location="cpu", weights_only=False)  # Locally created by this clean run.
    assert rng["iteration"] < target <= rng["iteration"] + 300
    batches = read_batches(train / "batch-order.jsonl", rng["iteration"], target)
    assert batches[target] == failure["batch_paths"]
    assert args.output.is_relative_to(args.attempt)
    args.output.mkdir(parents=True, exist_ok=False)
    upstream = args.root / "inputs/upstream"
    os.chdir(upstream)
    sys.path.insert(0, str(upstream))
    if not hasattr(collections, "Iterable"):
        collections.Iterable = Iterable
    import run_dataset as entry
    from trainer import _base_trainer

    get_trainer = entry.get_trainer

    def build(cfg):
        trainer = get_trainer(cfg)

        def replay():
            observer = TrainingDiagnostics(trainer, args.output, args.root / "dataset")
            record = {"production_checkpoint": str(checkpoint), "production_checkpoint_sha256": official.digest(checkpoint),
                      "start_epoch": epoch, "start_iteration": rng["iteration"], "target": target,
                      "purpose": "bounded_forensic_replay", "target_paths": batches[target],
                      "matching_batches": 0, "first_nonfinite": None}
            try:
                with ExitStack() as resources, ExitStack() as local_probe:
                    observer.install(_base_trainer, resources)
                    restore(trainer, checkpoint, rng)
                    observer.successful_updates = trainer.iter
                    iterator = iter(trainer.train_loader)
                    for step, paths in batches.items():
                        trainer.scheduler_step(trainer.iter)
                        trainer.iter = step
                        batch = next(iterator)
                        actual_paths = [Path(path).relative_to(observer.dataset_root).as_posix() for path in batch["img_path"]]
                        official.require(actual_paths == paths, f"sample order differs from clean training at {step}")
                        record["matching_batches"] += 1
                        if step == target:
                            record["attention_denominator"] = monitor_attention(trainer.net, local_probe)
                        trainer.set_input(batch, train=True)
                        trainer.optimize_parameters()
            except NonFiniteTrainingError:
                observed = official.read_record(args.output / "first-nonfinite.json")
                record["first_nonfinite"] = observed
                record["stage_matches_clean_failure"] = (observed["iteration"], observed["stage"]) == (
                    target, failure["stage"])
            except Exception as error:
                record["error"] = f"{type(error).__name__}: {error}"
                raise
            finally:
                official.write_json(args.output / "forensic-replay.json", record)
            raise ReplayFinished

        trainer.run = replay
        return trainer

    entry.get_trainer = build
    try:
        official.run(SimpleNamespace(
            upstream=upstream, weight=args.root / "inputs/weights" / official.WEIGHT_NAME,
            release_root=args.root, output=args.output, model="dinomaly", category="Doll",
            checkpoint=None, preflight=False, diagnose_doll=False,
        ), {})
    except ReplayFinished:
        pass


if __name__ == "__main__":
    main()
