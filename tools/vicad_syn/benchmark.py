"""Measure incremental uncertainty costs on frozen Car normal features."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from vicad_syn.experiment import read_json, write_json
from vicad_syn.training.train import frozen_model
from vicad_syn.uncertainty.signals import context_instability, epistemic_uncertainty


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    frozen = read_json(args.run / "frozen-protocol.json")["frozen"]
    cfg = frozen["config"]
    torch.set_num_threads(4)
    path = args.run / "Car"
    model, receipt = frozen_model(path / "conditioned", frozen, read_json(path / "train-cache/cache.json"),
                                  True, torch.device("cuda"))
    batch = cfg["batch_groups"]
    cache = np.load(path / "train-cache/features.npy", mmap_mode="r", allow_pickle=False)
    normal = torch.from_numpy(cache[:720].copy()).reshape(72, 10, 256, 768).cuda()
    records = {name: [] for name in ("full_prediction", "context_instability", "epistemic_statistics")}

    def measure(name, compute, record):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        start = perf_counter()
        value = compute()
        torch.cuda.synchronize()
        if record:
            records[name].append((perf_counter()-start, torch.cuda.max_memory_allocated()-before))
        return value

    for start in [-1, *range(0, len(normal), batch)]:
        x = F.normalize(normal[max(0,start):max(0,start)+batch].float(), dim=-1)
        full = measure("full_prediction", lambda: model(x), start >= 0)
        measure("context_instability", lambda: context_instability(model(x,"first"), model(x,"last")), start >= 0)
        measure("epistemic_statistics", lambda: epistemic_uncertainty(full), start >= 0)
    write_json(args.output, {"device": torch.cuda.get_device_name(), "checkpoint_sha256": receipt["sha256"],
        "groups": 72, "batch_groups": batch, "split": "normal training features only",
        "warmups": 1, "timings": {name: {"mean_batch_seconds": sum(row[0] for row in items)/len(items),
                                         "max_incremental_cuda_bytes": max(row[1] for row in items)}
                                  for name, items in records.items()}})


if __name__ == "__main__":
    main()
