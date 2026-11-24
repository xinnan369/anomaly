"""Fit the registered normal-only predictor and conditional reference prototypes."""

from time import perf_counter

import numpy as np
import torch

from vicad_refinement.data import as_tensor, view_indices
from vicad_refinement.model import estimate_prototypes
from vicad_syn.experiment import check, digest, read_json, write_json
from vicad_syn.training.train import build_predictor, prediction_loss, save_checkpoint


def fit(folder, cache, groups, features, ids, seed, base, protocol_binding, device):
    folder.mkdir(parents=True, exist_ok=True)
    binding = {"seed": seed, "feature_sha256": cache["sha256"], "training_objects": sorted(ids),
               "epochs": base["epochs"], "frozen_protocol": protocol_binding}
    check(ids and ids <= {g.identity for g in groups}, "train IDs must belong to official normal data")
    idx = [i for i, g in enumerate(groups) if g.identity in ids]
    check(len(idx) == 12 * len(ids), "invalid physical training set")
    final = folder / "checkpoint.pt"
    current = folder / "latest.pt"
    receipt = folder / "training.json"
    torch.manual_seed(seed)
    model = build_predictor(base, True, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base["learning_rate"], weight_decay=base["weight_decay"])
    if result := read_json(receipt):
        check(binding == result["binding"] and digest(final) == result["checkpoint_sha256"],
              "frozen checkpoint binding changed")
        state = torch.load(final, map_location=device, weights_only=True)
        check(state["binding"] == binding and state["epoch"] == base["epochs"], "wrong frozen epoch")
        model.load_state_dict(state["predictor"])
        return model.eval().requires_grad_(False), state["prototypes"], result
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    records = []
    if current.exists():
        saved = torch.load(current, map_location=device, weights_only=True)
        check(saved["binding"] == binding, "resume checkpoint binding changed")
        model.load_state_dict(saved["predictor"])
        optimizer.load_state_dict(saved["optimizer"])
        records = saved["records"]
    leaf = as_tensor(features, slice(None), device)[:, :, -1].contiguous()
    # Match the v1 AdamW leaf loss and epoch-order generator.
    for epoch in range(len(records) + 1, base["epochs"] + 1):
        when = perf_counter()
        model.train()
        order = np.random.default_rng(seed + epoch).permutation(idx)
        losses = []
        for offset in range(0, len(order), base["batch_groups"]):
            batch = leaf[order[offset:offset + base["batch_groups"]].tolist()]
            loss = prediction_loss(model, batch)
            check(bool(loss.isfinite()), f"nonfinite loss at {epoch}/{offset}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            optimizer.step()
            losses.append(loss.item())
        records.append({"epoch": epoch, "steps": len(losses), "loss": float(np.mean(losses)),
                        "seconds": perf_counter() - when, "peak_cuda_bytes": torch.cuda.max_memory_allocated()})
        save_checkpoint(current, {"binding": binding, "records": records, "predictor": model.state_dict(),
                                  "optimizer": optimizer.state_dict()})
        write_json(folder / "epochs.json", records)
        print(f"{folder.parent.name}/{folder.name}: epoch {epoch}/{base['epochs']} normal loss={records[-1]['loss']:.6f}", flush=True)
    del leaf
    views = view_indices(groups, device).tolist()
    # Evaluate prototypes one view at a time to bound working memory.
    prototypes = []
    for view in range(12):
        x = as_tensor(features, [i for i in idx if views[i] == view], device)
        prototypes.append(estimate_prototypes(x, torch.zeros(len(x), device=device), 1)[0])
    proto = torch.stack(prototypes)
    check(bool(proto.isfinite().all()), "nonfinite normal prototype")
    save_checkpoint(final, {"binding": binding, "predictor": model.state_dict(), "epoch": base["epochs"],
                            "prototypes": proto.cpu()})
    epoch_seconds = sum(row["seconds"] for row in records)
    result = {"status": "PASS", "binding": binding, "checkpoint_sha256": digest(final),
              "seconds": perf_counter() - started, "optimization_seconds": epoch_seconds,
              "steps": sum(row["steps"] for row in records),
              "peak_cuda_bytes": max(torch.cuda.max_memory_allocated(), *(row["peak_cuda_bytes"] for row in records)),
              "trainable_parameters": sum(p.numel() for p in model.parameters()),
              "prototype_float32_bytes": proto.numel() * proto.element_size()}
    write_json(receipt, result)
    return model.eval().requires_grad_(False), proto, result
