"""Verified category receipts and content-addressed frozen multi-layer features."""

from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from vicad_syn.data.cache import Images
from vicad_syn.data.manifest import VIEWS, groups
from vicad_syn.experiment import check, digest, read_json, write_json


def load_groups(root, category, base, expected, split):
    manifest = root / "dataset/jsons/meta_unsupervised.json"
    audit = read_json(root / "data_audit" / f"{category}.json")
    check(audit and audit["release_revision"] == base["data_revision"] and audit["category"] == category
          and audit["images"] == sum(expected) * 120 and audit["split_crc_size_overlap"] == 0,
          f"{category}: verified release assets required")
    check(digest(manifest) == base["manifest_sha256"], "fixed manifest hash mismatch")
    all_rows = read_json(manifest)
    train = groups(all_rows["train"][category], category)
    check(len(train) == expected[0] * 12 and all(
        row["object_anomaly"] == row["image_anomaly"] == 0 for group in train for row in group.records
    ), "invalid official normal training rows")
    if split == "train":
        return train
    check(split == "test", "invalid split")
    test = groups(all_rows["test"][category], category)
    check(len(test) == expected[1] * 12, "unexpected release test count")
    check({group.identity for group in train}.isdisjoint(group.identity for group in test),
          "physical instance crosses official splits")
    return test


def view_indices(groups, device):
    return torch.tensor([VIEWS.index(g.view) for g in groups], device=device)


def encode(root, groups, output, encoder, fixed, device):
    start = perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    paths = [item for group in groups for item in group.paths]
    shape = (len(paths), len(encoder.blocks), 256, 768)
    array_path = output / "features.npy"
    receipt_path = output / "features.json"
    binding = {
        "order_sha256": sha256(("\n".join(paths) + "\n").encode()).hexdigest(),
        "layers": [block + 1 for block in encoder.blocks], "dtype": "float16",
        "shape": list(shape), "source": fixed,
    }
    rec = read_json(receipt_path)
    if rec:
        check(rec["binding"] == binding, "feature cache protocol changed")
        if rec["status"] == "PASS":
            check(digest(array_path) == rec["sha256"], "cached feature digest changed")
            return rec
    else:
        check(not array_path.exists(), "unbound feature cache")
        rec = {"status": "RUNNING", "binding": binding, "position": 0, "seconds": 0}
        write_json(receipt_path, rec)
    buffer = np.lib.format.open_memmap(array_path, mode="r+" if array_path.exists() else "w+",
                                            dtype=np.float16, shape=shape)
    check(buffer.shape == shape and buffer.dtype == np.float16, "invalid feature cache layout")
    position = rec["position"]
    elapsed = rec["seconds"]
    torch.cuda.reset_peak_memory_stats()
    dataloader = DataLoader(Images(root / "dataset", paths[position:]), batch_size=24, num_workers=8, pin_memory=True)
    for index, images in enumerate(dataloader, 1):
        patch = encoder(images.to(device, non_blocking=True))
        check(bool(patch.isfinite().all()), f"nonfinite feature at {position}")
        buffer[position:position + len(patch)] = patch.half().cpu().numpy()
        position += len(patch)
        if index % 25 == 0 or position == len(paths):
            buffer.flush()
            rec.update(position=position, seconds=elapsed + perf_counter() - start)
            write_json(receipt_path, rec)
            print(f"{output.parent.name}/{output.name}: encoded {position}/{len(paths)}", flush=True)
    check(position == len(paths), "short feature cache")
    del buffer
    rec.update(status="PASS", sha256=digest(array_path), bytes=array_path.stat().st_size,
               peak_cuda_bytes=torch.cuda.max_memory_allocated())
    write_json(receipt_path, rec)
    return rec


def load_features(folder):
    return np.load(folder / "features.npy", allow_pickle=False, mmap_mode="r").reshape(-1, 10, 3, 256, 768)


def as_tensor(features, indices, device):
    return torch.from_numpy(np.array(features[indices], copy=True)).to(device)
