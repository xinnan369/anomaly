"""Persist and resume frozen patch features without test labels or masks."""

from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from torchvision.transforms import InterpolationMode

from vicad_syn.experiment import check, digest, read_json, write_json


IMAGE_SIZE = 224
TRANSFORM = Compose([Resize((IMAGE_SIZE, IMAGE_SIZE), InterpolationMode.BILINEAR), ToTensor(),
                     Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


class Images(Dataset):
    def __init__(self, dataset, files):
        self.dataset = dataset
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with Image.open(self.dataset / self.files[index]) as image:
            return TRANSFORM(image.convert("RGB"))


def cache_key(groups, config):
    files = [path for group in groups for path in group.paths]
    image_order = sha256(("\n".join(files) + "\n").encode()).hexdigest()
    return files, {
        "images": len(files), "groups": len(groups), "order_sha256": image_order,
        "backbone_sha256": config["backbone_sha256"], "release_revision": config["data_revision"],
        "extractor_sha256": digest(Path(__file__).parents[1] / "models/backbone.py"),
        "preprocess_sha256": digest(Path(__file__)),
        "feature_shape": [len(groups), 10, 256, 768], "feature_dtype": "float16",
    }


def encode(dataset, groups, folder, encoder, config, device):
    folder.mkdir(parents=True, exist_ok=True)
    files, expected = cache_key(groups, config)
    ledger = folder / "cache.json"
    array = folder / "features.npy"
    current = read_json(ledger)
    if current:
        check(current["identity"] == expected, "existing cache identity differs")
        if current["status"] == "PASS":
            check(digest(array) == current["sha256"], "cached feature file hash mismatch")
            return current
    else:
        check(not array.exists(), "unbound pre-existing feature file")
        current = {"identity": expected, "status": "RUNNING", "encoded_images": 0, "seconds": 0.0}
        write_json(ledger, current)
    start = perf_counter()
    prior = current["seconds"]
    begin = current["encoded_images"]
    encoded = np.lib.format.open_memmap(array, mode="r+" if array.exists() else "w+",
                                       dtype=np.float16, shape=(len(files), 256, 768))
    check(encoded.shape == (len(files), 256, 768) and encoded.dtype == np.float16, "existing cache array differs")
    data = DataLoader(Images(dataset, files[begin:]), batch_size=config["encode_batch"], shuffle=False,
                      num_workers=config["workers"], pin_memory=device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    position = begin
    for batch, images in enumerate(data, 1):
        patches = encoder(images.to(device, non_blocking=True))
        check(bool(patches.isfinite().all()), f"nonfinite frozen feature at image {position}")
        encoded[position:position + len(patches)] = patches.half().cpu().numpy()
        position += len(patches)
        if batch % 20 == 0 or position == len(files):
            encoded.flush()
            current.update(encoded_images=position, seconds=prior + perf_counter() - start)
            write_json(ledger, current)
            print(f"encode {folder.parent.name}/{folder.name}: {position}/{len(files)}", flush=True)
    check(position == len(files), "feature cache is incomplete")
    del encoded
    current.update(status="PASS", sha256=digest(array),
                   device=str(device), peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None)
    write_json(ledger, current)
    return current


def feature_tensor(folder, device):
    ledger = read_json(folder / "cache.json")
    check(ledger and ledger["status"] == "PASS", "feature cache isn't sealed")
    array = np.load(folder / "features.npy", allow_pickle=False).reshape(ledger["identity"]["feature_shape"])
    return torch.from_numpy(array).to(device)
