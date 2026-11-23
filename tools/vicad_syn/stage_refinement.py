"""Stage registered confirmation classes with published LFS hashes and HTTP resume."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import shutil
import sys
import time
from pathlib import Path
from threading import Lock
from urllib.parse import urlsplit

import requests

from vicad_syn.experiment import check, read_json

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "environment"))
from prepare_release import CHUNK, ARCHIVES, atomic_json, extract_and_audit, load_manifest, sha256
from sources import RELEASE_REVISION, RELEASE_URL

REGISTRY = Path(__file__).resolve().parents[2] / "vicad_syn/refinement/protocol.json"
LFS = "https://huggingface.co/datasets/ChengYuQi99/M2AD.git/info/lfs/objects/batch"


def lfs_download_source(checksum, size):
    """Resolve the publisher's content-addressed Git LFS download action."""
    lock = Lock()
    action, fetched = None, 0
    mime = "application/vnd.git-lfs+json"
    body = {"operation": "download", "transfers": ["basic"], "objects": [{"oid": checksum, "size": size}]}

    def source():
        nonlocal action, fetched
        with lock:
            if action is None or time.monotonic() - fetched >= 240:
                with requests.post(LFS, headers={"Accept": mime, "Content-Type": mime}, json=body,
                                   timeout=(15, 30)) as response:
                    response.raise_for_status()
                    item, = response.json()["objects"]
                check(item["oid"] == checksum and item["size"] == size, "different object from publisher LFS")
                action = item["actions"]["download"]
                parsed = urlsplit(action["href"])
                check(parsed.scheme == "https" and parsed.hostname is not None and (
                    parsed.hostname.endswith(".hf.co") or parsed.hostname.endswith(".huggingface.co")
                ), "unexpected publisher download origin")
                fetched = time.monotonic()
            return action["href"], action.get("header", {})

    return source

def get_part(url, total, bounds, location):
    start, stop = bounds
    path = location / f"{start:012d}.part"
    partial = path.with_suffix(".download")
    if path.exists():
        check(path.stat().st_size == stop - start, f"wrong completed range {bounds}")
        return path
    for attempt in range(8):
        size = partial.stat().st_size if partial.exists() else 0
        check(size <= stop - start, "partial segment is larger than its registered range")
        if size == stop - start:
            partial.replace(path)
            return path
        offset = start + size
        try:
            target, extra = url() if callable(url) else (url, {})
            headers = {**extra, "Range": f"bytes={offset}-{stop - 1}", "Accept-Encoding": "identity"}
            with requests.get(target, headers=headers, stream=True, timeout=(20, 50)) as response:
                response.raise_for_status()
                check(response.status_code == 206 and response.headers.get("Content-Range") ==
                      f"bytes {offset}-{stop - 1}/{total}", "unexpected HTTP content range")
                with partial.open("ab") as sink:
                    for block in response.iter_content(256 * 1024):
                        check(offset + len(block) <= stop, "HTTP range contains too many bytes")
                        sink.write(block)
                        offset += len(block)
                check(offset == stop, "HTTP range ended early")
            partial.replace(path)
            return path
        except (ValueError, KeyError, requests.RequestException) as error:
            print(f"range {start}-{stop - 1}: attempt {attempt + 1} {type(error).__name__}, "
                  f"retained {partial.stat().st_size if partial.exists() else 0} bytes", flush=True)
            if attempt == 7:
                raise RuntimeError(f"range {start}-{stop - 1} exhausted retries; restart to resume") from None
            time.sleep(min(2 ** attempt, 15))


def download(category, archives, checksum, workers, transport):
    archive = archives / f"{category}.zip"
    total = ARCHIVES[category]
    if archive.exists():
        check(archive.stat().st_size == total and sha256(archive) == checksum, "ZIP size/SHA-256 mismatch")
        return archive
    parts = archives / f"{category}-{RELEASE_REVISION}.parts"
    parts.mkdir(exist_ok=True)
    spans = [(i, min(i + CHUNK, total)) for i in range(0, total, CHUNK)]
    url = lfs_download_source(checksum, total) if transport == "lfs" else f"{RELEASE_URL}/{category}.zip"
    with ThreadPoolExecutor(max_workers=workers) as pool:
        finished = []
        for number, part in enumerate(pool.map(lambda span: get_part(url, total, span, parts), spans), 1):
            finished.append(part)
            if number % 8 == 0 or number == len(spans):
                print(f"{category}: {number}/{len(spans)} complete ranges", flush=True)
    pending = archive.with_suffix(".assembling")
    with pending.open("wb") as sink:
        for part in finished:
            with part.open("rb") as source:
                shutil.copyfileobj(source, sink, 4 * 1024 * 1024)
    check(pending.stat().st_size == total and sha256(pending) == checksum, "assembled ZIP size/SHA-256 mismatch")
    pending.replace(archive)
    for path in finished:
        path.unlink()
    parts.rmdir()
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest-cache", type=Path, required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--workers", type=int, default=12, choices=range(1, 13))
    parser.add_argument("--transport", choices=("resolve", "lfs"), default="resolve")
    args = parser.parse_args()
    root = args.root.resolve()
    check(not root.is_relative_to(REGISTRY.parents[2]), "stage data outside Git")
    protocol = read_json(REGISTRY)
    check(args.category in protocol["confirmatory_order"], "unregistered category")
    checksum = protocol["archive_sha256"][args.category]
    marker = root / "data_audit" / f"{args.category}.json"
    if marker.exists():
        audit = read_json(marker)
        check(audit["archive_sha256"] == checksum and audit["release_revision"] == RELEASE_REVISION,
              "wrong completed archive")
        return
    archives, dataset = root / "archives", root / "dataset"
    for directory in (archives, dataset / "jsons", marker.parent):
        directory.mkdir(parents=True, exist_ok=True)
    rows = load_manifest(args.manifest_cache)
    atomic_json(dataset / "jsons/meta_unsupervised.json", rows)
    print(f"{args.category}: staging fixed publication via {args.transport}", flush=True)
    archive = download(args.category, archives, checksum, args.workers, args.transport)
    # The original auditor records the expected hash in the receipt.
    from prepare_release import ARCHIVE_SHA256
    ARCHIVE_SHA256[args.category] = checksum
    audit = extract_and_audit(archive, rows, args.category, dataset)
    atomic_json(marker, {"category": args.category, "release_revision": RELEASE_REVISION, **audit})
    print(f"{args.category}: complete SHA-256/CRC/path-isolation audit passed", flush=True)


if __name__ == "__main__":
    main()
