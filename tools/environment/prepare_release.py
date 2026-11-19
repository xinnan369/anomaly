"""Download, verify and stage the four official M²AD-Synergy classes."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import time
import zipfile

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "audit"))
from manifest import check_manifest, load_manifest
from sources import ARCHIVES, CANDIDATES, RELEASE_REVISION, RELEASE_URL, require


ARCHIVE_SHA256 = {
    "Car": "27e77cfe0faad99065489ce5c5a785a2450da771372997dc9cfca160e778af36",
    "Cube": "72b5d96a1281c61d247771387b0a7bc66307c7d6e9c14112ae0553b5b89cafe7",
    "Doll": "1fd27c8aaf89d6bdde1d5061edcc5e45a5c8d7241c9d287402807a1985f8bcb3",
    "Ring": "20b2b0f12c338747208a914aa98e4f175eef7fa0e26de8eae0bc1f8ac7a27d7c",
}
CHUNK = 64 * 1024 * 1024


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def fetch_part(url, total, bounds, directory):
    start, stop = bounds
    part = directory / f"{start:012d}.part"
    size = stop - start
    if part.exists() and part.stat().st_size == size:
        return part
    headers = {"Range": f"bytes={start}-{stop-1}", "Accept-Encoding": "identity"}
    temp = part.with_suffix(".download")
    for attempt in range(5):
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(20, 90)) as response:
                response.raise_for_status()
                require(response.status_code == 206, "server did not honor Range")
                require(response.headers.get("Content-Range") == f"bytes {start}-{stop-1}/{total}",
                        "incorrect Content-Range")
                written = 0
                with temp.open("wb") as output:
                    for chunk in response.iter_content(1024 * 1024):
                        written += len(chunk)
                        require(written <= size, "oversized HTTP body")
                        output.write(chunk)
                require(written == size, "truncated HTTP body")
            temp.replace(part)
            return part
        except (ValueError, requests.RequestException) as error:
            if attempt == 4:
                raise RuntimeError(f"Range {start}-{stop-1} failed ({type(error).__name__}): "
                                   "rerun to resume completed parts") from None
            time.sleep(2 ** attempt)


def download(cls, root, workers):
    archive = root / f"{cls}.zip"
    size, digest = ARCHIVES[cls], ARCHIVE_SHA256[cls]
    if archive.exists():
        require(archive.stat().st_size == size and sha256(archive) == digest, f"{cls}: existing ZIP hash mismatch")
        return archive
    parts = root / f"{cls}-{RELEASE_REVISION}.parts"
    parts.mkdir(exist_ok=True)
    chunks = [(i, min(i + CHUNK, size)) for i in range(0, size, CHUNK)]
    url = f"{RELEASE_URL}/{cls}.zip"
    with ThreadPoolExecutor(max_workers=workers) as pool:
        files = []
        for count, part in enumerate(pool.map(lambda bounds: fetch_part(url, size, bounds, parts), chunks), 1):
            files.append(part)
            if count == len(chunks) or count % 8 == 0:
                print(f"{cls}: downloaded or resumed {count}/{len(chunks)} chunks", flush=True)
    pending = archive.with_suffix(".assembling")
    with pending.open("wb") as output:
        for path in files:
            with path.open("rb") as source:
                shutil.copyfileobj(source, output, 4 * 1024 * 1024)
    require(pending.stat().st_size == size and sha256(pending) == digest, f"{cls}: assembled ZIP hash mismatch")
    pending.replace(archive)
    for path in files:
        path.unlink()
    parts.rmdir()
    return archive


def extract_and_audit(archive_path, data, cls, dataset):
    rows = {split: data[split][cls] for split in ("train", "test")}
    images = {row["img_path"] for split in rows.values() for row in split}
    masks = {row["mask_path"] for split in rows.values() for row in split if row["mask_path"]}
    segments = {row["seg_path"] for split in rows.values() for row in split if row["seg_path"]}
    expected = images | masks | segments
    with zipfile.ZipFile(archive_path) as archive:
        members = [i for i in archive.infolist() if not i.is_dir()]
        files = {i.filename.removeprefix("./"): i for i in members}
        require(len(files) == len(members) and set(files) == expected, f"{cls}: ZIP paths differ from release manifest")
        fingerprints = [{(files[r["img_path"]].CRC, files[r["img_path"]].file_size) for r in rows[split]}
                        for split in ("train", "test")]
        require(not (fingerprints[0] & fingerprints[1]), f"{cls}: overlap in ZIP CRC32 and lengths")
        for number, (name, member) in enumerate(files.items(), 1):
            rel = PurePosixPath(name)
            require(not rel.is_absolute() and rel.parts[0] == cls and ".." not in rel.parts and "\\" not in name,
                    f"{cls}: invalid ZIP path")
            target = dataset.joinpath(*rel.parts)
            require(target.resolve().is_relative_to(dataset.resolve()), f"{cls}: extraction escaped data root")
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_suffix(".extracting")
            with archive.open(member) as source, temp.open("wb") as output:
                shutil.copyfileobj(source, output, 1024 * 1024)
            temp.replace(target)
            if number % 2000 == 0:
                print(f"{cls}: CRC-verified and extracted {number}/{len(files)} files", flush=True)
    require({p.relative_to(dataset).as_posix() for p in (dataset / cls).rglob("*") if p.is_file()} == expected,
            f"{cls}: extracted tree differs from manifest")
    return {"archive_sha256": ARCHIVE_SHA256[cls], "archive_bytes": ARCHIVES[cls],
            "crc_verified_files": len(files), "images": len(images), "masks": len(masks), "segments": len(segments),
            "counts": check_manifest(data)["classes"][cls], "split_crc_size_overlap": 0}


def atomic_json(path, data):
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--classes", choices=CANDIDATES, nargs="+", default=list(CANDIDATES))
    parser.add_argument("--workers", type=int, default=8, choices=range(1, 13))
    args = parser.parse_args()
    root = args.root.resolve()
    require(not root.is_relative_to(Path(__file__).resolve().parents[2]), "use a data root outside the repository")
    archives, dataset, audit = root / "archives", root / "dataset", root / "data_audit"
    for folder in (archives, dataset / "jsons", audit):
        folder.mkdir(parents=True, exist_ok=True)
    data = load_manifest(root / "cache")
    stats = check_manifest(data)
    atomic_json(audit / "manifest-summary.json", stats)
    atomic_json(dataset / "jsons" / "meta_unsupervised.json", data)
    for cls in dict.fromkeys(args.classes):
        print(f"{cls}: preparing {ARCHIVES[cls]} bytes from revision {RELEASE_REVISION}", flush=True)
        path = download(cls, archives, args.workers)
        marker = audit / f"{cls}.json"
        if marker.exists():
            marker.unlink()
        result = extract_and_audit(path, data, cls, dataset)
        atomic_json(marker, {"release_revision": RELEASE_REVISION, "category": cls, **result})
        print(f"{cls}: verified, staged and ready", flush=True)


if __name__ == "__main__":
    main()
