"""Plan or verify and stage four official M²AD-Synergy classes offline."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import zipfile

from filelock import FileLock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "audit"))
from tools.download_policy import assemble_local, describe
from manifest import check_manifest, load_manifest
from sources import ARCHIVES, CANDIDATES, RELEASE_REVISION, RELEASE_URL, require


ARCHIVE_SHA256 = {
    "Car": "27e77cfe0faad99065489ce5c5a785a2450da771372997dc9cfca160e778af36",
    "Cube": "72b5d96a1281c61d247771387b0a7bc66307c7d6e9c14112ae0553b5b89cafe7",
    "Doll": "1fd27c8aaf89d6bdde1d5061edcc5e45a5c8d7241c9d287402807a1985f8bcb3",
    "Ring": "20b2b0f12c338747208a914aa98e4f175eef7fa0e26de8eae0bc1f8ac7a27d7c",
}
def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def archive_item(root, category, checksum):
    mirror = RELEASE_URL.replace("https://huggingface.co/", "https://hf-mirror.com/", 1)
    name = f"{category}.zip"
    return {"name": name, "version": RELEASE_REVISION, "size_bytes": ARCHIVES[category],
            "sha256": checksum, "purpose": "M2AD-Synergy archive validation and evaluation",
            "mirror": f"{mirror}/{name}", "redirect_chain": [], "route_status": "unverified",
            "destination": str(root / "archives" / name)}


def extract_and_audit(archive_path, data, cls, dataset, checksum=None):
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
    return {"archive_sha256": checksum or ARCHIVE_SHA256[cls], "archive_bytes": ARCHIVES[cls],
            "crc_verified_files": len(files), "images": len(images), "masks": len(masks), "segments": len(segments),
            "counts": check_manifest(data)["classes"][cls], "split_crc_size_overlap": 0}


def atomic_json(path, data):
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def stage_archive(root, cache, category, checksum):
    archive = root / "archives" / f"{category}.zip"
    dataset = root / "dataset"
    marker = root / "data_audit" / f"{category}.json"
    if marker.is_file():
        receipt = json.loads(marker.read_text(encoding="utf-8"))
        require(receipt["archive_sha256"] == checksum and receipt["release_revision"] == RELEASE_REVISION,
                "existing receipt differs")
        require((dataset / category).is_dir() and (dataset / "jsons/meta_unsupervised.json").is_file(),
                "staged data missing; existing receipt preserved")
        return
    data = load_manifest(cache)
    path = assemble_local(archive, RELEASE_REVISION, ARCHIVES[category], checksum)
    manifest = dataset / "jsons/meta_unsupervised.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    expected = (json.dumps(data, indent=2) + "\n").replace("\n", "\r\n").encode()
    if not manifest.exists():
        temp = manifest.with_suffix(".json.tmp")
        temp.write_bytes(expected)
        temp.replace(manifest)
    require(manifest.read_bytes() == expected, "existing staged manifest differs")
    marker.parent.mkdir(parents=True, exist_ok=True)
    result = extract_and_audit(path, data, category, dataset, checksum)
    atomic_json(marker, {"release_revision": RELEASE_REVISION, "category": category, **result})


def main():
    parser = argparse.ArgumentParser(description="Plan or stage M2AD classes using local inputs")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--classes", choices=CANDIDATES, nargs="+", default=list(CANDIDATES))
    parser.add_argument("--task-id", default="TASK-002")
    parser.add_argument("--stage-local", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    require(not root.is_relative_to(Path(__file__).resolve().parents[2]), "use a data root outside the repository")
    categories = list(dict.fromkeys(args.classes))
    if args.stage_local:
        root.mkdir(parents=True, exist_ok=True)
        with FileLock(root / "staging.lock"):
            for category in categories:
                stage_archive(root, root / "cache", category, ARCHIVE_SHA256[category])
    else:
        plan = {"task_id": args.task_id, "files": [archive_item(root, cls, ARCHIVE_SHA256[cls])
                for cls in categories if not (root / "data_audit" / f"{cls}.json").is_file()]}
        print(json.dumps(describe(plan), indent=2))


if __name__ == "__main__":
    main()
