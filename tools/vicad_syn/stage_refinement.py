"""Stage registered full confirmation classes with published LFS hashes."""

import argparse
import sys
from pathlib import Path

from vicad_syn.experiment import check, read_json

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "environment"))
from prepare_release import ARCHIVE_SHA256, atomic_json, download, extract_and_audit, load_manifest

REGISTRY = Path(__file__).resolve().parents[2] / "vicad_syn/refinement/protocol.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest-cache", type=Path, required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--workers", type=int, default=12, choices=range(1, 13))
    args = parser.parse_args()
    root = args.root.resolve()
    check(not root.is_relative_to(REGISTRY.parents[2]), "stage data outside Git")
    protocol = read_json(REGISTRY)
    check(args.category in protocol["confirmatory_order"], "unregistered category")
    ARCHIVE_SHA256.update(protocol["archive_sha256"])
    marker = root / "data_audit" / f"{args.category}.json"
    if marker.exists():
        audit = read_json(marker)
        check(audit["archive_sha256"] == ARCHIVE_SHA256[args.category], "wrong completed archive")
        return
    archives, dataset = root / "archives", root / "dataset"
    for directory in (archives, dataset / "jsons", marker.parent):
        directory.mkdir(parents=True, exist_ok=True)
    rows = load_manifest(args.manifest_cache)
    atomic_json(dataset / "jsons/meta_unsupervised.json", rows)
    print(f"{args.category}: staging fixed publication", flush=True)
    archive = download(args.category, archives, args.workers)
    audit = extract_and_audit(archive, rows, args.category, dataset)
    from sources import RELEASE_REVISION
    atomic_json(marker, {"category": args.category, "release_revision": RELEASE_REVISION, **audit})
    print(f"{args.category}: complete SHA-256/CRC/path-isolation audit passed", flush=True)


if __name__ == "__main__":
    main()
