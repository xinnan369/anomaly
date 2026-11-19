"""Compare remote ZIP directories with the manifest and optionally sample alignment."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import struct
import zipfile

from manifest import load_manifest
from sources import ARCHIVES, CANDIDATES, RELEASE_REVISION, RELEASE_URL, read_https, require


class RangeReader:
    """Seek over a ZIP while retrieving only bounded ranges into memory."""

    def __init__(self, cls):
        self.size = ARCHIVES[cls]
        self.url = f"{RELEASE_URL}/{cls}.zip"
        self.position = 0
        self.blocks = []
        self.attempted_bytes = 0
        self.successful_bytes = 0

    def charge(self, size):
        require(self.attempted_bytes + size <= 24_000_000, "archive request budget exceeded")
        self.attempted_bytes += size

    def prefetch(self, start, size):
        require(0 <= start < start + size <= self.size, "range outside archive")
        require(size <= 5_000_000, "single archive range exceeds byte limit")
        for offset, block in self.blocks:
            if offset <= start and start + size <= offset + len(block):
                return
        block = read_https(self.url, size, (start, start + size - 1, self.size), charge=self.charge)
        self.blocks.append((start, block))
        self.successful_bytes += len(block)

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}
        self.position = base[whence] + offset
        require(self.position >= 0, "negative archive offset")
        return self.position

    def tell(self):
        return self.position

    def seekable(self):
        return True

    def read(self, size=-1):
        remaining = max(0, self.size - self.position)
        size = remaining if size < 0 else min(size, remaining)
        if size == 0:
            return b""
        start = self.position
        self.prefetch(start, size)
        self.position += size
        for offset, block in self.blocks:
            if offset <= start and start + size <= offset + len(block):
                return block[start - offset:start - offset + size]
        raise RuntimeError("missing cached range")


def alignment_sample(archive, reader, files, cls):
    import cv2
    import numpy as np

    shades = []
    for light in ("01", "05", "10"):
        member = files[f"{cls}/Good/000/A000_I{light}.png"]
        require(member.file_size <= 5_000_000, "sample PNG exceeds decoded byte limit")
        size = min(reader.size - member.header_offset, member.compress_size + 4096)
        reader.prefetch(member.header_offset, size)
        png = archive.read(member)  # zipfile verifies CRC on complete reads
        require(png.startswith(b"\x89PNG\r\n\x1a\n") and png[12:16] == b"IHDR", "invalid PNG header")
        width, height = struct.unpack(">II", png[16:24])
        require((width, height) == (1024, 1024), "unexpected sample PNG dimensions")
        gray = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        require(gray is not None and gray.shape == (height, width), "PNG decode failed")
        dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        shades.append(cv2.magnitude(dx, dy))

    window = cv2.createHanningWindow((1024, 1024), cv2.CV_32F)
    pairs = {}
    for light, gradient in zip(("05", "10"), shades[1:]):
        (dx, dy), response = cv2.phaseCorrelate(shades[0].copy(), gradient.copy(), window)
        require(np.isfinite([dx, dy, response]).all(), "non-finite phase correlation")
        pairs[f"I01_to_I{light}"] = {
            "dx_px": round(dx, 3), "dy_px": round(dy, 3), "response": round(response, 4),
        }
    return {"view": "000", "normal_objects": 1, "decoded_crc_checked_pngs": 3,
            "image_size": [1024, 1024], "pairs": pairs}


def check_archive(cls, data, sample):
    reader = RangeReader(cls)
    reader.prefetch(reader.size - 65_557, 65_557)
    with zipfile.ZipFile(reader) as archive:
        members = [info for info in archive.infolist() if not info.is_dir()]
        files = {info.filename.removeprefix("./"): info for info in members}
        require(len(files) == len(members), f"{cls}: duplicate ZIP filename")
        rows = data["train"][cls] + data["test"][cls]
        images = {row["img_path"] for row in rows}
        masks = {row["mask_path"] for row in rows if row["mask_path"]}
        segments = {row["seg_path"] for row in rows if row["seg_path"]}
        names = set(files)
        coverage = {
            "missing_images": len(images - names), "missing_masks": len(masks - names),
            "missing_segments": len(segments - names), "extra_members": len(names - images - masks - segments),
        }
        require(not any(coverage.values()), f"{cls}: ZIP/manifest coverage mismatch {coverage}")
        hashes = []
        for split in ("train", "test"):
            split_files = (files[row["img_path"]] for row in data[split][cls])
            hashes.append({(info.CRC, info.file_size) for info in split_files})
        collisions = len(hashes[0] & hashes[1])
        require(not collisions, f"{cls}: cross-split CRC32+size intersection {collisions}")
        result = {
            "members": len(files), "archive_bytes": reader.size,
            "member_uncompressed_bytes": sum(info.file_size for info in files.values()),
            **coverage, "cross_split_crc32_size": collisions,
        }
        if sample:
            result["alignment_sample"] = alignment_sample(archive, reader, files, cls)
        result["successful_range_bytes"] = reader.successful_bytes
        result["requested_range_bytes_including_retries"] = reader.attempted_bytes
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True, help="cache directory outside the repository")
    parser.add_argument("--classes", nargs="+", choices=tuple(ARCHIVES), default=list(ARCHIVES))
    parser.add_argument("--alignment", action="store_true", help="decode three images per selected candidate class")
    args = parser.parse_args()
    data = load_manifest(args.cache)
    classes = list(dict.fromkeys(args.classes))
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {cls: pool.submit(check_archive, cls, data, args.alignment and cls in CANDIDATES)
                for cls in classes}
        results, failures = {}, {}
        for cls, job in jobs.items():
            try:
                results[cls] = job.result()
            except (ValueError, RuntimeError, zipfile.BadZipFile) as error:
                failures[cls] = str(error)
    print(json.dumps({"release_revision": RELEASE_REVISION, "archives": results, "failures": failures}, indent=2))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
