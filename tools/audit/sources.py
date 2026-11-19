"""Pinned TASK-001 inputs and bounded HTTPS reads."""

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import requests


RELEASE_REVISION = "05abc5d03167354a7ef98541236b5c0bd00abadf"
RELEASE_URL = f"https://huggingface.co/datasets/ChengYuQi99/M2AD/resolve/{RELEASE_REVISION}"
CODE_REVISION = "a606aa0e1614f23dad70da26430dd66095192141"
ARCHIVES = {
    "Bird": 15506344175, "Car": 16318099753, "Cube": 17628365529,
    "Dice": 16681648758, "Doll": 16835932455, "Holder": 17471748248,
    "Motor": 14955220087, "Ring": 16564602406, "Teapot": 15758841927,
    "Tube": 16709077850,
}
CANDIDATES = ("Car", "Cube", "Doll", "Ring")


@dataclass(frozen=True)
class Resource:
    name: str
    url: str
    sha256: str
    max_bytes: int


MANIFEST = Resource(
    "jsons.zip",
    f"{RELEASE_URL}/jsons.zip",
    "bbeaa75c231c981e443c818859a701fcc0daf1769065abd91631f7e676725980",
    1_100_022,
)
FUSION = Resource(
    "m2ad_utils.py",
    f"https://raw.githubusercontent.com/hustCYQ/M2AD/{CODE_REVISION}/util/m2ad_utils.py",
    "e22c05165b65542d41f7f45828b0eb641614a0f84a7347ec9c1d17e1b855f009",
    20_000,
)
ADEVAL = Resource(
    "ADEval-1.1.0-py3-none-any.whl",
    "https://files.pythonhosted.org/packages/9d/64/56eb78b69199feb143633a21a4f585272616ba739e549e45715e13acd363/ADEval-1.1.0-py3-none-any.whl",
    "125caf991037a668a1f072f7e99c81b6bdf86dc8c30cf1699c731d4b79e0bd6e",
    26_044,
)


def require(condition, description):
    if not condition:
        raise ValueError(description)


def read_https(url, max_bytes, byte_range=None, charge=None):
    """Reject a server that ignores the range before reading its body."""
    headers = {"Accept-Encoding": "identity"}
    expected_range = None
    if byte_range is not None:
        start, end, total = byte_range
        headers["Range"] = f"bytes={start}-{end}"
        expected_range = f"bytes {start}-{end}/{total}"

    for attempt in range(3):
        if charge is not None:
            charge(max_bytes)
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(10, 35)) as response:
                response.raise_for_status()
                if expected_range is not None:
                    require(response.status_code == 206, "server ignored the byte range")
                    require(response.headers.get("Content-Range") == expected_range, "unexpected Content-Range")
                length = response.headers.get("Content-Length")
                if length is not None:
                    require(int(length) <= max_bytes, "response exceeds byte limit")
                body = bytearray()
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    require(len(body) + len(chunk) <= max_bytes, "response exceeds byte limit")
                    body.extend(chunk)
                if expected_range is not None:
                    require(len(body) == max_bytes, "incomplete byte range")
                return bytes(body)
        except requests.RequestException as error:
            if attempt == 2:
                raise RuntimeError(f"HTTPS read failed after 3 attempts ({type(error).__name__})") from None
            time.sleep(attempt + 1)


def verified_resource(cache: Path, resource: Resource) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / resource.name
    if path.exists():
        require(path.stat().st_size <= resource.max_bytes, f"{resource.name} exceeds byte limit")
        data = path.read_bytes()
    else:
        data = read_https(resource.url, resource.max_bytes)
    require(hashlib.sha256(data).hexdigest() == resource.sha256, f"{resource.name} SHA-256 mismatch")
    if not path.exists():
        path.write_bytes(data)
    return path
