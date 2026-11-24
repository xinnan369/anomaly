"""Pinned TASK-001 inputs and verified local cache access."""

import hashlib
from dataclasses import dataclass
from pathlib import Path


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


def verified_resource(cache: Path, resource: Resource) -> Path:
    path = cache / resource.name
    require(path.is_file(), f"{resource.name}: local cache missing; use the managed download plan")
    require(path.stat().st_size <= resource.max_bytes, f"{resource.name} exceeds byte limit")
    require(hashlib.sha256(path.read_bytes()).hexdigest() == resource.sha256, f"{resource.name} SHA-256 mismatch")
    return path
