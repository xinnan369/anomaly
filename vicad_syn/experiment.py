"""Frozen configuration, atomic receipts, and pinned upstream imports."""

from contextlib import contextmanager
from hashlib import file_digest, sha256
import json
import os
from pathlib import Path
import subprocess
import sys


CONFIG = Path(__file__).with_name("configs") / "experiment.json"


def digest(path):
    with Path(path).open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return Path(path).exists() and json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    temp.replace(path)


def check(condition, message):
    if not condition:
        raise ValueError(message)


def protocol():
    config = read_json(CONFIG)
    package = Path(__file__).parent
    sources = {p.relative_to(package).as_posix(): sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
               for p in sorted(package.rglob("*.py"))}
    return {"config": config, "config_sha256": sha256(CONFIG.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
            "sources": sources}


@contextmanager
def pinned_upstream(path, revision):
    head = subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
    check(head == revision, f"unexpected upstream revision {head}")
    subprocess.run(["git", "-c", "core.safecrlf=false", "-C", path, "diff", "--quiet", "HEAD", "--"], check=True)
    past = Path.cwd()
    try:
        os.chdir(path)
        sys.path.insert(0, str(path))
        yield
    finally:
        os.chdir(past)
        sys.path.pop(0)
