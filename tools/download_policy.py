"""Offline plans and owner-signed, budgeted downloads for local staging tools."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import subprocess
from urllib.parse import urljoin, urlsplit
import zipfile

from filelock import FileLock
import requests

LIMIT = 1_000_000_000
CHUNK = 64 * 1024 * 1024
DOMESTIC_MIRRORS = frozenset({
    "hf-mirror.com", "modelscope.cn", "www.modelscope.cn", "pypi.tuna.tsinghua.edu.cn",
    "mirrors.tuna.tsinghua.edu.cn", "mirrors.ustc.edu.cn", "mirrors.aliyun.com",
})
TRUST_DIRECTORY = Path.home() / ".codex" / "vicad-download"
SIGNER = "vicad-project-owner"
NAMESPACE = "vicad-download-v1"
REPO = Path(__file__).resolve().parents[1]


def require(valid, detail):
    if not valid:
        raise ValueError(detail)


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def endpoint(url):
    item = urlsplit(url)
    require(item.scheme == "https" and item.hostname and item.port in (None, 443)
            and not item.username and not item.password and not item.fragment, "invalid HTTPS source")
    return item.hostname, item.path


def clean_part(value):
    return isinstance(value, str) and bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value))


def local_path(value):
    path = Path(value)
    require(path.is_absolute() and not path.resolve().is_relative_to(REPO), "use an absolute path outside the repository")
    require(path == path.resolve(), "destination must remain a canonical path")
    return path


def spans(size):
    return [(start, min(start + CHUNK, size)) for start in range(0, size, CHUNK)]


def range_cache(target, revision):
    return target.parent / f"{target.stem}-{revision}.parts"


def retained(directory, start, stop):
    full = directory / f"{start:012d}.part"
    partial = full.with_suffix(".download")
    size = stop - start
    if full.exists():
        require(full.is_file() and full.stat().st_size == size, "invalid completed range")
        return full, size
    have = partial.stat().st_size if partial.exists() else 0
    require(have <= size, "oversized partial range")
    return partial, have


def verify_file(path, item):
    require(path.is_file() and path.stat().st_size == item["size_bytes"], "wrong file size")
    require(digest(path) == item["sha256"], "SHA-256 mismatch")
    if path.suffix.lower() in {".zip", ".whl"}:
        with zipfile.ZipFile(path) as bundle:
            require(bundle.testzip() is None, "archive CRC mismatch")


def file_matrix(plan):
    require(re.fullmatch(r"TASK-[0-9]{3}", plan.get("task_id", "")), "need canonical TASK-NNN ID")
    require(isinstance(plan.get("files"), list) and plan["files"], "no planned files")
    destinations = set()
    fingerprints = set()
    report = []
    for item in plan["files"]:
        name, revision = item.get("name"), item.get("version")
        require(clean_part(name) and clean_part(revision), "invalid file name or version")
        target = local_path(item["destination"])
        require(target.name == name and target not in destinations, "invalid or duplicate destination")
        destinations.add(target)
        checksum = item.get("sha256")
        if checksum:
            require(checksum not in fingerprints, "duplicate content: reuse the approved file locally")
            fingerprints.add(checksum)
        route = item["redirect_chain"]
        require(isinstance(route, list), "redirect chain must be listed")
        for url in [item["mirror"], *route]:
            endpoint(url)
            require(not urlsplit(url).query, "persist endpoints without tokens or queries")
        require(item["purpose"].strip(), "missing purpose")
        total = item.get("size_bytes")
        require(total is None or (type(total) is int and total > 0), "invalid size")
        parts = range_cache(target, revision)
        if total is None:
            remaining = None
        elif target.exists():
            verify_file(target, item)
            remaining = 0
        else:
            remaining = sum(stop - start - retained(parts, start, stop)[1] for start, stop in spans(total))
        report.append({**item, "cache": str(parts), "remaining_bytes": remaining,
                       "final_source": (route or [item["mirror"]])[-1] if item.get("route_status") == "verified" else None})
    return report


def verify_signature(raw, signature):
    require(TRUST_DIRECTORY.joinpath("owner.allowed_signers").is_file(), "project owner signing trust is not provisioned")
    command = ["ssh-keygen", "-Y", "verify", "-f", str(TRUST_DIRECTORY / "owner.allowed_signers"),
               "-I", SIGNER, "-n", NAMESPACE, "-s", str(signature)]
    result = subprocess.run(command, input=raw, capture_output=True, check=False)
    require(result.returncode == 0, "owner signature is invalid")


def approval(plan, matrix, raw, signature):
    require(signature is not None, "owner-signed approval is required before network access")
    verify_signature(raw, signature)
    require(isinstance(plan.get("owner_approval_record"), str) and plan["owner_approval_record"].strip(),
            "missing owner decision record")
    require(type(plan.get("task_budget_bytes")) is int and plan["task_budget_bytes"] >= 0, "invalid task budget")
    for item in matrix:
        require(item["remaining_bytes"] is not None, "unknown size: report and obtain approval with a verified size before transfer")
        if item["remaining_bytes"]:
            require(item.get("route_status") == "verified", "route and redirect chain must be verified before approval")
            index = item.get("direct_interface_index")
            require(type(index) is int and index > 0, "missing approved physical network interface index")
        require(re.fullmatch(r"[0-9a-f]{64}", item.get("sha256") or ""), "missing pinned SHA-256")
        require(type(item.get("file_budget_bytes")) is int and item["file_budget_bytes"] >= item["remaining_bytes"],
                "file budget is missing or insufficient")
        foreign = any(endpoint(url)[0] not in DOMESTIC_MIRRORS for url in [item["mirror"], *item["redirect_chain"]])
        if foreign:
            exception = item.get("owner_foreign_exception_record")
            require(isinstance(exception, str) and exception.strip() and exception != plan["owner_approval_record"],
                    "non-domestic endpoint requires a separate explicit owner decision bound by the signature")


class Ledger:
    """Persist conservative byte reservations across restarts and child processes."""
    def __init__(self, path, task_budget):
        self.path = path
        self.task_budget = task_budget
        self.state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"total": 0, "files": {}}

    def preflight(self, matrix):
        require(self.state["total"] + sum(item["remaining_bytes"] for item in matrix) <= self.task_budget,
                "task cumulative budget is insufficient")
        for item in matrix:
            used = self.state["files"].get(item["sha256"], 0)
            require(used + item["remaining_bytes"] <= item["file_budget_bytes"], "cumulative file budget is insufficient")

    def charge(self, key, size, file_budget, offset=None):
        total = self.state["total"] + size
        used = self.state["files"].get(key, 0) + size
        require(total <= self.task_budget and used <= file_budget, "approved cumulative download budget exhausted")
        self.state.update(total=total)
        self.state["files"][key] = used
        self.record("range_reserved", sha256=key, bytes=size, offset=offset)

    def record(self, event, **details):
        self.state.setdefault("events", []).append(
            {"utc": datetime.now(timezone.utc).isoformat(), "event": event, **details}
        )
        temp = self.path.with_suffix(".tmp")
        with temp.open("w", encoding="utf-8") as handle:
            json.dump(self.state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        temp.replace(self.path)


class DirectSession(requests.Session):
    def __init__(self):
        super().__init__()
        self.trust_env = False

    def resolve_redirects(self, *args, **kwargs):
        # Requests otherwise prepares the next redirect request and can drain its body.
        return iter(())


def verify_direct_route(host, approved_interface):
    """Fail closed if any current destination route leaves the approved physical adapter."""
    require(os.name == "nt", "managed downloads require the native Windows route check")
    addresses = {item[4][0] for item in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)}
    require(addresses, "no endpoint address available")
    script = (
        "$ErrorActionPreference='Stop'; "
        "$nic=Get-NetAdapter -InterfaceIndex $env:VICAD_APPROVED_INTERFACE; "
        "if (!$nic.HardwareInterface -or $nic.Status -ne 'Up') { exit 2 }; "
        "foreach ($ip in ($env:VICAD_ROUTE_IPS -split ',')) { "
        "$route=Find-NetRoute -RemoteIPAddress $ip | Where-Object { $_.DestinationPrefix }; "
        "if (@($route).Count -ne 1 -or $route.InterfaceIndex -ne $nic.ifIndex) { exit 3 } }"
    )
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
        env={**os.environ, "VICAD_APPROVED_INTERFACE": str(approved_interface),
             "VICAD_ROUTE_IPS": ",".join(sorted(addresses))},
        capture_output=True, check=False,
    )
    require(result.returncode == 0, "actual route diverges from the approved physical interface")


@contextmanager
def approved_request(session, item, headers, check_route):
    route = [item["mirror"], *item["redirect_chain"]]
    url = route[0]
    for i, allowed in enumerate(route):
        require(endpoint(url) == endpoint(allowed), "unapproved redirect endpoint")
        check_route(endpoint(url)[0], item["direct_interface_index"])
        with session.get(url, headers=headers, stream=True, timeout=(20, 60), allow_redirects=False) as response:
            if i < len(route) - 1:
                require(response.is_redirect and "Location" in response.headers, "approved redirect chain changed")
                url = urljoin(url, response.headers["Location"])
                continue
            require(not response.is_redirect, "unapproved final redirect")
            response.raise_for_status()
            yield response


def validate_range(response, start, stop, total):
    require(response.status_code == 206, "server ignored the byte range")
    require(response.headers.get("Content-Range") == f"bytes {start}-{stop - 1}/{total}", "unexpected Content-Range")
    length = response.headers.get("Content-Length")
    require(length is not None and int(length) == stop - start, "unexpected Content-Length")
    require(response.headers.get("Content-Encoding", "identity").lower() == "identity", "unexpected encoded body")


def assemble_local(target, revision, size, checksum):
    item = {"size_bytes": size, "sha256": checksum}
    if target.exists():
        verify_file(target, item)
        return target
    directory = range_cache(target, revision)
    complete = []
    for start, stop in spans(size):
        part, have = retained(directory, start, stop)
        require(have == stop - start, f"offline staging requires complete local range {start}-{stop - 1}")
        complete.append(part)
    pending = target.with_name(target.stem + ".assembling" + target.suffix)
    with pending.open("wb") as output:
        for part in complete:
            with part.open("rb") as source:
                shutil.copyfileobj(source, output, 4 * 1024 * 1024)
    verify_file(pending, item)
    pending.replace(target)
    return target


def transfer(session, item, ledger):
    target = local_path(item["destination"])
    size, revision = item["size_bytes"], item["version"]
    if target.exists():
        return
    directory = range_cache(target, revision)
    directory.mkdir(parents=True, exist_ok=True)
    for start, stop in spans(size):
        part, have = retained(directory, start, stop)
        if have == stop - start:
            continue
        offset = start + have
        ledger.charge(item["sha256"], stop - offset, item["file_budget_bytes"], offset)
        headers = {"Range": f"bytes={offset}-{stop - 1}", "Accept-Encoding": "identity"}
        with approved_request(session, item, headers, verify_direct_route) as response:
            validate_range(response, offset, stop, size)
            with part.open("ab") as output:
                for block in response.iter_content(256 * 1024):
                    require(offset + len(block) <= stop, "oversized response")
                    output.write(block)
                    offset += len(block)
            require(offset == stop, "truncated response; bounded reservation retained")
        part.replace(part.with_suffix(".part"))
        ledger.record("range_completed", sha256=item["sha256"], offset=start, bytes=stop - start)
    assemble_local(target, revision, size, item["sha256"])
    ledger.record("file_verified", name=item["name"], version=revision, sha256=item["sha256"], bytes=size)


def execute(raw, signature):
    plan = json.loads(raw)
    matrix = file_matrix(plan)
    approval(plan, matrix, raw, signature)
    # This workstation uses a single native Windows ledger for both Windows and WSL workflows.
    require("microsoft" not in platform.release().lower(), "run managed transfer on the native host; WSL stages offline")
    with FileLock(TRUST_DIRECTORY / "downloads.lock"):
        matrix = file_matrix(plan)
        ledger = Ledger(TRUST_DIRECTORY / f"{plan['task_id']}.json", plan["task_budget_bytes"])
        ledger.preflight(matrix)
        with DirectSession() as session:
            for item in matrix:
                transfer(session, item, ledger)


def describe(plan):
    matrix = file_matrix(plan) if plan.get("files") else []
    require(re.fullmatch(r"TASK-[0-9]{3}", plan.get("task_id", "")), "need canonical TASK-NNN ID")
    remaining = [item["remaining_bytes"] for item in matrix]
    known = all(n is not None for n in remaining)
    ledger_path = TRUST_DIRECTORY / f"{plan['task_id']}.json"
    spent = json.loads(ledger_path.read_text(encoding="utf-8"))["total"] if ledger_path.exists() else 0
    projection = spent + sum(remaining) if known else None
    return {"task_id": plan["task_id"], "files": matrix, "cumulative_reserved_bytes": spent,
            "projected_task_bytes": projection, "owner_threshold_triggered": not known or
            projection > LIMIT or any(item["size_bytes"] > LIMIT for item in matrix),
            "execution_requires_owner_signature": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--signature", type=Path)
    args = parser.parse_args()
    raw = args.plan.read_bytes()
    if args.execute:
        execute(raw, args.signature)
    else:
        print(json.dumps(describe(json.loads(raw)), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
