"""Leak guard for paths listed in the committed offload manifest.

Always fails when a listed path is tracked by git or exists at its repo-relative
path, and when the manifest is missing or malformed. In private mode every listed
file must also resolve through ``private_data.locate`` and match bytes and sha256.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from gates.findings import Finding
from private_data import (
    OFFLOAD_MANIFEST,
    OffloadManifestError,
    locate,
    missing_message,
    offloaded_entries,
    private_mode,
)


def _tracked(root: Path) -> set[str]:
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", "--"],
            capture_output=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return set()
    if proc.returncode != 0:
        return set()
    return {part.decode("utf-8", "replace") for part in proc.stdout.split(b"\0") if part}


def check_offload_manifest(root: Path) -> list[Finding]:
    try:
        entries = offloaded_entries(root)
    except OffloadManifestError as exc:
        return [Finding("offload-manifest", OFFLOAD_MANIFEST.as_posix(), str(exc))]
    findings: list[Finding] = []
    tracked = _tracked(root)
    for entry in entries:
        rel = str(entry["path"])
        if rel in tracked or (root / rel).exists():
            findings.append(
                Finding(
                    "offload-manifest",
                    rel,
                    "offloaded path is present in the working tree or tracked by git",
                )
            )
    if not private_mode():
        return findings
    leaked = {finding.path for finding in findings}
    for entry in entries:
        rel = str(entry["path"])
        if rel in leaked:
            continue
        path = locate(root, rel)
        if not path.is_file():
            findings.append(Finding("offload-manifest", rel, missing_message(rel)))
            continue
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if len(data) != entry["bytes"] or digest != entry["sha256"]:
            findings.append(
                Finding(
                    "offload-manifest",
                    rel,
                    "offloaded file bytes or sha256 do not match the manifest",
                )
            )
    return findings
