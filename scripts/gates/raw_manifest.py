"""Sha256 manifest for the untouched raw archive."""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

from gates.findings import Finding
from gates.records import RAW_ZIP, sha256_file
from private_data import locate, missing_message

MANIFEST = Path("data/raw/SHA256SUMS")
ZIP_NAME = "slm-datasets.zip"
_DIGEST = re.compile(r"^[0-9a-fA-F]{64}$")


def parse_manifest(text: str) -> dict[str, str]:
    listed: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        digest, name = stripped.split(None, 1)
        listed[name.strip().lstrip("*")] = digest
    return listed


def check_raw(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    zip_path = locate(root, RAW_ZIP.as_posix(), "SLM_DATASETS_ZIP")
    manifest_path = root / MANIFEST
    if not manifest_path.is_file():
        findings.append(
            Finding("raw", MANIFEST.as_posix(), "sha256 manifest is missing")
        )
        if not zip_path.is_file():
            findings.append(Finding("raw", RAW_ZIP.as_posix(), missing_message(RAW_ZIP.as_posix(), "SLM_DATASETS_ZIP")))
        return findings

    try:
        listed = parse_manifest(manifest_path.read_text(encoding="utf-8"))
    except ValueError:
        findings.append(
            Finding("raw", MANIFEST.as_posix(), "sha256 manifest line is not 'digest filename'")
        )
        return findings

    expected = listed.get(ZIP_NAME)
    if expected is None or _DIGEST.fullmatch(expected) is None:
        findings.append(
            Finding(
                "raw",
                MANIFEST.as_posix(),
                f"manifest must list {ZIP_NAME} with a sha256 digest",
            )
        )
        return findings

    if not zip_path.is_file():
        findings.append(Finding("raw", RAW_ZIP.as_posix(), missing_message(RAW_ZIP.as_posix(), "SLM_DATASETS_ZIP")))
        return findings

    actual = sha256_file(zip_path)
    if actual.lower() != expected.lower():
        findings.append(
            Finding(
                "raw",
                RAW_ZIP.as_posix(),
                f"sha256 mismatch: manifest {expected.lower()} archive {actual}",
            )
        )
    if not zipfile.is_zipfile(zip_path):
        findings.append(Finding("raw", RAW_ZIP.as_posix(), "raw archive is not a zip"))
    return findings
