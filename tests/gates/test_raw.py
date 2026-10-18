from __future__ import annotations
import pytest

import shutil
from pathlib import Path

from gates.raw_manifest import check_raw
from gates.records import sha256_file


@pytest.mark.private_data('data/raw/slm-datasets.zip')
def test_archive_matches_sha256_manifest(repo_root: Path) -> None:
    findings = check_raw(repo_root)
    assert findings == []
    digest = sha256_file(repo_root / "data" / "raw" / "slm-datasets.zip")
    manifest = (repo_root / "data" / "raw" / "SHA256SUMS").read_text(encoding="utf-8")
    assert digest in manifest
    assert "slm-datasets.zip" in manifest


@pytest.mark.private_data('data/raw/slm-datasets.zip')
def test_manifest_mismatch_is_red(tmp_path: Path, repo_root: Path) -> None:
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    shutil.copy(repo_root / "data" / "raw" / "slm-datasets.zip", raw / "slm-datasets.zip")
    (raw / "SHA256SUMS").write_text("0" * 64 + "  slm-datasets.zip\n", encoding="utf-8")
    findings = check_raw(tmp_path)
    assert findings
    assert any("sha256 mismatch" in finding.message for finding in findings)
