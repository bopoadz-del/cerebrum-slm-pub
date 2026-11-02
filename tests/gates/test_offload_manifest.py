"""Offload manifest gate and NOT-RUN for absent private inputs. Synthetic files only."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from gates.absent import classify_absent
from gates.findings import Finding, NotRun
from gates.held_out_shards import check_held_out_shards
from gates.offload_manifest import check_offload_manifest
from gates.raw_manifest import check_raw
from gates.run import collect, main
from private_data import PRIVATE_ENV, is_offloaded, offloaded_paths, private_mode


def _clear(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in PRIVATE_ENV:
        monkeypatch.delenv(name, raising=False)


def _manifest(root: Path, files: list[dict]) -> None:
    path = root / "configs" / "private_data" / "offloaded_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": 1, "files": files}), encoding="utf-8")


def _entry(rel: str, payload: bytes) -> dict:
    return {
        "path": rel,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "category": "owner_data",
    }


def test_leak_guard_red_when_planted_green_when_removed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    rel = "data/private/planted.bin"
    payload = b"synthetic"
    _manifest(tmp_path, [_entry(rel, payload)])
    assert check_offload_manifest(tmp_path) == []
    target = tmp_path / rel
    target.parent.mkdir(parents=True)
    target.write_bytes(payload)
    found = check_offload_manifest(tmp_path)
    assert [item.path for item in found] == [rel]
    assert "working tree" in found[0].message
    target.unlink()
    assert check_offload_manifest(tmp_path) == []


def test_tracked_path_is_red_after_the_worktree_file_is_removed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    rel = "data/private/tracked.bin"
    payload = b"synthetic"
    target = tmp_path / rel
    target.parent.mkdir(parents=True)
    target.write_bytes(payload)
    _manifest(tmp_path, [_entry(rel, payload)])
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "add", "--", rel], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.email=ci@example.com", "-c", "user.name=ci", "commit", "-m", "add"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    target.unlink()
    assert not target.exists()
    found = check_offload_manifest(tmp_path)
    assert any(item.path == rel for item in found)


def test_private_mode_missing_and_mismatched_file_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    rel = "data/private/sample.bin"
    payload = b"synthetic-bytes"
    _manifest(tmp_path, [_entry(rel, payload)])
    external = tmp_path / "external"
    external.mkdir()
    monkeypatch.setenv("SLM_PRIVATE_DATA_DIR", str(external))
    assert private_mode()
    missing = check_offload_manifest(tmp_path)
    assert any(item.path == rel and "absent" in item.message for item in missing)
    dest = external / rel
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"other-bytes")
    mismatched = check_offload_manifest(tmp_path)
    assert any(item.path == rel and "sha256" in item.message for item in mismatched)
    dest.write_bytes(payload)
    assert check_offload_manifest(tmp_path) == []


def test_missing_and_malformed_manifest_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    missing = check_offload_manifest(tmp_path)
    assert missing and "missing" in missing[0].message
    path = tmp_path / "configs" / "private_data" / "offloaded_manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text("{", encoding="utf-8")
    malformed = check_offload_manifest(tmp_path)
    assert malformed and "malformed" in malformed[0].message


def test_public_absent_manifest_path_is_not_run_and_other_paths_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear(monkeypatch)
    held = "data/held_out/shards.json"
    _manifest(tmp_path, [_entry(held, b"synthetic")])
    assert is_offloaded("data/held_out", tmp_path)
    assert is_offloaded(held, tmp_path)
    assert not is_offloaded("data/held_out/extra.json", tmp_path)
    not_run = check_held_out_shards(tmp_path)
    assert len(not_run) == 1 and isinstance(not_run[0], NotRun)
    assert not_run[0].path == "data/held_out"
    assert not_run[0].env == "SLM_HELD_OUT_DIR"

    other = tmp_path / "other"
    other.mkdir()
    _manifest(other, [_entry("data/other/file.bin", b"synthetic")])
    failed = check_held_out_shards(other)
    assert len(failed) == 1 and isinstance(failed[0], Finding)
    assert failed[0].path == "data/held_out"
    assert "absent" in failed[0].message


def test_public_absent_zip_is_not_run_private_mode_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    (raw / "SHA256SUMS").write_text("ab" * 32 + "  slm-datasets.zip\n", encoding="utf-8")
    rel = "data/raw/slm-datasets.zip"
    _manifest(tmp_path, [_entry(rel, b"synthetic")])
    not_run = check_raw(tmp_path)
    assert len(not_run) == 1 and isinstance(not_run[0], NotRun)
    assert not_run[0].path == rel
    assert not_run[0].env == "SLM_DATASETS_ZIP"

    monkeypatch.setenv("SLM_DATASETS_ZIP", str(tmp_path / "missing.zip"))
    failed = check_raw(tmp_path)
    assert any(isinstance(item, Finding) and item.path == rel and "absent" in item.message for item in failed)
    assert not any(isinstance(item, NotRun) for item in failed)

    bare = tmp_path / "bare"
    (bare / "data" / "raw").mkdir(parents=True)
    (bare / "data" / "raw" / "SHA256SUMS").write_text("ab" * 32 + "  slm-datasets.zip\n", encoding="utf-8")
    _manifest(bare, [_entry("data/other/file.bin", b"synthetic")])
    _clear(monkeypatch)
    other = check_raw(bare)
    assert any(isinstance(item, Finding) and item.path == rel for item in other)
    assert not any(isinstance(item, NotRun) for item in other)


def test_classify_absent_private_mode_is_a_finding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    _manifest(tmp_path, [_entry("data/held_out/shards.json", b"synthetic")])
    monkeypatch.setenv("SLM_PRIVATE_ROOT", str(tmp_path / "external"))
    result = classify_absent(tmp_path, "held-out-shards", "data/held_out", "SLM_HELD_OUT_DIR")
    assert isinstance(result, Finding)


def test_public_repo_lists_not_run_and_stays_green(repo_root: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    findings, not_run = collect(repo_root)
    assert findings == []
    assert not_run
    assert {item.gate for item in not_run} >= {"held-out-shards", "raw"}
    assert main(["--root", str(repo_root), "--skip-proof"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert "NOT-RUN [held-out-shards] data/held_out set SLM_HELD_OUT_DIR" in captured.out
    assert "NOT-RUN [raw] data/raw/slm-datasets.zip set SLM_DATASETS_ZIP" in captured.out
    assert f"gates: green ({len(not_run)} not run: private data absent)" in captured.out
    assert offloaded_paths(repo_root)
