"""collection_provenance --discover: NOT-RUN, FAIL, and validated. Synthetic rows only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from private_data import PRIVATE_ENV
from router.collection_provenance import main

REL = "data/router/additions_sample.jsonl"


def _clear(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in PRIVATE_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("SLM_OWNER_ADDITIONS", raising=False)


def _manifest(root: Path, paths: list[str]) -> None:
    path = root / "configs" / "private_data" / "offloaded_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    files = [{"path": rel, "bytes": 4, "sha256": "ab" * 32, "category": "owner_data"} for rel in paths]
    path.write_text(json.dumps({"schema": 1, "files": files}), encoding="utf-8")


def _row() -> dict:
    return {
        "source_url": "https://example.gov/meeting/attachment-001.pdf",
        "fetch_timestamp": "2026-10-07T19:24:54Z",
        "http_status": 200,
        "content_sha256": "c" * 64,
        "page_title": "Example meeting packet",
    }


def _write_additions(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_row()) + "\n", encoding="utf-8")


def test_discover_public_absent_manifest_file_is_not_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _clear(monkeypatch)
    _manifest(tmp_path, [REL])
    assert main(["--discover", "--root", str(tmp_path)]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert f"NOT-RUN [collection-provenance] {REL} set SLM_PRIVATE_DATA_DIR" in captured.out
    assert "provenance: green (1 not run: private data absent)" in captured.out


def test_discover_validates_in_tree_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _clear(monkeypatch)
    _manifest(tmp_path, [REL])
    _write_additions(tmp_path / REL)
    assert main(["--discover", "--root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert f"{REL}: 1 rows, 0 no-provenance" in out
    assert "NOT-RUN" not in out
    assert "provenance: green (0 not run: private data absent)" in out


def test_explicit_absent_path_fails_even_when_not_in_the_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _clear(monkeypatch)
    _manifest(tmp_path, [REL])
    missing = "data/router/not_listed.jsonl"
    assert main(["--discover", "--root", str(tmp_path), missing]) == 1
    captured = capsys.readouterr()
    assert missing in captured.err
    assert "absent" in captured.err
    assert "NOT-RUN [collection-provenance]" in captured.out


def test_explicit_absent_manifest_path_still_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _clear(monkeypatch)
    _manifest(tmp_path, [REL])
    assert main(["--root", str(tmp_path), REL]) == 1
    captured = capsys.readouterr()
    assert "absent" in captured.err
    assert "NOT-RUN" not in captured.out
    assert "provenance: green" not in captured.out


def test_discover_private_mode_missing_fails_and_located_file_validates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _clear(monkeypatch)
    external = tmp_path / "external"
    external.mkdir()
    monkeypatch.setenv("SLM_PRIVATE_DATA_DIR", str(external))
    _manifest(tmp_path, [REL])
    assert main(["--discover", "--root", str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert REL in captured.err and "absent" in captured.err
    assert "NOT-RUN" not in captured.out

    _write_additions(external / REL)
    assert main(["--discover", "--root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert f"{REL}: 1 rows, 0 no-provenance" in out
    assert "provenance: green (0 not run: private data absent)" in out
