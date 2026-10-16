"""Fail-closed sealed/private path resolution (no file contents)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from scripts.private_data import PrivateDataError, env_for, require


ROOT = Path(__file__).resolve().parents[1]


def test_env_for_sealed_uses_sealed_dir() -> None:
    assert env_for("data/eval/router_unseen/sealed_labels.jsonl") == "SLM_SEALED_DIR"


def test_require_sealed_fails_closed_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "SLM_SEALED_DIR",
        "SLM_PRIVATE_DATA_DIR",
        "SLM_PRIVATE_ROOT",
        "SLM_HELD_OUT_DIR",
    ):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(PrivateDataError) as exc:
        require(ROOT, "data/eval/router_unseen/sealed_labels.jsonl")
    msg = str(exc.value)
    assert "sealed_labels.jsonl" in msg
    assert "SLM_SEALED_DIR" in msg
    # no absolute hardcoded machine path required
    assert "/home/" not in msg


def test_require_sealed_resolves_under_sealed_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target = tmp_path / "sealed_labels.jsonl"
    target.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("SLM_SEALED_DIR", str(tmp_path))
    for key in ("SLM_PRIVATE_DATA_DIR", "SLM_PRIVATE_ROOT"):
        monkeypatch.delenv(key, raising=False)
    got = require(ROOT, "data/eval/router_unseen/sealed_labels.jsonl")
    assert got == target
