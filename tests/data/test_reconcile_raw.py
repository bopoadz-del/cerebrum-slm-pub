"""Every raw line lands in exactly one destination, and the committed table is current."""

from __future__ import annotations
import pytest

import importlib.util
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "data" / "reconcile_raw.py"
DESTINATIONS = {"train", "held_out", "regression", "retrieval", "dropped"}


def _load():
    spec = importlib.util.spec_from_file_location("reconcile_raw", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reconcile_raw = _load()


@pytest.mark.private_data(
    'data/raw/slm-datasets.zip',
    'data/held_out/pairs.jsonl',
)
def test_every_line_has_one_destination_and_dropped_rows_have_a_reason():
    rows, summary = reconcile_raw.reconcile(ROOT)
    lines = [row for row in rows if row["kind"] != "file"]
    assert lines
    assert all(row["destination"] in DESTINATIONS for row in lines)
    assert all(row["reason"] for row in lines)
    refs = Counter((row["member"], row["line"]) for row in lines)
    assert max(refs.values()) == 1


@pytest.mark.private_data(
    'data/raw/slm-datasets.zip',
    'data/held_out/pairs.jsonl',
)
def test_split_files_are_fully_accounted_for():
    rows, summary = reconcile_raw.reconcile(ROOT)
    pairs = [row for row in rows if row["kind"] == "pair"]
    for split in ("train", "held_out", "regression"):
        traced = {row["content_hash"] for row in pairs if row["destination"] == split}
        assert len(traced) == summary["split_counts"][split]
    facts = {row["content_hash"] for row in pairs if row["destination"] == "retrieval"}
    assert len(facts) == summary["split_counts"]["retrieval_fact"]


@pytest.mark.private_data(
    'data/raw/slm-datasets.zip',
    'data/held_out/pairs.jsonl',
)
def test_committed_table_is_current():
    assert reconcile_raw.main(["--root", str(ROOT), "--check"]) == 0


@pytest.mark.private_data(
    'data/raw/slm-datasets.zip',
    'data/held_out/pairs.jsonl',
)
def test_table_carries_no_item_text():
    table = (ROOT / reconcile_raw.OUT / "raw_lines.jsonl").read_text(encoding="utf-8")
    for line in table.splitlines():
        row = json.loads(line)
        assert set(row) == {"content_hash", "destination", "id", "kind", "line", "member", "reason"}
