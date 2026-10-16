"""Generic addendum sealing: unsure rows are held (never sealed), provenance is required, isolation failures are held,
the per-(project, class) cap counts existing sealed rows, and exhausted pairs are derived from data only."""

import hashlib
import json
from pathlib import Path

import pytest

from router.guard import RouterWall
from router.labels import load_labels
from router.seal_addendum import (
    check_against_exhaustion,
    classify,
    enforce_addendum_cap,
    exhausted_pairs,
    pair_counts,
    write_addendum,
)


def _sheet(path: Path, labels: list[str], rows: list[dict[str, str]]) -> Path:
    columns = ["id", "file name", "excerpt", *labels, "unsure", "notes", "owner_tick"]
    lines = [",".join(columns)]
    for row in rows:
        lines.append(",".join(row.get(column, "") for column in columns))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _meta(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _url(host: str, first: str, n: int) -> str:
    return f"https://{host}/{first}/doc{n}.pdf"


def test_unsure_row_is_held_with_its_tentative_tick_and_never_sealed(tmp_path: Path, repo_root: Path) -> None:
    labels = load_labels(repo_root / "configs" / "router" / "labels.json")
    sheet = _sheet(
        tmp_path / "s.csv",
        labels,
        [
            {"id": "A1", "file name": f"{labels[0]}.pdf", "excerpt": "body one", labels[3]: "True", "unsure": "False", "notes": "clear"},
            {"id": "A2", "excerpt": "body two", labels[4]: "True", "unsure": "True", "notes": "owner not sure"},
        ],
    )
    meta = _meta(tmp_path / "m.jsonl", [
        {"sheet_id": "A1", "content_sha256": "a" * 64, "isolation_key": "aaaaaaaa", "source_url": _url("h.example", "x", 1)},
        {"sheet_id": "A2", "content_sha256": "b" * 64, "isolation_key": "bbbbbbbb", "source_url": _url("h.example", "x", 2)},
    ])
    sealed, held, log = classify([sheet], labels, [meta])
    assert [r["sheet_id"] for r in sealed] == ["A1"]
    assert sealed[0]["label"] == labels[3]  # the tick, never the file name
    assert sealed[0]["source_url"] and sealed[0]["file_sha256"] == "a" * 64
    assert held == [
        {
            "sheet_id": "A2",
            "file_sha256": "b" * 64,
            "isolation_key": "bbbbbbbb",
            "source_url": _url("h.example", "x", 2),
            "owner_notes": "owner not sure",
            "held_reason": "unsure",
            "owner_tentative_tick": labels[4],
        }
    ]
    assert len(log) == 1 and log[0].startswith("HELD A2")


def test_two_label_ticks_or_no_tick_is_a_wall(tmp_path: Path, repo_root: Path) -> None:
    labels = load_labels(repo_root / "configs" / "router" / "labels.json")
    meta = _meta(tmp_path / "m.jsonl", [{"sheet_id": "A1", "content_sha256": "a" * 64, "source_url": _url("h", "x", 1)}])
    two = _sheet(tmp_path / "two.csv", labels, [{"id": "A1", labels[0]: "True", labels[1]: "True", "unsure": "True"}])
    with pytest.raises(RouterWall, match="more than one label tick"):
        classify([two], labels, [meta])
    none = _sheet(tmp_path / "none.csv", labels, [{"id": "A1", "unsure": "False"}])
    with pytest.raises(RouterWall, match="exactly one tick"):
        classify([none], labels, [meta])


def test_row_without_url_or_sha_provenance_is_held(tmp_path: Path, repo_root: Path) -> None:
    labels = load_labels(repo_root / "configs" / "router" / "labels.json")
    sheet = _sheet(tmp_path / "s.csv", labels, [
        {"id": "A1", labels[2]: "True", "unsure": "False"},
        {"id": "A2", labels[2]: "True", "unsure": "False"},
        {"id": "A3", labels[2]: "True", "unsure": "False"},
    ])
    meta = _meta(tmp_path / "m.jsonl", [
        {"sheet_id": "A1", "content_sha256": "a" * 64},
        {"sheet_id": "A2", "source_url": _url("h", "x", 2)},
        {"sheet_id": "A3", "content_sha256": "c" * 64, "source_url": _url("h", "x", 3)},
    ])
    sealed, held, log = classify([sheet], labels, [meta])
    assert [r["sheet_id"] for r in sealed] == ["A3"]
    assert [r["sheet_id"] for r in held] == ["A1", "A2"]
    assert all(r["held_reason"].startswith("no provenance") for r in held)
    assert len(log) == 2


def test_isolation_failure_is_held_not_swapped(tmp_path: Path, repo_root: Path) -> None:
    labels = load_labels(repo_root / "configs" / "router" / "labels.json")
    sheet = _sheet(tmp_path / "s.csv", labels, [{"id": "A1", labels[2]: "True"}, {"id": "A2", labels[2]: "True"}])
    meta = _meta(tmp_path / "m.jsonl", [
        {"sheet_id": "A1", "content_sha256": "a" * 64, "source_url": _url("h", "x", 1)},
        {"sheet_id": "A2", "content_sha256": "b" * 64, "source_url": _url("h", "x", 2)},
    ])
    sealed, held, _ = classify([sheet], labels, [meta], {"A2": ["train_sha_hit"]})
    assert [r["sheet_id"] for r in sealed] == ["A1"]
    assert held[0]["sheet_id"] == "A2" and "train_sha_hit" in held[0]["held_reason"]


def _row(label: str, host: str, first: str, n: int) -> dict:
    return {"sheet_id": f"{host}-{n}", "label": label, "file_sha256": hashlib.sha256(f"{host}{first}{n}".encode()).hexdigest(),
            "project": "web_public", "source_url": _url(host, first, n)}


def test_cap_counts_existing_sealed_rows_and_legacy_pairs_take_no_new_row() -> None:
    existing = [_row("NCR", "p.example", "docs", i) for i in range(2)] + [_row("BOQ", "q.example", "home", i) for i in range(5)]
    enforce_addendum_cap([_row("NCR", "p.example", "docs", 10)], existing)  # 2 + 1 = 3: allowed
    with pytest.raises(RouterWall, match="limit 3"):
        enforce_addendum_cap([_row("NCR", "p.example", "docs", 10), _row("NCR", "p.example", "docs", 11)], existing)
    with pytest.raises(RouterWall, match="limit 3"):
        enforce_addendum_cap([_row("BOQ", "q.example", "home", 99)], existing)
    enforce_addendum_cap([_row("BOQ", "p.example", "docs", 7)], existing)  # other project: fine
    with pytest.raises(RouterWall):
        enforce_addendum_cap([_row("RFI", "r.example", "u", i) for i in range(4)], [])


def test_exhausted_pairs_come_from_data_and_block_later_batches() -> None:
    rows = [_row("NCR", "p.example", "docs", i) for i in range(3)] + [_row("BOQ", "q.example", "home", i) for i in range(5)]
    rows += [_row("RFI", "r.example", "u", 0)]
    rows.append(dict(rows[0]))  # the same document registered twice counts once
    assert pair_counts(rows)[("p.example/docs", "NCR")] == 3
    record = exhausted_pairs(rows)
    assert record == [
        {"project": "p.example/docs", "label": "NCR", "count": 3, "status": "at_cap"},
        {"project": "q.example/home", "label": "BOQ", "count": 5, "status": "over_cap_legacy"},
    ]
    later = [_row("NCR", "p.example", "docs", 50), _row("RFI", "p.example", "docs", 51), _row("BOQ", "q.example", "home", 52)]
    assert [r["sheet_id"] for r in check_against_exhaustion(later, record)] == ["p.example-50", "q.example-52"]


def test_write_addendum_seals_match_their_files(tmp_path: Path) -> None:
    out = tmp_path / "add"
    names = write_addendum(out, [_row("NCR", "p", "d", 1)], [{"sheet_id": "x", "held_reason": "unsure"}], {"isolation.json": {"rows": 1}})
    assert names == ["sealed_labels.jsonl", "held.jsonl", "isolation.json"]
    sums = []
    for name in names:
        raw = (out / name).read_bytes()
        assert b"\r\n" not in raw
        line = f"{hashlib.sha256(raw).hexdigest()}  {name}\n"
        assert (out / f"{name}.sha256").read_text(encoding="utf-8") == line
        sums.append(line)
    assert (out / "SHA256SUMS").read_text(encoding="utf-8") == "".join(sums)
