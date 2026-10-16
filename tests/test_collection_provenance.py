"""Collection-provenance rule tests — structural, content-blind."""

from pathlib import Path

from router.collection_provenance import (
    NO_PROVENANCE,
    REQUIRED_FIELDS,
    validate_file,
    validate_row,
)


def _full_row() -> dict:
    return {
        "doc_id": "d" * 64,
        "file_sha256": "f" * 64,
        "source_url": "https://example.gov/meeting/attachment-001.pdf",
        "fetch_timestamp": "2026-10-07T19:24:54Z",
        "http_status": 200,
        "content_sha256": "c" * 64,
        "page_title": "Example meeting packet",
        "excerpt": "some text",
        "label": "report",
    }


def test_complete_row_passes():
    assert validate_row(_full_row(), 1).ok


def test_missing_url_fails():
    row = _full_row()
    del row["source_url"]
    res = validate_row(row, 2)
    assert not res.ok and res.missing == ("source_url",)


def test_all_required_fields_are_checked():
    row = _full_row()
    for field in REQUIRED_FIELDS:
        del row[field]
    res = validate_row(row, 3)
    assert set(res.missing) == set(REQUIRED_FIELDS)


def test_no_provenance_flag_allows_legacy_row():
    row = _full_row()
    for field in REQUIRED_FIELDS:
        del row[field]
    row["provenance_status"] = NO_PROVENANCE
    res = validate_row(row, 4)
    assert res.ok and res.legacy


def test_bad_field_shapes_fail():
    row = _full_row()
    row["source_url"] = "not-a-url"
    row["http_status"] = "200"  # string, not int
    row["content_sha256"] = "short"
    res = validate_row(row, 5)
    assert set(res.missing) == {"source_url", "http_status", "content_sha256"}


def test_file_validation_reports_rows(tmp_path: Path):
    good = _full_row()
    bad = _full_row()
    del bad["page_title"]
    legacy = _full_row()
    for f in REQUIRED_FIELDS:
        del legacy[f]
    legacy["provenance_status"] = NO_PROVENANCE
    p = tmp_path / "additions.jsonl"
    import json
    p.write_text(
        "\n".join(json.dumps(r) for r in (good, bad, legacy)) + "\n",
        encoding="utf-8",
    )
    res = validate_file(p)
    assert res.rows == 3
    assert res.legacy_rows == 1
    assert len(res.violations) == 1
    assert res.violations[0].row_index == 2
