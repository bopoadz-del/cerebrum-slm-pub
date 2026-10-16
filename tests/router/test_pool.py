import json

from router.pool import integrity, normalize_row


def test_normalize_drops_train_label_and_duplicate_text() -> None:
    row = normalize_row(
        {
            "doc_id": "d",
            "project_id": "p",
            "filename": "drawing.pdf",
            "first_512": "excerpt words",
            "text": "excerpt words",
            "declared_type": "document",
            "label": "document",
            "content_sha256": None,
            "doc_id_source": "content_sha256",
        }
    )
    assert row["first_512_tokens"] == "excerpt words"
    assert row["fork_declared_type"] == "document"
    assert row["filename"] == "drawing.pdf"
    assert "label" not in row
    assert "text" not in row
    assert "declared_type" not in row
    assert "ingested_at" not in row
    assert row["content_sha256"] is None


def test_filename_does_not_set_the_reference_type() -> None:
    row = normalize_row({"doc_id": "d", "filename": "schedule.pdf", "first_512_tokens": "body"})
    assert row["fork_declared_type"] is None
    assert row["first_512_tokens"] == "body"


def test_integrity_keeps_duplicate_rows() -> None:
    rows = [
        normalize_row({"doc_id": "same", "project_id": "a", "first_512": "one", "declared_type": "document"}),
        normalize_row({"doc_id": "same", "project_id": "b", "first_512": "two", "declared_type": "document"}),
        normalize_row({"doc_id": "other", "project_id": "a", "first_512": "three", "declared_type": "boq"}),
    ]
    report = integrity(rows)
    assert report["rows"] == 3
    assert report["distinct_doc_id"] == 2
    assert report["duplicate_doc_ids"] == 1
    assert report["extra_rows"] == 1
    assert report["doc_id_groups_with_excerpt_conflict"] == 1
    assert report["doc_id_groups_with_project_conflict"] == 1
    assert report["fork_declared_type_counts"]["document"] == 2
    payload = json.dumps(report)
    assert "one" not in payload
    assert "two" not in payload
