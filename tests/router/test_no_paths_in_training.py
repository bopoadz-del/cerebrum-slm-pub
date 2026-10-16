"""Guard: no training row may carry a file path or a [source: ...] tag."""

import json
from pathlib import Path

import pytest

from router.content import content_text, model_user_text
from router.guard import RouterWall
from router.pool import normalize_row
from router.strip_source_tags import row_has_path
from router.train_router import require_no_paths, training_examples

TRAINING_FILES = (
    "data/router/teacher_labels.jsonl",
    "data/router/train.jsonl",
    "data/router/val.jsonl",
    "data/router/additions_owner_laptop.jsonl",
    "data/router/additions_owner_sources_pr2.jsonl",
    "data/router/additions_change_request_v9.jsonl",
    "data/router/additions_thin_v9.jsonl",
    "data/router/additions_thin_v9b.jsonl",
    "data/router/additions_thin_v9f.jsonl",
    "data/router/v5_snapshot/teacher_labels.jsonl",
    "data/router/v5_snapshot/train.jsonl",
    "data/router/v5_snapshot/val.jsonl",
)


@pytest.mark.parametrize("relpath", TRAINING_FILES)
def test_committed_training_rows_hold_no_path(repo_root: Path, relpath: str) -> None:
    from private_data import locate
    from private_paths import skip_if_missing

    skip_if_missing(repo_root, relpath)
    path = locate(repo_root, relpath)
    bad = [
        index
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if line.strip() and row_has_path(json.loads(line))
    ]
    assert bad == [], f"{relpath}: {len(bad)} rows match a path pattern, first lines {bad[:5]}"


def test_raw_pool_excerpts_hold_no_path(repo_root: Path) -> None:
    from private_data import locate
    from private_paths import skip_if_missing

    skip_if_missing(repo_root, "data/raw/router/labels.jsonl")
    path = locate(repo_root, "data/raw/router/labels.jsonl")
    bad = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            text = json.loads(line).get("first_512_tokens")
            bad += row_has_path({"first_512_tokens": text})
    assert bad == 0


def test_guard_refuses_a_row_with_a_path() -> None:
    rows = [{"messages": [{"role": "user", "content": r"[source: G:\a\b.pdf] body"}]}]
    with pytest.raises(RouterWall, match="file path"):
        require_no_paths(rows)
    with pytest.raises(RouterWall):
        require_no_paths([{"messages": [{"role": "user", "content": "see /srv/data/x/report.pdf"}]}])
    assert require_no_paths([{"messages": [{"role": "user", "content": "plain body"}]}])


def test_training_examples_strip_the_tag() -> None:
    rows = [{"label": "report", "excerpt": r"[source: G:\My Drive\Folder\r.pdf] Monthly progress report"}]
    examples = training_examples(rows, ["report"])
    assert examples[0]["messages"][1]["content"] == "Monthly progress report"


def test_teacher_input_and_pool_normalise_strip_the_tag() -> None:
    row = {"first_512_tokens": r"[source: X:/a/b.pdf] Letter body"}
    assert content_text(row) == "Letter body"
    assert model_user_text(row) == "Letter body"
    assert content_text({"first_512_tokens": r"[source: X:/a/b.pdf]"}) is None
    assert normalize_row({"doc_id": "d", "first_512_tokens": r"[source: X:\a\b.pdf] body"})["first_512_tokens"] == "body"
