"""Encoded-bytes content rule: text that is mostly encoded bytes has no content."""
import pytest

import base64
import json
import random
from pathlib import Path

from router.content import ENCODED_RATIO_MAX, ENCODED_RUN_MIN, encoded_ratio, is_mostly_encoded
from router.train_router import training_examples


def _b64(n: int, seed: int = 0) -> str:
    rng = random.Random(seed)
    return base64.b64encode(bytes(rng.randrange(256) for _ in range(n))).decode()


def test_threshold_is_stated_and_generic() -> None:
    assert ENCODED_RATIO_MAX == 0.5
    assert ENCODED_RUN_MIN == 64


def test_base64_and_hex_dumps_are_encoded() -> None:
    assert is_mostly_encoded(_b64(3000))
    assert is_mostly_encoded("PK header " + _b64(2000))
    hexdump = "".join(random.Random(1).choice("0123456789abcdef") for _ in range(4000))
    assert is_mostly_encoded(hexdump)


def test_document_text_is_not_encoded() -> None:
    prose = "Monthly progress report. Works on level 3 slab are 80% complete; see RFI-0042 and DE4-B-MU-01. " * 40
    assert encoded_ratio(prose) == 0.0
    assert not is_mostly_encoded(prose)
    assert not is_mostly_encoded("混凝土浇筑检查记录，第三层楼板。" * 200)
    assert not is_mostly_encoded("Báo cáo không phù hợp, hạng mục kết cấu. " * 100)
    url = "https://example.org/" + "a" * 70
    assert not is_mostly_encoded(("Site visit notes and actions. " * 10) + url)


def test_ratio_boundary() -> None:
    run = "A" * 100
    words = "word " * 25  # 100 visible characters
    assert encoded_ratio(run + " " + words) == 0.5
    assert not is_mostly_encoded(run + " " + words)
    assert is_mostly_encoded(run + "A " + words)


def test_builder_drops_encoded_rows_only() -> None:
    rows = [
        {"label": "other", "excerpt": _b64(3000), "doc_id": "a"},
        {"label": "report", "excerpt": "Monthly progress report for the works.", "doc_id": "b"},
    ]
    examples = training_examples(rows, ["other", "report"])
    assert [e["doc_id"] for e in examples] == ["b"]


@pytest.mark.private_data(
    'data/router/train.jsonl',
    'data/router/val.jsonl',
)
def test_committed_train_and_val_hold_no_encoded_rows(repo_root: Path) -> None:
    for split in ("train", "val"):
        path = repo_root / "data" / "router" / f"{split}.jsonl"
        bad = [
            index
            for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
            if line.strip() and is_mostly_encoded(json.loads(line)["messages"][1]["content"])
        ]
        assert bad == [], f"{split}: encoded rows at lines {bad[:5]}"
