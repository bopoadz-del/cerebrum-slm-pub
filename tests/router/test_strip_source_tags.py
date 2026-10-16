import json
from pathlib import Path

import pytest

from router.strip_source_tags import clean_file, clean_value, has_path, main, strip_source_tags


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (r"[source: G:\My Drive\Some Folder\Pack 1\PDF\AB-CD-001-02.pdf] AB-CD-001 -- KEY PLAN", "AB-CD-001 -- KEY PLAN"),
        (r"[source: X:/a/b.pdf]body", "body"),
        (r"one [source: Q:\x\y.pdf] two [source: Q:\x\z.pdf] three", "one two three"),
        (r"IS SUBJECT TO AND[source: G:\My Drive\Folder A\Design dr", "IS SUBJECT TO AND"),
        (r"cut-off date[source: G:\My Drive\Fo", "cut-off date"),
        (r"[SOURCE : D:\p\q.docx] Letter", "Letter"),
        (r"see C:\Users\someone\Documents\Final Report v2.docx for details", "see for details"),
        (r"see C:\temp\notes for details", "see for details"),
        ("stored at /home/user/data/file.pdf ok", "stored at ok"),
        ("cfg ~/x/y/z.txt ok", "cfg ok"),
        ("rel ./a/b.json ok", "rel ok"),
        ("deep /srv/one/two/three ok", "deep ok"),
        ("source: X:/Dubai/f.pdf rest", "rest"),
        (r"unc \\server\share\folder\doc name.pdf next", "unc next"),
        ("uri file:///tmp/a.pdf ok", "uri ok"),
    ],
)
def test_strips_tags_and_paths(text: str, expected: str) -> None:
    out = strip_source_tags(text)
    assert out == expected
    assert not has_path(out)


@pytest.mark.parametrize(
    "text",
    [
        "Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
        "and/or 12/05/2024 kN/m/m N/A 1/2/3",
        "https://example.com/a/b/c.pdf and www.site.org/x/y/z",
        "Ref DGDA/AB2/INF/0123 dated 3 May",
        "Sub-Clause 20.1 [Contractor's Claims] applies",
        "ratio 3:1 at 10:30 hrs",
        "2 Roof cap /Joint/trims/Ridge cap 7.0%",
        "Remove all small signs for B3 /2/1/ etc",
    ],
)
def test_leaves_content_alone(text: str) -> None:
    assert strip_source_tags(text) == text
    assert not has_path(text)


def test_clean_value_skips_identifier_keys() -> None:
    row = {
        "doc_id": "abc",
        "filename": "x.pdf",
        "excerpt": r"[source: G:\d\x.pdf] body",
        "messages": [{"role": "user", "content": r"[source: G:\d\x.pdf] body"}],
    }
    out = clean_value(row)
    assert out["doc_id"] == "abc"
    assert out["filename"] == "x.pdf"
    assert out["excerpt"] == "body"
    assert out["messages"][0]["content"] == "body"


def test_clean_file_and_check(tmp_path: Path) -> None:
    src = tmp_path / "rows.jsonl"
    src.write_text(
        json.dumps({"excerpt": r"[source: G:\a\b.pdf] text"}) + "\n" + json.dumps({"excerpt": "plain"}) + "\n",
        encoding="utf-8",
    )
    rep = clean_file(src, tmp_path / "rows_clean.jsonl")
    assert rep == {"file": "rows.jsonl", "out": "rows_clean.jsonl", "rows": 2, "rows_changed": 1, "rows_still_matching": 0}
    assert main(["--check", str(tmp_path / "rows_clean.jsonl")]) == 0
    assert main(["--check", str(src)]) == 1
