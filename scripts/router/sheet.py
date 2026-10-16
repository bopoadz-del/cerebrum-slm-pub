"""Practitioner tick sheets. One column per declared label, plus unsure."""

from __future__ import annotations

import csv
from pathlib import Path


UNSURE = "unsure"
ID_COLUMN = "sheet_id"
EXCERPT_COLUMN = "excerpt"
FILENAME_COLUMN = "filename_optional_blank"


def header(labels: list[str]) -> list[str]:
    if UNSURE in labels:
        raise ValueError("unsure is an abstain column, not a declared label")
    return [ID_COLUMN, EXCERPT_COLUMN, FILENAME_COLUMN, *labels, UNSURE]


def blank_row(sheet_id: str, labels: list[str], excerpt: str = "") -> list[str]:
    """A tick row. The filename column stays blank so it cannot be used as a label."""
    return [sheet_id, excerpt, "", *[""] * len(labels), ""]


def sealed_ids(n: int) -> list[str]:
    if n < 1:
        raise ValueError("sealed sheet needs at least one row")
    return [f"SEAL-{index:03d}" for index in range(1, n + 1)]


def write_sheet(path: Path, labels: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header(labels))
        writer.writerows(rows)


def sealed_rows(labels: list[str], n: int) -> list[list[str]]:
    return [blank_row(sheet_id, labels) for sheet_id in sealed_ids(n)]
