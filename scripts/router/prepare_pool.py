"""Prepare the raw pool for teacher labelling.

Drops empty excerpts and duplicate doc ids. Does not assign a label.
"""

from __future__ import annotations

from dataclasses import dataclass

from router.content import content_text
from router.guard import RouterWall
from router.isolation import Exclusions, rejected_rows


@dataclass(frozen=True)
class PoolPrep:
    rows: list[dict]
    input_rows: int
    empty_text: int
    duplicate_doc_ids: int
    sealed_overlaps: int
    kept: int


def prepare_pool(rows: list[dict], exclusions: Exclusions) -> PoolPrep:
    """Keep one content row per doc_id. Refuse a sealed sha256 or project."""
    kept: list[dict] = []
    seen: set[str] = set()
    empty = 0
    duplicates = 0
    for row in rows:
        if content_text(row) is None:
            empty += 1
            continue
        doc_id = row.get("doc_id")
        if isinstance(doc_id, str) and doc_id in seen:
            duplicates += 1
            continue
        if isinstance(doc_id, str):
            seen.add(doc_id)
        kept.append(row)
    overlaps = rejected_rows(kept, exclusions)
    if overlaps:
        raise RouterWall("WALL: pool sha256 or project matches a sealed candidate")
    return PoolPrep(
        rows=kept,
        input_rows=len(rows),
        empty_text=empty,
        duplicate_doc_ids=duplicates,
        sealed_overlaps=0,
        kept=len(kept),
    )
