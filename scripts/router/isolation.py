"""Keep sealed router documents out of training pools and splits.

``file_sha256`` and ``project`` are exclusion keys. They are not labels.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from router.guard import RouterWall

_SHA_FIELDS = ("file_sha256", "doc_sha256", "content_sha256", "doc_id")
_PROJECT_FIELDS = ("project", "project_id")


@dataclass(frozen=True)
class Exclusions:
    file_sha256: frozenset[str]
    projects: frozenset[str]


def load_exclusions(path: Path) -> Exclusions:
    """Read sealed rows. A missing or empty file excludes nothing."""
    if not path.is_file() or path.stat().st_size == 0:
        return Exclusions(frozenset(), frozenset())
    shas: set[str] = set()
    projects: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            continue
        sha = _text(obj.get("file_sha256"))
        if sha:
            shas.add(sha.casefold())
        project = _text(obj.get("project"))
        if project:
            projects.add(project)
    return Exclusions(frozenset(shas), frozenset(projects))


def overlaps(row: dict, exclusions: Exclusions) -> bool:
    """True when the row's document sha or project is in the sealed set."""
    if not exclusions.file_sha256 and not exclusions.projects:
        return False
    for source in _sources(row):
        for field in _SHA_FIELDS:
            sha = _text(source.get(field))
            if sha and sha.casefold() in exclusions.file_sha256:
                return True
        for field in _PROJECT_FIELDS:
            project = _text(source.get(field))
            if project and project in exclusions.projects:
                return True
    return False


def rejected_rows(rows: list[dict], exclusions: Exclusions) -> list[dict]:
    return [row for row in rows if overlaps(row, exclusions)]


def require_isolated(rows: list[dict], exclusions: Exclusions) -> list[dict]:
    """Return ``rows`` unchanged, or refuse the pool when a sealed row is present."""
    if rejected_rows(rows, exclusions):
        raise RouterWall("WALL: sealed doc or project in training pool")
    return rows


def split_training_pool(rows: list[dict], sealed_path: Path) -> list[dict]:
    """Split/training-pool entry. Refuses sealed documents. Writes no pairs."""
    return require_isolated(rows, load_exclusions(sealed_path))


def _sources(row: dict) -> list[dict]:
    sources = [row]
    meta = row.get("meta")
    if isinstance(meta, dict):
        sources.append(meta)
    return sources


def _text(value: object) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped
    return None
