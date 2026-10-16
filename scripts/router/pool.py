"""Raw document pool. Fork type is a reference column, never a training target."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from router.guard import RouterWall, refuse_sealed_input
from router.strip_source_tags import strip_source_tags

# Integrity columns carried through when the source row has them.
_OPTIONAL = ("content_sha256", "doc_id_source")
_EXCERPT_IN = ("first_512_tokens", "first_512")
_TYPE_IN = ("fork_declared_type", "declared_type")


def normalize_row(row: dict) -> dict:
    """Map one pool row onto the raw schema. Drops train-like label and extra text."""
    if not isinstance(row, dict):
        raise ValueError("pool row must be an object")
    excerpt = None
    for key in _EXCERPT_IN:
        value = row.get(key)
        if isinstance(value, str):
            excerpt = strip_source_tags(value)
            break
    declared = None
    for key in _TYPE_IN:
        if key in row and row[key] not in (None, ""):
            declared = row[key]
            break
    out: dict = {
        "doc_id": row.get("doc_id"),
        "project_id": row.get("project_id"),
        "filename": row.get("filename"),
        "first_512_tokens": excerpt,
        "fork_declared_type": declared,
    }
    ingested = row.get("ingested_at")
    if ingested not in (None, ""):
        out["ingested_at"] = ingested
    for key in _OPTIONAL:
        if key in row:
            out[key] = row[key]
    if "label" in out or "text" in out or "declared_type" in out:
        raise ValueError("normalized row still carries a train-like column")
    return out


def integrity(rows: list[dict]) -> dict:
    """Counts for the pool note. Does not drop or clone rows."""
    groups: dict[object, list[dict]] = {}
    for row in rows:
        groups.setdefault(row.get("doc_id"), []).append(row)
    conflict_excerpt = 0
    conflict_type = 0
    conflict_project = 0
    for group in groups.values():
        if len(group) < 2:
            continue
        if len({item.get("first_512_tokens") for item in group}) > 1:
            conflict_excerpt += 1
        if len({item.get("fork_declared_type") for item in group}) > 1:
            conflict_type += 1
        if len({item.get("project_id") for item in group}) > 1:
            conflict_project += 1
    extra = sum(len(group) - 1 for group in groups.values() if len(group) > 1)
    return {
        "rows": len(rows),
        "distinct_doc_id": len(groups),
        "duplicate_doc_ids": sum(1 for group in groups.values() if len(group) > 1),
        "extra_rows": extra,
        "doc_id_groups_with_excerpt_conflict": conflict_excerpt,
        "doc_id_groups_with_type_conflict": conflict_type,
        "doc_id_groups_with_project_conflict": conflict_project,
        "fork_declared_type_counts": dict(Counter(row.get("fork_declared_type") for row in rows)),
        "projects": len({row.get("project_id") for row in rows}),
    }


def normalize_jsonl(source: Path, dest: Path) -> dict:
    refuse_sealed_input(source)
    refuse_sealed_input(dest)
    rows: list[dict] = []
    text = source.read_text(encoding="utf-8")
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source}:{lineno}: {exc}") from exc
        rows.append(normalize_row(obj))
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    dest.write_text(payload, encoding="utf-8")
    report = integrity(rows)
    report["sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Normalize the raw router document pool")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = normalize_jsonl(args.input, args.output)
    except (RouterWall, ValueError) as exc:
        print(str(exc))
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
