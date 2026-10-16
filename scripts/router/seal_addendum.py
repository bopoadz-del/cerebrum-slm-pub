"""Seal an owner-ticked addendum sheet into a NEW sealed file, and hold every row that may not be sealed.

Rules (all generic; no row id, project or document name appears in this code):

* The label is the owner's single label tick, copied unedited. It never comes from a filename, folder, id or meta file.
* A row whose ``unsure`` column is ticked is HELD, never sealed. The owner may also have put a tentative label tick on it;
  that tick is kept in the held record for reference only. Two or more label ticks are still a wall.
* A row without provenance (content sha256 and source URL from the meta file) is HELD with one log line.
* A row named in the isolation-failure file (sheet id -> failing checks, produced by the isolation check) is HELD.
* Per-(source project, class) cap: existing sealed rows of the same pair plus the new rows may not exceed the cap.
  Pairs already over the cap in older sets are not re-judged, but they can never take a new row.
* ``exhausted_pairs`` derives, from data only, every (project, class) pair that has reached the cap; a later batch is
  checked against that record with ``check_against_exhaustion``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from router.guard import RouterWall
from router.labels import load_labels, repo_root
from router.seal_sealed import (
    MAX_PER_PROJECT_PER_CLASS,
    _collect_rows,
    _excluded,
    _is_tick,
    _read_meta,
    _sealed_row,
    _ticked,
    enforce_project_diversity,
    project_key,
)
from router.sheet import UNSURE

NOTES_COLUMN = "notes"


def classify(
    sheets: list[Path],
    labels: list[str],
    meta: list[Path],
    isolation_failures: dict[str, list[str]] | None = None,
) -> tuple[list[dict], list[dict], list[str]]:
    """Return (sealed rows, held rows, log lines). Raises on a malformed tick."""
    rows = _collect_rows(list(sheets), labels)
    joined = _read_meta(list(meta))
    failures = isolation_failures or {}
    sealed: list[dict] = []
    held: list[dict] = []
    log: list[str] = []
    for row in rows:
        sheet_id = row["sheet_id"]
        ticks = _ticked(row, labels)
        if len(ticks) > 1:
            raise RouterWall(f"WALL: {sheet_id} has more than one label tick")
        unsure = _is_tick(row.get(UNSURE))
        notes = (row.get(NOTES_COLUMN) or "").strip()
        meta_row = joined.get(sheet_id) or {}
        base = {
            "sheet_id": sheet_id,
            "file_sha256": meta_row.get("file_sha256"),
            "isolation_key": meta_row.get("isolation_key"),
            "source_url": meta_row.get("source_url"),
            "owner_notes": notes,
        }
        if unsure:
            held.append({**base, "held_reason": "unsure", "owner_tentative_tick": ticks[0] if ticks else None})
            log.append(f"HELD {sheet_id}: owner ticked unsure")
            continue
        if not ticks:
            raise RouterWall(f"WALL: {sheet_id} does not have exactly one tick")
        if _excluded(row.get("excluded_from_eval")):
            held.append({**base, "held_reason": "excluded_from_eval", "owner_tentative_tick": ticks[0]})
            log.append(f"HELD {sheet_id}: excluded_from_eval")
            continue
        if not (meta_row.get("file_sha256") and meta_row.get("source_url")):
            held.append({**base, "held_reason": "no provenance (content sha256 + source URL required)", "owner_tentative_tick": ticks[0]})
            log.append(f"HELD {sheet_id}: missing URL/sha256 provenance")
            continue
        if failures.get(sheet_id):
            held.append({**base, "held_reason": "isolation: " + ", ".join(sorted(failures[sheet_id])), "owner_tentative_tick": ticks[0]})
            log.append(f"HELD {sheet_id}: isolation check failed ({', '.join(sorted(failures[sheet_id]))})")
            continue
        sealed.append(_sealed_row(row, ticks[0], sheet_id, meta_row))
    return sealed, held, log


def pair_counts(rows: list[dict]) -> Counter:
    """Distinct documents per (source project, label). A document sealed in two sets counts once."""
    seen: set[tuple[str, str, str]] = set()
    counts: Counter = Counter()
    for row in rows:
        key = (project_key(row), str(row.get("label") or ""))
        doc = str(row.get("file_sha256") or row.get("sheet_id") or "")
        if (key[0], key[1], doc) in seen:
            continue
        seen.add((key[0], key[1], doc))
        counts[key] += 1
    return counts


def enforce_addendum_cap(new_rows: list[dict], existing_rows: list[dict], limit: int = MAX_PER_PROJECT_PER_CLASS) -> None:
    """Within the addendum and against every existing sealed row: no pair the addendum touches may exceed ``limit``."""
    enforce_project_diversity(new_rows, limit)
    before = pair_counts(existing_rows)
    after = pair_counts([*existing_rows, *new_rows])
    for key in pair_counts(new_rows):
        if after[key] > limit:
            raise RouterWall(
                f"WALL: {after[key]} sealed documents for {key[1]!r} from project {key[0]!r} "
                f"(limit {limit}; {before[key]} already sealed)"
            )


def exhausted_pairs(rows: list[dict], limit: int = MAX_PER_PROJECT_PER_CLASS) -> list[dict]:
    """Every (project, label) pair at or over the cap, from data only."""
    out = []
    for (project, label), count in sorted(pair_counts(rows).items()):
        if count >= limit:
            out.append({"project": project, "label": label, "count": count, "status": "at_cap" if count == limit else "over_cap_legacy"})
    return out


def check_against_exhaustion(candidates: list[dict], record: list[dict]) -> list[dict]:
    """Candidate rows (with label + source_url/project) whose (project, label) pair is already exhausted."""
    closed = {(item["project"], item["label"]) for item in record}
    return [row for row in candidates if (project_key(row), str(row.get("label") or "")) in closed]


def write_addendum(output_dir: Path, sealed: list[dict], held: list[dict], extra: dict[str, object] | None = None) -> list[str]:
    """Write sealed_labels.jsonl, held.jsonl and any extra JSON records, with per-file .sha256 and SHA256SUMS."""
    output_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {
        "sealed_labels.jsonl": "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in sealed),
        "held.jsonl": "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in held),
    }
    for name, obj in (extra or {}).items():
        files[name] = json.dumps(obj, ensure_ascii=False, indent=2) + "\n"
    lines = []
    for name, text in files.items():
        (output_dir / name).write_text(text, encoding="utf-8", newline="\n")
        line = f"{hashlib.sha256(text.encode('utf-8')).hexdigest()}  {name}\n"
        (output_dir / f"{name}.sha256").write_text(line, encoding="utf-8", newline="\n")
        lines.append(line)
    (output_dir / "SHA256SUMS").write_text("".join(lines), encoding="utf-8", newline="\n")
    return list(files)


def _existing_sealed(root: Path) -> list[dict]:
    config = json.loads((root / "configs" / "router" / "sealed_eval.json").read_text(encoding="utf-8"))
    rows: list[dict] = []
    for item in config["v9_sealed"]["sets"]:
        from private_data import locate, missing_message

        path = locate(root, item["sealed_labels"])
        if not path.is_file():
            raise RouterWall(missing_message(item["sealed_labels"]))
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seal an owner-ticked addendum sheet (held rows recorded, never sealed)")
    parser.add_argument("--sheet", type=Path, action="append", required=True)
    parser.add_argument("--meta", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--isolation-failures", type=Path, default=None)
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args(argv)
    root = (args.root or repo_root()).resolve()
    try:
        labels = load_labels(root / "configs" / "router" / "labels.json")
        failures = json.loads(args.isolation_failures.read_text(encoding="utf-8")) if args.isolation_failures else {}
        sealed, held, log = classify(args.sheet, labels, args.meta, failures)
        enforce_addendum_cap(sealed, _existing_sealed(root))
        write_addendum(args.output_dir, sealed, held)
    except RouterWall as exc:
        print(str(exc))
        return 2
    for line in log:
        print(line)
    print(json.dumps({"sealed": len(sealed), "held": len(held), "labels": dict(Counter(r["label"] for r in sealed))}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
