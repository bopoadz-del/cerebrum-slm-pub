"""Drop a candidate before it is written to a labelling sheet when its document hash is already known.

Known hashes are every sha256 already stored for a sealed row, a held row, or a candidate
that was already accepted (this batch or an earlier one). A row is identified by its file
hash and its source-document hash. The second row of one source document is rejected even
when the extracted file hash differs. Nothing here names a project, a form or a row.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HEX = set("0123456789abcdef")
_FIELDS = ("sha256", "file_sha256", "source_sha256", "content_sha256")


def identity_hashes(row: dict) -> set[str]:
    """Lower-case sha256 values that identify the document. Text hashes are not identity."""
    found: set[str] = set()
    for key in _FIELDS:
        value = row.get(key)
        if not isinstance(value, str):
            continue
        text = value.strip().lower()
        if len(text) == 64 and set(text) <= _HEX:
            found.add(text)
    return found


def hashes_from_records(records: list[dict]) -> set[str]:
    known: set[str] = set()
    for row in records:
        known |= identity_hashes(row)
    return known


def keep_for_sheet(candidates: list[dict], known: set[str]) -> tuple[list[dict], list[dict]]:
    """Return (kept, rejected). A candidate with no identity hash is rejected.

    Rejected rows never reach the sheet. Accepting a row adds its hashes to the known set,
    so a later row of the same file or the same source document is rejected too.
    """
    seen = set(known)
    kept: list[dict] = []
    rejected: list[dict] = []
    for row in candidates:
        ident = identity_hashes(row)
        if not ident or ident & seen:
            rejected.append(row)
            continue
        kept.append(row)
        seen |= ident
    return kept, rejected


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def load_known(paths: list[Path]) -> set[str]:
    known: set[str] = set()
    for path in paths:
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".json":
            obj = json.loads(text)
            records = obj["candidates"] if isinstance(obj, dict) and "candidates" in obj else obj
            if isinstance(records, dict) and "rows" in records:
                records = records["rows"]
            if isinstance(records, list):
                known |= hashes_from_records([r for r in records if isinstance(r, dict)])
            continue
        known |= hashes_from_records(_read_jsonl(path))
    return known


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reject candidates whose sha256 is already known")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--known", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rejected", type=Path, required=True)
    args = parser.parse_args(argv)
    candidates = _read_jsonl(args.candidates)
    kept, rejected = keep_for_sheet(candidates, load_known(args.known))
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8")
    args.rejected.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rejected), encoding="utf-8")
    print(json.dumps({"kept": len(kept), "rejected": len(rejected)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
