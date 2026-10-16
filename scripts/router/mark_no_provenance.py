"""One-shot marker: flag legacy additions rows as no-provenance.

Rows collected before the 2026-10-08 collection rule recorded no source URL
("domain sha256 only" era). This script writes ``provenance_status:
"no-provenance"`` onto rows that lack a valid source_url. It never touches a
label, an id, an excerpt or any other field, and it never removes a row.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from router.collection_provenance import _is_url, NO_PROVENANCE


def mark(path: Path) -> int:
    lines = path.read_text(encoding="utf-8").splitlines()
    changed = 0
    out: list[str] = []
    for line in lines:
        if not line.strip():
            out.append(line)
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            out.append(line)
            continue
        if not _is_url(obj.get("source_url")) and obj.get("provenance_status") != NO_PROVENANCE:
            obj["provenance_status"] = NO_PROVENANCE
            changed += 1
        out.append(json.dumps(obj, ensure_ascii=False))
    if changed:
        path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return changed


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: python -m router.mark_no_provenance <additions.jsonl> [...]", file=sys.stderr)
        return 2
    for raw in argv:
        p = Path(raw)
        n = mark(p)
        print(f"{p}: flagged {n} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
