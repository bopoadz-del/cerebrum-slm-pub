"""Collection provenance rule — every collected router row records its source.

The rule (owner order 2026-10-08): a row may enter a labelling sheet only when
it records, at fetch time:

    source_url       the full URL the document was fetched from
    fetch_timestamp  ISO-8601 UTC fetch time
    http_status      the HTTP status of the fetch
    content_sha256   sha256 of the fetched bytes
    page_title       the fetched page's title

"domain sha256 only" provenance is retired. Legacy rows that predate the rule
carry ``provenance_status: "no-provenance"`` and are validation-visible but
never count as rule-compliant.

Validation is structural: it checks field presence/types on the recorded row.
It never reads document content, filenames or labels.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from private_data import (
    OffloadManifestError,
    env_for,
    is_offloaded,
    locate,
    offloaded_paths,
    private_mode,
)

ADDITIONS_GLOB = "data/router/additions_*.jsonl"
OWNER_ADDITIONS_ENV = "SLM_OWNER_ADDITIONS"

REQUIRED_FIELDS: tuple[str, ...] = (
    "source_url",
    "fetch_timestamp",
    "http_status",
    "content_sha256",
    "page_title",
)

NO_PROVENANCE = "no-provenance"


@dataclass(frozen=True)
class RowResult:
    row_index: int
    missing: tuple[str, ...] = ()
    legacy: bool = False

    @property
    def ok(self) -> bool:
        return not self.missing


@dataclass
class FileResult:
    path: str
    rows: int = 0
    violations: list[RowResult] = field(default_factory=list)
    legacy_rows: int = 0

    @property
    def ok(self) -> bool:
        return not self.violations


def _is_url(value: object) -> bool:
    return isinstance(value, str) and value.startswith(("http://", "https://"))


def _is_timestamp(value: object) -> bool:
    # ISO-8601: 2026-10-07T19:24:54Z or with offset — structural only
    if not isinstance(value, str) or len(value) < 16:
        return False
    return value[4] == "-" and value[10] in ("T", " ") and value[13] == ":"


def _is_http_status(value: object) -> bool:
    return isinstance(value, int) and 100 <= value <= 599


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        c in "0123456789abcdefABCDEF" for c in value
    )


_FIELD_CHECKS = {
    "source_url": _is_url,
    "fetch_timestamp": _is_timestamp,
    "http_status": _is_http_status,
    "content_sha256": _is_sha256,
    "page_title": lambda v: isinstance(v, str) and bool(v.strip()),
}


def validate_row(row: dict, index: int) -> RowResult:
    """Structural check of one additions row. Content is never read."""
    missing = tuple(
        name for name in REQUIRED_FIELDS if not _FIELD_CHECKS[name](row.get(name))
    )
    legacy = row.get("provenance_status") == NO_PROVENANCE
    if missing and legacy:
        return RowResult(row_index=index, missing=(), legacy=True)
    return RowResult(row_index=index, missing=missing, legacy=legacy)


def validate_file(path: Path) -> FileResult:
    result = FileResult(path=str(path))
    if not path.is_file():
        result.violations.append(RowResult(row_index=0, missing=("file is absent",)))
        return result
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            result.violations.append(RowResult(row_index=index, missing=("<not an object>",)))
            continue
        row = validate_row(obj, index)
        result.rows += 1
        if row.legacy:
            result.legacy_rows += 1
        elif not row.ok:
            result.violations.append(row)
    return result


def validate_paths(paths: list[Path]) -> list[FileResult]:
    return [validate_file(p) for p in paths]


def _report(res: FileResult, label: str) -> bool:
    """Print one file result. Return True when the file failed. Never prints row text."""
    if res.rows == 0 and any(v.missing == ("file is absent",) for v in res.violations):
        print(
            f"{label}: private data file is absent. "
            "Pass an external path. Refusing to continue without the file.",
            file=sys.stderr,
        )
        return True
    print(f"{label}: {res.rows} rows, {res.legacy_rows} no-provenance")
    failed = False
    for v in res.violations:
        failed = True
        print(f"  row {v.row_index}: missing/invalid {', '.join(v.missing)}")
    return failed


def _norm_rel(relative: str) -> str:
    return relative.replace("\\", "/").lstrip("/")


def _matches_additions(relative: str) -> bool:
    return fnmatch.fnmatch(_norm_rel(relative), ADDITIONS_GLOB)


def discover_targets(root: Path) -> tuple[list[str], list[str], list[str]]:
    """In-tree additions, manifest additions not in the tree, and owner-additions paths.

    Raises ``OffloadManifestError`` when the manifest cannot be read.
    """
    tree: list[str] = []
    router = root / "data" / "router"
    if router.is_dir():
        for path in sorted(router.glob("additions_*.jsonl")):
            if path.is_file():
                tree.append(path.relative_to(root).as_posix())
    listed = sorted(rel for rel in offloaded_paths(root) if _matches_additions(rel))
    manifest = [rel for rel in listed if rel not in set(tree)]
    owner = [part for part in os.environ.get(OWNER_ADDITIONS_ENV, "").split() if part.strip()]
    return tree, manifest, owner


def _owner_label(index: int, path: str) -> str:
    if Path(path).is_absolute():
        return f"{OWNER_ADDITIONS_ENV}#{index}"
    return path


def _run_discover(root: Path) -> tuple[bool, list[str]]:
    """Return (failed, not-run lines)."""
    try:
        tree, manifest, owner = discover_targets(root)
    except OffloadManifestError as exc:
        print(str(exc), file=sys.stderr)
        return True, []
    failed = False
    not_run: list[str] = []
    for rel in tree:
        if _report(validate_file(root / rel), rel):
            failed = True
    for rel in manifest:
        if not private_mode() and is_offloaded(rel, root):
            not_run.append(f"NOT-RUN [collection-provenance] {rel} set {env_for(rel)}")
            continue
        located = locate(root, rel)
        if not located.is_file():
            print(
                f"{rel}: private data file is absent. Set {env_for(rel)}. "
                "Refusing to continue without the file.",
                file=sys.stderr,
            )
            failed = True
            continue
        if _report(validate_file(located), rel):
            failed = True
    for index, path in enumerate(owner, start=1):
        label = _owner_label(index, path)
        target = Path(path)
        if not target.is_file():
            print(
                f"{label}: private data file is absent. Set {OWNER_ADDITIONS_ENV}. "
                "Refusing to continue without the file.",
                file=sys.stderr,
            )
            failed = True
            continue
        if _report(validate_file(target), label):
            failed = True
    return failed, not_run


def _run_explicit(root: Path, paths: list[str]) -> bool:
    failed = False
    for raw in paths:
        rel = _norm_rel(raw)
        target = Path(raw)
        if not target.is_file() and not Path(raw).is_absolute():
            target = root / rel
        if target.is_file():
            if _report(validate_file(target), raw):
                failed = True
            continue
        # Absent explicit paths fail. A manifest hit does not turn this into NOT-RUN.
        print(
            f"{raw}: private data file is absent. "
            "Pass an external path. Refusing to continue without the file.",
            file=sys.stderr,
        )
        failed = True
    return failed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate collection provenance on additions files.")
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Validate in-tree additions_*.jsonl, manifest matches, and SLM_OWNER_ADDITIONS.",
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args(argv)
    if not args.discover and not args.paths:
        print(
            "usage: python -m router.collection_provenance [--discover] [--root DIR] <additions.jsonl> [...]",
            file=sys.stderr,
        )
        return 2
    root = (args.root or Path.cwd()).resolve()
    failed = False
    not_run: list[str] = []
    if args.discover:
        failed, not_run = _run_discover(root)
    if args.paths:
        failed = _run_explicit(root, list(args.paths)) or failed
    for line in not_run:
        print(line)
    if failed:
        return 1
    if args.discover:
        print(f"provenance: green ({len(not_run)} not run: private data absent)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
