"""Remove file-path metadata from document text.

The Fork export prepended ``[source: <drive path>]`` tags to stored excerpts.
A path, folder or filename must never decide a label or reach training text,
so this module strips:

* any bracketed ``[source: ...]`` tag (closed, or cut off by the 512-token limit),
* an unbracketed ``source:`` marker when a file path follows it,
* any Windows drive path (``X:\\a\\b``, ``X:/a/b``), UNC path (``\\\\host\\share``),
  ``file://`` URI, or POSIX path (``/a/b/c``, ``~/a/b``, ``./a/b.ext``),

wherever it appears in excerpt or prompt text. The rules are structural regexes
only: no word lists and no project or folder names. Plain words such as
``Source: Esri`` and ``and/or``, dates such as ``12/05/2024`` and web URLs are
left alone.

CLI: ``python scripts/router/strip_source_tags.py IN.jsonl [IN2.jsonl ...]``
writes ``IN_clean.jsonl`` next to each input (or ``--out`` for one file) and
prints per-file counts. ``--check`` exits 1 when any text field still matches.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

# Characters Windows forbids inside a path segment, plus brackets and line breaks.
_WIN_SEG = r"[^\\/:*?\"<>|\r\n\[\]]"
_WIN_SEG_NOSPACE = r"[^\\/:*?\"<>|\s\[\]]"
_EXT = r"\.[A-Za-z][A-Za-z0-9]{0,5}"

# 1) Bracketed tag. Ends at "]", before the next "[", or at end of text (truncated tag).
_BRACKET_TAG = r"\[\s*source\s*:[^\[\]]*(?:\]|(?=\[)|$)"

# 2) Windows drive path with backslashes. Folder segments may hold spaces because each
#    one is closed by a separator. The last segment may hold spaces only when it ends
#    in a file extension; otherwise it stops at whitespace.
_WIN_BACKSLASH = (
    rf"\b[A-Za-z]:\\(?:{_WIN_SEG}{{1,120}}?\\)*"
    rf"(?:{_WIN_SEG}{{0,120}}?{_EXT}\b|{_WIN_SEG_NOSPACE}*)"
)
# 3) Windows drive path with forward slashes: no spaces in segments.
_WIN_FWD = rf"\b[A-Za-z]:/(?:{_WIN_SEG_NOSPACE}+/)*{_WIN_SEG_NOSPACE}*"
# 4) UNC path.
_UNC = (
    rf"\\\\{_WIN_SEG_NOSPACE}+\\(?:{_WIN_SEG}{{1,120}}?\\)*"
    rf"(?:{_WIN_SEG}{{0,120}}?{_EXT}\b|{_WIN_SEG_NOSPACE}*)"
)
# 5) file:// URI.
_FILE_URI = r"\bfile:/{1,3}[^\s\]\[\"'<>]+"
# 6) POSIX path: not glued to a word, a slash, a dot or a colon (so URLs, dates and
#    "and/or" are skipped). Every segment holds a letter. Either two or more segments
#    ending in a file extension, or four or more segments ("/a/b/c" in a table cell
#    such as "cap /Joint/trims/Ridge" is too short to call a path).
_PSEG = r"[\w.\-]*[A-Za-z][\w.\-]*"
_POSIX = (
    rf"(?<![\w/.:\-])(?:~|\.{{1,2}})?/(?:{_PSEG}/)+{_PSEG}{_EXT}\b"
    rf"|(?<![\w/.:\-])(?:~|\.{{1,2}})?/(?:{_PSEG}/){{3,}}(?:{_PSEG})?"
)

_PATH = rf"(?:{_WIN_BACKSLASH}|{_WIN_FWD}|{_UNC}|{_FILE_URI}|{_POSIX})"
# 7) Unbracketed "source:" marker followed by a path: drop marker and path together.
_SOURCE_PATH = rf"\bsource\s*:\s*{_PATH}"

PATH_RE = re.compile(rf"{_BRACKET_TAG}|{_SOURCE_PATH}|{_PATH}", re.IGNORECASE)
"""Matches anything this module removes. Use ``has_path`` for guards."""

_SPACES = re.compile(r"[ \t]{2,}")

# Row keys that are identifiers or grouping keys, not document text. A filename is a
# split grouping key only and is never model input; it is left as-is.
ID_KEYS = frozenset({
    "doc_id", "project_id", "project", "file_sha256", "content_sha256", "doc_sha256",
    "doc_id_source", "filename", "fork_declared_type", "label", "idx", "confidence",
    "group", "gold", "pred", "raw", "role",
})


def strip_source_tags(text: str) -> str:
    """Return ``text`` with every source tag and file path removed."""
    if not isinstance(text, str) or not text:
        return text
    out = PATH_RE.sub(" ", text)
    if out == text:
        return text
    out = _SPACES.sub(" ", out)
    return out.strip()


def has_path(text: Any) -> bool:
    """True when ``text`` still holds a source tag or a file path."""
    return isinstance(text, str) and PATH_RE.search(text) is not None


def clean_value(value: Any, key: str | None = None) -> Any:
    """Strip text inside nested rows. Identifier keys are not touched."""
    if key in ID_KEYS:
        return value
    if isinstance(value, str):
        return strip_source_tags(value)
    if isinstance(value, list):
        return [clean_value(item) for item in value]
    if isinstance(value, dict):
        return {k: clean_value(v, k) for k, v in value.items()}
    return value


def row_has_path(value: Any, key: str | None = None) -> bool:
    if key in ID_KEYS:
        return False
    if isinstance(value, str):
        return has_path(value)
    if isinstance(value, list):
        return any(row_has_path(item) for item in value)
    if isinstance(value, dict):
        return any(row_has_path(v, k) for k, v in value.items())
    return False


def clean_file(src: Path, dest: Path) -> dict:
    rows = changed = still = 0
    lines_out: list[str] = []
    for line in src.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        new = clean_value(row)
        rows += 1
        if new != row:
            changed += 1
        if row_has_path(new):
            still += 1
        lines_out.append(json.dumps(new, ensure_ascii=False))
    dest.write_text("".join(item + "\n" for item in lines_out), encoding="utf-8")
    return {"file": src.name, "out": dest.name, "rows": rows, "rows_changed": changed, "rows_still_matching": still}


def check_file(path: Path) -> dict:
    rows = bad = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows += 1
            bad += row_has_path(json.loads(line))
    return {"file": path.name, "rows": rows, "rows_matching": bad}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Strip [source: ...] tags and file paths from JSONL text fields")
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="output path (one input only)")
    ap.add_argument("--check", action="store_true", help="report matches only; exit 1 if any")
    args = ap.parse_args(argv)
    if args.out and len(args.inputs) != 1:
        ap.error("--out needs exactly one input")
    failed = False
    for src in args.inputs:
        if args.check:
            rep = check_file(src)
            failed |= rep["rows_matching"] > 0
        else:
            dest = args.out or src.with_name(f"{src.stem}_clean{src.suffix}")
            rep = clean_file(src, dest)
            failed |= rep["rows_still_matching"] > 0
        print(json.dumps(rep))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
