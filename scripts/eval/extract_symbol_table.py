"""Extract the corpus notation list (LIST OF SYMBOLS) from the blind corpus PDF.

Run once; the output is committed as data and read by the refusal grader:

    python scripts/eval/extract_symbol_table.py --pdf hif12026.pdf

The corpus is the one named in data/eval/unseen/README.md (HDS-5, third
edition, FHWA-HIF-12-026, public URL below). The PDF is not committed. Text is
read with poppler's ``pdftotext -layout`` from the pages whose text carries the
LIST OF SYMBOLS heading. Each entry is the symbol column and the definition
column as printed; subscripts printed apart (``HW a``) are joined (``HWa``),
and a row that lists several symbols (``L1, L2, L3, L4``) gives each symbol the
same definition. Nothing is typed by hand.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "eval" / "graders" / "data" / "hds5_symbols.json"
SOURCE = {
    "title": "Hydraulic Design of Highway Culverts, Hydraulic Design Series Number 5, Third Edition (April 2012)",
    "publication": "FHWA-HIF-12-026",
    "url": "https://www.fhwa.dot.gov/engineering/hydraulics/pubs/12026/hif12026.pdf",
    "section": "LIST OF SYMBOLS",
}
HEADING = "LIST OF SYMBOLS"
_ROMAN = re.compile(r"^[ivxlcdm]+$", re.IGNORECASE)
_SPLIT = re.compile(r"\S(\s{2,}|\s(?=[A-Z][a-z]))")


def _page_text(pdf: Path, page: int) -> str:
    return subprocess.run(
        ["pdftotext", "-layout", "-f", str(page), "-l", str(page), str(pdf), "-"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _page_count(pdf: Path) -> int:
    info = subprocess.run(["pdfinfo", str(pdf)], check=True, capture_output=True, text=True).stdout
    return int(re.search(r"^Pages:\s+(\d+)", info, re.MULTILINE).group(1))


def symbol_pages(pdf: Path, limit: int = 60) -> list[tuple[int, str]]:
    pages = []
    for page in range(1, min(limit, _page_count(pdf)) + 1):
        text = _page_text(pdf, page)
        if HEADING in text and "...." not in text.split(HEADING, 1)[1].splitlines()[0]:
            pages.append((page, text))
    return pages


def _definition_column(lines: list[str]) -> int:
    starts: Counter[int] = Counter()
    for line in lines:
        if not line or line[0].isspace():
            continue
        match = _SPLIT.search(line)
        if match:
            starts[match.end()] += 1
    return starts.most_common(1)[0][0]


def parse_page(text: str) -> list[tuple[list[str], str]]:
    body = text.split(HEADING, 1)[1].splitlines()[1:]
    lines = [line.rstrip() for line in body if line.strip() and not _ROMAN.match(line.strip())]
    lines = [line for line in lines if "intentionally left blank" not in line]
    column = _definition_column(lines)
    entries: list[tuple[list[str], str]] = []
    for line in lines:
        head, tail = line[:column].strip(), line[column:].strip()
        if not head:
            if entries:
                symbols, definition = entries[-1]
                entries[-1] = (symbols, f"{definition} {tail}".strip())
            continue
        symbols = [re.sub(r"\s+", "", part) for part in head.split(",") if part.strip()]
        if not tail and entries:
            entries[-1] = (entries[-1][0] + symbols, entries[-1][1])
            continue
        entries.append((symbols, tail))
    return entries


def extract(pdf: Path) -> dict:
    digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
    pages = symbol_pages(pdf)
    symbols: list[dict] = []
    for page, text in pages:
        for names, definition in parse_page(text):
            for name in names:
                symbols.append({"symbol": name, "definition": definition, "pdf_page": page})
    return {
        "source": dict(SOURCE, pdf_sha256=digest, pdf_pages=[page for page, _ in pages]),
        "extracted_by": "scripts/eval/extract_symbol_table.py",
        "count": len(symbols),
        "symbols": symbols,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    table = extract(args.pdf)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(table, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"symbols: {table['count']} from pages {table['source']['pdf_pages']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
