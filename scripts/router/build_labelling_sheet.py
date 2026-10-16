"""Labelling-sheet generator — Definitions emitted from LABEL_SET_v9.md verbatim.

The Definitions tab carries the approved v9 label definitions byte-for-byte
(same bytes as docs/router/LABEL_SET_v9.md). The previous ad-hoc builder
hand-copied definitions and drifted ("Regional names:" vs "Regional names
that map here:", a missing Examples line, a missing Regional mappings
section). This generator cannot drift: it reads the file at build time.

The Label tab is the standard 27-column layout (id, file name, excerpt,
22 labels, unsure, notes) with blank rows — owner ticks, never generator
ticks.

``regenerate_definitions`` rewrites ONLY the Definitions tab of an existing
sheet (Label tab untouched) so a shipped sheet can be re-based onto the
approved text in place.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from openpyxl import Workbook, load_workbook

from router.labels import load_labels

REPO = Path(__file__).resolve().parent.parent.parent
DEFINITIONS_MD = REPO / "docs" / "router" / "LABEL_SET_v9.md"

INSTRUCTIONS = "\n".join(
    [
        "INSTRUCTIONS",
        "Label each document by its CONTENT only (open the PDF; the excerpt column is a convenience copy of its text). Ignore the file name and where it came from.",
        'On the Label tab, tick exactly ONE label column per row by setting it to TRUE.',
        'If you are not sure, set the "unsure" column to TRUE (you may also tick your best label) and add a note.',
        "These rows are candidates for the router sealed thin-class addendum. Nothing enters the sealed set until you have ticked it; rows are added as an addendum, never swapped in.",
        "Label set: v9 (22 labels). Definitions below are the approved v9 text.",
    ]
)


def definitions_bytes(md_path: Path = DEFINITIONS_MD) -> str:
    return md_path.read_text(encoding="utf-8")


def header(labels: list[str]) -> list[str]:
    return ["id", "file name", "excerpt", *labels, "unsure", "notes"]


def write_sheet(
    path: Path,
    labels: list[str],
    rows: list[list[str]],
    md_path: Path = DEFINITIONS_MD,
) -> None:
    wb = Workbook()
    defs = wb.active
    defs.title = "Definitions"
    defs["A1"] = INSTRUCTIONS
    defs["A2"] = definitions_bytes(md_path)

    lab = wb.create_sheet("Label")
    lab.append(header(labels))
    for row in rows:
        rid, filename, excerpt = (list(row) + ["", ""])[:3]
        lab.append([rid, filename, excerpt, *(["False"] * len(labels)), None, None])
    wb.save(path)


def regenerate_definitions(path: Path, md_path: Path = DEFINITIONS_MD) -> None:
    """Rewrite only the Definitions tab of an existing sheet, in place."""
    wb = load_workbook(path)
    if "Definitions" not in wb.sheetnames:
        wb.create_sheet("Definitions", 0)
    defs = wb["Definitions"]
    for row in defs.iter_rows():
        for cell in row:
            cell.value = None
    defs["A1"] = INSTRUCTIONS
    defs["A2"] = definitions_bytes(md_path)
    wb.save(path)


def _rows_from_csv(path: Path) -> list[list[str]]:
    import csv

    with path.open(newline="", encoding="utf-8") as f:
        return [list(row) for row in csv.reader(f)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build a router labelling sheet.")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--rows", type=Path, help="CSV of id,file name,excerpt rows")
    ap.add_argument("--definitions", type=Path, default=DEFINITIONS_MD)
    ap.add_argument(
        "--regen-definitions", action="store_true",
        help="rewrite only the Definitions tab of an existing --out sheet",
    )
    args = ap.parse_args(argv)
    labels = load_labels()
    if args.regen_definitions:
        regenerate_definitions(args.out, args.definitions)
        print(f"regenerated Definitions in {args.out}")
        return 0
    rows = _rows_from_csv(args.rows) if args.rows else []
    write_sheet(args.out, labels, rows, args.definitions)
    print(f"wrote {args.out} ({len(labels)} labels, {len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
