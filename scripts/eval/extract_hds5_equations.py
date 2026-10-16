"""Extract the HDS-5 equation list (id, name, inputs) from the blind corpus PDF.

Same approach as ``scripts/eval/extract_symbol_table.py``: poppler ``pdftotext
-layout`` on the pages that carry the chapter-3 and Appendix-A equation numbers
printed in FHWA-HIF-12-026. Output is committed as data and attached as the
blind routing registry. Nothing is typed by hand.

Names come from the Where-block description of the left-hand symbol when
present, otherwise from the nearest preceding section heading or equation
intro sentence. Math fragments from the equation line are not used as names.

    python scripts/eval/extract_hds5_equations.py --pdf hif12026.pdf
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "eval" / "graders" / "data" / "hds5_equations.json"
SYMBOLS = ROOT / "eval" / "graders" / "data" / "hds5_symbols.json"
SOURCE = {
    "title": "Hydraulic Design of Highway Culverts, Hydraulic Design Series Number 5, Third Edition (April 2012)",
    "publication": "FHWA-HIF-12-026",
    "url": "https://www.fhwa.dot.gov/engineering/hydraulics/pubs/12026/hif12026.pdf",
    "section": "Chapter 3 Culvert Hydraulic Design; Appendix A Inlet Control Equations",
}

TARGET_IDS = (
    "3.1", "3.2", "3.3", "3.4a", "3.4b", "3.4c", "3.4d", "3.4e",
    "3.5", "3.6a", "3.6b", "3.7", "3.8", "3.9",
    "A.1", "A.2", "A.3", "A.4",
)
PAGE_RANGES = ((91, 97), (190, 192))

_EQ_MARK = re.compile(r"\(\s*(?P<num>(?:3\.\d+[a-z]?|A\.\d+))\s*\)\s*$")
_JOIN_SUB = re.compile(r"\b([A-Z])\s+([a-z0-9]+)\b")
_SECTION = re.compile(
    r"^(?:"
    r"\d+\.\d+(?:\.\d+)?\s+\S|"  # 3.1.5 Roadway...
    r"A\.\d+(?:\.\d+)?\s+\S|"    # A.2.1 Unsubmerged...
    r"APPENDIX\s+[A-Z]\b|"
    r"INLET CONTROL\b|"
    r"Hydraulics of\b|"
    r"Composite Roughness\b|"
    r"Unsubmerged\b|"
    r"Submerged\b"
    r")"
)
_INTRO = re.compile(
    r"(?i)("
    r"is calculated as follows|"
    r"the velocity head is|"
    r"the entrance loss is|"
    r"the friction loss is|"
    r"the exit loss is|"
    r"the following equation|"
    r"the following relationship results|"
    r"equation 3\.6a becomes|"
    r"barrel losses \(H\) is obtained|"
    r"following equation for barrel losses|"
    r"full flow friction slope is|"
    r"defines the flow across the roadway|"
    r"form\s*\(\s*1\s*\) is based|"
    r"form\s*\(\s*2\s*\)|"
    r"submerged equation|"
    r"specific head at critical depth|"
    r"total energy \(HL\) required|"
    r"made up of the"
    r")"
)
_EXTRA = {
    "Hg", "Ku", "Ks", "Vu", "HV", "Hc", "hc", "yh", "Ap", "Tp", "Fr",
    "Qc", "Sf", "ke", "n1", "n2", "p1", "p2", "nc", "G", "Y", "M", "K",
    "HWr", "HWi", "HWo", "HL", "TW", "LS", "Qo", "Cd",
}


def _page_text(pdf: Path, page: int) -> str:
    return subprocess.run(
        ["pdftotext", "-layout", "-f", str(page), "-l", str(page), str(pdf), "-"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _load_allowed() -> frozenset[str]:
    payload = json.loads(SYMBOLS.read_text(encoding="utf-8"))
    names = {row["symbol"] for row in payload["symbols"] if isinstance(row.get("symbol"), str)}
    names.update(_EXTRA)
    return frozenset(names)


def _normalize_symbol(text: str) -> str:
    text = _JOIN_SUB.sub(r"\1\2", text.strip())
    text = re.sub(r"\s+", "", text)
    return text


def _clean_prose(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip(" \t.:;")
    text = re.sub(r"\s+\(\d+\.\d+[a-z]?\)\s*$", "", text)
    if len(text) > 160:
        text = text[:157].rstrip() + "..."
    return text


def _lhs_symbol(marker_line: str) -> str | None:
    body = _EQ_MARK.sub("", marker_line)
    body = _JOIN_SUB.sub(r"\1\2", body)
    # Prefer text before '='.
    if "=" in body:
        left = body.split("=", 1)[0]
    else:
        left = body
    left = re.sub(r"[^A-Za-z0-9]+", " ", left).strip()
    tokens = left.split()
    if not tokens:
        return None
    # Last token before '=' is usually the defined quantity (Ho, HV, Sf, nc...).
    cand = tokens[-1]
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", cand):
        return cand
    return None


def _where_definitions(lines: list[str], start: int) -> dict[str, str]:
    """Map symbol -> description from a Where-style block after an equation."""
    found: dict[str, str] = {}
    started = False
    for line in lines[start : start + 30]:
        stripped = line.rstrip()
        if not stripped.strip():
            if started and found:
                break
            continue
        if _EQ_MARK.search(stripped):
            break
        left = stripped.lstrip()
        if left.startswith("Where"):
            started = True
            continue
        # 'HW i             Headwater depth...'
        match = re.match(
            r"^([A-Za-z](?:\s+[a-z0-9]+)?[A-Za-z0-9]*)\s{2,}(=?\s*)(.+)$",
            left,
        )
        if match is None:
            match = re.match(
                r"^([A-Za-z][A-Za-z0-9]*)\s+=\s+(.+)$",
                left,
            )
            if match is None:
                if started and found and left[:1].isupper() and " " in left[:20]:
                    # continuation of previous definition
                    continue
                if started and found:
                    break
                continue
            sym = match.group(1)
            desc = match.group(2)
        else:
            sym = _normalize_symbol(match.group(1))
            desc = match.group(3)
            if desc.startswith("="):
                desc = desc.lstrip("= ").strip()
        started = True
        # Multi-symbol header like 'K,,M, c, Y'
        if "," in sym or sym in {"K,,M"}:
            for part in re.split(r"[,,\s]+", left.split()[0] if False else re.split(r"\s{2,}", left)[0]):
                part = part.strip(" ,")
                if part in {"K", "M", "c", "Y"} and part not in found:
                    found[part] = desc.strip()
            continue
        if sym and sym not in found:
            found[sym] = re.sub(r"\s+", " ", desc).strip()
    return found


def _nearest_name(lines: list[str], index: int, number: str, lhs: str | None) -> str:
    # 1) Where-block description for the LHS symbol, scanning after the marker.
    defs = _where_definitions(lines, index + 1)
    if lhs and lhs in defs and len(defs[lhs]) >= 8:
        return _clean_prose(defs[lhs])

    # 2) Nearest preceding intro sentence (join wrapped lines; skip math crumbs).
    collected: list[str] = []
    for j in range(index - 1, max(-1, index - 25), -1):
        raw = lines[j].rstrip()
        if not raw.strip():
            # Only end the chunk once we have real prose (not math crumbs).
            if collected and any(re.findall(r"[A-Za-z]{3,}", item) for item in collected):
                break
            continue
        if _EQ_MARK.search(raw):
            break
        stripped = raw.strip()
        # Skip math-only crumbs (no 3+ letter word).
        letter_runs = re.findall(r"[A-Za-z]{3,}", stripped)
        if not letter_runs:
            continue
        collected.append(stripped)
        joined = " ".join(reversed(collected))
        if _INTRO.search(joined) or joined.rstrip().endswith(":"):
            # USU method wraps; keep one prior line so "Utah State" is included.
            if "USU" in joined and "Utah State" not in joined:
                continue
            break
        if len(collected) >= 6:
            break
    if collected:
        joined = _clean_prose(" ".join(reversed(collected)))
        # If the chunk begins mid-sentence, keep from the intro match onward.
        m = _INTRO.search(joined)
        if m and m.start() > 0 and joined[0].islower():
            cut = re.search(r"(?:^|[.!?]\s+)([A-Z].+)$", joined[m.start():])
            if cut:
                joined = _clean_prose(cut.group(1))
            else:
                joined = _clean_prose(joined[m.start():])
                joined = joined[0].upper() + joined[1:] if joined else joined
        if joined.lower().startswith("equation, the "):
            joined = joined.split(", ", 1)[1]
            joined = joined[0].upper() + joined[1:]
        # Utah State University line often wraps across two PDF lines.
        if "USU" in joined and "Utah State" in " ".join(collected):
            full = _clean_prose(" ".join(reversed(collected)))
            if "Utah State" in full:
                joined = full[full.index("Utah State"):]
        if (
            _INTRO.search(joined)
            or joined.endswith("follows")
            or any(
                key in joined.lower()
                for key in (
                    "loss", "velocity", "head", "energy", "form", "overtopping",
                    "roughness", "friction slope", "specific head", "relationship",
                    "barrel", "manning",
                )
            )
        ):
            if ":" in joined:
                head, _tail = joined.split(":", 1)
                if len(head) > 20:
                    joined = head
            # Reject the "Form (1) is preferable..." comparative aside for A.1.
            if "preferable" not in joined.lower() and "easier to apply" not in joined.lower():
                return _clean_prose(joined)

    # 3) Form (1)/(2) labels on the marker line itself.
    marker = lines[index]
    if re.search(r"Form\s*\(\s*1\s*\)", marker, re.I) or number == "A.1":
        buf: list[str] = []
        for j in range(index - 1, max(-1, index - 35), -1):
            s = lines[j].strip()
            if not s:
                continue
            buf.append(s)
            joined = " ".join(reversed(buf))
            if re.search(r"Form\s*\(\s*1\s*\) is based on the specific head", joined, re.I):
                m = re.search(r"Form\s*\(\s*1\s*\) is based on the specific head[^.]*", joined, re.I)
                return _clean_prose(m.group(0) if m else joined)
            if s.startswith("A.2.1"):
                return _clean_prose(s)
            if len(buf) >= 8:
                break
        # Fall through to section heading / Equation (N) if PDF prose not found.
    if re.search(r"Form\s*\(\s*2\s*\)", marker, re.I) or number == "A.2":
        buf2: list[str] = []
        for j in range(index - 1, max(-1, index - 35), -1):
            s = lines[j].strip()
            if not s:
                continue
            buf2.append(s)
            joined = " ".join(reversed(buf2))
            if re.search(r"Form\s*\(\s*2\s*\) is an exponential", joined, re.I):
                m = re.search(r"Form\s*\(\s*2\s*\) is an exponential[^.]*", joined, re.I)
                return _clean_prose(m.group(0) if m else joined)
            if s.startswith("A.2.1"):
                return _clean_prose(s)
            if len(buf2) >= 8:
                break
        # Fall through to section heading / Equation (N) if PDF prose not found.

    # 4) Nearest preceding section heading.
    for j in range(index - 1, max(-1, index - 40), -1):
        stripped = lines[j].strip()
        if not stripped:
            continue
        if _SECTION.match(stripped):
            heading = stripped.split(". ", 1)[0]
            return _clean_prose(heading)

    # 5) Fallback: Equation (N) — still from the marker, not hand-typed body.
    return f"Equation ({number})"


def _definition_inputs(defs: dict[str, str], allowed: frozenset[str]) -> list[str]:
    names = [sym for sym in defs if sym in allowed]
    return names


def _expr_inputs(marker_line: str, allowed: frozenset[str]) -> list[str]:
    body = _EQ_MARK.sub("", marker_line)
    body = _JOIN_SUB.sub(r"\1\2", body)
    body = re.sub(r"\b([a-z])\s+([a-z0-9]+)\b", r"\1\2", body)  # d c -> dc, y h -> yh
    body = re.sub(r"\bHW r\b", "HWr", body)
    body = re.sub(r"\bk e\b", "ke", body)
    body = re.sub(r"\bK s\b", "Ks", body)
    body = re.sub(r"\bH v\b", "Hv", body)
    names: list[str] = []
    seen: set[str] = set()
    for sym in sorted(allowed, key=len, reverse=True):
        pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(sym)}(?![A-Za-z0-9])")
        if pattern.search(body) and sym not in seen:
            seen.add(sym)
            names.append(sym)
            # Consume matched span so shorter symbols inside longer ones are not re-hit
            # (boundary regex already guards; still drop lone letters if longer sibling present).
    # Drop single-letter leftovers when a longer symbol sharing that letter was found,
    # except canonical hydraulic singles.
    keep_singles = {"A", "D", "L", "n", "g", "S", "K", "M", "Y", "c", "p", "Q", "V", "R"}
    longer = {s for s in names if len(s) > 1}
    names = [s for s in names if len(s) > 1 or s in keep_singles]
    return names


def parse_pages(pages: list[tuple[int, str]], allowed: frozenset[str]) -> dict[str, dict]:
    """Parse equations across pages with one continuous line buffer.

    Names can resolve from a section heading or intro on an earlier page
    (for example Form (1)/(2) prose on page 190 for markers on page 191).
    """
    found: dict[str, dict] = {}
    shared_where: dict[str, str] = {}
    all_lines: list[str] = []
    line_page: list[int] = []
    for page, text in pages:
        for line in text.splitlines():
            all_lines.append(line)
            line_page.append(page)
        # Keep a blank separator so page-local Where scans still terminate.
        all_lines.append("")
        line_page.append(page)

    for index, line in enumerate(all_lines):
        match = _EQ_MARK.search(line.rstrip())
        if match is None:
            continue
        number = match.group("num")
        if number not in TARGET_IDS:
            continue
        formula_id = f"HDS5-{number}"
        if formula_id in found:
            continue
        page = line_page[index]
        lhs = _lhs_symbol(line)
        defs = _where_definitions(all_lines, index + 1)
        name = _nearest_name(all_lines, index, number, lhs)
        inputs = _definition_inputs(defs, allowed)
        if not inputs:
            inputs = _expr_inputs(line, allowed)
        if len(inputs) < 2:
            window = "\n".join(all_lines[max(0, index - 2) : index + 3])
            inputs = list(dict.fromkeys(inputs + _expr_inputs(window, allowed)))
        found[formula_id] = {
            "id": formula_id,
            "number": number,
            "name": name,
            "inputs": inputs,
            "pdf_page": page,
        }
        if number == "A.2" and defs:
            shared_where = defs

    if not shared_where:
        for index, line in enumerate(all_lines):
            if "Where:" in line or line.strip() == "Where:":
                shared_where = _where_definitions(all_lines, index)
                if shared_where:
                    break

    if shared_where:
        shared_inputs = _definition_inputs(shared_where, allowed)
        for num in ("A.1", "A.2", "A.3"):
            key = f"HDS5-{num}"
            if key in found and shared_inputs:
                found[key]["inputs"] = list(dict.fromkeys(found[key]["inputs"] + shared_inputs))
                if found[key]["name"].startswith("Equation") and "HWi" in shared_where:
                    found[key]["name"] = _clean_prose(shared_where["HWi"])
    return found


def extract(pdf: Path) -> dict:
    digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
    allowed = _load_allowed()
    pages: list[tuple[int, str]] = []
    for start, end in PAGE_RANGES:
        for page in range(start, end + 1):
            pages.append((page, _page_text(pdf, page)))
    equations = parse_pages(pages, allowed)
    missing = [f"HDS5-{num}" for num in TARGET_IDS if f"HDS5-{num}" not in equations]
    if missing:
        raise RuntimeError(f"equation extraction missed: {missing}")
    empty = [row["id"] for row in equations.values() if not row["inputs"]]
    if empty:
        raise RuntimeError(f"equation extraction produced empty inputs for: {empty}")
    ordered = [equations[f"HDS5-{num}"] for num in TARGET_IDS]
    return {
        "source": dict(SOURCE, pdf_sha256=digest, pdf_pages=sorted({row["pdf_page"] for row in ordered})),
        "extracted_by": "scripts/eval/extract_hds5_equations.py",
        "count": len(ordered),
        "equations": ordered,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    table = extract(args.pdf)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(table, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"equations: {table['count']} from pages {table['source']['pdf_pages']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
