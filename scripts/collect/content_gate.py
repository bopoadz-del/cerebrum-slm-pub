"""Content gate for collected router rows — structural only, no word lists.

A row may enter a labelling sheet only when its text is long enough to read
(>= 200 chars) and carries at least one filled field value: a date, an amount,
or a name-shaped value line. Blank forms carry labels and no values, so they
route to the template bucket automatically.

Detection is structural:
  - date:   dd/mm/yyyy, dd-mmm-yyyy (month name), yyyy-mm-dd
  - amount: $ 12,345.67 or any two-decimal number
  - name:   a mixed-case line with a digit and two or more Titlecase words
            (typed values are mixed-case; form labels are not)

The gate never reads filenames, source URLs or labels.
"""
from __future__ import annotations

import re

MIN_CHARS = 200

_DATE = re.compile(
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
    r"|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.? \d{1,2},? \d{4}\b"
    r"|\b\d{4}-\d{2}-\d{2}\b",
    re.I,
)
_AMOUNT = re.compile(r"\$\s?[\d,]+(?:\.\d{2})?\b|\b[\d,]+\.\d{2}\b")
_TITLECASE_WORD = re.compile(r"\b[A-Z][a-z][A-Za-z.'-]*\b")


def _name_evidence(text: str) -> bool:
    """True when a mixed-case line carries a digit and >=2 Titlecase words."""
    for line in text.splitlines():
        if not line.strip() or not re.search(r"\d", line):
            continue
        if line == line.upper():  # all-caps line is a label row, not a value
            continue
        if len(_TITLECASE_WORD.findall(line)) >= 2:
            return True
    return False


def filled_field_evidence(text: str) -> dict[str, bool]:
    return {
        "date": bool(_DATE.search(text)),
        "amount": bool(_AMOUNT.search(text)),
        "name": _name_evidence(text),
    }


def gate(text: str) -> tuple[bool, str]:
    """Return (pass, reason). Structural only; content is never labelled."""
    if len(text.strip()) < MIN_CHARS:
        return False, f"text below {MIN_CHARS} chars"
    evidence = filled_field_evidence(text)
    if not any(evidence.values()):
        return False, "no filled field value (date/amount/name) outside form labels"
    return True, ""
