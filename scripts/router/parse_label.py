"""One output parser for every router model and baseline.

A declared label matches as written. An alias matches the whole string after
whitespace is collapsed, with case ignored. A substring, a plural, or a
near-miss stays unmatched. There is no fuzzy match. Every alias target is a
declared label. Conditional names (engineer's instruction, architect's
instruction) are not aliases. "rfi" is not mapped to inspection request.

Parsing rules are fixed now, before any v8 scores exist.
The v9 aliases for "change request" and the construction change directive are
fixed before any v9 scores exist.
"""

from __future__ import annotations

# Whole-string aliases. Keys are casefolded. Targets are declared v9 labels.
_ALIASES = {
    "change order": "VO/CO",
    "variation order": "VO/CO",
    "variation": "VO/CO",
    "co": "VO/CO",
    "vo": "VO/CO",
    "variation instruction": "VO/CO",
    "construction change directive": "VO/CO",
    "change order request": "change request",
    "variation request": "change request",
    "potential change order": "change request",
    "pco": "change request",
    "cor": "change request",
    "request for modification": "change request",
    "rfm": "change request",
    "proposal request": "change request",
    "variation proposal": "change request",
    "change proposal": "change request",
    "pay application": "payment application",
    "pay app": "payment application",
    "application for payment": "payment application",
    "interim application": "payment application",
    "progress claim": "payment application",
    "payment claim": "payment application",
    "ipc": "payment certificate",
    "interim payment certificate": "payment certificate",
    "final payment certificate": "payment certificate",
    "certificate of payment": "payment certificate",
    "spec": "specification",
    "request for inspection": "inspection request",
    "inspection and test request": "inspection request",
    "itr": "inspection request",
    "wir": "inspection request",
    "mir": "inspection request",
    "work inspection request": "inspection request",
    "material inspection request": "inspection request",
    "register": "register/log",
    "log": "register/log",
    "notice of claim": "claim",
    "eot claim": "claim",
    "request for information": "RFI",
    "non-conformance report": "NCR",
    "non-compliance report": "NCR",
    "bill of quantities": "BOQ",
    "schedule of rates": "BOQ",
    "minutes of meeting": "meeting minutes",
}


def canonical_label(text: str, labels: list[str]) -> str | None:
    """Return a declared label, or None when ``text`` is not one."""
    cleaned = " ".join(text.split())
    if not cleaned:
        return None
    allowed = set(labels)
    if cleaned in allowed:
        return cleaned
    target = _ALIASES.get(cleaned.casefold())
    if target in allowed:
        return target
    return None


def parse_router_output(text: str, labels: list[str]) -> str:
    """Parse one model or baseline string into a declared label or unsure.

    A bare label and a JSON teacher reply share ``canonical_label``. A constant
    baseline emits a canonical label, which this function returns unchanged.
    """
    direct = canonical_label(text, labels)
    if direct is not None:
        return direct
    from router.teacher_label import parse_teacher_response

    return parse_teacher_response(text, labels)["label"]
