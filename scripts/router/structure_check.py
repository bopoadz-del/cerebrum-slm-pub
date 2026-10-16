"""Rule-based structure check: a second method next to the content read of an ingested document.

It reads only the document's own structure (field labels, amounts, dates, signature blocks) and returns the
v9 labels whose structural rule matches, strongest first. Generic rules only: no file, project or party
specific overrides.

Payment certificate vs payment application (ORDERS_1940Z item 1): a payment certificate needs a FILLED
certifier block, i.e. a certifier role (certified / approved / recommended / reviewed / checked / verified by,
authorised signatory, employer's or engineer's representative) followed by a name or signature value, plus a
certified amount next to the certification statement and a date. A form that carries the certification wording
but whose certifier block is blank (empty, underscores, or straight into the next field label) has not been
certified yet and is a payment application.

Known limit: the date is read at document level, because dates inside a signature row are often redacted or
cut off in a page excerpt. Amount tokens produced by the generic redactor (`<AMT>` / `<AMT:~1eN>`) count as
amounts. Batch-fitted label rules (e.g. lesson-number or insurance-whomsoever shortcuts) are not used; only
general structural patterns plus the certifier-block rule.
"""

from __future__ import annotations

import re

AMT = r"\d{1,3}(?:,\d{3})+(?:\.\d{2})?|<AMT(?::~1e-?\d+)?>"
_MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|November|December"
)
DATE = (
    r"\b\d{1,2}[-/ ](?:\d{1,2}|[A-Za-z]{3,9})[-/ ]\d{2,4}\b"
    rf"|\b(?:{_MONTHS})\s+\d{{1,2}},?\s+\d{{4}}\b"
    rf"|\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{_MONTHS})\s+\d{{4}}\b"
)
CERT_STATEMENT = re.compile(
    r"(?i)gross works certified|amount certified|certified (?:for payment|amount)"
    r"|we (?:hereby )?certify|payment certificate no"
)
CERTIFIER_ROLE = re.compile(
    r"(?i)\b(?:certified|approved|recommended|reviewed|checked|verified)\s+by\b\s*:?"
    r"|\bauthori[sz]ed\s+signatory\b\s*:?"
    r"|\b(?:employer|engineer)(?:'|\u2019)?s\s+representative\b\s*:?"
    r"|\bon\s+behalf\s+of\s+the\s+(?:engineer|employer|consultant)\b\s*:?"
)
# a field label that starts the next cell of a signature block
FIELD_LABEL = re.compile(
    r"(?i)\b(?:name|signature|sign|date|position|designation|title|stamp|prepared\s+by)\b\s*:?"
)
REDACTION_BAR = "\u2588"
_FILLER = re.compile(r"[\s_.\-:\u2013\u2014|/]+")
_AMOUNT_AFTER_STATEMENT = 160
_VALUE_WINDOW = 60


def _has(pattern: str, text: str, n: int = 1) -> bool:
    return len(re.findall(pattern, text, re.I)) >= n


def _role_value(text: str, end: int) -> str:
    """Text written after a certifier role label, up to the next role or field label."""
    seg = text[end : end + _VALUE_WINDOW]
    cut = len(seg)
    for rx in (CERTIFIER_ROLE, FIELD_LABEL):
        m = rx.search(seg)
        if m and m.start() < cut:
            cut = m.start()
    return seg[:cut]


def _value_is_filled(value: str) -> bool:
    if REDACTION_BAR in value:
        return True
    rest = _FILLER.sub(" ", value).strip()
    return bool(re.search(r"[A-Za-z]{2,}", rest))


def certifier_block_filled(text: str) -> bool:
    """True when at least one certifier role in the text carries a name or signature value."""
    return any(
        _value_is_filled(_role_value(text, m.end())) for m in CERTIFIER_ROLE.finditer(text)
    )


def certified_amount(text: str) -> bool:
    """A certification statement followed closely by an amount."""
    return any(
        re.search(AMT, text[m.end() : m.end() + _AMOUNT_AFTER_STATEMENT])
        for m in CERT_STATEMENT.finditer(text)
    )


def is_payment_certificate(text: str) -> bool:
    return certified_amount(text) and certifier_block_filled(text) and _has(DATE, text)


def rules(text: str) -> list[str]:
    t = text
    out: list[str] = []
    amt = _has(AMT, t)
    certificate = is_payment_certificate(t)
    if certificate:
        out.append("payment certificate")
    if (
        _has(r"payment application|application for payment|interim payment|tax invoice|invoice", t)
        and amt
        and _has(r"claimed|claim|submit|previous|cumulative|to.?date|invoiced|amount due", t)
        and not certificate
    ):
        out.append("payment application")
    if (
        _has(r"extension of time|\bEOT\b|prolongation|entitle", t)
        and _has(r"clause", t)
        and _has(r"delay event|particulars|\bdays\b", t)
    ):
        out.append("claim")
    if _has(r"\bagreement\b|\bdeed\b", t) and _has(
        r"\bbetween\b|\bparties\b|hereby agreed|whereas", t, 2
    ):
        out.append("contract")
    if (
        _has(r"\bFrom\s*:|\bDear\b", t)
        and _has(r"\bTo\s*:|\bSent\s*:|\bAttention\b", t)
        and _has(r"Subject|\bRe\s*:", t)
        and _has(r"regards|yours (?:faithfully|sincerely)|thank", t)
    ):
        out.append("correspondence")
    codes = set(re.findall(r"\b[A-Z]{2,3}-\d{2}\b", t))
    if t.count("|") >= 20 and len(codes) >= 4:
        out.append("register/log")
    if len(re.findall(DATE, t)) >= 15 and _has(
        r"\bactivity\b|\bstart\b|\bfinish\b|programme|duration", t, 3
    ):
        out.append("schedule")
    return out


def primary(text: str) -> str:
    found = rules(text)
    return found[0] if found else "none"
