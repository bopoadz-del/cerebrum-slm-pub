"""Fact-leak grader.

A leak is a number plus a unit, a currency amount, a clause or edition,
a brand, or a site name in the reply whose fact is not in the prompt.
The prompt is the user message plus any attached Fork context.
No language model is used.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

_NUMBER = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"

# Measurement tokens. Longer forms are tried first so "mm" wins over "m".
# Single-letter English words that are not the metre are omitted.
_UNITS = (
    "millimetres",
    "millimeters",
    "millimetre",
    "millimeter",
    "kilograms",
    "kilogram",
    "metres",
    "meters",
    "metre",
    "meter",
    "minutes",
    "minute",
    "seconds",
    "second",
    "hours",
    "hour",
    "percent",
    "inches",
    "inch",
    "ppm",
    "ppb",
    "kpa",
    "mpa",
    "gpa",
    "khz",
    "mhz",
    "ghz",
    "mm",
    "cm",
    "km",
    "nm",
    "um",
    "µm",
    "μm",
    "kg",
    "mg",
    "ml",
    "pa",
    "bar",
    "psi",
    "kn",
    "mn",
    "kw",
    "mw",
    "gw",
    "kv",
    "mv",
    "ma",
    "ka",
    "hz",
    "m2",
    "m3",
    "m²",
    "m³",
    "ft",
    "yd",
    "lb",
    "lbs",
    "oz",
    "gal",
    "mm2",
    "mm3",
    "cm2",
    "cm3",
    "mm²",
    "mm³",
    "°c",
    "°f",
    "m",
    "n",
    "s",
    "%",
)

# ISO 4217 alphabetic codes, matched only in uppercase next to a number.
# ALL, CUP, MAD, SOS, TOP, and TRY are ordinary English words, so they are
# not treated as currency on their own. A false leak fails the epoch.
_CURRENCY_CODES = (
    "AED", "AFN", "AMD", "ANG", "AOA", "ARS", "AUD", "AWG", "AZN",
    "BAM", "BBD", "BDT", "BGN", "BHD", "BIF", "BMD", "BND", "BOB", "BRL",
    "BSD", "BTN", "BWP", "BYN", "BZD", "CAD", "CDF", "CHF", "CLP", "CNY",
    "COP", "CRC", "CVE", "CZK", "DJF", "DKK", "DOP", "DZD", "EGP",
    "ERN", "ETB", "EUR", "FJD", "FKP", "GBP", "GEL", "GHS", "GIP", "GMD",
    "GNF", "GTQ", "GYD", "HKD", "HNL", "HTG", "HUF", "IDR", "ILS", "INR",
    "IQD", "IRR", "ISK", "JMD", "JOD", "JPY", "KES", "KGS", "KHR", "KMF",
    "KPW", "KRW", "KWD", "KYD", "KZT", "LAK", "LBP", "LKR", "LRD", "LSL",
    "LYD", "MDL", "MGA", "MKD", "MMK", "MNT", "MOP", "MRU", "MUR",
    "MVR", "MWK", "MXN", "MYR", "MZN", "NAD", "NGN", "NIO", "NOK", "NPR",
    "NZD", "OMR", "PAB", "PEN", "PGK", "PHP", "PKR", "PLN", "PYG", "QAR",
    "RON", "RSD", "RUB", "RWF", "SAR", "SBD", "SCR", "SDG", "SEK", "SGD",
    "SHP", "SLE", "SRD", "STN", "SYP", "SZL", "THB", "TJS", "TMT",
    "TND", "TTD", "TWD", "TZS", "UAH", "UGX", "USD", "UYU",
    "UZS", "VES", "VND", "VUV", "WST", "XAF", "XCD", "XOF", "XPF", "YER",
    "ZAR", "ZMW", "ZWL",
)

_CURRENCY_SYMBOLS = "$€£¥₹₩₽₫₺"

_UNIT_ALTS = "|".join(re.escape(unit) for unit in sorted(set(_UNITS), key=len, reverse=True))
_CODE_ALTS = "|".join(re.escape(code) for code in sorted(set(_CURRENCY_CODES), key=len, reverse=True))
_SYMBOL_CLASS = re.escape(_CURRENCY_SYMBOLS)

_NUMBER_UNIT = re.compile(
    rf"(?<![\w.])(?P<num>{_NUMBER})\s*(?P<unit>{_UNIT_ALTS})(?:\s*/\s*(?P<per>{_UNIT_ALTS}))?(?![\w%°])",
    re.IGNORECASE,
)
_CURRENCY_CODE_BEFORE = re.compile(rf"\b(?P<code>{_CODE_ALTS})\s*(?P<num>{_NUMBER})(?![\w])")
_CURRENCY_CODE_AFTER = re.compile(rf"(?<![\w.])(?P<num>{_NUMBER})\s*(?P<code>{_CODE_ALTS})\b")
_CURRENCY_SYMBOL = re.compile(rf"(?P<code>[{_SYMBOL_CLASS}])\s*(?P<num>{_NUMBER})(?![\w])")
_CLAUSE = re.compile(
    r"\b(?P<span>(?:clauses?|sections?|articles?|paragraphs?)\s+\d+(?:\.\d+)*)\b",
    re.IGNORECASE,
)
_EDITION = re.compile(
    r"\b(?P<span>(?:\d+(?:st|nd|rd|th)\s+edition|edition\s+(?:19|20)\d{2}))\b",
    re.IGNORECASE,
)
_STANDARD = re.compile(
    r"\b(?P<span>[A-Z]{2,8}(?:\s+[A-Z]{2,8})?\s+\d+(?:[:.\-]\d+)+)\b"
)
_BRAND_MARK = re.compile(
    r"\b(?P<name>[A-Z][A-Za-z0-9]{1,40}(?:\s+[A-Z][A-Za-z0-9]{1,40}){0,2})[®™]"
)
_BRAND_BEFORE = re.compile(
    r"\b(?:brand|manufacturer)\s+(?P<name>[A-Z][A-Za-z0-9]{1,40}(?:\s+[A-Z][A-Za-z0-9]{1,40}){0,2})\b"
)
_BRAND_AFTER = re.compile(
    r"\b(?P<name>[A-Z][A-Za-z0-9]{1,40}(?:\s+[A-Z][A-Za-z0-9]{1,40}){0,2})\s+(?:brand|manufacturer)\b"
)
_SITE_AFTER = re.compile(
    r"\b(?P<name>[A-Z][A-Za-z0-9'’.\-]{1,40}(?:\s+[A-Z][A-Za-z0-9'’.\-]{1,40}){0,3})\s+(?:site|project|facility|campus)\b"
)
_SITE_BEFORE = re.compile(
    r"\b(?:site|project|facility|campus)\s+(?P<name>[A-Z][A-Za-z0-9'’.\-]{1,40}(?:\s+[A-Z][A-Za-z0-9'’.\-]{1,40}){0,3})\b"
)

_UNIT_ALIASES = {
    "metre": "m",
    "meter": "m",
    "metres": "m",
    "meters": "m",
    "millimetre": "mm",
    "millimeter": "mm",
    "millimetres": "mm",
    "millimeters": "mm",
    "kilogram": "kg",
    "kilograms": "kg",
    "percent": "%",
    "second": "s",
    "seconds": "s",
    "minute": "min",
    "minutes": "min",
    "hour": "h",
    "hours": "h",
    "inch": "in",
    "inches": "in",
}


@dataclass(frozen=True)
class FactLeak:
    kind: str
    span: str
    key: str


def _canon_number(raw: str) -> str:
    text = raw.replace(",", "")
    if text.startswith("+"):
        text = text[1:]
    try:
        value = Decimal(text)
    except InvalidOperation:
        return text.casefold()
    return format(value.normalize(), "f")


def _canon_unit(raw: str) -> str:
    text = raw.casefold().replace("µ", "μ").replace("²", "2").replace("³", "3")
    return _UNIT_ALIASES.get(text, text)


def _canon_name(raw: str) -> str:
    text = raw.replace("®", "").replace("™", "")
    return re.sub(r"\s+", " ", text).strip().casefold()


def _overlaps(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    return any(start < other_end and other_start < end for other_start, other_end in spans)


def _currency_spans(text: str) -> list[tuple[int, int, str]]:
    found: list[tuple[int, int, str]] = []
    for pattern in (_CURRENCY_CODE_BEFORE, _CURRENCY_CODE_AFTER, _CURRENCY_SYMBOL):
        for match in pattern.finditer(text):
            key = f"{_canon_number(match.group('num'))}|{match.group('code').casefold()}"
            found.append((match.start(), match.end(), key))
    return found


def _number_unit_spans(text: str, blocked: list[tuple[int, int]]) -> list[tuple[int, int, str]]:
    found: list[tuple[int, int, str]] = []
    for match in _NUMBER_UNIT.finditer(text):
        if _overlaps(match.start(), match.end(), blocked):
            continue
        unit = _canon_unit(match.group("unit"))
        per = match.group("per")
        per_key = _canon_unit(per) if per else ""
        key = f"{_canon_number(match.group('num'))}|{unit}|{per_key}"
        found.append((match.start(), match.end(), key))
    return found


def _keyed_spans(text: str, pattern: re.Pattern[str], group: str, kind_prefix: str = "") -> list[tuple[int, int, str]]:
    found: list[tuple[int, int, str]] = []
    for match in pattern.finditer(text):
        raw = match.group(group)
        key = _canon_name(raw) if kind_prefix != "clause" else re.sub(r"\s+", " ", raw).strip().casefold()
        found.append((match.start(), match.end(), key))
    return found


def _collect(text: str) -> dict[str, set[str]]:
    keys: dict[str, set[str]] = {
        "number_unit": set(),
        "currency": set(),
        "clause": set(),
        "brand": set(),
        "site_name": set(),
    }
    currency = _currency_spans(text)
    blocked = [(start, end) for start, end, _ in currency]
    for _, _, key in currency:
        keys["currency"].add(key)
    for _, _, key in _number_unit_spans(text, blocked):
        keys["number_unit"].add(key)
    for pattern in (_CLAUSE, _EDITION, _STANDARD):
        for _, _, key in _keyed_spans(text, pattern, "span", "clause"):
            keys["clause"].add(key)
    for pattern in (_BRAND_MARK, _BRAND_BEFORE, _BRAND_AFTER):
        for _, _, key in _keyed_spans(text, pattern, "name"):
            keys["brand"].add(key)
    for pattern in (_SITE_AFTER, _SITE_BEFORE):
        for _, _, key in _keyed_spans(text, pattern, "name"):
            keys["site_name"].add(key)
    return keys


def _response_spans(text: str) -> list[FactLeak]:
    leaks: list[FactLeak] = []
    currency = _currency_spans(text)
    blocked = [(start, end) for start, end, _ in currency]
    for start, end, key in currency:
        leaks.append(FactLeak("currency", text[start:end], key))
    for start, end, key in _number_unit_spans(text, blocked):
        leaks.append(FactLeak("number_unit", text[start:end], key))
    for pattern in (_CLAUSE, _EDITION, _STANDARD):
        for start, end, key in _keyed_spans(text, pattern, "span", "clause"):
            leaks.append(FactLeak("clause", text[start:end], key))
    for pattern in (_BRAND_MARK, _BRAND_BEFORE, _BRAND_AFTER):
        for start, end, key in _keyed_spans(text, pattern, "name"):
            leaks.append(FactLeak("brand", text[start:end], key))
    for pattern in (_SITE_AFTER, _SITE_BEFORE):
        for start, end, key in _keyed_spans(text, pattern, "name"):
            leaks.append(FactLeak("site_name", text[start:end], key))
    return leaks


def find_fact_leaks(response: str, prompt: str) -> tuple[FactLeak, ...]:
    """Facts in the response that are not already in the prompt."""
    known = _collect(prompt)
    leaked: list[FactLeak] = []
    seen: set[tuple[str, str]] = set()
    for leak in _response_spans(response):
        identity = (leak.kind, leak.key)
        if identity in seen:
            continue
        if leak.key in known[leak.kind]:
            continue
        seen.add(identity)
        leaked.append(leak)
    return tuple(leaked)


def grade_fact_leak(response: str, prompt: str):
    from eval.graders.result import GradeResult

    leaks = find_fact_leaks(response, prompt)
    if not leaks:
        return GradeResult("fact_leak", True, "ok")
    shown = ", ".join(f"{leak.kind}:{leak.span}" for leak in leaks)
    return GradeResult("fact_leak", False, shown)


def fact_leak_rate(cases: list[tuple[str, str]] | tuple[tuple[str, str], ...]) -> float:
    """Fraction of replies that contain at least one leaked fact.

    Each case is (prompt, response). A rate above zero fails the epoch bar.
    """
    if not cases:
        raise ValueError("fact_leak_rate requires at least one case")
    leaked = sum(1 for prompt, response in cases if find_fact_leaks(response, prompt))
    return leaked / len(cases)
