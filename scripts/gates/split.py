"""THE SPLIT: a fact in an assistant turn is retrieval data, not a training pair."""

from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

from gates.findings import Finding
from gates.records import RAW_ZIP, RecordParseError, iter_training_files, load_records, pair_view, relpath

# Longer tokens first so "mm" wins over "m" and "hours" wins over "hr".
_UNIT_TOKENS = (
    "minutes",
    "minute",
    "seconds",
    "second",
    "months",
    "month",
    "hours",
    "tonnes",
    "tonne",
    "weeks",
    "week",
    "m²",
    "m³",
    "mm²",
    "days",
    "mins",
    "secs",
    "GPa",
    "MPa",
    "kPa",
    "kWh",
    "MWh",
    "kN/m",
    "kNm",
    "ton",
    "tons",
    "hour",
    "hrs",
    "day",
    "min",
    "sec",
    "mm",
    "cm",
    "km",
    "m2",
    "m3",
    "kg",
    "kN",
    "kW",
    "MW",
    "GW",
    "Pa",
    "hr",
    "°C",
    "℃",
    "°F",
    "m/s",
    "l/s",
    "bar",
    "psi",
    "ha",
    "m",
    "%",
)
_NUM_UNIT = re.compile(
    r"(?<![A-Za-z0-9])\d+(?:\.\d+)?(?:\s*|-)?(?:"
    + "|".join(re.escape(token) for token in sorted(_UNIT_TOKENS, key=len, reverse=True))
    + r")(?![A-Za-z0-9])"
)
_ISO = r"USD|SAR|AED|QAR|EUR|GBP|KWD|BHD|OMR"
_CURRENCY_WORD = r"riyals?|dirhams?|dollars?|euros?|pounds?"
_AMOUNT = r"\d[\d,]*(?:\.\d+)?"
# A bare code is not an amount. "cement price sar t" names a missing input.
_CURRENCY = re.compile(
    rf"(?i)(?:[$€£]\s?{_AMOUNT}"
    rf"|(?<![A-Za-z])(?:{_ISO})\s*{_AMOUNT}(?![A-Za-z0-9])"
    rf"|(?<![A-Za-z0-9]){_AMOUNT}\s*(?:{_ISO})(?![A-Za-z])"
    rf"|(?<![A-Za-z0-9]){_AMOUNT}\s*(?:{_CURRENCY_WORD})(?![A-Za-z]))"
)
# BS 8110 and CIRIA have no edition. ACI 318-19 and EN 1992-1-1 do.
_STANDARD = re.compile(
    r"\b(?:BS\s*EN|BS|EN|ISO|IEC|ACI|ASTM|ASME|AASHTO|DIN|CIRIA|NFPA|IBC|UBC)"
    r"\s*[A-Z]?\d[\w]*[-:.][\w./:-]*[A-Za-z0-9]",
    re.IGNORECASE,
)
_CLAUSE = re.compile(r"(?i)\b(?:clauses?|cl\.|sections?|tables?)\s+\d[\d.]*")
_EDITION = re.compile(r"(?i)\b(?:editions?|ed\.)\b|\b(?:19|20)\d{2}\b")
_HOST = re.compile(r"\b(?:[a-z0-9-]+\.)+(?:com|sa|net|org|ae|qa|uk)\b", re.IGNORECASE)
_MONTHS = r"Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
_DROP_WORDS = re.compile(
    r"\b(?:Handbook|guide|Property|Construction|Middle|East|KSAMI)\b",
    re.IGNORECASE,
)


class TermsError(Exception):
    def __init__(self, path: str, message: str) -> None:
        super().__init__(message)
        self.path = path


@dataclass(frozen=True)
class RetrievalTerms:
    brands: frozenset[str]
    sites: frozenset[str]
    clause_names: tuple[tuple[str, str], ...]


EMPTY_TERMS = RetrievalTerms(frozenset(), frozenset(), ())


def leak_kinds(text: str, terms: RetrievalTerms) -> list[str]:
    """Kinds of retrieval fact carried by one assistant turn."""
    kinds: list[str] = []
    if _NUM_UNIT.search(text):
        kinds.append("number-unit")
    if _CURRENCY.search(text):
        kinds.append("currency")
    if _has_clause_edition(text, terms):
        kinds.append("clause-edition")
    if _matched_phrase(text, terms.brands):
        kinds.append("brand")
    if _matched_phrase(text, terms.sites):
        kinds.append("site")
    return kinds


def check_split(root: Path) -> list[Finding]:
    try:
        terms = load_retrieval_terms(root)
    except TermsError as exc:
        return [Finding("split", exc.path, str(exc))]
    findings: list[Finding] = []
    for path in iter_training_files(root):
        where = relpath(path, root)
        try:
            rows = load_records(path)
        except RecordParseError as exc:
            findings.append(Finding("split", where, str(exc)))
            continue
        for lineno, obj in rows:
            view = pair_view(obj)
            if view is None:
                continue
            for turn in view.assistant_turns:
                kinds = leak_kinds(turn, terms)
                if kinds:
                    findings.append(
                        Finding(
                            "split",
                            where,
                            f"line {lineno}: assistant turn contains {', '.join(kinds)}",
                        )
                    )
    return findings


def load_retrieval_terms(root: Path) -> RetrievalTerms:
    brands: set[str] = set()
    sites: set[str] = set()
    clauses: set[tuple[str, str]] = set()
    _consume_zip_terms(root, brands, sites, clauses)
    _consume_retrieval_tree(root, brands, sites, clauses)
    sites_folded = {site.casefold() for site in sites}
    brands = {brand for brand in brands if brand.casefold() not in sites_folded}
    return RetrievalTerms(
        brands=frozenset(brands),
        sites=frozenset(sites),
        clause_names=tuple(sorted(clauses)),
    )


def _consume_zip_terms(
    root: Path,
    brands: set[str],
    sites: set[str],
    clauses: set[tuple[str, str]],
) -> None:
    zip_path = root / RAW_ZIP
    if not zip_path.is_file():
        return
    try:
        zf = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile as exc:
        raise TermsError(RAW_ZIP.as_posix(), "raw archive is not a zip") from exc
    with zf:
        for info in zf.infolist():
            if info.is_dir() or not info.filename.endswith(".jsonl"):
                continue
            try:
                text = zf.read(info.filename).decode("utf-8")
            except UnicodeError as exc:
                raise TermsError(
                    f"{RAW_ZIP.as_posix()}!{info.filename}",
                    "member is not UTF-8",
                ) from exc
            for lineno, line in enumerate(text.splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    where = f"{RAW_ZIP.as_posix()}!{info.filename}"
                    raise TermsError(where, f"{where}:{lineno}: {exc}") from exc
                if isinstance(obj, dict):
                    _harvest_row(obj, brands, sites, clauses)


def _consume_retrieval_tree(
    root: Path,
    brands: set[str],
    sites: set[str],
    clauses: set[tuple[str, str]],
) -> None:
    base = root / "data" / "retrieval"
    if not base.is_dir():
        return
    for path in sorted(item for item in base.rglob("*") if item.is_file()):
        if path.suffix.lower() not in {".json", ".jsonl"}:
            continue
        try:
            rows = load_records(path)
        except RecordParseError as exc:
            raise TermsError(path.as_posix(), str(exc)) from exc
        for _lineno, obj in rows:
            _harvest_row(obj, brands, sites, clauses)


def _harvest_row(
    obj: dict,
    brands: set[str],
    sites: set[str],
    clauses: set[tuple[str, str]],
) -> None:
    market = obj.get("market")
    if isinstance(market, str) and len(market.strip()) >= 4:
        sites.add(market.strip())
    source = obj.get("source")
    if isinstance(source, str):
        for host in _HOST.findall(source):
            sites.add(host)
        for part in re.split(r"[;/]", source):
            brand = _clean_brand(part.strip().strip("."))
            if brand:
                brands.add(brand)
    name = obj.get("code_name")
    year = obj.get("edition_year")
    if isinstance(name, str) and isinstance(year, str) and name.strip() and year.strip():
        clauses.add((name.strip(), str(year).strip()))


def _clean_brand(part: str) -> str | None:
    if not part or re.match(r"(?i)RFP\b", part):
        return None
    if _HOST.search(part):
        return None
    cleaned = re.sub(r"\(.*?\)", "", part)
    cleaned = re.sub(rf"\b(?:{_MONTHS})\b\.?", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(?:19|20)\d{2}\b", "", cleaned)
    cleaned = _DROP_WORDS.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,-&")
    if len(cleaned) < 4:
        return None
    return cleaned


def _has_clause_edition(text: str, terms: RetrievalTerms) -> bool:
    if _STANDARD.search(text):
        return True
    if _CLAUSE.search(text) and _EDITION.search(text):
        return True
    for name, year in terms.clause_names:
        if _phrase_in(text, name) and (
            _year_in(text, year) or _EDITION.search(text) is not None
        ):
            return True
    return False


def _matched_phrase(text: str, phrases: frozenset[str]) -> str | None:
    for phrase in sorted(phrases, key=len, reverse=True):
        if len(phrase) < 4:
            continue
        if _phrase_in(text, phrase):
            return phrase
    return None


def _phrase_in(text: str, phrase: str) -> bool:
    return (
        re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text, flags=re.IGNORECASE)
        is not None
    )


def _year_in(text: str, year: str) -> bool:
    return re.search(rf"(?<!\d){re.escape(year)}(?!\d)", text) is not None
