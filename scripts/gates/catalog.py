"""Sealed catalog derived at runtime. Probe ids are not checked in as a list."""

from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

from gates.records import (
    RAW_ZIP,
    PairView,
    RecordParseError,
    content_hash,
    iter_jsonl,
    load_records,
    pair_view,
    record_hash,
)

PROBE_FIELDS = ("source_case", "probe_id", "probe", "case_id")
PROBE_STOP = {"all", "none", "true", "false", "null", "test", "case"}
_PROBE_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{3,}\Z")
_DOMAIN_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}\Z")
_INV_PROBE = re.compile(r"INV-\d+\Z")
_YAML_DOMAIN = re.compile(
    r"(?m)^[ \t]*domain:[ \t]*['\"]?([A-Za-z][A-Za-z0-9_-]{2,})"
)
_YAML_PROBE = re.compile(
    r"(?m)^[ \t]*(?:source_case|probe_id|probe|case_id):[ \t]*['\"]?([A-Za-z0-9][A-Za-z0-9_.:-]{3,})"
)


class CatalogError(Exception):
    def __init__(self, path: str, message: str) -> None:
        super().__init__(message)
        self.path = path


@dataclass(frozen=True)
class Catalog:
    probe_ids: frozenset[str]
    domains: frozenset[str]
    content_hashes: frozenset[str]
    record_hashes: frozenset[str]


def _add_probe(bucket: set[str], value: object) -> None:
    if not isinstance(value, str):
        return
    token = value.strip()
    if len(token) < 5 or token.casefold() in PROBE_STOP:
        return
    if _PROBE_TOKEN.fullmatch(token) is None:
        return
    bucket.add(token)


def _add_domain(bucket: set[str], value: object) -> None:
    if not isinstance(value, str):
        return
    token = value.strip()
    if len(token) < 4 or _DOMAIN_TOKEN.fullmatch(token) is None:
        return
    # governance, datacenter: a single word is a training-domain label.
    # Sealed item names carry an underscore or a digit.
    if token.isalpha():
        return
    bucket.add(token)


def _harvest_identifiers(obj: dict, probes: set[str], domains: set[str]) -> None:
    sources: list[dict] = [obj]
    meta = obj.get("meta")
    if isinstance(meta, dict):
        sources.append(meta)
        pair_type = meta.get("pair_type")
        if isinstance(pair_type, str) and _INV_PROBE.fullmatch(pair_type):
            _add_probe(probes, pair_type)
    record = obj.get("record")
    if isinstance(record, dict):
        sources.append(record)
    for source in sources:
        for field in PROBE_FIELDS:
            _add_probe(probes, source.get(field))
        _add_domain(domains, source.get("domain"))


def _record_is_eval(obj: dict) -> bool:
    meta = obj.get("meta")
    return isinstance(meta, dict) and meta.get("eval_only") is True


def _member_is_eval_file(name: str) -> bool:
    base = name.rsplit("/", 1)[-1].lower()
    return (
        "_adversarial" in base
        or "governance_blind" in base
        or base.endswith("_blind.jsonl")
        or "unroutable" in base
    )


def _seal_record(
    obj: dict,
    content_hashes: set[str],
    record_hashes: set[str],
) -> None:
    record_hashes.add(record_hash(obj))
    view: PairView | None = pair_view(obj)
    if view is not None:
        content_hashes.add(content_hash(view.user, view.assistant))


def _read_zip_member(zf: zipfile.ZipFile, name: str) -> str:
    try:
        return zf.read(name).decode("utf-8")
    except UnicodeError as exc:
        raise CatalogError(f"{RAW_ZIP.as_posix()}!{name}", "member is not UTF-8") from exc


def _consume_zip(
    root: Path,
    probes: set[str],
    domains: set[str],
    content_hashes: set[str],
    record_hashes: set[str],
) -> None:
    zip_path = root / RAW_ZIP
    if not zip_path.is_file():
        return
    try:
        zf = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile as exc:
        raise CatalogError(RAW_ZIP.as_posix(), "raw archive is not a zip") from exc
    with zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = info.filename
            # Facility notes are not sealed eval items. Names come from eval records.
            if name.endswith((".yaml", ".yml")):
                _read_zip_member(zf, name)
                continue
            if not name.endswith(".jsonl"):
                continue
            text = _read_zip_member(zf, name)
            seal_file = _member_is_eval_file(name)
            for lineno, line in enumerate(text.splitlines(), 1):
                if not line.strip():
                    continue
                where = f"{RAW_ZIP.as_posix()}!{name}"
                try:
                    obj = _loads_object(line, where, lineno)
                except RecordParseError as exc:
                    raise CatalogError(where, str(exc)) from exc
                if seal_file or _record_is_eval(obj):
                    _harvest_identifiers(obj, probes, domains)
                    _seal_record(obj, content_hashes, record_hashes)


def _loads_object(line: str, where: str, lineno: int) -> dict:
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as exc:
        raise RecordParseError(where, lineno, str(exc)) from exc
    if not isinstance(obj, dict):
        raise RecordParseError(where, lineno, "record is not an object")
    return obj


def _consume_eval_tree(
    root: Path,
    probes: set[str],
    domains: set[str],
    content_hashes: set[str],
    record_hashes: set[str],
) -> None:
    base = root / "data" / "eval"
    if not base.is_dir():
        return
    for path in sorted(item for item in base.rglob("*") if item.is_file()):
        if path.suffix.lower() in {".yaml", ".yml"}:
            text = path.read_text(encoding="utf-8")
            for match in _YAML_DOMAIN.finditer(text):
                _add_domain(domains, match.group(1))
            for match in _YAML_PROBE.finditer(text):
                _add_probe(probes, match.group(1))
            continue
        if path.suffix.lower() not in {".json", ".jsonl"}:
            continue
        try:
            rows = load_records(path) if path.suffix.lower() == ".json" else iter_jsonl(path)
        except RecordParseError as exc:
            raise CatalogError(path.as_posix(), str(exc)) from exc
        for _lineno, obj in rows:
            _harvest_identifiers(obj, probes, domains)
            _seal_record(obj, content_hashes, record_hashes)


def build_catalog(root: Path) -> Catalog:
    """Names and hashes from sealed eval records only, not from training rows."""
    probes: set[str] = set()
    domains: set[str] = set()
    content_hashes: set[str] = set()
    record_hashes: set[str] = set()
    _consume_zip(root, probes, domains, content_hashes, record_hashes)
    _consume_eval_tree(root, probes, domains, content_hashes, record_hashes)
    return Catalog(
        probe_ids=frozenset(probes),
        domains=frozenset(domains),
        content_hashes=frozenset(content_hashes),
        record_hashes=frozenset(record_hashes),
    )
