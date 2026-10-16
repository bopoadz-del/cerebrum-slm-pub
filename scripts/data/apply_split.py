#!/usr/bin/env python3
"""Normalise the construction dataset and apply the Split.

Reads a source zip (the bytes are not modified) and writes:

- data/train/                  posture pairs, zero facts
- data/held_out/               one full domain, plus a stratified 15 percent
- data/retrieval/              rates, formula registry, clause store, facilities
- data/eval/regression/        adversarial, blind, unroutable, eval-only seal

The script is idempotent: the same zip produces the same bytes. Domain names
are copied from source fields into data. They are not selected by a list in
this file. Re-running overwrites only the four trees above.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

CHAT_KEYS = ("system", "user", "assistant", "meta")
HOLDOUT_FRACTION_NUMERATOR = 15
HOLDOUT_FRACTION_DENOMINATOR = 100

# A number bound to a unit. Mixed-case and no-vowel forms stay case-sensitive
# so ordinary words are not treated as units. Spelled measures are the Split.
_NUMBER_UNIT_ABBREV = re.compile(
    r"(?<![\w])\d+(?:[.,]\d+)?\s*(?:"
    r"[A-Za-z]{0,3}[A-Z][A-Za-z]{0,6}"
    r"|[bcdfghjklmnpqrstvwxyz]{2,4}"
    r"|m[23]"
    r")\b"
)
_NUMBER_UNIT_WORD = re.compile(
    r"(?i)(?<![\w])\d+(?:[.,]\d+)?\s*(?:"
    r"hours?|minutes?|seconds?|days?|weeks?|months?|years?"
    r"|metres?|meters?|tonnes?|tons?|litres?|liters?|hectares?"
    r"|min|hrs?|sec|ha"
    r")\b"
)
_NUMBER_PERCENT = re.compile(r"(?<![\w])\d+(?:[.,]\d+)?\s*%")
_EDITION = re.compile(
    r"\b(?:[A-Z]{2,8}\s+){1,3}[A-Z]?\d{2,5}(?:[.\-]\d+)*[A-Z]?\b"
)
_CLAUSE_WORD = re.compile(r"(?i)\b(?:clause|edition)\s+\d")
_CURRENCY_SYMBOL = re.compile(r"[$€£]\s*\d|\d[\d,]*(?:\.\d+)?\s*[$€£]")

_OWNED = (
    Path("data/train"),
    Path("data/retrieval"),
    Path("data/held_out"),
    Path("data/eval/regression"),
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_hash(obj: Any) -> str:
    return sha256_bytes(canonical_json(obj).encode("utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [canonical_json(row) for row in rows]
    text = ("\n".join(lines) + "\n") if lines else ""
    path.write_text(text, encoding="utf-8")


def holdout_count(population: int) -> int:
    """Round half up of 15 percent, with no minimum of one."""
    if population <= 0:
        return 0
    return (population * HOLDOUT_FRACTION_NUMERATOR + HOLDOUT_FRACTION_DENOMINATOR // 2) // HOLDOUT_FRACTION_DENOMINATOR


def looks_like_unit_fact(text: str) -> bool:
    return bool(
        _NUMBER_UNIT_ABBREV.search(text)
        or _NUMBER_UNIT_WORD.search(text)
        or _NUMBER_PERCENT.search(text)
    )


def looks_like_edition(text: str) -> bool:
    return _EDITION.search(text) is not None or _CLAUSE_WORD.search(text) is not None


def assistant_has_fact(text: str, lexicon: dict[str, Any]) -> bool:
    """True when an assistant turn states a fact the Split keeps out of weights."""
    if not text:
        return False
    if looks_like_unit_fact(text) or looks_like_edition(text):
        return True
    if _CURRENCY_SYMBOL.search(text):
        return True
    for code in lexicon.get("currencies", ()):
        if re.search(rf"\b{re.escape(code)}\b", text):
            return True
    for phrase in lexicon.get("sites", ()):
        if re.search(rf"(?i)(?<!\w){re.escape(phrase)}(?!\w)", text):
            return True
    return False


def _truthy_flag(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str) and value.strip().lower() == "true":
        return True
    return False


def _source_group(relpath: str) -> str:
    parts = PurePosix(relpath).parts
    return parts[0] if parts else ""


class PurePosix:
    """Tiny posix path helper so classification does not depend on the host OS."""

    def __init__(self, text: str) -> None:
        self.parts = tuple(part for part in text.split("/") if part and part != ".")

    @property
    def name(self) -> str:
        return self.parts[-1] if self.parts else ""

    @property
    def stem(self) -> str:
        name = self.name
        return name.rsplit(".", 1)[0] if "." in name else name


def _regression_path(relpath: str) -> bool:
    name = PurePosix(relpath).name.lower()
    return "adversarial" in name or "unroutable" in name or "_blind" in name


def _join_instruction(instruction: str, extra: str) -> str:
    instruction = instruction or ""
    extra = extra or ""
    if instruction and extra:
        return instruction + "\n\n" + extra
    return instruction or extra


def to_chat(relpath: str, record: dict[str, Any], schema: str) -> dict[str, Any]:
    meta_in = record.get("meta") if isinstance(record.get("meta"), dict) else {}
    meta: dict[str, Any] = dict(meta_in)
    if schema == "instruction_input_output":
        user = _join_instruction(str(record.get("instruction") or ""), str(record.get("input") or ""))
        assistant = str(record.get("output") or "")
    elif schema == "prompt_completion":
        user = str(record.get("prompt") or "")
        assistant = str(record.get("completion") or "")
        if record.get("type") and "pair_type" not in meta:
            meta["pair_type"] = record["type"]
    elif schema == "instruction_response":
        user = str(record.get("instruction") or "")
        assistant = str(record.get("response") or "")
        if record.get("pair_type") and "pair_type" not in meta:
            meta["pair_type"] = record["pair_type"]
        if record.get("formula_id"):
            meta["formula_id"] = record["formula_id"]
    elif schema == "unroutable":
        user = canonical_json(record.get("record", record))
        assistant = str(record.get("reason") or "")
        meta["reason"] = record.get("reason")
        meta["source_kind"] = record.get("source_kind")
    else:
        raise ValueError(f"unknown schema {schema}")

    if "domain" in meta_in and meta_in.get("domain"):
        meta["domain"] = meta_in["domain"]
        meta["domain_from_source"] = True
    else:
        meta.pop("domain", None)
        meta["domain_from_source"] = False

    meta["source_relpath"] = relpath
    meta["source_schema"] = schema
    meta["source_group"] = meta["domain"] if meta.get("domain_from_source") else _source_group(relpath)
    pair = {
        "system": "",
        "user": user,
        "assistant": assistant,
        "meta": meta,
    }
    meta["content_hash"] = content_hash(
        {"system": pair["system"], "user": pair["user"], "assistant": pair["assistant"]}
    )
    return pair


def _chat_schema(record: dict[str, Any]) -> str | None:
    keys = set(record)
    if {"instruction", "output"} <= keys:
        return "instruction_input_output"
    if {"prompt", "completion"} <= keys:
        return "prompt_completion"
    if {"instruction", "response"} <= keys:
        return "instruction_response"
    return None


def _is_regression_record(relpath: str, record: dict[str, Any], schema: str) -> bool:
    if schema == "unroutable" or _regression_path(relpath):
        return True
    meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
    if _truthy_flag(meta.get("eval_only")) or _truthy_flag(record.get("eval_only")):
        return True
    pair_type = str(meta.get("pair_type") or record.get("pair_type") or "")
    if pair_type.upper() == "ADVERSARIAL":
        return True
    return False


def _prefer_rank(relpath: str) -> tuple[int, str]:
    parts = PurePosix(relpath).parts
    from_fork_tree = 0 if "from-source" in parts else 1
    return (from_fork_tree, relpath)


def _load_zip(zip_path: Path) -> tuple[str, list[tuple[str, bytes]]]:
    raw = zip_path.read_bytes()
    digest = sha256_bytes(raw)
    members: list[tuple[str, bytes]] = []
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            name = info.filename
            if name.startswith("__MACOSX/") or name.endswith("/"):
                continue
            prefix = "slm-datasets/"
            rel = name[len(prefix):] if name.startswith(prefix) else name
            if not rel or rel.endswith("/"):
                continue
            members.append((rel, archive.read(info)))
    members.sort(key=lambda item: item[0])
    return digest, members


def _parse_members(members: list[tuple[str, bytes]]) -> dict[str, list[Any]]:
    buckets: dict[str, list[Any]] = {
        "pairs": [],
        "rates": [],
        "clauses": [],
        "formulas": [],
        "indexes": [],
        "facilities": [],
        "facility_notes": [],
        "licences": [],
        "notes": [],
        "rate_notes": [],
    }
    for rel, payload in members:
        suffix = PurePosix(rel).name.rsplit(".", 1)[-1].lower()
        if suffix == "jsonl":
            _consume_jsonl(buckets, rel, payload)
        elif suffix in {"yaml", "yml"}:
            _consume_yaml(buckets, rel, payload)
        elif suffix == "json":
            _consume_json(buckets, rel, payload)
        elif suffix == "txt":
            buckets["notes"].append({"source_relpath": rel, "text": payload.decode("utf-8")})
        else:
            raise ValueError(f"unrecognised source file: {rel}")
    return buckets


def _consume_jsonl(buckets: dict[str, list[Any]], rel: str, payload: bytes) -> None:
    text = payload.decode("utf-8")
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"{rel}:{line_no} is not an object")
        schema = _chat_schema(record)
        if schema:
            chat = to_chat(rel, record, schema)
            chat["meta"]["regression"] = _is_regression_record(rel, record, schema)
            buckets["pairs"].append(chat)
            continue
        if {"market", "year", "source"} <= set(record) and ("rate" in record or "currency" in record):
            record = dict(record)
            record["source_relpath"] = rel
            record["content_hash"] = content_hash(
                {key: record.get(key) for key in sorted(record) if key != "content_hash"}
            )
            buckets["rates"].append(record)
            continue
        if {"clause_number", "text"} <= set(record):
            record = dict(record)
            record["source_relpath"] = rel
            record["content_hash"] = content_hash(
                {key: record[key] for key in record if key != "content_hash"}
            )
            buckets["clauses"].append(record)
            continue
        if "reason" in record and "record" in record:
            chat = to_chat(rel, record, "unroutable")
            chat["meta"]["regression"] = True
            buckets["pairs"].append(chat)
            continue
        raise ValueError(f"{rel}:{line_no} did not match a known record shape")


def _consume_yaml(buckets: dict[str, list[Any]], rel: str, payload: bytes) -> None:
    document = yaml.safe_load(payload.decode("utf-8"))
    if isinstance(document, list):
        buckets["licences"].append({"source_relpath": rel, "entries": document})
        return
    if not isinstance(document, dict):
        raise ValueError(f"{rel} yaml is not a mapping")
    if "expression" in document or ("id" in document and "inputs" in document):
        buckets["formulas"].append(
            {
                "source_relpath": rel,
                "document": document,
                "id": document.get("id"),
                "content_hash": content_hash(document),
            }
        )
        return
    if "formulas" in document and "expression" not in document:
        buckets["indexes"].append({"source_relpath": rel, "document": document})
        return
    if "values" in document and isinstance(document["values"], list):
        for value in document["values"]:
            if isinstance(value, dict):
                buckets["facilities"].append(_facility_row(rel, value))
        return
    if "records" in document and isinstance(document["records"], list):
        for value in document["records"]:
            if isinstance(value, dict) and value.get("facility_id"):
                buckets["facilities"].append(_facility_row(rel, value))
        return
    buckets["facility_notes"].append({"source_relpath": rel, "document": document})


def _facility_row(rel: str, value: dict[str, Any]) -> dict[str, Any]:
    row = dict(value)
    row["source_relpath"] = rel
    row["content_hash"] = _facility_content_hash(row)
    return row


def _facility_content_hash(row: dict[str, Any]) -> str:
    return content_hash({key: row[key] for key in row if key not in {"content_hash", "source_relpath"}})


# Facility B's figures were invented so a lookup for A cannot return B.
# They are isolation data, not a measured sheet, so the worked-example label is replaced.
_ISOLATION_FACILITY_ID = "B"
_ISOLATION_SOURCE = "facility B is isolation data, not a measured example"


def _relabel_isolation_facility(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    relabeled: list[dict[str, Any]] = []
    for row in rows:
        if row.get("facility_id") != _ISOLATION_FACILITY_ID:
            relabeled.append(row)
            continue
        updated = dict(row)
        updated["source"] = _ISOLATION_SOURCE
        updated["content_hash"] = _facility_content_hash(updated)
        relabeled.append(updated)
    return relabeled


# The pack note attributed the worked-example source to A and B together.
# A remains that worked example. B is isolation data.
_PROVENANCE_GROUPS_B_WITH_WORKED_EXAMPLE = (
    "facility_id A and a second facility B, domain data_centre, source worked-example, date_stated 2026-10-04."
)
_PROVENANCE_A_WORKED_EXAMPLE_B_ISOLATION = (
    "facility A, domain data_centre, source worked-example, date_stated 2026-10-04. "
    "Facility B is isolation data, not a worked example."
)


def _correct_facility_provenance(notes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    corrected: list[dict[str, Any]] = []
    for note in notes:
        document = note.get("document")
        if not isinstance(document, dict) or "provenance_notes" not in document:
            corrected.append(note)
            continue
        provenance = []
        for text in document["provenance_notes"]:
            if isinstance(text, str):
                text = text.replace(
                    _PROVENANCE_GROUPS_B_WITH_WORKED_EXAMPLE,
                    _PROVENANCE_A_WORKED_EXAMPLE_B_ISOLATION,
                )
            provenance.append(text)
        document = dict(document)
        document["provenance_notes"] = provenance
        updated = dict(note)
        updated["document"] = document
        corrected.append(updated)
    return corrected


def _consume_json(buckets: dict[str, list[Any]], rel: str, payload: bytes) -> None:
    document = json.loads(payload.decode("utf-8"))
    note = {"source_relpath": rel, "document": document}
    if rel.startswith("rates/"):
        buckets["rate_notes"].append(note)
    else:
        buckets["facility_notes"].append(note)


def _dedupe_exact(pairs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    by_hash: dict[str, dict[str, Any]] = {}
    collapsed = 0
    for pair in pairs:
        digest = pair["meta"]["content_hash"]
        current = by_hash.get(digest)
        if current is None:
            by_hash[digest] = pair
            continue
        collapsed += 1
        if _prefer_rank(pair["meta"]["source_relpath"]) < _prefer_rank(current["meta"]["source_relpath"]):
            by_hash[digest] = pair
    kept = sorted(by_hash.values(), key=lambda pair: pair["meta"]["content_hash"])
    return kept, collapsed


def _prefer_formula_keys(pairs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Keep one row when both formula trees ask the same question.

    Content that hashes the same is already collapsed. When the wording
    differs, the from-source row is the Fork copy and the other is dropped.
    """
    grouped: dict[tuple[Any, Any, str], dict[str, Any]] = {}
    passthrough: list[dict[str, Any]] = []
    preferred = 0
    for pair in pairs:
        formula_id = pair["meta"].get("formula_id")
        if not formula_id:
            passthrough.append(pair)
            continue
        key = (formula_id, pair["meta"].get("pair_type"), pair["user"])
        current = grouped.get(key)
        if current is None:
            grouped[key] = pair
            continue
        preferred += 1
        if _prefer_rank(pair["meta"]["source_relpath"]) < _prefer_rank(current["meta"]["source_relpath"]):
            grouped[key] = pair
    kept = passthrough + list(grouped.values())
    kept.sort(key=lambda pair: pair["meta"]["content_hash"])
    return kept, preferred


def _dedupe_by_hash(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chosen: dict[str, dict[str, Any]] = {}
    for row in rows:
        digest = row["content_hash"]
        current = chosen.get(digest)
        if current is None or _prefer_rank(str(row.get("source_relpath", ""))) < _prefer_rank(str(current.get("source_relpath", ""))):
            chosen[digest] = row
    return sorted(chosen.values(), key=lambda row: row["content_hash"])


def _build_lexicon(rates: list[dict[str, Any]]) -> dict[str, Any]:
    currencies = set()
    sites = set()
    for row in rates:
        currency = str(row.get("currency") or "").strip()
        if re.fullmatch(r"[A-Z]{3}", currency):
            currencies.add(currency)
        market = str(row.get("market") or "").strip()
        if len(market) >= 4:
            sites.add(market)
    return {"currencies": sorted(currencies), "sites": sorted(sites, key=len, reverse=True)}


def _choose_holdout_domain(pairs: list[dict[str, Any]]) -> str | None:
    grouped: dict[str, int] = defaultdict(int)
    for pair in pairs:
        if not pair["meta"].get("domain_from_source"):
            continue
        domain = pair["meta"].get("domain")
        if isinstance(domain, str) and domain:
            grouped[domain] += 1
    if not grouped:
        return None
    ranked = sorted(grouped.items(), key=lambda item: (item[1], item[0]))
    index = (len(ranked) - 1) // 2
    return ranked[index][0]


def _formula_registry(formulas: list[dict[str, Any]], indexes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    excluded: set[str] = set()
    citations: dict[str, str] = {}
    repo = ""
    for index in indexes:
        document = index["document"]
        for helper in document.get("excluded_helpers") or []:
            excluded.add(str(helper))
        if document.get("repo") and "from-source" in PurePosix(index["source_relpath"]).parts:
            repo = str(document["repo"])
        for entry in document.get("formulas") or []:
            if not isinstance(entry, dict) or "id" not in entry:
                continue
            if entry.get("citation") and (
                entry["id"] not in citations or "from-source" in PurePosix(index["source_relpath"]).parts
            ):
                citations[str(entry["id"])] = str(entry["citation"])

    by_id: dict[str, dict[str, Any]] = {}
    for formula in formulas:
        formula_id = str(formula.get("id") or "")
        if not formula_id or formula_id in excluded:
            continue
        current = by_id.get(formula_id)
        if current is None or _prefer_rank(formula["source_relpath"]) < _prefer_rank(current["source_relpath"]):
            by_id[formula_id] = formula

    registry = []
    for formula_id in sorted(by_id):
        formula = by_id[formula_id]
        entry = {
            "id": formula_id,
            "content_hash": formula["content_hash"],
            "source_relpath": formula["source_relpath"],
            "definition": formula["document"],
        }
        if formula_id in citations:
            entry["citation"] = citations[formula_id]
        if repo:
            entry["repo"] = repo
        registry.append(entry)
    return registry


def _attach_fact_statements(registry: list[dict[str, Any]], facts: list[dict[str, Any]]) -> None:
    by_id = {entry["id"]: entry for entry in registry}
    for pair in facts:
        formula_id = pair["meta"].get("formula_id")
        if not formula_id or formula_id not in by_id:
            continue
        entry = by_id[formula_id]
        statements = entry.setdefault("excluded_routing_statements", [])
        text = pair["assistant"]
        if text not in statements:
            statements.append(text)
    for entry in registry:
        if "excluded_routing_statements" in entry:
            entry["excluded_routing_statements"].sort()


def _strip_internal(pair: dict[str, Any], split: str, holdout: str | None = None) -> dict[str, Any]:
    meta = dict(pair["meta"])
    meta.pop("regression", None)
    meta["split"] = split
    if holdout:
        meta["holdout"] = holdout
    else:
        meta.pop("holdout", None)
    return {
        "system": pair["system"],
        "user": pair["user"],
        "assistant": pair["assistant"],
        "meta": meta,
    }


def _refuse_router_sealed(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Refuse a training row whose document sha256 or project is in the router seal.

    The seal lives under data/eval and is read here, before any pair is written.
    The keys are exclusion keys. They do not assign a label.
    """
    scripts = Path(__file__).resolve().parents[1]
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from router.isolation import load_exclusions, require_isolated

    sealed = Path(__file__).resolve().parents[2] / "data" / "eval" / "router_unseen" / "sealed_labels.jsonl"
    return require_isolated(rows, load_exclusions(sealed))


def _prepare_output(root: Path) -> None:
    for relative in _OWNED:
        target = root / relative
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)


def apply_split(zip_path: Path, out_root: Path) -> dict[str, Any]:
    zip_path = zip_path.resolve()
    out_root = out_root.resolve()
    zip_hash, members = _load_zip(zip_path)
    buckets = _parse_members(members)

    rates = _dedupe_by_hash(buckets["rates"])
    rates.sort(key=lambda row: (str(row.get("market")), row.get("year"), str(row.get("description")), row["content_hash"]))
    for row in rates:
        for field in ("market", "year", "source"):
            if row.get(field) in (None, ""):
                raise ValueError(f"rate row missing {field}")

    clauses = _dedupe_by_hash(buckets["clauses"])
    clauses.sort(key=lambda row: (str(row.get("code_name")), str(row.get("clause_number")), row.get("page") or 0, row["content_hash"]))

    facilities = _dedupe_by_hash(buckets["facilities"])
    facilities = _relabel_isolation_facility(facilities)
    facilities.sort(key=lambda row: (str(row.get("facility_id")), str(row.get("parameter")), row["content_hash"]))

    registry = _formula_registry(buckets["formulas"], buckets["indexes"])

    raw_regression = [pair for pair in buckets["pairs"] if pair["meta"].get("regression")]
    raw_candidates = [pair for pair in buckets["pairs"] if not pair["meta"].get("regression")]
    regression, regression_collapsed = _dedupe_exact(raw_regression)
    candidates, candidate_collapsed = _dedupe_exact(raw_candidates)
    candidates, formula_preferred = _prefer_formula_keys(candidates)
    regression_hashes = {pair["meta"]["content_hash"] for pair in regression}
    sealed_overlap = 0
    kept_candidates: list[dict[str, Any]] = []
    for pair in candidates:
        if pair["meta"]["content_hash"] in regression_hashes:
            sealed_overlap += 1
            continue
        kept_candidates.append(pair)
    candidates = _refuse_router_sealed(kept_candidates)
    dedupe_stats = {
        "candidate_exact_collapsed": candidate_collapsed,
        "formula_key_preferred": formula_preferred,
        "regression_exact_collapsed": regression_collapsed,
        "sealed_overlap_removed": sealed_overlap,
    }
    lexicon = _build_lexicon(rates)

    facts: list[dict[str, Any]] = []
    posture: list[dict[str, Any]] = []
    for pair in candidates:
        if assistant_has_fact(pair["assistant"], lexicon):
            facts.append(pair)
        else:
            posture.append(pair)
    candidates = posture
    _attach_fact_statements(registry, facts)

    holdout_domain = _choose_holdout_domain(candidates)
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in candidates:
        by_stratum[str(pair["meta"].get("source_group") or "")].append(pair)

    held_domain: list[dict[str, Any]] = []
    held_stratified: list[dict[str, Any]] = []
    training: list[dict[str, Any]] = []
    stratum_report: dict[str, dict[str, int]] = {}
    for stratum in sorted(by_stratum):
        group = sorted(by_stratum[stratum], key=lambda pair: pair["meta"]["content_hash"])
        full_domain = bool(holdout_domain) and stratum == holdout_domain
        if full_domain:
            taken = group
            rest: list[dict[str, Any]] = []
            stratified_n = 0
        else:
            stratified_n = holdout_count(len(group))
            taken = group[:stratified_n]
            rest = group[stratified_n:]
        held_domain.extend(taken if full_domain else [])
        held_stratified.extend([] if full_domain else taken)
        training.extend(rest)
        stratum_report[stratum] = {
            "candidates": len(group),
            "train": len(rest),
            "held_out": len(taken),
            "full_domain": int(full_domain),
            "stratified": 0 if full_domain else stratified_n,
        }

    def _ordered(rows: list[dict[str, Any]], split: str, holdout: str | None = None) -> list[dict[str, Any]]:
        sealed = [_strip_internal(row, split, holdout) for row in rows]
        sealed.sort(key=lambda row: row["meta"]["content_hash"])
        return sealed

    train_rows = _ordered(training, "train")
    held_rows = _ordered(held_domain, "held_out", "domain") + _ordered(held_stratified, "held_out", "stratified")
    held_rows.sort(key=lambda row: row["meta"]["content_hash"])
    regression_rows = _ordered(regression, "regression")
    fact_rows = _ordered(facts, "retrieval")

    for row in train_rows + held_rows:
        if assistant_has_fact(row["assistant"], lexicon):
            raise ValueError("fact survived into a non-retrieval split")
        if holdout_domain and row["meta"].get("domain") == holdout_domain and row["meta"]["split"] == "train":
            raise ValueError("held-out domain survived into training")

    _prepare_output(out_root)

    train_path = out_root / "data/train/pairs.jsonl"
    held_path = out_root / "data/held_out/pairs.jsonl"
    regression_path = out_root / "data/eval/regression/pairs.jsonl"
    fact_path = out_root / "data/retrieval/fact_pairs.jsonl"
    rates_path = out_root / "data/retrieval/rates/rates.jsonl"
    clauses_path = out_root / "data/retrieval/clauses/clauses.jsonl"
    facilities_path = out_root / "data/retrieval/facilities/records.jsonl"
    registry_path = out_root / "data/retrieval/formula_registry.json"

    write_jsonl(train_path, train_rows)
    write_jsonl(held_path, held_rows)
    write_jsonl(regression_path, regression_rows)
    write_jsonl(fact_path, fact_rows)
    write_jsonl(rates_path, rates)
    write_jsonl(clauses_path, clauses)
    write_jsonl(facilities_path, facilities)
    write_json(registry_path, {"count": len(registry), "formulas": registry})

    if buckets["notes"]:
        note_dir = out_root / "data/retrieval/clauses/sources"
        note_dir.mkdir(parents=True, exist_ok=True)
        for note in buckets["notes"]:
            name = PurePosix(note["source_relpath"]).name
            (note_dir / name).write_text(note["text"], encoding="utf-8")

    regression_hash = sha256_file(regression_path)
    seal = {
        "algorithm": "sha256",
        "artifact": "pairs.jsonl",
        "count": len(regression_rows),
        "sha256": regression_hash,
        "source_zip_sha256": zip_hash,
    }
    write_json(out_root / "data/eval/regression/seal.json", seal)

    train_manifest = {
        "count": len(train_rows),
        "fact_count": 0,
        "schema": list(CHAT_KEYS),
        "source_zip_sha256": zip_hash,
        "by_stratum": {key: value["train"] for key, value in stratum_report.items()},
        "dedupe": dedupe_stats,
    }
    write_json(out_root / "data/train/manifest.json", train_manifest)

    held_manifest = {
        "count": len(held_rows),
        "domain_pair_count": len(held_domain),
        "fraction_numerator": HOLDOUT_FRACTION_NUMERATOR,
        "fraction_denominator": HOLDOUT_FRACTION_DENOMINATOR,
        "held_out_domain": holdout_domain,
        "holdout_rule": "median explicit domain ordered by (count ascending, name ascending); index (n-1)//2",
        "schema": list(CHAT_KEYS),
        "source_zip_sha256": zip_hash,
        "stratified_pair_count": len(held_stratified),
        "by_stratum": stratum_report,
    }
    write_json(out_root / "data/held_out/manifest.json", held_manifest)

    rate_sources = sorted({str(row.get("source")) for row in rates})
    write_json(
        out_root / "data/retrieval/rates/manifest.json",
        {
            "count": len(rates),
            "required_fields": ["market", "year", "source"],
            "sources": rate_sources,
            "source_manifests": buckets["rate_notes"],
            "source_zip_sha256": zip_hash,
        },
    )
    write_json(
        out_root / "data/retrieval/clauses/manifest.json",
        {
            "count": len(clauses),
            "licences": buckets["licences"],
            "source_zip_sha256": zip_hash,
        },
    )
    write_json(
        out_root / "data/retrieval/facilities/manifest.json",
        {
            "count": len(facilities),
            "notes": _correct_facility_provenance(buckets["facility_notes"]),
            "source_zip_sha256": zip_hash,
        },
    )
    write_json(
        out_root / "data/retrieval/manifest.json",
        {
            "clauses": len(clauses),
            "facilities": len(facilities),
            "fact_pairs": len(fact_rows),
            "formula_registry": len(registry),
            "rates": len(rates),
            "source_zip_sha256": zip_hash,
        },
    )

    summary = {
        "train": len(train_rows),
        "held_out": len(held_rows),
        "held_out_domain": holdout_domain,
        "held_out_domain_pairs": len(held_domain),
        "held_out_stratified": len(held_stratified),
        "regression": len(regression_rows),
        "regression_sha256": regression_hash,
        "rates": len(rates),
        "clauses": len(clauses),
        "facilities": len(facilities),
        "formulas": len(registry),
        "fact_pairs": len(fact_rows),
        "source_zip_sha256": zip_hash,
        "by_stratum": stratum_report,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalise pairs and apply the Split.")
    parser.add_argument("--zip", required=True, type=Path, help="Path to the source zip. Bytes are only read.")
    parser.add_argument("--out", required=True, type=Path, help="Repository root that receives data/ trees.")
    args = parser.parse_args()
    summary = apply_split(args.zip, args.out)
    print(canonical_json(summary))


if __name__ == "__main__":
    main()
