"""Reconcile every record in data/raw/slm-datasets.zip with its destination.

Each JSONL line in the archive is traced by its source id (member:line when the
record carries no id field) and content hash to exactly one of
train / held_out / regression / retrieval / dropped (with a reason). YAML and
JSON members are retrieval inputs (formula registry, indexes, facilities,
manifests) or notes; they are listed per file. The table is written to
docs/audit/raw-reconciliation/raw_lines.jsonl and a summary to
docs/audit/raw-reconciliation/summary.json. Item text is not copied; rows carry the
member path, line number, content hash, split, and reason only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "data"))

import apply_split as A  # noqa: E402

TRAIN = Path("data/train/pairs.jsonl")
HELD = Path("data/held_out/pairs.jsonl")
REGRESSION = Path("data/eval/regression/pairs.jsonl")
FACTS = Path("data/retrieval/fact_pairs.jsonl")
OUT = Path("docs/audit/raw-reconciliation")


def _from_raw(meta: dict) -> bool:
    """A row built from a raw-archive line (not augmentation, not a data/sources file)."""
    return not meta.get("aug") and not str(meta.get("source_relpath", "")).startswith("data/")


def _hashes(root: Path, rel: Path) -> set[str]:
    path = root / rel
    if not path.is_file():
        return set()
    found = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            meta = json.loads(line)["meta"]
            if _from_raw(meta):
                found.add(meta["content_hash"])
    return found


def _non_raw(root: Path, rel: Path) -> dict[str, int]:
    path = root / rel
    counts: Counter = Counter()
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                meta = json.loads(line)["meta"]
                if not _from_raw(meta):
                    counts[meta.get("aug") or meta.get("source_relpath")] += 1
    return dict(sorted(counts.items()))


def _record_id(record: dict) -> str:
    meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
    for key in ("id", "case_id", "probe_id", "pair_id"):
        for source in (record, meta):
            value = source.get(key)
            if isinstance(value, (str, int)) and str(value).strip():
                return str(value)
    return ""


def reconcile(root: Path) -> tuple[list[dict], dict]:
    _, members = A._load_zip(root / "data/raw/slm-datasets.zip")
    splits = {
        "train": _hashes(root, TRAIN),
        "held_out": _hashes(root, HELD),
        "regression": _hashes(root, REGRESSION),
        "retrieval_fact": _hashes(root, FACTS),
    }
    rows: list[dict] = []
    pairs: list[dict] = []
    for rel, payload in members:
        suffix = rel.rsplit(".", 1)[-1].lower()
        if suffix != "jsonl":
            kind = {"yaml": "retrieval_or_note", "yml": "retrieval_or_note", "json": "manifest_note", "txt": "clause_source_note"}.get(suffix, "unknown")
            rows.append({"member": rel, "line": None, "id": "", "content_hash": "", "kind": "file", "destination": "retrieval", "reason": kind})
            continue
        for line_no, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            schema = A._chat_schema(record)
            row = {"member": rel, "line": line_no, "id": _record_id(record) or f"{rel}:{line_no}", "content_hash": "", "kind": "", "destination": "", "reason": ""}
            if schema is None and "reason" in record and "record" in record:
                schema = "unroutable"
            if schema is None:
                row["kind"] = "rate" if {"market", "year", "source"} <= set(record) else ("clause" if {"clause_number", "text"} <= set(record) else "other")
                row["destination"] = "retrieval"
                row["reason"] = f"{row['kind']} record"
                rows.append(row)
                continue
            chat = A.to_chat(rel, record, schema)
            regression = True if schema == "unroutable" else A._is_regression_record(rel, record, schema)
            row["content_hash"] = chat["meta"]["content_hash"]
            row["kind"] = "pair"
            row["_regression"] = regression
            row["_formula_key"] = (chat["meta"].get("formula_id"), chat["meta"].get("pair_type"), chat["user"]) if chat["meta"].get("formula_id") else None
            rows.append(row)
            pairs.append(row)

    # Exact-duplicate and formula-key collapses, mirroring apply_split.
    for group_regression in (True, False):
        by_hash: dict[str, list[dict]] = defaultdict(list)
        for row in pairs:
            if row["_regression"] == group_regression:
                by_hash[row["content_hash"]].append(row)
        for digest, group in by_hash.items():
            keeper = min(group, key=lambda r: A._prefer_rank(r["member"]))
            for row in group:
                if row is not keeper:
                    row["destination"] = "dropped"
                    row["reason"] = f"exact duplicate of {keeper['member']}:{keeper['line']} (same user+assistant hash)"
    keyed: dict[tuple, list[dict]] = defaultdict(list)
    for row in pairs:
        if not row["_regression"] and not row["destination"] and row["_formula_key"]:
            keyed[row["_formula_key"]].append(row)
    for key, group in keyed.items():
        keeper = min(group, key=lambda r: A._prefer_rank(r["member"]))
        for row in group:
            if row is not keeper:
                row["destination"] = "dropped"
                row["reason"] = f"same formula_id, pair_type and question as {keeper['member']}:{keeper['line']}; the from-source copy is kept"

    for row in pairs:
        if row["destination"]:
            continue
        found = [name for name, digests in splits.items() if row["content_hash"] in digests]
        if row["_regression"]:
            if "regression" in found:
                row["destination"], row["reason"] = "regression", "eval-only (adversarial / blind / unroutable / eval_only)"
            else:
                row["destination"], row["reason"] = "MISSING", "regression record not in the sealed regression file"
        elif "regression" in found:
            row["destination"], row["reason"] = "dropped", "same content as a sealed regression record"
        elif "retrieval_fact" in found:
            row["destination"], row["reason"] = "retrieval", "assistant carries a fact (number with unit, currency, clause+edition, brand, or site)"
        elif "held_out" in found:
            row["destination"], row["reason"] = "held_out", "held-out (full held-out domain or 15% stratified)"
        elif "train" in found:
            row["destination"], row["reason"] = "train", "posture pair"
        else:
            row["destination"], row["reason"] = "MISSING", "not found in any split"
    for row in rows:
        row.pop("_regression", None)
        row.pop("_formula_key", None)

    summary: dict = {"lines": sum(1 for r in rows if r["kind"] != "file"), "files_non_jsonl": sum(1 for r in rows if r["kind"] == "file")}
    by_member: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        if row["kind"] != "file":
            by_member[row["member"]][row["destination"]] += 1
    summary["by_member"] = {member: dict(sorted(counts.items())) for member, counts in sorted(by_member.items())}
    summary["by_destination"] = dict(sorted(Counter(r["destination"] for r in rows if r["kind"] != "file").items()))
    summary["dropped_reasons"] = dict(sorted(Counter(r["reason"].split(" of ")[0].split(" as ")[0] for r in rows if r["destination"] == "dropped").items()))
    summary["split_counts"] = {name: len(digests) for name, digests in splits.items()}
    summary["non_raw_rows"] = {"train": _non_raw(root, TRAIN), "held_out": _non_raw(root, HELD)}
    return rows, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--check", action="store_true", help="Fail if a line is missing or the committed table is stale")
    args = parser.parse_args(argv)
    rows, summary = reconcile(args.root)
    table = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows)
    summary_text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    out = args.root / OUT
    missing = [r for r in rows if r["destination"] == "MISSING"]
    if args.check:
        stale = (out / "raw_lines.jsonl").read_text(encoding="utf-8") != table or (out / "summary.json").read_text(encoding="utf-8") != summary_text
        if missing or stale:
            print(f"reconciliation: missing={len(missing)} stale={stale}")
            return 1
        print("reconciliation: ok")
        return 0
    out.mkdir(parents=True, exist_ok=True)
    (out / "raw_lines.jsonl").write_text(table, encoding="utf-8")
    (out / "summary.json").write_text(summary_text, encoding="utf-8")
    print(summary_text)
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
