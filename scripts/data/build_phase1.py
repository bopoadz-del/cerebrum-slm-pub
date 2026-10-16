#!/usr/bin/env python3
"""Phase-1 training set: confirmed parents, user-turn paraphrases, hard negatives.

Order of trust, as the owner set it:

1. Confirmed parents. The posture pairs ``apply_split`` produces from the raw
   archive, plus the owner-reviewed temporary-works pairs in
   ``data/sources/temporary-works/`` (``eval_only`` false). The temporary-works
   adversarial file is eval only and lives sealed under ``data/eval/``.
2. Question-side augmentation of confirmed train parents only. Each child keeps
   its parent's assistant turn byte for byte and carries ``meta.parent_id``.
   A paraphrase is dropped when it adds a number, unit, currency, clause,
   edition, brand, or site the parent user turn does not carry, when it repeats
   another user turn, or when it overlaps a sealed eval item.
3. Hard negatives. For each refuse-labelled train parent whose formula is in the
   committed registry, one near miss where the formula record is attached as
   Fork context and the right reply routes to it. The user turn and the target
   are a fixed frame filled from the registry record. The target cites only the
   formula id present in the attached context. No model writes either side.

Held-out is re-stratified over parents: the held-out domain stays whole, and
15 percent of every other source group (first by content hash) is held out.
Children of a held-out parent are never written to train. The blind set and
everything under ``data/eval/`` are not modified.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for _entry in (ROOT / "scripts", ROOT / "src", ROOT):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from eval.graders.fact_leak import find_fact_leaks  # noqa: E402
from eval.labels import _REFUSE_PAIR_TYPES  # noqa: E402
from train.posture_target import registry_inputs, target_for  # noqa: E402

_SPLIT_PATH = ROOT / "scripts" / "data" / "apply_split.py"
_spec = importlib.util.spec_from_file_location("apply_split_module", _SPLIT_PATH)
split = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules["apply_split_module"] = split
_spec.loader.exec_module(split)

RAW_ZIP = Path("data/raw/slm-datasets.zip")
TEMPORARY_WORKS = Path("data/sources/temporary-works/temporary_works_pairs.jsonl")
PARAPHRASES = Path("data/augment/user_paraphrases.jsonl")
REGISTRY = Path("data/retrieval/formula_registry.json")
SEALED_TREE = Path("data/eval")
MAX_CHILDREN = 3
MIN_CHILDREN = 3
OVERLAP_LIMIT = 0.8
HARD_NEGATIVE_PAIR_TYPE = "which_formula_applies"
AUG_PARAPHRASE = "paraphrase"
AUG_HARD_NEGATIVE = "hard_negative"
_TOKEN = re.compile(r"[a-z0-9]+")
_DIGITS = re.compile(r"\d+(?:[.,]\d+)?")


def tokens(text: str) -> frozenset[str]:
    return frozenset(_TOKEN.findall(text.casefold()))


def token_overlap(a: str, b: str) -> float:
    """Jaccard overlap of lower-cased word tokens."""
    left, right = tokens(a), tokens(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def normalise_user(text: str) -> str:
    return " ".join(_TOKEN.findall(text.casefold()))


def pair_hash(user: str, assistant: str) -> str:
    return split.content_hash({"system": "", "user": user, "assistant": assistant})


def new_facts(child: str, parent: str) -> list[str]:
    """Facts or bare numbers in the child user turn that the parent does not carry."""
    found = [f"{leak.kind}:{leak.span}" for leak in find_fact_leaks(child, parent)]
    parent_numbers = set(_DIGITS.findall(parent))
    found.extend(f"number:{number}" for number in _DIGITS.findall(child) if number not in parent_numbers)
    return found


def sealed_user_texts(root: Path) -> list[str]:
    """User-side text of every sealed eval record and of the raw archive's eval-only records."""
    texts: list[str] = []
    for path in sorted((root / SEALED_TREE).rglob("*")):
        if path.suffix not in {".jsonl", ".json"} or not path.is_file():
            continue
        lines = path.read_text(encoding="utf-8").splitlines() if path.suffix == ".jsonl" else []
        for line in lines:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = _user_side(obj)
            if text:
                texts.append(text)
    _, members = split._load_zip(root / RAW_ZIP)
    for rel, payload in members:
        if not rel.endswith(".jsonl"):
            continue
        for line in payload.decode("utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            schema = split._chat_schema(obj)
            if schema is None and not ("reason" in obj and "record" in obj):
                continue
            schema = schema or "unroutable"
            if split._is_regression_record(rel, obj, schema):
                text = _user_side(obj)
                if text:
                    texts.append(text)
    return texts


def _user_side(obj: dict) -> str:
    for key in ("user", "question", "prompt"):
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value
    instruction = obj.get("instruction")
    if isinstance(instruction, str) and instruction.strip():
        extra = obj.get("input") if isinstance(obj.get("input"), str) else ""
        return split._join_instruction(instruction, extra)
    if "record" in obj:
        return split.canonical_json(obj.get("record"))
    return ""


def max_sealed_overlap(text: str, sealed: list[str]) -> float:
    return max((token_overlap(text, other) for other in sealed), default=0.0)


def _refuse_router_sealed(rows: list[dict]) -> list[dict]:
    """Same exclusion as the split: a sealed doc sha256 or project cannot be trained."""
    from router.isolation import load_exclusions, require_isolated

    sealed = root_sealed_labels()
    return require_isolated(rows, load_exclusions(sealed))


def root_sealed_labels() -> Path:
    return ROOT / "data" / "eval" / "router_unseen" / "sealed_labels.jsonl"


def base_parents(root: Path) -> tuple[list[dict], str]:
    """Train and held-out rows ``apply_split`` writes from the raw archive."""
    with tempfile.TemporaryDirectory() as tmp:
        summary = split.apply_split(root / RAW_ZIP, Path(tmp))
        rows = _read_jsonl(Path(tmp) / "data/train/pairs.jsonl") + _read_jsonl(Path(tmp) / "data/held_out/pairs.jsonl")
    return _refuse_router_sealed(rows), summary["held_out_domain"]


def temporary_works_parents(root: Path) -> list[dict]:
    rel = "temporary-works/temporary_works_pairs.jsonl"
    rows: list[dict] = []
    for record in _read_jsonl(root / TEMPORARY_WORKS):
        if split._truthy_flag((record.get("meta") or {}).get("eval_only")):
            continue
        schema = split._chat_schema(record)
        if schema is None:
            raise ValueError("temporary-works record has no chat schema")
        chat = split.to_chat(rel, record, schema)
        chat["meta"]["source_relpath"] = TEMPORARY_WORKS.as_posix()
        chat["meta"]["source_origin"] = "owner_reviewed_local"
        rows.append(chat)
    return _refuse_router_sealed(rows)


def strip_parent(row: dict) -> dict:
    meta = {key: value for key, value in row["meta"].items() if key not in {"split", "holdout", "regression"}}
    return {"system": "", "user": row["user"], "assistant": row["assistant"], "meta": meta}


def stratify(parents: list[dict], holdout_domain: str) -> tuple[list[dict], list[dict], dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in parents:
        groups[str(row["meta"].get("source_group") or "")].append(row)
    train: list[dict] = []
    held: list[dict] = []
    report: dict[str, dict[str, int]] = {}
    for name in sorted(groups):
        group = sorted(groups[name], key=lambda row: row["meta"]["content_hash"])
        whole = bool(holdout_domain) and name == holdout_domain
        n_held = len(group) if whole else split.holdout_count(len(group))
        for row in group[:n_held]:
            row["meta"]["split"] = "held_out"
            row["meta"]["holdout"] = "domain" if whole else "stratified"
            held.append(row)
        for row in group[n_held:]:
            row["meta"]["split"] = "train"
            train.append(row)
        report[name] = {
            "candidates": len(group),
            "full_domain": int(whole),
            "held_out": n_held,
            "stratified": 0 if whole else n_held,
            "train": len(group) - n_held,
        }
    return train, held, report


def load_paraphrases(root: Path) -> dict[str, list[str]]:
    path = root / PARAPHRASES
    if not path.exists():
        return {}
    found: dict[str, list[str]] = {}
    for row in _read_jsonl(path):
        variants = [text for text in row.get("variants", []) if isinstance(text, str)]
        found[str(row["parent_id"])] = variants
    return found


def paraphrase_children(
    train_parents: list[dict],
    variants: dict[str, list[str]],
    sealed: list[str],
    taken_users: set[str],
) -> tuple[list[dict], dict[str, int]]:
    stats = {"offered": 0, "kept": 0, "dup": 0, "new_fact": 0, "sealed_overlap": 0, "split_fact": 0, "blank": 0}
    children: list[dict] = []
    lexicon_free = {}
    for parent in sorted(train_parents, key=lambda row: row["meta"]["content_hash"]):
        parent_id = parent["meta"]["content_hash"]
        kept = 0
        # Deterministic selection: stable hash order, then keep first MAX_CHILDREN.
        ordered = sorted(
            variants.get(parent_id, []),
            key=lambda text: hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest(),
        )
        for text in ordered:
            stats["offered"] += 1
            user = " ".join(text.split())
            if kept >= MAX_CHILDREN:
                break
            if not user:
                stats["blank"] += 1
                continue
            key = normalise_user(user)
            if key in taken_users:
                stats["dup"] += 1
                continue
            if new_facts(user, parent["user"]):
                stats["new_fact"] += 1
                continue
            if max_sealed_overlap(user, sealed) >= OVERLAP_LIMIT:
                stats["sealed_overlap"] += 1
                continue
            if split.assistant_has_fact(parent["assistant"], lexicon_free):
                stats["split_fact"] += 1
                continue
            taken_users.add(key)
            meta = dict(parent["meta"])
            meta.update(
                {
                    "aug": AUG_PARAPHRASE,
                    "parent_id": parent_id,
                    "split": "train",
                    "content_hash": pair_hash(user, parent["assistant"]),
                }
            )
            meta.pop("holdout", None)
            children.append({"system": "", "user": user, "assistant": parent["assistant"], "meta": meta})
            kept += 1
            stats["kept"] += 1
    return children, stats


def hard_negative_context(formula_id: str, entry: dict, required: tuple[str, ...]) -> str:
    computes = str((entry.get("definition") or {}).get("computes") or "").strip()
    lines = [f"[formula:{formula_id}]", f"formula_id: {formula_id}"]
    if computes:
        lines.append(f"computes: {computes}")
    lines.append("required inputs: " + (", ".join(required) if required else "none"))
    return "\n".join(lines)


def hard_negatives(train_parents: list[dict], registry: dict, sealed: list[str], taken_users: set[str]) -> tuple[list[dict], dict[str, int]]:
    inputs = registry_inputs(registry)
    entries = {entry["id"]: entry for entry in registry["formulas"]}
    stats = {"refuse_parents": 0, "built": 0, "no_registry_record": 0, "sealed_overlap": 0, "dup": 0}
    rows: list[dict] = []
    for parent in sorted(train_parents, key=lambda row: row["meta"]["content_hash"]):
        if parent["meta"].get("pair_type") not in _REFUSE_PAIR_TYPES:
            continue
        stats["refuse_parents"] += 1
        formula_id = str(parent["meta"].get("formula_id") or "")
        if formula_id not in entries:
            stats["no_registry_record"] += 1
            continue
        required = inputs[formula_id]
        task = formula_id.replace("_", " ")
        question = (
            f"Which formula applies to {task}, and which inputs does it require? "
            "Use the attached Fork context."
        )
        context = hard_negative_context(formula_id, entries[formula_id], required)
        user = f"{question}\n\nFork context:\n{context}"
        key = normalise_user(user)
        if key in taken_users:
            stats["dup"] += 1
            continue
        if max_sealed_overlap(user, sealed) >= OVERLAP_LIMIT:
            stats["sealed_overlap"] += 1
            continue
        taken_users.add(key)
        assistant = target_for(HARD_NEGATIVE_PAIR_TYPE, formula_id, inputs).render()
        meta = {
            "aug": AUG_HARD_NEGATIVE,
            "context_ids": [f"formula:{formula_id}"],
            "content_hash": pair_hash(user, assistant),
            "domain": parent["meta"].get("domain"),
            "domain_from_source": parent["meta"].get("domain_from_source"),
            "eval_only": False,
            "formula_id": formula_id,
            "pair_type": HARD_NEGATIVE_PAIR_TYPE,
            "parent_id": parent["meta"]["content_hash"],
            "source_group": parent["meta"].get("source_group"),
            "source_relpath": REGISTRY.as_posix(),
            "split": "train",
        }
        rows.append({"system": "", "user": user, "assistant": assistant, "meta": meta})
        stats["built"] += 1
    return rows, stats


def build(root: Path, write: bool = True) -> dict[str, Any]:
    base, holdout_domain = base_parents(root)
    parents = [strip_parent(row) for row in base] + [strip_parent(row) for row in temporary_works_parents(root)]
    hashes = [row["meta"]["content_hash"] for row in parents]
    if len(hashes) != len(set(hashes)):
        raise ValueError("duplicate parent content hash")
    lexicon = split._build_lexicon(_read_jsonl(root / "data/retrieval/rates/rates.jsonl"))
    for row in parents:
        if split.assistant_has_fact(row["assistant"], lexicon):
            raise ValueError("a parent assistant turn carries a fact")
    train_parents, held, report = stratify(parents, holdout_domain)
    sealed = sealed_user_texts(root)
    taken = {normalise_user(row["user"]) for row in parents}
    children, child_stats = paraphrase_children(train_parents, load_paraphrases(root), sealed, taken)
    registry = json.loads((root / REGISTRY).read_text(encoding="utf-8"))
    negatives, negative_stats = hard_negatives(train_parents, registry, sealed, taken)

    train_rows = sorted(train_parents + children + negatives, key=lambda row: row["meta"]["content_hash"])
    held_rows = sorted(held, key=lambda row: row["meta"]["content_hash"])
    per_parent = defaultdict(int)
    for child in children:
        per_parent[child["meta"]["parent_id"]] += 1
    short = sum(1 for row in train_parents if per_parent[row["meta"]["content_hash"]] < MIN_CHILDREN)
    counts = {
        "original_confirmed_train": sum(1 for row in train_parents if row["meta"].get("source_origin") is None),
        "temporary_works_train": sum(1 for row in train_parents if row["meta"].get("source_origin") == "owner_reviewed_local"),
        "paraphrase_children": len(children),
        "hard_negatives": len(negatives),
    }
    zip_hash = split.sha256_file(root / RAW_ZIP)
    sources = {
        "raw_zip_sha256": zip_hash,
        "temporary_works_sha256": split.sha256_file(root / TEMPORARY_WORKS),
        "paraphrases_sha256": split.sha256_file(root / PARAPHRASES) if (root / PARAPHRASES).exists() else None,
        "registry_sha256": split.sha256_file(root / REGISTRY),
    }
    train_manifest = {
        "build": "scripts/data/build_phase1.py",
        "by_source": counts,
        "by_stratum": {name: value["train"] for name, value in report.items()},
        "count": len(train_rows),
        "fact_count": 0,
        "hard_negative_stats": negative_stats,
        "paraphrase_stats": child_stats,
        "parents_under_min_children": short,
        "schema": list(split.CHAT_KEYS),
        "sources": sources,
        "source_zip_sha256": zip_hash,
    }
    held_manifest = {
        "build": "scripts/data/build_phase1.py",
        "by_stratum": report,
        "count": len(held_rows),
        "domain_pair_count": sum(1 for row in held_rows if row["meta"]["holdout"] == "domain"),
        "fraction_denominator": split.HOLDOUT_FRACTION_DENOMINATOR,
        "fraction_numerator": split.HOLDOUT_FRACTION_NUMERATOR,
        "group_by": "parent_id; children of a held-out parent are not written to train",
        "held_out_domain": holdout_domain,
        "holdout_rule": "held-out domain whole; 15 percent of every other source group, first by content hash, round half up",
        "schema": list(split.CHAT_KEYS),
        "source_zip_sha256": zip_hash,
        "stratified_pair_count": sum(1 for row in held_rows if row["meta"]["holdout"] == "stratified"),
    }
    if write:
        split.write_jsonl(root / "data/train/pairs.jsonl", train_rows)
        split.write_jsonl(root / "data/held_out/pairs.jsonl", held_rows)
        split.write_json(root / "data/train/manifest.json", train_manifest)
        split.write_json(root / "data/held_out/manifest.json", held_manifest)
    return {"train": len(train_rows), "held_out": len(held_rows), **counts, "parents_under_min_children": short,
            "paraphrase_stats": child_stats, "hard_negative_stats": negative_stats}


def train_parents_for_paraphrase(root: Path) -> list[dict]:
    base, holdout_domain = base_parents(root)
    parents = [strip_parent(row) for row in base] + [strip_parent(row) for row in temporary_works_parents(root)]
    train_parents, _, _ = stratify(parents, holdout_domain)
    return [{"parent_id": row["meta"]["content_hash"], "user": row["user"]} for row in train_parents]


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--emit-parents", type=Path, help="Write train parents (id, user) for the paraphrase step and stop")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.emit_parents:
        rows = train_parents_for_paraphrase(root)
        args.emit_parents.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        print(json.dumps({"train_parents": len(rows)}))
        return
    print(json.dumps(build(root), sort_keys=True))


if __name__ == "__main__":
    main()
