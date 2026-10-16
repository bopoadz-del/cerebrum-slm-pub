"""Augmentation gate: question-side children and hard negatives stay honest.

Checks every row in the train split that carries ``meta.aug``:

1. It hashes to a parent: ``meta.parent_id`` is the content hash of a parent row
   (no ``meta.aug``) in the train split, and its own ``content_hash`` is the hash
   of its system, user, and assistant text.
2. Group split: a child of a held-out parent never lands in train, no parent is
   in both splits, and held-out rows carry no augmentation. Held-out means
   every shard listed in data/held_out/shards.json.
3. A paraphrase keeps the parent assistant turn byte for byte, the same
   pair_type and formula_id, and adds no number, unit, currency, clause,
   edition, brand, or site that the parent user turn does not carry.
4. A hard negative routes to a formula whose id is present in its attached Fork
   context, cites only context ids present in that context, and comes from a
   refuse-labelled parent.
5. No augmented user turn matches a sealed eval item by content hash or reaches
   a token overlap of 0.8 or more (Jaccard over lower-cased word tokens) with
   any sealed eval item's user side.

The Split check (``gates.split``) runs on these rows like on any training row.
"""

from __future__ import annotations

import json
import re
import sys
import zipfile
from pathlib import Path

from gates.catalog import _member_is_eval_file, _record_is_eval
from gates.findings import Finding
from gates.records import RAW_ZIP, canonical_json, sha256_bytes
from private_data import locate

TRAIN = Path("data/train/pairs.jsonl")
HELD_OUT = Path("data/held_out/pairs.jsonl")
SHARDS = Path("data/held_out/shards.json")
OVERLAP_LIMIT = 0.8
REFUSE_PARENT_TYPES = frozenset(
    {"REFUSAL_SCOPE", "REFUSAL_VS_VACUUM", "FORCING_THE_NUMBER", "bare_rate_not_answer", "why_formula_cannot_answer"}
)
_TOKEN = re.compile(r"[a-z0-9]+")
_DIGITS = re.compile(r"\d+(?:[.,]\d+)?")
_CONTEXT_MARK = "\n\nFork context:\n"


def pair_content_hash(system: str, user: str, assistant: str) -> str:
    return sha256_bytes(canonical_json({"system": system, "user": user, "assistant": assistant}).encode("utf-8"))


def token_set(text: str) -> frozenset[str]:
    return frozenset(_TOKEN.findall(text.casefold()))


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _fact_leaks():
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.append(str(root))
    from eval.graders.fact_leak import find_fact_leaks

    return find_fact_leaks


def added_facts(child: str, parent: str) -> list[str]:
    found = [f"{leak.kind}" for leak in _fact_leaks()(child, parent)]
    parent_numbers = set(_DIGITS.findall(parent))
    found.extend("number" for number in _DIGITS.findall(child) if number not in parent_numbers)
    return found


def _user_side(obj: dict) -> str:
    for key in ("user", "question", "prompt"):
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value
    instruction = obj.get("instruction")
    if isinstance(instruction, str) and instruction.strip():
        extra = obj.get("input") if isinstance(obj.get("input"), str) else ""
        return f"{instruction}\n\n{extra}" if extra else instruction
    if "record" in obj:
        return canonical_json(obj.get("record"))
    return ""


def sealed_users(root: Path) -> list[frozenset[str]]:
    texts: list[str] = []
    base = root / "data" / "eval"
    if base.is_dir():
        for path in sorted(base.rglob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        texts.append(_user_side(obj))
    zip_path = locate(root, RAW_ZIP.as_posix(), "SLM_DATASETS_ZIP")
    if zip_path.is_file():
        with zipfile.ZipFile(zip_path) as zf:
            for name in zf.namelist():
                if not name.endswith(".jsonl"):
                    continue
                eval_file = _member_is_eval_file(name)
                for line in zf.read(name).decode("utf-8").splitlines():
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    if isinstance(obj, dict) and (eval_file or _record_is_eval(obj)):
                        texts.append(_user_side(obj))
    return [token_set(text) for text in texts if text.strip()]


def _rows(path: Path) -> list[tuple[int, dict]]:
    if not path.is_file():
        return []
    rows = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            rows.append((lineno, json.loads(line)))
    return rows


def _held_shards(root: Path) -> list[tuple[str, list[tuple[int, dict]]]]:
    """Every held-out shard as (path, rows). Rows carry meta.content_hash.

    Shards come from data/held_out/shards.json when it exists (see
    gates.held_out_shards); otherwise the original held-out split alone. A
    shard row without a content_hash gets the hash of its chat-schema text.
    """
    if not locate(root, SHARDS.as_posix(), "SLM_HELD_OUT_SHARDS").is_file():
        return [(HELD_OUT.as_posix(), _rows(locate(root, HELD_OUT.as_posix(), "SLM_HELD_OUT_PAIRS")))]
    repo = Path(__file__).resolve().parents[2]
    for entry in (repo, repo / "src"):
        if str(entry) not in sys.path:
            sys.path.append(str(entry))
    from eval.shards import ShardError, load_shards, shard_record

    try:
        shards = load_shards(root, verify=False)
    except ShardError:
        # gates.held_out_shards reports a broken shard list.
        return [(HELD_OUT.as_posix(), _rows(locate(root, HELD_OUT.as_posix(), "SLM_HELD_OUT_PAIRS")))]
    found: list[tuple[str, list[tuple[int, dict]]]] = []
    for shard in shards:
        rows = []
        for lineno, row in _rows(locate(root, shard.path)):
            meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
            if not meta.get("content_hash"):
                try:
                    chat = shard_record(row, shard.record_schema, line=lineno, name=shard.name)
                except ShardError:
                    continue
                meta = dict(meta)
                meta["content_hash"] = pair_content_hash(chat["system"], chat["user"], chat["assistant"])
                row = dict(row, meta=meta)
            rows.append((lineno, row))
        found.append((shard.path, rows))
    return found


def check_augmentation(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    train = _rows(root / TRAIN)
    train_rel = TRAIN.as_posix()

    def fail(path: str, lineno: int, message: str) -> None:
        findings.append(Finding("augment", path, f"line {lineno}: {message}"))

    parents = {row["meta"]["content_hash"]: row for _, row in train if not row["meta"].get("aug")}
    held_hashes = set()
    for held_rel, held in _held_shards(root):
        for lineno, row in held:
            held_hashes.add(row["meta"]["content_hash"])
            if row["meta"].get("aug") or row["meta"].get("parent_id"):
                fail(held_rel, lineno, "held-out row carries augmentation")
    for digest in set(parents) & held_hashes:
        findings.append(Finding("augment", train_rel, f"parent {digest[:12]} is in both train and held-out"))

    augmented = [(lineno, row) for lineno, row in train if row["meta"].get("aug")]
    sealed = sealed_users(root) if augmented else []
    for lineno, row in augmented:
        meta = row["meta"]
        kind = meta.get("aug")
        parent_id = meta.get("parent_id")
        digest = pair_content_hash(row.get("system", ""), row["user"], row["assistant"])
        if meta.get("content_hash") != digest:
            fail(train_rel, lineno, "content_hash does not hash the row text")
        if parent_id in held_hashes:
            fail(train_rel, lineno, "child of a held-out parent is in train")
            continue
        parent = parents.get(parent_id)
        if parent is None:
            fail(train_rel, lineno, "parent_id does not hash to a train parent")
            continue
        if kind == "paraphrase":
            if row["assistant"] != parent["assistant"]:
                fail(train_rel, lineno, "paraphrase changed the assistant turn")
            for field in ("pair_type", "formula_id"):
                if meta.get(field) != parent["meta"].get(field):
                    fail(train_rel, lineno, f"paraphrase changed {field}")
            added = added_facts(row["user"], parent["user"])
            if added:
                fail(train_rel, lineno, "paraphrase adds " + ", ".join(sorted(set(added))))
        elif kind == "hard_negative":
            if parent["meta"].get("pair_type") not in REFUSE_PARENT_TYPES:
                fail(train_rel, lineno, "hard negative parent is not refuse-labelled")
            if _CONTEXT_MARK not in row["user"]:
                fail(train_rel, lineno, "hard negative has no attached Fork context")
                continue
            context = row["user"].split(_CONTEXT_MARK, 1)[1]
            ids = meta.get("context_ids") or []
            if not ids or any(f"[{cid}]" not in context for cid in ids):
                fail(train_rel, lineno, "hard negative cites an id not in its context")
            try:
                reply = json.loads(row["assistant"])
            except json.JSONDecodeError:
                fail(train_rel, lineno, "hard negative target is not the reply contract")
                continue
            route = reply.get("formula_id") if isinstance(reply, dict) else None
            if not route or f"[formula:{route}]" not in context or route != meta.get("formula_id"):
                fail(train_rel, lineno, "hard negative routes to a formula not in its context")
        else:
            fail(train_rel, lineno, f"unknown augmentation kind {kind!r}")
        user_tokens = token_set(row["user"])
        if any(jaccard(user_tokens, other) >= OVERLAP_LIMIT for other in sealed):
            fail(train_rel, lineno, "augmented user turn overlaps a sealed eval item (>= 0.8)")
    return findings
