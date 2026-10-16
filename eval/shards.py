"""Held-out shards listed in ``data/held_out/shards.json``.

The shard list is data. Each entry names a frozen file under
``data/held_out/`` with its sha256, row count, record schema and the routing
registry its routed items use. The original shard is the phase-1 held-out
split (``eval.labels.HELD_OUT_RELATIVE``); its labels come from
``label_eval_sets`` unchanged. Every other shard is labelled with the same
pair_type rule (``eval.labels.label_held_out_record``) after its records are
read onto the chat schema. No item text is held here. See docs/eval/HELD_OUT.md.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from eval.labels import EvalLabel, EvalLabelError, label_eval_sets, label_held_out_record
from eval.readers import EvalReadError, read_records
from eval.registry import load_registry_inputs
from model.identity import repo_root
from private_data import locate, missing_message

SHARDS_RELATIVE = "data/held_out/shards.json"
SHARDS_SCHEMA = "held_out_shards/v1"
HELD_OUT_DIR = "data/held_out/"
RECORD_SCHEMAS = ("system_user_assistant", "instruction_input_output")
_REQUIRED = (
    "name",
    "path",
    "sha256",
    "count",
    "frozen",
    "created",
    "original",
    "record_schema",
    "routing_registry",
    "near_dup_at_freeze",
)


class ShardError(RuntimeError):
    """The shard list or a shard file is not usable. The message has no item text."""


@dataclass(frozen=True)
class HeldOutShard:
    name: str
    path: str
    sha256: str
    count: int
    frozen: bool
    created: str
    original: bool
    record_schema: str
    routing_registry: str
    near_dup_at_freeze: int


def shards_listed(root: Path | None = None) -> bool:
    base = repo_root() if root is None else root
    return locate(base, SHARDS_RELATIVE).is_file()


def load_shards(root: Path | None = None, *, verify: bool = True) -> tuple[HeldOutShard, ...]:
    """Read the shard list. With ``verify`` every frozen shard must match its sha256 and count."""
    base = repo_root() if root is None else root
    path = locate(base, SHARDS_RELATIVE)
    if not path.is_file():
        raise ShardError(missing_message(SHARDS_RELATIVE))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ShardError(missing_message(SHARDS_RELATIVE)) from exc
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ShardError("shard list is not JSON") from exc
    if not isinstance(payload, dict) or payload.get("schema") != SHARDS_SCHEMA:
        raise ShardError("shard list schema is not held_out_shards/v1")
    entries = payload.get("shards")
    if not isinstance(entries, list) or not entries:
        raise ShardError("shard list names no shard")
    shards = tuple(_shard(entry, index) for index, entry in enumerate(entries, 1))
    names = [shard.name for shard in shards]
    if len(set(names)) != len(names):
        raise ShardError("shard names are not unique")
    paths = [shard.path for shard in shards]
    if len(set(paths)) != len(paths):
        raise ShardError("two shards name the same file")
    if sum(1 for shard in shards if shard.original) != 1:
        raise ShardError("exactly one shard must be the original")
    if verify:
        for shard in shards:
            verify_shard(base, shard)
    return shards


def original_shard(shards: tuple[HeldOutShard, ...]) -> HeldOutShard:
    return next(shard for shard in shards if shard.original)


def added_shards(shards: tuple[HeldOutShard, ...]) -> tuple[HeldOutShard, ...]:
    return tuple(shard for shard in shards if not shard.original)


def select_shards(shards: tuple[HeldOutShard, ...], names) -> tuple[HeldOutShard, ...]:
    wanted = list(names)
    known = {shard.name: shard for shard in shards}
    missing = [name for name in wanted if name not in known]
    if missing:
        raise ShardError("shard is not listed in the shard list: " + ", ".join(missing))
    return tuple(known[name] for name in wanted)


def verify_shard(root: Path, shard: HeldOutShard) -> None:
    target = locate(root, shard.path)
    if not target.is_file():
        raise ShardError(missing_message(shard.path))
    data = target.read_bytes()
    if shard.frozen and hashlib.sha256(data).hexdigest() != shard.sha256.lower():
        raise ShardError(f"frozen shard does not match its sha256: {shard.name}")
    rows = sum(1 for line in data.decode("utf-8").splitlines() if line.strip())
    if rows != shard.count:
        raise ShardError(f"shard row count does not match the shard list: {shard.name}")


def shard_record(record: dict, record_schema: str, *, line: int, name: str) -> dict:
    """Return the record on the chat schema (system, user, assistant, meta)."""
    meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
    if record_schema == "system_user_assistant":
        user = record.get("user")
        assistant = record.get("assistant")
        system = record.get("system", "")
    elif record_schema == "instruction_input_output":
        user = record.get("instruction")
        extra = record.get("input")
        if isinstance(user, str) and isinstance(extra, str) and extra.strip():
            user = f"{user}\n{extra}"
        assistant = record.get("output")
        system = ""
    else:
        raise ShardError(f"shard {name} has an unknown record schema")
    if not isinstance(user, str) or not user.strip():
        raise ShardError(f"shard {name} line {line} has no user text")
    if not isinstance(assistant, str) or not assistant.strip():
        raise ShardError(f"shard {name} line {line} has no assistant text")
    if not isinstance(system, str):
        raise ShardError(f"shard {name} line {line} system is not text")
    return {"system": system, "user": user, "assistant": assistant, "meta": meta}


def read_shard_records(root: Path, shard: HeldOutShard) -> list[tuple[int, dict]]:
    """Shard rows on the chat schema, with their file line numbers."""
    try:
        rows = read_records(root, shard.path)
    except EvalReadError as exc:
        raise ShardError(str(exc)) from None
    return [
        (line, shard_record(row, shard.record_schema, line=line, name=shard.name))
        for line, row in rows
    ]


def label_shard(root: Path | None, shard: HeldOutShard) -> tuple[EvalLabel, ...]:
    """Label one shard with the held-out pair_type rule."""
    base = repo_root() if root is None else root
    verify_shard(base, shard)
    if shard.original:
        try:
            sets = dict(label_eval_sets(base))
        except (EvalReadError, EvalLabelError) as exc:
            raise ShardError(str(exc)) from None
        return sets["held_out"]
    inputs = load_registry_inputs(base)
    labels = []
    for line, record in read_shard_records(base, shard):
        try:
            label = label_held_out_record(record, line=line, inputs=inputs)
        except EvalLabelError as exc:
            raise ShardError(f"shard {shard.name}: {exc}") from None
        labels.append(
            EvalLabel(
                shard.name,
                label.line,
                label.prompt,
                label.expected_block,
                label.named_inputs,
                label.formula_id,
                label.required_inputs,
            )
        )
    return tuple(labels)


def _shard(entry: object, index: int) -> HeldOutShard:
    if not isinstance(entry, dict):
        raise ShardError(f"shard entry {index} is not an object")
    missing = [key for key in _REQUIRED if key not in entry]
    if missing:
        raise ShardError(f"shard entry {index} is missing " + ", ".join(missing))
    name = entry["name"]
    path = entry["path"]
    digest = entry["sha256"]
    count = entry["count"]
    near = entry["near_dup_at_freeze"]
    if not isinstance(name, str) or not name.strip():
        raise ShardError(f"shard entry {index} has no name")
    if not isinstance(path, str) or not path.startswith(HELD_OUT_DIR) or ".." in Path(path).parts:
        raise ShardError(f"shard entry {index} path is not under {HELD_OUT_DIR}")
    if not path.endswith(".jsonl"):
        raise ShardError(f"shard entry {index} path is not a .jsonl file")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdefABCDEF" for c in digest):
        raise ShardError(f"shard entry {index} sha256 is not a digest")
    for key, value in (("count", count), ("near_dup_at_freeze", near)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ShardError(f"shard entry {index} {key} is not a count")
    if count < 1:
        raise ShardError(f"shard entry {index} is empty")
    if not isinstance(entry["frozen"], bool) or not isinstance(entry["original"], bool):
        raise ShardError(f"shard entry {index} frozen/original flags are not booleans")
    if entry["record_schema"] not in RECORD_SCHEMAS:
        raise ShardError(f"shard entry {index} record_schema is unknown")
    created = entry["created"]
    if not isinstance(created, str) or len(created) != 10:
        raise ShardError(f"shard entry {index} created is not a YYYY-MM-DD date")
    registry = entry["routing_registry"]
    if not isinstance(registry, str) or not registry.strip():
        raise ShardError(f"shard entry {index} names no routing registry")
    return HeldOutShard(
        name.strip(),
        path,
        digest.lower(),
        count,
        entry["frozen"],
        created,
        entry["original"],
        entry["record_schema"],
        registry.strip(),
        near,
    )
