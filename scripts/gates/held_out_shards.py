"""Held-out shards: frozen, sealed by sha256, and never in train.

The shard list is data (``data/held_out/shards.json``). For every listed shard:

1. The file exists, its sha256 matches the list (frozen) and its row count
   matches. An added shard also carries a ``<file>.sha256`` seal beside it in
   the sha256sum style, and that seal must match.
2. Every ``.jsonl`` under ``data/held_out/`` is a listed shard, and the
   original shard is the held-out split the eval labels read.
3. No shard row is in train: no shared content hash and no shared user turn
   (exact, after case and whitespace folding) with any row of a file a
   trainer reads. Train files are every ``train_file`` named in
   ``configs/lora/*.json`` plus every file under ``data/train/``. No shard row
   is in another shard.
4. Near-duplicates use the augmentation gate's rule (Jaccard over lower-cased
   word tokens of the user side >= ``OVERLAP_LIMIT``) against train rows,
   other shards and the sealed eval items. The count per shard must not
   exceed ``near_dup_at_freeze`` recorded in the shard list when it was frozen.

Findings name a shard and a line number, never item text.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

from gates.augment import OVERLAP_LIMIT, jaccard, pair_content_hash, sealed_users, token_set
from gates.findings import Finding
from gates.records import RecordParseError, iter_training_files, load_records, pair_view, relpath
from private_data import locate, missing_message

SHARDS = Path("data/held_out/shards.json")
HELD_DIR = Path("data/held_out")
TRAIN_DIR = Path("data/train")
LORA_CONFIGS = Path("configs/lora")
_SPACE = re.compile(r"\s+")


def _eval_shards():
    root = Path(__file__).resolve().parents[2]
    for entry in (root, root / "src"):
        if str(entry) not in sys.path:
            sys.path.append(str(entry))
    import eval.shards as shards

    return shards


def _fold(text: str) -> str:
    return _SPACE.sub(" ", text.casefold()).strip()


def shard_rows(root: Path) -> tuple[list, dict[str, list[tuple[int, dict]]], list[Finding]]:
    """Listed shards and their rows on the chat schema. Findings for any defect."""
    module = _eval_shards()
    findings: list[Finding] = []
    where = SHARDS.as_posix()
    try:
        shards = module.load_shards(root, verify=False)
    except module.ShardError as exc:
        return [], {}, [Finding("held-out-shards", where, str(exc))]
    rows: dict[str, list[tuple[int, dict]]] = {}
    for shard in shards:
        target = locate(root, shard.path)
        if not target.is_file():
            findings.append(Finding("held-out-shards", shard.path, missing_message(shard.path)))
            continue
        data = target.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if shard.frozen and digest != shard.sha256:
            findings.append(
                Finding("held-out-shards", shard.path, f"frozen shard sha256 {digest} != listed {shard.sha256}")
            )
        if not shard.frozen:
            findings.append(Finding("held-out-shards", shard.path, "held-out shards are frozen; this one is not"))
        findings.extend(_sidecar(root, shard, digest))
        try:
            parsed = load_records(target)
        except (RecordParseError, UnicodeError) as exc:
            findings.append(Finding("held-out-shards", shard.path, f"shard does not parse: {exc}"))
            continue
        if len(parsed) != shard.count:
            findings.append(
                Finding("held-out-shards", shard.path, f"shard has {len(parsed)} rows, list says {shard.count}")
            )
        chat: list[tuple[int, dict]] = []
        for line, obj in parsed:
            try:
                chat.append((line, module.shard_record(obj, shard.record_schema, line=line, name=shard.name)))
            except module.ShardError as exc:
                findings.append(Finding("held-out-shards", shard.path, str(exc)))
            if obj.get("meta", {}).get("aug") or obj.get("meta", {}).get("parent_id"):
                findings.append(Finding("held-out-shards", shard.path, f"line {line}: held-out row carries augmentation"))
        rows[shard.name] = chat
    return list(shards), rows, findings


def _sidecar(root: Path, shard, digest: str) -> list[Finding]:
    seal = locate(root, shard.path + ".sha256")
    if not seal.is_file():
        if shard.original:
            return []
        return [Finding("held-out-shards", shard.path, "added shard has no .sha256 seal beside it")]
    parts = seal.read_text(encoding="utf-8").split()
    name = Path(shard.path).name
    if len(parts) != 2 or parts[1].lstrip("*") != name:
        return [Finding("held-out-shards", relpath(seal, root), "seal line is not 'digest filename'")]
    if parts[0].lower() != digest:
        return [Finding("held-out-shards", relpath(seal, root), "seal does not match the shard file")]
    return []


def train_files(root: Path) -> list[Path]:
    """Files a trainer reads: every lora config ``train_file`` and everything under data/train/."""
    named: set[Path] = set()
    for config in sorted((root / LORA_CONFIGS).glob("*.json")):
        try:
            value = json.loads(config.read_text(encoding="utf-8")).get("train_file")
        except (UnicodeError, json.JSONDecodeError, AttributeError):
            continue
        if isinstance(value, str) and value.strip():
            named.add((root / value).resolve())
    named.update(path.resolve() for path in iter_training_files(root) if _under(path, root / TRAIN_DIR))
    return sorted(path for path in named if path.is_file())


def _under(path: Path, folder: Path) -> bool:
    try:
        path.resolve().relative_to(folder.resolve())
    except ValueError:
        return False
    return True


def _train_side(root: Path, shard_paths: set[str]) -> list[tuple[str, int, str, str, str]]:
    """(path, line, system, user, assistant) for every pair in a train file."""
    found: list[tuple[str, int, str, str, str]] = []
    for path in train_files(root):
        where = relpath(path, root)
        if where in shard_paths or _under(path, root / HELD_DIR):
            continue
        try:
            rows = load_records(path)
        except (RecordParseError, UnicodeError):
            continue  # other gates report unreadable training files
        for line, obj in rows:
            user = obj.get("user")
            assistant = obj.get("assistant")
            if isinstance(user, str) and isinstance(assistant, str):
                system = obj.get("system") if isinstance(obj.get("system"), str) else ""
                found.append((where, line, system, user, assistant))
                continue
            view = pair_view(obj)
            if view is not None:
                found.append((where, line, "", view.user, view.assistant))
    return found


def check_held_out_shards(root: Path) -> list[Finding]:
    held_dir = locate(root, HELD_DIR.as_posix(), "SLM_HELD_OUT_DIR")
    if not held_dir.is_dir():
        return [Finding("held-out-shards", HELD_DIR.as_posix(), missing_message(HELD_DIR.as_posix(), "SLM_HELD_OUT_DIR"))]
    if not locate(root, SHARDS.as_posix(), "SLM_HELD_OUT_SHARDS").is_file():
        if any(held_dir.glob("*.jsonl")):
            return [
                Finding(
                    "held-out-shards",
                    SHARDS.as_posix(),
                    "held-out shard list is missing. " + missing_message(SHARDS.as_posix(), "SLM_HELD_OUT_SHARDS"),
                )
            ]
        return []
    shards, rows, findings = shard_rows(root)
    if not shards:
        return findings
    listed = {shard.path for shard in shards}
    for path in sorted(held_dir.rglob("*.jsonl")):
        try:
            where = (HELD_DIR / path.resolve().relative_to(held_dir.resolve())).as_posix()
        except ValueError:
            where = relpath(path, root)
        if where not in listed:
            findings.append(Finding("held-out-shards", where, "held-out .jsonl is not a listed shard"))
    module = _eval_shards()
    from eval.labels import HELD_OUT_RELATIVE

    original = module.original_shard(tuple(shards))
    if original.path != HELD_OUT_RELATIVE:
        findings.append(Finding("held-out-shards", original.path, "original shard is not the held-out split"))

    train = _train_side(root, listed)
    train_hash = {}
    train_user = {}
    train_tokens = []
    for where, line, system, user, assistant in train:
        tag = f"{where}:{line}"
        train_hash.setdefault(pair_content_hash(system, user, assistant), tag)
        train_user.setdefault(_fold(user), tag)
        train_tokens.append(token_set(user))
    sealed = sealed_users(root)

    shard_tokens = {
        name: [(line, token_set(record["user"])) for line, record in chat] for name, chat in rows.items()
    }
    seen_hash: dict[str, str] = {}
    seen_user: dict[str, str] = {}
    for shard in shards:
        for line, record in rows.get(shard.name, []):
            here = f"line {line}"
            digest = pair_content_hash(record["system"], record["user"], record["assistant"])
            user = _fold(record["user"])
            if digest in train_hash:
                findings.append(Finding("held-out-shards", shard.path, f"{here}: pair is in train ({train_hash[digest]})"))
            if user in train_user:
                findings.append(Finding("held-out-shards", shard.path, f"{here}: user turn is in train ({train_user[user]})"))
            for table, key, kind in ((seen_hash, digest, "pair"), (seen_user, user, "user turn")):
                other = table.get(key)
                if other is not None and not other.startswith(shard.name + ":"):
                    findings.append(Finding("held-out-shards", shard.path, f"{here}: {kind} is also in shard {other}"))
                table.setdefault(key, f"{shard.name}:{line}")

    for shard in shards:
        near = 0
        for line, tokens in shard_tokens.get(shard.name, []):
            near += sum(1 for other in train_tokens if jaccard(tokens, other) >= OVERLAP_LIMIT)
            near += sum(1 for other in sealed if jaccard(tokens, other) >= OVERLAP_LIMIT)
            for other_name, other_rows in shard_tokens.items():
                if other_name == shard.name:
                    continue
                near += sum(1 for _, other in other_rows if jaccard(tokens, other) >= OVERLAP_LIMIT)
        if near > shard.near_dup_at_freeze:
            findings.append(
                Finding(
                    "held-out-shards",
                    shard.path,
                    f"{near} near-duplicate(s) at Jaccard >= {OVERLAP_LIMIT} with train, sealed eval or "
                    f"another shard; {shard.near_dup_at_freeze} recorded at freeze",
                )
            )
    return findings


def near_dup_counts(root: Path) -> dict[str, int]:
    """Near-duplicate count per shard, the number recorded as near_dup_at_freeze."""
    shards, rows, _ = shard_rows(root)
    listed = {shard.path for shard in shards}
    train_tokens = [token_set(user) for _, _, _, user, _ in _train_side(root, listed)]
    sealed = sealed_users(root)
    tokens = {name: [token_set(record["user"]) for _, record in chat] for name, chat in rows.items()}
    counts: dict[str, int] = {}
    for name, mine in tokens.items():
        total = 0
        for item in mine:
            total += sum(1 for other in train_tokens if jaccard(item, other) >= OVERLAP_LIMIT)
            total += sum(1 for other in sealed if jaccard(item, other) >= OVERLAP_LIMIT)
            for other_name, theirs in tokens.items():
                if other_name != name:
                    total += sum(1 for other in theirs if jaccard(item, other) >= OVERLAP_LIMIT)
        counts[name] = total
    return counts


def main() -> int:
    root = Path.cwd()
    print(json.dumps(near_dup_counts(root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
