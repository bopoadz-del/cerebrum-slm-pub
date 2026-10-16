"""Pair records, training-file discovery, and the hashes the gates compare."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

TRAINING_SKIP_PREFIXES = ("data/raw/", "data/eval/", "data/retrieval/")
RAW_ZIP = Path("data/raw/slm-datasets.zip")


class RecordParseError(Exception):
    def __init__(self, path: str, line: int, detail: str) -> None:
        super().__init__(f"{path}:{line}: {detail}")
        self.path = path
        self.line = line


@dataclass(frozen=True)
class PairView:
    user: str
    assistant_turns: tuple[str, ...]

    @property
    def assistant(self) -> str:
        return "\n".join(self.assistant_turns)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_json(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def record_hash(obj: object) -> str:
    return sha256_bytes(canonical_json(obj).encode("utf-8"))


def content_hash(user: str, assistant: str) -> str:
    payload = canonical_json({"assistant": assistant, "user": user})
    return sha256_bytes(payload.encode("utf-8"))


def pair_view(obj: dict) -> PairView | None:
    messages = obj.get("messages")
    if isinstance(messages, list):
        users: list[str] = []
        assistants: list[str] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "")).lower()
            content = message.get("content")
            text = content if isinstance(content, str) else ""
            if role in {"user", "human"}:
                users.append(text)
            elif role in {"assistant", "model"}:
                assistants.append(text)
        if users or assistants:
            return PairView("\n".join(users), tuple(assistants))
        return None

    if "instruction" in obj and any(key in obj for key in ("output", "response")):
        user = obj.get("instruction") if isinstance(obj.get("instruction"), str) else ""
        extra = obj.get("input") if isinstance(obj.get("input"), str) else ""
        if extra.strip():
            user = f"{user}\n{extra}" if user else extra
        assistant = obj.get("output", obj.get("response"))
        assistant_text = assistant if isinstance(assistant, str) else ""
        return PairView(user, (assistant_text,))

    if "prompt" in obj and "completion" in obj:
        user = obj.get("prompt") if isinstance(obj.get("prompt"), str) else ""
        assistant = obj.get("completion") if isinstance(obj.get("completion"), str) else ""
        return PairView(user, (assistant,))
    return None


def looks_like_pair(obj: dict) -> bool:
    return pair_view(obj) is not None


def iter_jsonl(path: Path) -> list[tuple[int, dict]]:
    rows: list[tuple[int, dict]] = []
    text = path.read_text(encoding="utf-8")
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RecordParseError(path.as_posix(), lineno, str(exc)) from exc
        if not isinstance(obj, dict):
            raise RecordParseError(path.as_posix(), lineno, "record is not an object")
        rows.append((lineno, obj))
    return rows


def load_records(path: Path) -> list[tuple[int, dict]]:
    if path.suffix.lower() == ".jsonl":
        return iter_jsonl(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RecordParseError(path.as_posix(), 1, str(exc)) from exc
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and isinstance(data.get("pairs"), list):
        items = data["pairs"]
    elif isinstance(data, dict):
        items = [data]
    else:
        raise RecordParseError(path.as_posix(), 1, "unsupported JSON value")
    rows: list[tuple[int, dict]] = []
    for index, obj in enumerate(items, 1):
        if not isinstance(obj, dict):
            raise RecordParseError(path.as_posix(), index, "record is not an object")
        rows.append((index, obj))
    return rows


def is_training_path(root: Path, path: Path) -> bool:
    if path.suffix.lower() not in {".json", ".jsonl"}:
        return False
    rel = path.relative_to(root).as_posix()
    if not rel.startswith("data/"):
        return False
    return not rel.startswith(TRAINING_SKIP_PREFIXES)


def iter_training_files(root: Path) -> list[Path]:
    data = root / "data"
    if not data.is_dir():
        return []
    files = [
        path
        for path in data.rglob("*")
        if path.is_file() and is_training_path(root, path)
    ]
    return sorted(files)


def relpath(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()
