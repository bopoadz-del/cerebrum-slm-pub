"""Locate private data that is not stored in the working tree.

Sealed-eval and owner-data loaders resolve paths only via environment variables.
There is no hardcoded absolute path. Fail closed when unset.

Resolution order for a repo-relative path ``relative``:

1. Specific env var for that path (see SPECIFIC_ENV), when set.
2. ``SLM_SEALED_DIR`` for paths under ``data/eval/router_unseen`` / ``data/eval/unseen`` /
   ``data/held_out`` (joined as the suffix after those roots).
3. ``SLM_PRIVATE_DATA_DIR`` or ``SLM_PRIVATE_ROOT`` / ``relative``.
4. In-tree ``root / relative`` only when no env root applies (dev convenience).
   ``require`` still fails closed if the resolved path is absent.

Messages name the relative path and env var only — never file contents.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

SPECIFIC_ENV = {
    "data/raw/slm-datasets.zip": "SLM_DATASETS_ZIP",
    "data/held_out": "SLM_HELD_OUT_DIR",
    "data/held_out/shards.json": "SLM_HELD_OUT_SHARDS",
    "data/held_out/pairs.jsonl": "SLM_HELD_OUT_PAIRS",
    "data/held_out/manifest.json": "SLM_HELD_OUT_MANIFEST",
    "docs/serve/transcripts-epoch2.jsonl": "SLM_TRANSCRIPTS_JSONL",
}

# Any non-empty value selects private mode. Public CI sets none of these.
PRIVATE_ENV = tuple(
    dict.fromkeys(
        (
            "SLM_PRIVATE_DATA_DIR",
            "SLM_PRIVATE_ROOT",
            "SLM_SEALED_DIR",
            "SLM_HELD_OUT_DIR",
            *SPECIFIC_ENV.values(),
        )
    )
)

OFFLOAD_MANIFEST = Path("configs/private_data/offloaded_manifest.json")


class PrivateDataError(FileNotFoundError):
    """A private file or directory is not available. The message has no file contents."""


class OffloadManifestError(ValueError):
    """The committed offload manifest is missing or malformed. No file contents."""


def _norm(relative: str) -> str:
    return relative.replace("\\", "/").lstrip("/").rstrip("/")


def env_for(relative: str) -> str:
    text = _norm(relative)
    if text in SPECIFIC_ENV:
        return SPECIFIC_ENV[text]
    if text == "data/held_out" or text.startswith("data/held_out/"):
        return "SLM_HELD_OUT_DIR"
    if (
        text == "data/eval/router_unseen"
        or text.startswith("data/eval/router_unseen/")
        or text == "data/eval/unseen"
        or text.startswith("data/eval/unseen/")
    ):
        return "SLM_SEALED_DIR"
    return "SLM_PRIVATE_DATA_DIR"


def missing_message(relative: str, env: str | None = None) -> str:
    variable = env or env_for(relative)
    return (
        f"private data is absent: {relative}. "
        f"Set {variable} (or SLM_PRIVATE_DATA_DIR / SLM_PRIVATE_ROOT) to an external "
        f"path that contains {relative}. Refusing to continue without the file."
    )


def _join_sealed(base: str, rel: str) -> Path:
    rel = _norm(rel)
    for prefix in (
        "data/eval/router_unseen/",
        "data/eval/unseen/",
        "data/held_out/",
    ):
        if rel.startswith(prefix):
            return Path(base) / rel[len(prefix) :]
    for prefix in ("data/eval/router_unseen", "data/eval/unseen", "data/held_out"):
        if rel == prefix:
            return Path(base)
    return Path(base) / rel


def locate(root: Path, relative: str, env: str | None = None) -> Path:
    """Return the path to use. Does not require the file to exist."""
    rel = _norm(relative)
    variable = env or env_for(rel)

    override = os.environ.get(variable, "").strip()
    if override:
        if variable == "SLM_HELD_OUT_DIR" and rel.startswith("data/held_out/"):
            return Path(override) / rel[len("data/held_out/") :]
        if variable == "SLM_SEALED_DIR":
            return _join_sealed(override, rel)
        if variable in {"SLM_PRIVATE_DATA_DIR", "SLM_PRIVATE_ROOT"}:
            return Path(override) / rel
        if variable in SPECIFIC_ENV.values():
            return Path(override)
        return Path(override)

    sealed = os.environ.get("SLM_SEALED_DIR", "").strip()
    if sealed and (
        rel.startswith("data/eval/router_unseen")
        or rel.startswith("data/eval/unseen")
        or rel.startswith("data/held_out")
    ):
        return _join_sealed(sealed, rel)

    for key in ("SLM_PRIVATE_DATA_DIR", "SLM_PRIVATE_ROOT"):
        private = os.environ.get(key, "").strip()
        if private:
            return Path(private) / rel

    held = os.environ.get("SLM_HELD_OUT_DIR", "").strip()
    if held and (rel == "data/held_out" or rel.startswith("data/held_out/")):
        if rel == "data/held_out":
            return Path(held)
        return Path(held) / rel[len("data/held_out/") :]

    return root / rel


def require(root: Path, relative: str, env: str | None = None) -> Path:
    """Return an existing file or directory, or raise ``PrivateDataError``."""
    rel = _norm(relative)
    variable = env or env_for(rel)
    path = locate(root, rel, env)
    if path.exists():
        return path
    raise PrivateDataError(missing_message(rel, variable))


def private_mode() -> bool:
    """True when any private-data env var is set to a non-empty value."""
    return any(os.environ.get(name, "").strip() for name in PRIVATE_ENV)


def offloaded_entries(root: Path) -> list[dict[str, object]]:
    """Load the committed manifest. Raises ``OffloadManifestError`` when unusable."""
    path = root / OFFLOAD_MANIFEST
    if not path.is_file():
        raise OffloadManifestError(f"offloaded manifest is missing: {OFFLOAD_MANIFEST.as_posix()}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OffloadManifestError(f"offloaded manifest is malformed: {OFFLOAD_MANIFEST.as_posix()}") from exc
    files = payload.get("files") if isinstance(payload, dict) else None
    if not isinstance(files, list) or not files:
        raise OffloadManifestError(f"offloaded manifest is malformed: {OFFLOAD_MANIFEST.as_posix()}")
    entries: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in files:
        if not isinstance(item, dict):
            raise OffloadManifestError(f"offloaded manifest is malformed: {OFFLOAD_MANIFEST.as_posix()}")
        rel = item.get("path")
        size = item.get("bytes")
        digest = item.get("sha256")
        if not isinstance(rel, str) or not rel.strip():
            raise OffloadManifestError(f"offloaded manifest is malformed: {OFFLOAD_MANIFEST.as_posix()}")
        norm = _norm(rel)
        parts = Path(norm).parts
        if not norm or norm.startswith("/") or ".." in parts or norm in seen:
            raise OffloadManifestError(f"offloaded manifest is malformed: {OFFLOAD_MANIFEST.as_posix()}")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise OffloadManifestError(f"offloaded manifest is malformed: {OFFLOAD_MANIFEST.as_posix()}")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdefABCDEF" for c in digest):
            raise OffloadManifestError(f"offloaded manifest is malformed: {OFFLOAD_MANIFEST.as_posix()}")
        seen.add(norm)
        entries.append({"path": norm, "bytes": size, "sha256": digest.lower()})
    return entries


def offloaded_paths(root: Path) -> set[str]:
    """Repo-relative paths listed in the committed offload manifest."""
    return {str(entry["path"]) for entry in offloaded_entries(root)}


def is_offloaded(relative: str, root: Path | None = None) -> bool:
    """True for an exact manifest path, a path under a listed directory, or a directory that contains listed files.

    A file that is not itself listed is not offloaded, even when sibling files are.
    A missing or malformed manifest matches nothing.
    """
    base = Path.cwd() if root is None else root
    try:
        listed = offloaded_paths(base)
    except OffloadManifestError:
        return False
    rel = _norm(relative)
    if rel in listed:
        return True
    prefix = rel.rstrip("/") + "/"
    if any(path.startswith(prefix) for path in listed):
        return True
    for entry in listed:
        directory = entry.rstrip("/")
        if rel.startswith(directory + "/"):
            return True
    return False
