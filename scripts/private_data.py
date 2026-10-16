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


class PrivateDataError(FileNotFoundError):
    """A private file or directory is not available. The message has no file contents."""


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
