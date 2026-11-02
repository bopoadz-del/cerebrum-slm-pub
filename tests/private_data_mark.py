"""Skip decision for the private_data pytest marker. No node ids."""

from __future__ import annotations

from pathlib import Path

from private_data import env_for, is_offloaded, private_mode


def private_data_skip_reason(root: Path, paths: tuple[str, ...]) -> str | None:
    """Return a skip reason, or None when the test must run.

    Public mode skips only when at least one declared path is absent and every
    absent declared path is offloaded. Private mode never skips. The reason
    names the absent path(s) and the env var only.
    """
    if private_mode():
        return None
    absent = [rel for rel in paths if not (root / rel).exists()]
    if not absent:
        return None
    if not all(is_offloaded(rel, root) for rel in absent):
        return None
    named = ", ".join(f"{rel} ({env_for(rel)})" for rel in absent)
    return f"private_data: {named}"
