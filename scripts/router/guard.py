"""Stops labelling tools from reading the sealed owner corpus."""

from __future__ import annotations

from pathlib import Path

from router.labels import repo_root


class RouterWall(Exception):
    """A blocking condition. No training labels were written."""


def refuse_sealed_input(path: Path, root: Path | None = None) -> None:
    base = (root or repo_root()).resolve()
    sealed_roots = (
        (base / "data" / "sealed").resolve(),
        (base / "data" / "eval" / "router_unseen").resolve(),
    )
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    for sealed in sealed_roots:
        try:
            resolved.relative_to(sealed)
        except ValueError:
            continue
        raise RouterWall("WALL: sealed corpus")
