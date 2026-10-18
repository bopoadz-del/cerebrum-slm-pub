"""Turn an absent private input into NOT-RUN or the original failing finding."""

from __future__ import annotations

from pathlib import Path

from gates.findings import Finding, NotRun
from private_data import _norm, env_for, is_offloaded, missing_message, private_mode


def classify_absent(root: Path, gate: str, relative: str, env: str | None = None) -> Finding | NotRun:
    """NOT-RUN only in public mode when the path is offloaded. Otherwise the failing finding."""
    rel = _norm(relative)
    variable = env or env_for(rel)
    if not private_mode() and is_offloaded(rel, root):
        return NotRun(gate, rel, variable)
    return Finding(gate, rel, missing_message(rel, variable))
