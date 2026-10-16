"""Gate findings reported by CI."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Finding:
    gate: str
    path: str
    message: str

    def format(self) -> str:
        where = f"{self.path}: " if self.path else ""
        return f"[{self.gate}] {where}{self.message}"
