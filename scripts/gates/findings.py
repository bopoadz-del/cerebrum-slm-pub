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


@dataclass(frozen=True)
class NotRun:
    """A gate input was not checked because private data is absent in public mode."""

    gate: str
    path: str
    env: str

    def format(self) -> str:
        return f"NOT-RUN [{self.gate}] {self.path} set {self.env}"
