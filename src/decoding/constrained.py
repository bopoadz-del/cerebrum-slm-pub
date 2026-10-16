"""Emit citation, clause, source, and formula ids only from attached context.

An empty allowed set means no citation or formula id is possible, so the decode
refuses. An identifier that is not in the attached context is not emitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AbstractSet, Sequence

_KINDS = frozenset({"citation", "clause", "source", "formula"})


@dataclass(frozen=True)
class EmitResult:
    value: str | None
    refused: bool
    reason: str | None


@dataclass(frozen=True)
class CitationToken:
    kind: str
    identifier: str


@dataclass(frozen=True)
class DecodeResult:
    refused: bool
    text: str | None
    emitted: tuple[str, ...]
    reason: str | None


def emit(kind: str, identifier: str, allowed: AbstractSet[str]) -> EmitResult:
    """Return the identifier only when the attached context contains it."""

    if kind not in _KINDS:
        raise ValueError(f"unknown citation kind: {kind}")
    if not allowed:
        return EmitResult(
            None,
            True,
            "empty context: no citation is possible",
        )
    if identifier not in allowed:
        return EmitResult(
            None,
            True,
            "identifier is not in the attached context",
        )
    return EmitResult(identifier, False, None)


def decode(
    text: str,
    tokens: Sequence[CitationToken],
    allowed: AbstractSet[str],
) -> DecodeResult:
    """Decode prose plus citation tokens.

    If any citation, clause, source, or formula id cannot be emitted, the text is
    dropped and no identifier is emitted.
    """

    emitted: list[str] = []
    for token in tokens:
        result = emit(token.kind, token.identifier, allowed)
        if result.refused or result.value is None:
            return DecodeResult(True, None, (), result.reason)
        emitted.append(result.value)
    return DecodeResult(False, text, tuple(emitted), None)
