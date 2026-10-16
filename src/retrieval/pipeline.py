"""Cite source ids from context The Fork attached, and call formulas.

The model does not choose retrieval, the grounding verdict, or escalation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from src.decoding.constrained import CitationToken, decode
from src.policy.escalation import decide
from src.retrieval.fork import attach_context, call_formula
from src.retrieval.grounding import blocked, grounding_gate


@dataclass(frozen=True)
class TurnResult:
    answer: str | None
    citations: tuple[str, ...]
    reason: str | None
    grounding_status: str
    escalated: bool
    escalation_reasons: tuple[str, ...]
    formula_called: str | None


def answer(
    query: str,
    project_id: str,
    proposed_text: str,
    *,
    citations: Sequence[str] = (),
    clauses: Sequence[str] = (),
    sources: Sequence[str] = (),
    formula_id: str | None = None,
    formula_params: dict[str, Any] | None = None,
    confidence: float,
    task_type: str,
    k: int | None = None,
    intent: str | None = None,
) -> TurnResult:
    """Accept a proposed answer only with context-backed citations and figures."""

    context = attach_context(query, project_id, k=k, intent=intent)
    policy = decide(confidence, task_type, context.present)
    if not context.present:
        verdict = blocked("empty context: no answer is possible")
        return _finish(
            verdict,
            policy,
            citations=(),
            formula_called=None,
        )

    formula_result: Any = None
    formula_called: str | None = None
    if formula_id and context.text_contains(formula_id):
        try:
            formula_result = call_formula(formula_id, formula_params)
            formula_called = formula_id
        except Exception as exc:
            verdict = blocked(f"formula call failed: {type(exc).__name__}")
            return _finish(verdict, policy, citations=(), formula_called=None)
    tokens = _tokens(citations, clauses, sources, formula_called)
    allowed = frozenset(
        token.identifier for token in tokens if context.allows(token.kind, token.identifier)
    )

    decoded = decode(proposed_text, tokens, allowed)
    if decoded.refused:
        verdict = blocked(decoded.reason or "identifier is not in the attached context")
        return _finish(verdict, policy, citations=(), formula_called=formula_called)

    verdict = grounding_gate(
        decoded.text,
        context,
        formula_result,
        cited_source_ids=decoded.emitted,
    )
    return _finish(
        verdict,
        policy,
        citations=decoded.emitted,
        formula_called=formula_called,
    )


def _tokens(
    citations: Sequence[str],
    clauses: Sequence[str],
    sources: Sequence[str],
    formula_called: str | None,
) -> tuple[CitationToken, ...]:
    raw: list[CitationToken] = []
    raw.extend(CitationToken("citation", item) for item in citations)
    raw.extend(CitationToken("clause", item) for item in clauses)
    raw.extend(CitationToken("source", item) for item in sources)
    if formula_called is not None:
        raw.append(CitationToken("source", formula_called))
    seen: set[tuple[str, str]] = set()
    unique: list[CitationToken] = []
    for token in raw:
        key = (token.kind, token.identifier)
        if key in seen:
            continue
        seen.add(key)
        unique.append(token)
    return tuple(unique)


def _finish(
    verdict,
    policy,
    *,
    citations: tuple[str, ...],
    formula_called: str | None,
) -> TurnResult:
    """Escalation records a decision. It does not restore a blocked answer."""

    withheld = verdict.blocked or verdict.status != "PASS"
    return TurnResult(
        answer=None if withheld else verdict.answer,
        citations=() if withheld else citations,
        reason=verdict.reason,
        grounding_status=verdict.status,
        escalated=policy.escalate,
        escalation_reasons=policy.reasons,
        formula_called=None if withheld else formula_called,
    )
