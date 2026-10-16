"""Frontier pass from confidence, task type, and whether context is present.

The same inputs always produce the same decision. The decision is logged
whether or not it escalates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

LOGGER = logging.getLogger("src.policy.escalation")

CONFIDENCE_FLOOR = 0.5


@dataclass(frozen=True)
class EscalationDecision:
    escalate: bool
    reasons: tuple[str, ...]
    confidence: float
    task_type: str
    context_present: bool


def decide(confidence: float, task_type: str, context_present: bool) -> EscalationDecision:
    """Decide a frontier pass without asking a model."""

    reasons: list[str] = []
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        reasons.append("confidence_not_numeric")
        confidence_value = 0.0
    else:
        confidence_value = float(confidence)
        if confidence_value < 0.0 or confidence_value > 1.0:
            reasons.append("confidence_out_of_range")
        elif confidence_value < CONFIDENCE_FLOOR:
            reasons.append("confidence_below_floor")
    label = task_type if isinstance(task_type, str) else ""
    if not label.strip():
        reasons.append("task_type_unspecified")
    if not context_present:
        reasons.append("context_absent")
    decision = EscalationDecision(
        escalate=bool(reasons),
        reasons=tuple(reasons),
        confidence=confidence_value,
        task_type=label.strip(),
        context_present=bool(context_present),
    )
    LOGGER.info(
        "escalation escalated=%s reasons=%s confidence=%s task_type=%s context_present=%s",
        decision.escalate,
        ",".join(decision.reasons),
        decision.confidence,
        decision.task_type,
        decision.context_present,
    )
    return decision
