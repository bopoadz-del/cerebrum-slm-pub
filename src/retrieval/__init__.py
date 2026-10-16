"""Retrieval attachment for the posture model.

Context comes from The Fork's retriever. This package does not retrieve.
"""

from src.retrieval.context import AttachedContext
from src.retrieval.fork import attach_context, call_formula, call_retrieve
from src.retrieval.grounding import GroundingVerdict, grounding_gate
from src.retrieval.pipeline import TurnResult, answer

__all__ = [
    "AttachedContext",
    "GroundingVerdict",
    "TurnResult",
    "answer",
    "attach_context",
    "call_formula",
    "call_retrieve",
    "grounding_gate",
]
