"""Call The Fork. Do not retrieve here.

The retriever is ``app.core.rag.retriever.retrieve``.
Formulas are ``app.agents.formulas.run_formula``.
"""

from __future__ import annotations

import importlib
from typing import Any

from src.retrieval.context import AttachedContext, context_from_retrieve

RETRIEVER_MODULE = "app.core.rag.retriever"
RETRIEVE_NAME = "retrieve"
AVAILABLE_NAME = "available"
FORMULA_MODULE = "app.agents.formulas"
RUN_FORMULA_NAME = "run_formula"


def call_retrieve(
    query: str,
    project_id: str,
    *,
    k: int | None = None,
    intent: str | None = None,
) -> Any:
    """Call The Fork's retriever and return its payload unchanged."""

    module = importlib.import_module(RETRIEVER_MODULE)
    available = getattr(module, AVAILABLE_NAME, None)
    if callable(available) and available() is False:
        return []
    retrieve = getattr(module, RETRIEVE_NAME)
    kwargs: dict[str, Any] = {}
    if k is not None:
        kwargs["k"] = k
    if intent is not None:
        kwargs["intent"] = intent
    return retrieve(query, project_id, **kwargs)


def attach_context(
    query: str,
    project_id: str,
    *,
    k: int | None = None,
    intent: str | None = None,
) -> AttachedContext:
    """Attach the context returned by The Fork's retriever."""

    return context_from_retrieve(
        call_retrieve(query, project_id, k=k, intent=intent)
    )


def call_formula(formula_id: str, params: dict[str, Any] | None = None) -> Any:
    """Call a formula by id. The value is the callable's return, not a memory."""

    module = importlib.import_module(FORMULA_MODULE)
    run_formula = getattr(module, RUN_FORMULA_NAME)
    return run_formula(formula_id, params)
