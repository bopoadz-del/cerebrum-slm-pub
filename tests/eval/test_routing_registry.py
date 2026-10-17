"""Routing requires an attached registry; bare routing fails."""

from __future__ import annotations

import pytest

from eval.routing_registry import (
    RoutingRegistryError,
    load_blind_routing_registry,
    load_held_out_routing_registry,
    require_routing_registry,
)
from eval.tuned import grade_routing_or_raise
from model.identity import repo_root


@pytest.mark.private_data('data/retrieval/formula_registry.json')
def test_held_out_and_blind_registries_load():
    held = load_held_out_routing_registry(repo_root())
    blind = load_blind_routing_registry(repo_root())
    assert len(held.entries) >= 1
    assert len(blind.entries) == 18
    assert all(e.formula_id.startswith("HDS5-") for e in blind.entries)
    assert "id:" in held.format_context()
    assert "id:" in blind.format_context()


def test_bare_routing_is_refused():
    with pytest.raises(RoutingRegistryError, match="without an attached registry"):
        require_routing_registry(None)
    with pytest.raises(RoutingRegistryError, match="without an attached registry"):
        grade_routing_or_raise(
            '{"formula_id":"x","required_inputs":[]}',
            formula_id="",
            required_inputs=(),
            registry=None,
        )


@pytest.mark.private_data('data/retrieval/formula_registry.json')
def test_routing_with_registry_grades():
    registry = load_held_out_routing_registry(repo_root())
    entry = registry.entries[0]
    response = (
        '{"block":"none","formula_id":"%s","required_inputs":%s}'
        % (entry.formula_id, list(entry.inputs))
    ).replace("'", '"')
    # Build JSON properly
    import json
    response = json.dumps(
        {"block": "none", "formula_id": entry.formula_id, "required_inputs": list(entry.inputs)}
    )
    result = grade_routing_or_raise(
        response,
        formula_id=entry.formula_id,
        required_inputs=entry.inputs,
        registry=registry,
    )
    assert result.passed
