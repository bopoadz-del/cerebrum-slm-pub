"""Formula registries attached when scoring routing.

Held-out routing attaches the committed formula registry
(``data/retrieval/formula_registry.json``). Blind routing attaches the HDS-5
equation list extracted by ``scripts/eval/extract_hds5_equations.py``.

Routing must never be scored without an attached registry. Refusal/qualification
and fact-leak stay context-free.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from eval.registry import REGISTRY_RELATIVE
from model.identity import repo_root

HDS5_EQUATIONS_RELATIVE = "eval/graders/data/hds5_equations.json"


class RoutingRegistryError(RuntimeError):
    """A routing registry cannot be loaded or is empty."""


@dataclass(frozen=True)
class FormulaEntry:
    formula_id: str
    name: str
    inputs: tuple[str, ...]


@dataclass(frozen=True)
class RoutingRegistry:
    """Attached formula list for one eval split."""

    split: str
    source: str
    entries: tuple[FormulaEntry, ...]

    @property
    def ids(self) -> frozenset[str]:
        return frozenset(entry.formula_id for entry in self.entries)

    def format_context(self) -> str:
        """Render the attached registry text. Empty registries are refused."""
        if not self.entries:
            raise RoutingRegistryError(f"{self.split} routing registry is empty")
        lines = [
            "Formula registry (id, name, inputs).",
            "Route only to an id listed here. required_inputs must be that id's inputs.",
            f"source: {self.source}",
        ]
        for entry in self.entries:
            inputs = ", ".join(entry.inputs) if entry.inputs else "(none)"
            lines.append(f"- id: {entry.formula_id} | name: {entry.name} | inputs: {inputs}")
        return "\n".join(lines)


def load_held_out_routing_registry(root: Path | None = None) -> RoutingRegistry:
    base = repo_root() if root is None else root
    path = base / REGISTRY_RELATIVE
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RoutingRegistryError("held-out formula registry is missing") from exc
    except json.JSONDecodeError as exc:
        raise RoutingRegistryError("held-out formula registry is not JSON") from exc
    formulas = payload.get("formulas")
    if not isinstance(formulas, list) or not formulas:
        raise RoutingRegistryError("held-out formula registry has no formulas")
    entries: list[FormulaEntry] = []
    for item in formulas:
        if not isinstance(item, dict):
            continue
        formula_id = item.get("id")
        definition = item.get("definition") if isinstance(item.get("definition"), dict) else {}
        if not isinstance(formula_id, str) or not formula_id.strip():
            continue
        name = definition.get("computes") if isinstance(definition.get("computes"), str) else formula_id
        inputs = []
        for field in definition.get("inputs") or []:
            if isinstance(field, dict) and field.get("required") is True and isinstance(field.get("name"), str):
                inputs.append(field["name"].strip())
        entries.append(FormulaEntry(formula_id.strip(), name.strip() or formula_id.strip(), tuple(inputs)))
    if not entries:
        raise RoutingRegistryError("held-out formula registry has no usable formulas")
    return RoutingRegistry("held_out", REGISTRY_RELATIVE, tuple(entries))


def load_blind_routing_registry(root: Path | None = None) -> RoutingRegistry:
    base = repo_root() if root is None else root
    path = base / HDS5_EQUATIONS_RELATIVE
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RoutingRegistryError("blind HDS-5 equation list is missing") from exc
    except json.JSONDecodeError as exc:
        raise RoutingRegistryError("blind HDS-5 equation list is not JSON") from exc
    equations = payload.get("equations")
    if not isinstance(equations, list) or not equations:
        raise RoutingRegistryError("blind HDS-5 equation list is empty")
    entries: list[FormulaEntry] = []
    for item in equations:
        if not isinstance(item, dict):
            continue
        formula_id = item.get("id")
        name = item.get("name")
        inputs = item.get("inputs")
        if not isinstance(formula_id, str) or not formula_id.strip():
            continue
        if not isinstance(name, str) or not name.strip():
            continue
        if not isinstance(inputs, list) or any(not isinstance(x, str) or not x.strip() for x in inputs):
            raise RoutingRegistryError(f"blind equation {formula_id!r} has no usable inputs")
        entries.append(
            FormulaEntry(
                formula_id.strip(),
                name.strip(),
                tuple(x.strip() for x in inputs),
            )
        )
    if not entries:
        raise RoutingRegistryError("blind HDS-5 equation list has no usable equations")
    return RoutingRegistry("blind", HDS5_EQUATIONS_RELATIVE, tuple(entries))


def load_routing_registry(split: str, root: Path | None = None) -> RoutingRegistry:
    if split == "held_out":
        return load_held_out_routing_registry(root)
    if split == "blind":
        return load_blind_routing_registry(root)
    raise RoutingRegistryError(f"unknown split for routing registry: {split}")


def require_routing_registry(registry: RoutingRegistry | None) -> RoutingRegistry:
    """Refuse bare routing. Call before any formula-routing grade."""
    if registry is None:
        raise RoutingRegistryError("routing cannot be scored without an attached registry")
    if not registry.entries:
        raise RoutingRegistryError("routing cannot be scored with an empty registry")
    return registry



def constrain_formula_response(response: str, registry: RoutingRegistry) -> str:
    """Apply constrained decoding to formula_id using the attached registry ids."""
    import json

    from eval.graders.parse import extract_object, field_from_label
    from src.decoding.constrained import emit

    require_routing_registry(registry)
    obj = extract_object(response)
    found: str | None = None
    if obj is not None and "formula_id" in obj:
        value = obj["formula_id"]
        if isinstance(value, str):
            found = value.strip()
    if found is None:
        labelled = field_from_label(response, "formula_id")
        if labelled is not None:
            found = labelled.strip().strip("'\"")
    if found is None or found == "":
        return response
    result = emit("formula", found, registry.ids)
    if not result.refused and result.value == found:
        return response
    if obj is not None and isinstance(obj, dict):
        rewritten = dict(obj)
        rewritten["formula_id"] = ""
        return json.dumps(rewritten, ensure_ascii=False)
    return response.replace(found, "", 1)
