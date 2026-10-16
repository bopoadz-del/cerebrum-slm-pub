"""Required inputs per formula id, read from the committed formula registry.

One rule for the held-out labels (``eval/labels.py``) and the training targets
(``scripts/train/posture_target.py``): a formula's required inputs are the
``definition.inputs`` entries marked ``required: true``, in registry order.
An id the registry does not carry has no required inputs.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

REGISTRY_RELATIVE = "data/retrieval/formula_registry.json"


class RegistryError(ValueError):
    """The formula registry has no usable formulas list."""


def registry_inputs(registry: Mapping) -> dict[str, tuple[str, ...]]:
    """Formula id -> required input names, in registry order."""
    formulas = registry.get("formulas") if isinstance(registry, Mapping) else None
    if not isinstance(formulas, list):
        raise RegistryError("formula registry has no formulas list")
    found: dict[str, tuple[str, ...]] = {}
    for entry in formulas:
        if not isinstance(entry, Mapping):
            continue
        formula_id = entry.get("id")
        definition = entry.get("definition")
        if not isinstance(formula_id, str) or not isinstance(definition, Mapping):
            continue
        names = []
        for item in definition.get("inputs") or []:
            if isinstance(item, Mapping) and item.get("required") is True and isinstance(item.get("name"), str):
                names.append(item["name"].strip())
        found[formula_id.strip()] = tuple(names)
    return found


def load_registry_inputs(root: Path) -> dict[str, tuple[str, ...]]:
    path = Path(root) / REGISTRY_RELATIVE
    return registry_inputs(json.loads(path.read_text(encoding="utf-8")))
