"""Shared labels for the recorded baselines and a tuned run.

Held-out gold has no block field. Refusal labels come from pair_type:
refuse for REFUSAL_SCOPE, REFUSAL_VS_VACUUM, FORCING_THE_NUMBER,
bare_rate_not_answer, and why_formula_cannot_answer; block none for
which_formula_applies; otherwise qualify. A formula id is expected only
for which_formula_applies, what_must_be_known_first, and
why_formula_cannot_answer. Held-out gold has no required-input list, so
the expected required inputs of a routed held-out item are that formula's
required inputs in the committed formula registry (``eval/registry.py``),
the same rule the training targets use. An empty route is a blank formula id
with an empty set. A held-out formula id outside those three pair types,
including which_standard_governs, still expects an empty route.

Blind labels use each item's shape. Refusal uses expected_block and
named_inputs. formula_id null or blank means an empty route. When a
formula id is set, named_inputs are the required inputs.

The prompt graded for fact-leak is the user text only. No Fork context
is attached. Item text is not part of a label's repr.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from eval.readers import EvalReadError, read_records
from eval.registry import load_registry_inputs
from model.identity import repo_root

HELD_OUT_RELATIVE = "data/held_out/pairs.jsonl"
HELD_OUT_MANIFEST = "data/held_out/manifest.json"
BLIND_RELATIVE = "data/eval/unseen/questions.jsonl"

_REFUSE_PAIR_TYPES = frozenset(
    {
        "REFUSAL_SCOPE",
        "REFUSAL_VS_VACUUM",
        "FORCING_THE_NUMBER",
        "bare_rate_not_answer",
        "why_formula_cannot_answer",
    }
)
_NONE_PAIR_TYPE = "which_formula_applies"
_FORMULA_PAIR_TYPES = frozenset(
    {
        "which_formula_applies",
        "what_must_be_known_first",
        "why_formula_cannot_answer",
    }
)


class EvalLabelError(RuntimeError):
    """A record has no usable label. The message has no item text."""


@dataclass(frozen=True)
class EvalLabel:
    split: str
    line: int
    prompt: str
    expected_block: str
    named_inputs: tuple[str, ...]
    formula_id: str
    required_inputs: tuple[str, ...]

    def __repr__(self) -> str:
        route = "empty" if self.formula_id == "" else "formula"
        return (
            f"EvalLabel(split={self.split!r}, line={self.line}, "
            f"block={self.expected_block!r}, route={route})"
        )


def label_eval_sets(root: Path | None = None) -> tuple[tuple[str, tuple[EvalLabel, ...]], ...]:
    """Label held-out and the blind set. No model is called."""
    base = repo_root() if root is None else root
    held_rows = read_records(base, HELD_OUT_RELATIVE)
    manifest = read_records(base, HELD_OUT_MANIFEST)
    if len(manifest) != 1:
        raise EvalLabelError("held-out manifest is not one object")
    expected = manifest[0][1].get("count")
    if expected != len(held_rows):
        raise EvalLabelError("held-out row count does not match its manifest")
    blind_rows = read_records(base, BLIND_RELATIVE)
    inputs = load_registry_inputs(base)
    return (
        ("held_out", tuple(_label_held_out(line, row, inputs) for line, row in held_rows)),
        ("blind", tuple(_label_blind(line, row) for line, row in blind_rows)),
    )


def label_held_out_record(record: dict, *, line: int = 1, inputs: dict | None = None) -> EvalLabel:
    return _label_held_out(line, record, inputs or {})


def label_blind_record(record: dict, *, line: int = 1) -> EvalLabel:
    return _label_blind(line, record)


def _label_held_out(line: int, record: dict, inputs: dict) -> EvalLabel:
    pair_type = _pair_type(record)
    if pair_type is None:
        raise EvalLabelError(f"held-out line {line} has no pair_type")
    if pair_type in _REFUSE_PAIR_TYPES:
        block = "refuse"
    elif pair_type == _NONE_PAIR_TYPE:
        block = "none"
    else:
        block = "qualify"
    formula_id = ""
    required: tuple[str, ...] = ()
    if pair_type in _FORMULA_PAIR_TYPES:
        formula_id = _required_formula_id(record, line, "held-out")
        required = tuple(inputs.get(formula_id, ()))
    prompt = _text_field(record, "user", line, "held-out")
    label = EvalLabel(
        "held_out",
        line,
        prompt,
        block,
        (),
        formula_id,
        required,
    )
    _validate(label)
    return label


def _label_blind(line: int, record: dict) -> EvalLabel:
    shape = record.get("shape")
    if not isinstance(shape, dict):
        raise EvalLabelError(f"blind line {line} has no shape")
    block = shape.get("expected_block")
    if not isinstance(block, str) or not block.strip():
        raise EvalLabelError(f"blind line {line} has no expected_block")
    named = _string_tuple(shape.get("named_inputs"), line, "named_inputs")
    raw_id = shape.get("formula_id")
    if raw_id is None or (isinstance(raw_id, str) and not raw_id.strip()):
        formula_id = ""
        required: tuple[str, ...] = ()
    elif isinstance(raw_id, str):
        formula_id = raw_id.strip()
        required = named
    else:
        raise EvalLabelError(f"blind line {line} has a formula_id that is not text")
    prompt = _text_field(record, "question", line, "blind")
    label = EvalLabel("blind", line, prompt, block.strip(), named, formula_id, required)
    _validate(label)
    return label


def _pair_type(record: dict) -> str | None:
    meta = record.get("meta")
    if isinstance(meta, dict):
        value = meta.get("pair_type")
        if isinstance(value, str) and value.strip():
            return value.strip()
    value = record.get("pair_type")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _required_formula_id(record: dict, line: int, split: str) -> str:
    meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
    value = meta.get("formula_id", record.get("formula_id"))
    if not isinstance(value, str) or not value.strip():
        raise EvalLabelError(f"{split} line {line} expects a formula id and has none")
    return value.strip()


def _text_field(record: dict, key: str, line: int, split: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise EvalLabelError(f"{split} line {line} has no {key}")
    return value


def _string_tuple(value: object, line: int, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise EvalLabelError(f"blind line {line} {field} is not a list of strings")
    if any(not item.strip() for item in value):
        raise EvalLabelError(f"blind line {line} {field} contains a blank name")
    return tuple(item.strip() for item in value)


def _validate(label: EvalLabel) -> None:
    if not label.expected_block.strip() and not label.named_inputs:
        raise EvalLabelError(f"{label.split} line {label.line} has no refusal shape")
    if any(not name.strip() for name in label.required_inputs):
        raise EvalLabelError(f"{label.split} line {label.line} has a blank required input")
