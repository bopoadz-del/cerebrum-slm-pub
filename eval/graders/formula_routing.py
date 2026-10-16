"""Formula-routing grader.

Pass only when the reply's formula id equals the expected id and the
required-inputs set equals the expected set. Order does not matter.
Extra or missing names fail. A blank expected id means no formula:
an empty route passes, and any returned id fails. No language model is used.
"""

from __future__ import annotations

from collections.abc import Sequence

from eval.graders.parse import extract_object, field_from_label, parse_string_list
from eval.graders.result import GradeResult


def _formula_id(response: str, obj: dict | None) -> str | None:
    if obj is not None and "formula_id" in obj:
        value = obj["formula_id"]
        if isinstance(value, str):
            return value.strip()
        return None
    labelled = field_from_label(response, "formula_id")
    if labelled is None:
        return None
    return labelled.strip().strip("'\"")


def _required_inputs(response: str, obj: dict | None) -> set[str] | None:
    if obj is not None and "required_inputs" in obj:
        value = obj["required_inputs"]
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return {item.strip() for item in value}
        return None
    labelled = field_from_label(response, "required_inputs")
    if labelled is None:
        return None
    parsed = parse_string_list(labelled)
    if parsed is None:
        return None
    return {item for item in parsed}


def grade_formula_routing(
    response: str,
    *,
    formula_id: str,
    required_inputs: Sequence[str],
) -> GradeResult:
    expected_id = "" if formula_id is None else str(formula_id).strip()
    expected = {str(name).strip() for name in required_inputs}
    if any(not name for name in expected):
        raise ValueError("required input must not be blank")

    obj = extract_object(response)
    reasons: list[str] = []
    found_id = _formula_id(response, obj)
    if found_id != expected_id:
        reasons.append(f"formula id {found_id!r} != {expected_id!r}")
    found_inputs = _required_inputs(response, obj)
    if found_inputs is None:
        reasons.append("required-inputs set not present")
    elif found_inputs != expected:
        reasons.append(
            f"required-inputs {sorted(found_inputs)!r} != {sorted(expected)!r}"
        )

    passed = not reasons
    detail = "ok" if passed else "; ".join(reasons)
    return GradeResult("formula_routing", passed, detail)
