"""Refusal / qualification grader.

A case supplies an expected block, a list of named inputs, or both.
The block must be present when it is supplied. Every named input must be
present when they are supplied. Surrounding wording is ignored.
No language model is used.

Two authors wrote the gold and the reply contract with different words for
the same three blocks. The grader compares blocks by class, not by spelling:

- refuse:  ``refuse`` (reply contract), ``refusal`` and ``false_premise``
  (blind shape)
- qualify: ``qualify`` (reply contract), ``qualifier`` and ``missing_input``
  (blind shape)
- none:    ``none`` (reply contract), ``formula`` (blind shape)

A block word outside those classes is compared by its own spelling.
Classes never merge: refuse never matches qualify or none.

Named inputs are read from every name-list field the reply carries
(``missing_inputs`` and ``required_inputs``), so a name listed under the
other label still counts. Names compare after case folding and after runs of
spaces, hyphens, and underscores collapse to one separator. A descriptive name
also matches the symbol it identifies in the corpus notation list
(``eval/graders/symbols.py``): the one symbol whose printed definition holds
every word of the name. Either form passes for a named input.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from eval.graders.parse import block_from_label, extract_object, field_from_label, parse_string_list
from eval.graders.result import GradeResult
from eval.graders.symbols import name_keys

_SEPARATORS = re.compile(r"[\s\-_]+")
_NAME_FIELDS = ("missing_inputs", "required_inputs")

# Block word -> block class. Keys are normalised with _norm.
_BLOCK_CLASS = {
    "refuse": "refuse",
    "refusal": "refuse",
    "false_premise": "refuse",
    "qualify": "qualify",
    "qualifier": "qualify",
    "missing_input": "qualify",
    "none": "none",
    "formula": "none",
}


def _norm(text: str) -> str:
    return _SEPARATORS.sub("_", str(text).strip().casefold()).strip("_")


def block_class(word: str) -> str:
    """The block class of a block word. An unknown word is its own class."""
    key = _norm(word)
    return _BLOCK_CLASS.get(key, key)


def _token_present(text: str, token: str) -> bool:
    parts = [re.escape(part) for part in _SEPARATORS.split(token.strip()) if part]
    if not parts:
        return False
    body = r"[\s\-_]+".join(parts)
    return re.search(rf"(?<![\w]){body}(?![\w])", text, flags=re.IGNORECASE) is not None


def _block_value(response: str, obj: dict | None) -> str | None:
    if obj is not None and obj.get("block") is not None:
        return str(obj["block"]).strip()
    return block_from_label(response)


def _list_field(response: str, obj: dict | None, field: str) -> list[str] | None:
    if obj is not None and field in obj:
        raw = obj[field]
        if isinstance(raw, list) and all(isinstance(item, str) for item in raw):
            return [item.strip() for item in raw]
        return []
    labelled = field_from_label(response, field)
    if labelled is None:
        return None
    parsed = parse_string_list(labelled)
    return parsed if parsed is not None else []


def _named_values(response: str, obj: dict | None) -> list[str] | None:
    """Names from every name-list field, or None when the reply has none."""
    found: list[str] | None = None
    for field in _NAME_FIELDS:
        values = _list_field(response, obj, field)
        if values is None:
            continue
        found = (found or []) + values
    return found


def grade_refusal_or_qualification(
    response: str,
    *,
    expected_block: str | None = None,
    named_inputs: Sequence[str] | None = None,
) -> GradeResult:
    names = tuple(named_inputs or ())
    if expected_block is None and not names:
        raise ValueError("case must supply expected_block or named_inputs")
    if expected_block is not None and not str(expected_block).strip():
        raise ValueError("expected_block must not be blank")
    if any(not str(name).strip() for name in names):
        raise ValueError("named input must not be blank")

    obj = extract_object(response)
    reasons: list[str] = []

    if expected_block is not None:
        wanted = expected_block.strip()
        wanted_class = block_class(wanted)
        found = _block_value(response, obj)
        if found is not None:
            present = block_class(found) == wanted_class
        else:
            present = _token_present(response, wanted) or _token_present(response, wanted_class)
        if not present:
            reasons.append(f"expected block {wanted!r} not present")

    if names:
        found_names = _named_values(response, obj)
        if found_names is None:
            for name in names:
                if not _token_present(response, name.strip()):
                    reasons.append(f"named input {name.strip()!r} not present")
        else:
            have: set[str] = set()
            for name in found_names:
                have |= name_keys(name)
            for name in names:
                if not name_keys(name) & have:
                    reasons.append(f"named input {name.strip()!r} not present")

    passed = not reasons
    detail = "ok" if passed else "; ".join(reasons)
    return GradeResult("refusal_or_qualification", passed, detail)
