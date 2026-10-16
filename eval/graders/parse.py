"""Pull a structured object out of a model reply. No model is called."""

from __future__ import annotations

import json
import re

_GRADE_KEYS = {"block", "missing_inputs", "formula_id", "required_inputs"}
_BLOCK_LINE = re.compile(
    r"""(?im)^\s*["']?block["']?\s*[:=]\s*["']?([A-Za-z_]+)["']?\s*,?\s*$"""
)
_FIELD_LINE = re.compile(
    r"""(?im)^\s*["']?(missing_inputs|formula_id|required_inputs)["']?\s*[:=]\s*(.+?)\s*,?\s*$"""
)


def extract_object(text: str) -> dict | None:
    """Return the first JSON object that carries a grader field, if any."""
    decoder = json.JSONDecoder()
    index = 0
    while True:
        start = text.find("{", index)
        if start < 0:
            return None
        try:
            obj, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            index = start + 1
            continue
        if isinstance(obj, dict) and _GRADE_KEYS.intersection(obj):
            return obj
        index = start + 1


def block_from_label(text: str) -> str | None:
    match = _BLOCK_LINE.search(text)
    if match is None:
        return None
    return match.group(1).strip()


def field_from_label(text: str, name: str) -> str | None:
    for match in _FIELD_LINE.finditer(text):
        if match.group(1) == name:
            return match.group(2).strip()
    return None


def parse_string_list(raw: str) -> list[str] | None:
    text = raw.strip().rstrip(",")
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = None
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return [item.strip() for item in value]
        parts = re.split(r"[,;]", text[1:-1])
        return [part.strip().strip("'\"") for part in parts if part.strip()]
    parts = re.split(r"[,;]", text)
    return [part.strip().strip("'\"") for part in parts if part.strip()]
