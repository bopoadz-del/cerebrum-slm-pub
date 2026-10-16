"""The label set is the JSON file. Callers load it; they do not keep a copy."""

from __future__ import annotations

import json
from pathlib import Path


class LabelConfigError(Exception):
    pass


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def labels_path(root: Path | None = None) -> Path:
    return (root or repo_root()) / "configs" / "router" / "labels.json"


def label_set_path(root: Path | None = None) -> Path:
    """The approved label set in force (v9). Older sets stay on disk for replay."""
    return (root or repo_root()) / "configs" / "router" / "label_set_v9.json"


def label_set_v8_path(root: Path | None = None) -> Path:
    return (root or repo_root()) / "configs" / "router" / "label_set_v8.json"


def sealed_config_path(root: Path | None = None) -> Path:
    return (root or repo_root()) / "configs" / "router" / "sealed_eval.json"


def load_labels(path: Path | None = None) -> list[str]:
    """Return the declared label strings in file order."""
    source = path or labels_path()
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LabelConfigError(f"label file is unreadable: {source}") from exc
    if not isinstance(data, dict):
        raise LabelConfigError("label file must be an object")
    labels = data.get("labels")
    if not isinstance(labels, list) or not labels:
        raise LabelConfigError("label file has no labels array")
    cleaned: list[str] = []
    for item in labels:
        if not isinstance(item, str) or not item.strip() or item != item.strip():
            raise LabelConfigError("each label must be a non-empty string")
        cleaned.append(item)
    if len(set(cleaned)) != len(cleaned):
        raise LabelConfigError("label file repeats a label")
    return cleaned


def load_label_set(path: Path | None = None) -> dict:
    """Return the approved label-set document. Callers do not copy it into code."""
    source = path or label_set_path()
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LabelConfigError(f"label set is unreadable: {source}") from exc
    if not isinstance(data, dict):
        raise LabelConfigError("label set must be an object")
    rules = data.get("design_rules")
    entries = data.get("labels")
    if not isinstance(rules, list) or not rules:
        raise LabelConfigError("label set has no design_rules")
    if not isinstance(entries, list) or not entries:
        raise LabelConfigError("label set has no labels")
    for rule in rules:
        if not isinstance(rule, str) or not rule.strip():
            raise LabelConfigError("each design rule must be a non-empty string")
    for item in entries:
        if not isinstance(item, dict):
            raise LabelConfigError("each label entry must be an object")
        for key in ("label", "definition", "is_not"):
            value = item.get(key)
            if not isinstance(value, str) or not value.strip():
                raise LabelConfigError(f"label entry {key} must be a non-empty string")
    return data


def load_sealed_config(path: Path | None = None) -> dict:
    source = path or sealed_config_path()
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LabelConfigError(f"sealed eval config is unreadable: {source}") from exc
    if not isinstance(data, dict):
        raise LabelConfigError("sealed eval config must be an object")
    _require_int(data, "n")
    _require_int(data, "per_label")
    threshold = data.get("agreement_threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise LabelConfigError("agreement_threshold must be a number")
    if not isinstance(data.get("majority_class_fail"), bool):
        raise LabelConfigError("majority_class_fail must be a boolean")
    teacher = data.get("teacher")
    if not isinstance(teacher, dict):
        raise LabelConfigError("teacher settings must be an object")
    for key in ("provider", "model", "temperature", "confidence_floor"):
        if key not in teacher:
            raise LabelConfigError(f"teacher.{key} is missing")
    note = data.get("note")
    if not isinstance(note, str) or not note.strip():
        raise LabelConfigError("sealed eval config needs a note")
    for key in ("sealed_dir", "sealed_labels"):
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            raise LabelConfigError(f"{key} must be a path")
    return data


def confidence_floor(config: dict) -> float | None:
    value = config["teacher"]["confidence_floor"]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LabelConfigError("teacher.confidence_floor must be a number or null")
    return float(value)


def _require_int(data: dict, key: str) -> None:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise LabelConfigError(f"{key} must be an integer")
