"""Validation metrics for the router. This module does not read the sealed set."""

from __future__ import annotations

from pathlib import Path

from router.guard import RouterWall, refuse_sealed_input
from router.labels import repo_root
from router.majority_gate import GateResult, evaluate


def per_label_recall(
    gold: list[str],
    pred: list[str],
    labels: list[str],
) -> dict[str, float | None]:
    """Recall for each declared label. A missing class stays None. Rows are not added."""
    if len(gold) != len(pred):
        raise ValueError("gold and pred lengths differ")
    scores: dict[str, float | None] = {}
    for label in labels:
        indexes = [index for index, item in enumerate(gold) if item == label]
        if not indexes:
            scores[label] = None
            continue
        hits = sum(1 for index in indexes if pred[index] == label)
        scores[label] = hits / len(indexes)
    return scores


def majority_gate_on_val(
    gold: list[str],
    pred: list[str],
    *,
    order: list[str],
    majority_class_fail: bool,
) -> GateResult:
    return evaluate(gold, pred, majority_class_fail=majority_class_fail, order=order)


def refuse_sealed_score(path: Path, root: Path | None = None) -> None:
    """Scoring the sealed set is not part of this run."""
    try:
        refuse_sealed_input(path, root or repo_root())
    except RouterWall as exc:
        raise RouterWall("WALL: sealed set is not scored") from exc
