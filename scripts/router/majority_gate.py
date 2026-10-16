"""Eval gate. A predictor that emits the majority label on every row fails."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class GateResult:
    passed: bool
    reason: str
    accuracy: float | None


def majority_label(gold: list[str], order: list[str]) -> str | None:
    """Most frequent declared label. Ties use the order in the label file."""
    rank = {label: index for index, label in enumerate(order)}
    counts = Counter(label for label in gold if label in rank)
    if not counts:
        return None
    best_count = max(counts.values())
    tied = [label for label, count in counts.items() if count == best_count]
    return min(tied, key=lambda label: rank[label])


def evaluate(
    gold: list[str],
    pred: list[str],
    *,
    majority_class_fail: bool,
    order: list[str],
) -> GateResult:
    """Fail a constant majority predictor even when its accuracy is 1."""
    if not gold or len(gold) != len(pred):
        return GateResult(False, "empty or mismatched", None)
    accuracy = sum(left == right for left, right in zip(gold, pred)) / len(gold)
    if majority_class_fail and _is_majority_constant(gold, pred, order):
        return GateResult(False, "majority-class predictor", accuracy)
    return GateResult(True, "not a majority-class predictor", accuracy)


def _is_majority_constant(gold: list[str], pred: list[str], order: list[str]) -> bool:
    majority = majority_label(gold, order)
    if majority is None:
        return False
    return all(label == majority for label in pred)
