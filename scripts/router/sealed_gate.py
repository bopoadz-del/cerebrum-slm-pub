"""Sealed-set gate, version 2.

This module does not read ``data/eval/`` and does not score a checkpoint.
The val gate in ``majority_gate.py`` is unchanged. Callers pass gold and
predictions in memory. Version 2 applies only to a future fresh sealed set.
The spent v5 set is not re-scored here.
"""

from __future__ import annotations

from router.majority_gate import evaluate
from router.val_metrics import per_label_recall

RULE_V1 = (
    "The model passes only if macro recall beats every constant single-label "
    "predictor on the sealed rows, and each of those constant predictors fails "
    "the existing majority gate. Macro recall is the mean of per-label recall "
    "over labels present in the sealed rows. The constant predictors include "
    "always-drawing, the training majority, and always-report, the sealed-set "
    "majority. A tie does not beat a constant predictor. The majority gate is "
    "scripts/router/majority_gate.py with majority_class_fail left unchanged. "
    "sealed_scoring stays false."
)

RULE = (
    "Version 2. The model passes only if macro recall beats every constant "
    "single-label predictor on the sealed rows, and each of those constant "
    "predictors fails the existing majority-gate thresholds on its own. Each "
    "constant is evaluated per label, not only when that label is the set "
    "majority. Macro recall is the mean of per-label recall over labels "
    "present in the sealed rows. The constant predictors include always-drawing, "
    "the training majority, and always-report, the sealed-set majority. A tie "
    "does not beat a constant predictor. The majority gate is "
    "scripts/router/majority_gate.py with majority_class_fail left unchanged. "
    "The val gate is unchanged. sealed_scoring stays false. Version 2 applies "
    "only to a future fresh sealed set. The v5 sealed set stays FAIL under "
    "version 1 and is not re-scored."
)


def macro_recall(gold: list[str], pred: list[str], labels: list[str]) -> float | None:
    """Mean recall over labels that occur in ``gold``. A missing label stays out."""
    scores = per_label_recall(gold, pred, labels)
    present = [score for score in scores.values() if score is not None]
    if not present:
        return None
    return sum(present) / len(present)


def evaluate_sealed_gate(
    gold: list[str],
    pred: list[str],
    labels: list[str],
    *,
    majority_class_fail: bool,
) -> dict:
    """Apply sealed gate version 2. Writes nothing."""
    if not labels or not gold or len(gold) != len(pred):
        return {
            "passed": False,
            "reason": "empty or mismatched",
            "macro_recall": None,
            "beats_every_constant": False,
            "every_constant_fails_majority_gate": False,
        }
    model = macro_recall(gold, pred, labels)
    beats: dict[str, bool] = {}
    constant_fails: dict[str, bool] = {}
    for label in labels:
        constant = [label] * len(gold)
        baseline = macro_recall(gold, constant, labels)
        beats[label] = model is not None and baseline is not None and model > baseline
        constant_fails[label] = _constant_fails_on_its_own(
            label,
            len(gold),
            labels,
            majority_class_fail=majority_class_fail,
        )
    beats_every = all(beats.values())
    constants_fail = all(constant_fails.values())
    passed = beats_every and constants_fail
    return {
        "passed": passed,
        "reason": "sealed gate" if passed else "sealed gate failed",
        "macro_recall": model,
        "beats_every_constant": beats_every,
        "every_constant_fails_majority_gate": constants_fail,
    }


MIN_SEALED_PER_LABEL = 12


def evaluate_sealed_gate_v8(
    gold: list[str],
    pred: list[str],
    labels: list[str],
    *,
    majority_class_fail: bool,
) -> dict:
    """Version 2, plus a per-label count rule. Version 2 itself is unchanged.

    ``router_passed`` is true only when the version-2 gate passed and every
    declared label has at least ``MIN_SEALED_PER_LABEL`` sealed gold documents.
    A label with fewer is marked ``not passable``. Nothing else is loosened.
    """
    gate_v2 = evaluate_sealed_gate(
        gold,
        pred,
        labels,
        majority_class_fail=majority_class_fail,
    )
    aligned = bool(gold) and len(gold) == len(pred)
    recalls = per_label_recall(gold, pred, labels) if aligned else {}
    per_label: list[dict] = []
    for label in labels:
        n = sum(1 for item in gold if item == label)
        recall = None if n == 0 or not aligned else recalls[label]
        passable = n >= MIN_SEALED_PER_LABEL
        row: dict = {
            "label": label,
            "n": n,
            "recall": recall,
            "passable": passable,
        }
        if not passable:
            row["status"] = "not passable"
        per_label.append(row)
    every_label_passable = bool(labels) and all(row["passable"] for row in per_label)
    return {
        "router_passed": bool(gate_v2["passed"]) and every_label_passable,
        "min_sealed_per_label": MIN_SEALED_PER_LABEL,
        "per_label": per_label,
        "gate_v2": gate_v2,
    }


def _constant_fails_on_its_own(
    label: str,
    n: int,
    order: list[str],
    *,
    majority_class_fail: bool,
) -> bool:
    """Apply the val majority-gate threshold to this constant by itself.

    ``majority_gate.evaluate`` fails a predictor when every prediction is the
    majority label of the rows it is given. On a row set whose gold is all
    ``label``, the constant predictor of ``label`` is that majority predictor,
    so it fails when ``majority_class_fail`` is true. Version 1 called
    ``evaluate`` on the real sealed gold instead, which fails only the
    set-majority constant.
    """
    own = [label] * n
    gate = evaluate(own, own, majority_class_fail=majority_class_fail, order=order)
    return gate.passed is False
