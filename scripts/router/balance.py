"""Router v6 training mechanism: class-weighted loss, epoch selection and early stop.

Every class weight is computed from the label counts of the training rows. No label
name, project, document or row is named here. A label is only a key in a count.

Weight rule (inverse frequency with a cap, then renormalised):
    N = training rows, K = classes present in training, n_c = rows of class c
    raw_c   = N / (K * n_c)            1.0 for every class when classes are balanced
    capped_c = min(raw_c, cap)          a rare class gets at most ``cap`` times the balanced weight
    w_c     = capped_c / mean_rows(capped)   the mean weight over training rows is 1.0, so the total
                                       gradient mass per epoch matches an unweighted run

The weight multiplies the per-token loss weights of the assistant label tokens
(Tinker cross_entropy: loss = sum(-logprob * weights)). Prompt tokens stay at 0.

Selection and early stop read one number per epoch: validation macro recall over the
classes present in validation. No other metric is read.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

SELECTION_METRIC = "val_macro_recall_present_classes"
WEIGHTING_METHOD = "inverse_frequency_loss_weight"
MAX_EPOCHS = 3


class BalanceConfigError(Exception):
    pass


def class_weights(row_labels: Iterable[str], cap: float) -> dict[str, float]:
    """Return one loss weight per class, computed from the counts of ``row_labels`` only."""
    if isinstance(cap, bool) or not isinstance(cap, (int, float)) or cap < 1:
        raise BalanceConfigError("cap must be a number >= 1")
    counts = Counter(row_labels)
    total = sum(counts.values())
    if not total:
        raise BalanceConfigError("no training rows")
    classes = len(counts)
    capped = {label: min(total / (classes * n), float(cap)) for label, n in counts.items()}
    mean = sum(counts[label] * weight for label, weight in capped.items()) / total
    return {label: weight / mean for label, weight in capped.items()}


def scale_label_weights(token_weights: Sequence[float], factor: float) -> list[float]:
    """Multiply the loss weights of one row. Zero (prompt) tokens stay zero."""
    return [float(weight) * float(factor) for weight in token_weights]


def macro_recall_present(gold: Sequence[str], pred: Sequence[str]) -> float | None:
    """Mean recall over the classes present in ``gold``. A class absent from gold is left out."""
    if len(gold) != len(pred):
        raise ValueError("gold and pred lengths differ")
    present = sorted(set(gold))
    if not present:
        return None
    recalls = []
    for label in present:
        idx = [i for i, item in enumerate(gold) if item == label]
        recalls.append(sum(1 for i in idx if pred[i] == label) / len(idx))
    return sum(recalls) / len(recalls)


def _metric(entry: dict) -> float:
    value = entry.get(SELECTION_METRIC)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BalanceConfigError(f"epoch entry has no {SELECTION_METRIC}")
    return float(value)


def select_epoch(history: Sequence[dict]) -> int:
    """Epoch number with the highest selection metric. A tie keeps the earlier epoch."""
    if not history:
        raise BalanceConfigError("no epochs to select from")
    scores = [_metric(entry) for entry in history]
    best = max(range(len(history)), key=lambda index: (scores[index], -index))
    return int(history[best]["epoch"])


def should_stop(history: Sequence[dict], patience: int, max_epochs: int) -> bool:
    """Stop at ``max_epochs`` or when the last ``patience`` epochs did not beat the best before them."""
    if len(history) >= max_epochs:
        return True
    if len(history) <= patience:
        return False
    best_before = max(_metric(entry) for entry in history[:-patience])
    return all(_metric(entry) <= best_before for entry in history[-patience:])


def load_v6_config(root: Path) -> dict:
    """Read configs/router/train_v6.json, merge it onto the LoRA file, and refuse anything off-rule."""
    data = json.loads((root / "configs" / "router" / "train_v6.json").read_text(encoding="utf-8"))
    lora = json.loads((root / data["lora_config"]).read_text(encoding="utf-8"))
    lora.update(data["overrides"])
    epochs = lora.get("epochs")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= MAX_EPOCHS:
        raise BalanceConfigError(f"epochs must be 1..{MAX_EPOCHS}")
    weighting = data.get("class_weighting", {})
    if weighting.get("method") != WEIGHTING_METHOD:
        raise BalanceConfigError("class_weighting.method is not the inverse-frequency loss weight")
    class_weights(["probe"], weighting.get("cap"))
    selection = data.get("selection", {})
    if selection.get("metric") != SELECTION_METRIC:
        raise BalanceConfigError(f"selection.metric must be {SELECTION_METRIC}")
    patience = selection.get("early_stop_patience")
    if isinstance(patience, bool) or not isinstance(patience, int) or patience < 1:
        raise BalanceConfigError("selection.early_stop_patience must be an integer >= 1")
    return {"lora": lora, "cap": float(weighting["cap"]), "patience": patience, "max_epochs": epochs, "config": data}
