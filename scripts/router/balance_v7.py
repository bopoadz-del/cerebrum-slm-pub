"""Router v7 class weights: tempered inverse frequency with a share cap, stored in a data file.

Every weight is computed from the label counts of the v7 training examples (base train rows plus
generated pairs). No label, project, document or row is named here; a label is only a key in a count.

Rule:
    N = training examples, K = classes present, n_c = examples of class c, natural share s_c = n_c / N
    raw_c = (N / (K * n_c)) ** exponent        exponent 0.5: the square root of the v6 inverse frequency
    w_c   = raw_c / mean_rows(raw)             mean weight over examples is 1.0, so gradient mass per epoch is unchanged
    cap   : the effective share n_c * w_c / N may not exceed share_cap * s_c, i.e. w_c <= share_cap.
            Capped classes are fixed at share_cap and the rest are rescaled so the mean stays 1.0; repeated until stable.

The weight multiplies the loss weights of the assistant label tokens (Tinker cross_entropy), exactly as in v6.
The computed weights are written to ``data/router/v7/class_weights.json`` (v7b: ``data/router/v7b/class_weights.json``,
named by ``configs/router/train_v7b.json``). Training reads that file and refuses it when it does not equal a
recomputation from the counts of the examples it is about to train on.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable

from router.balance import MAX_EPOCHS, SELECTION_METRIC, BalanceConfigError

WEIGHTING_METHOD = "tempered_inverse_frequency_loss_weight"
WEIGHTS_FILE = "data/router/v7/class_weights.json"
CONFIG_FILE = "configs/router/train_v7.json"
_TOL = 1e-9


def tempered_class_weights(row_labels: Iterable[str], exponent: float, share_cap: float) -> dict[str, float]:
    """One loss weight per class from the counts of ``row_labels`` only."""
    for name, value in (("exponent", exponent), ("share_cap", share_cap)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise BalanceConfigError(f"{name} must be a number")
    if not 0 <= exponent <= 1:
        raise BalanceConfigError("exponent must be in [0, 1]")
    if share_cap < 1:
        raise BalanceConfigError("share_cap must be >= 1")
    counts = Counter(row_labels)
    total = sum(counts.values())
    if not total:
        raise BalanceConfigError("no training rows")
    classes = len(counts)
    raw = {label: (total / (classes * n)) ** exponent for label, n in counts.items()}
    capped: set[str] = set()
    while True:
        fixed_mass = sum(counts[label] * float(share_cap) for label in capped)
        free_mass = sum(counts[label] * raw[label] for label in counts if label not in capped)
        scale = (total - fixed_mass) / free_mass if free_mass else 0.0
        weights = {label: (float(share_cap) if label in capped else raw[label] * scale) for label in counts}
        over = {label for label, w in weights.items() if label not in capped and w > share_cap + _TOL}
        if not over:
            return weights
        capped |= over


def weights_file_for(data: dict) -> str:
    """The class-weight data file a train config must name: data/router/<run>/class_weights.json."""
    run = data.get("run")
    if not isinstance(run, str) or not run.replace("_", "").isalnum():
        raise BalanceConfigError("train config run must be a plain run name")
    return f"data/router/{run}/class_weights.json"


def load_v7_config(root: Path, config_file: str = CONFIG_FILE) -> dict:
    """Read a v7-family train config (default configs/router/train_v7.json) onto the LoRA file and refuse anything off-rule."""
    data = json.loads((root / config_file).read_text(encoding="utf-8"))
    lora = json.loads((root / data["lora_config"]).read_text(encoding="utf-8"))
    lora.update(data["overrides"])
    epochs = lora.get("epochs")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= MAX_EPOCHS:
        raise BalanceConfigError(f"epochs must be 1..{MAX_EPOCHS}")
    weighting = data.get("class_weighting", {})
    if weighting.get("method") != WEIGHTING_METHOD:
        raise BalanceConfigError("class_weighting.method is not the tempered inverse-frequency loss weight")
    tempered_class_weights(["probe"], weighting.get("exponent"), weighting.get("share_cap"))
    if weighting.get("weights_file") != weights_file_for(data):
        raise BalanceConfigError("class_weighting.weights_file is not the run's data file")
    selection = data.get("selection", {})
    if selection.get("metric") != SELECTION_METRIC:
        raise BalanceConfigError(f"selection.metric must be {SELECTION_METRIC}")
    patience = selection.get("early_stop_patience")
    if isinstance(patience, bool) or not isinstance(patience, int) or patience < 1:
        raise BalanceConfigError("selection.early_stop_patience must be an integer >= 1")
    return {
        "lora": lora,
        "exponent": float(weighting["exponent"]),
        "share_cap": float(weighting["share_cap"]),
        "patience": patience,
        "max_epochs": epochs,
        "weights_file": weighting["weights_file"],
        "config": data,
    }


def v7_example_labels(base_rows: list[dict], pair_refs: list[dict]) -> list[str]:
    return [row["messages"][-1]["content"] for row in base_rows] + [ref["label"] for ref in pair_refs]


def weights_payload(root: Path, example_labels: list[str], config_file: str = CONFIG_FILE) -> dict:
    """The data file content: counts, natural and effective shares, and the weights."""
    cfg = load_v7_config(root, config_file)
    counts = Counter(example_labels)
    total = sum(counts.values())
    weights = tempered_class_weights(example_labels, cfg["exponent"], cfg["share_cap"])
    ordered = sorted(counts, key=lambda label: (-counts[label], label))
    return {
        "method": WEIGHTING_METHOD,
        "exponent": cfg["exponent"],
        "share_cap": cfg["share_cap"],
        "source": "label counts of the {0} training examples (data/router/train.jsonl + data/router/{0}/pairs_*.jsonl)".format(
            cfg["config"]["run"]
        ),
        "examples": total,
        "classes": len(counts),
        "counts": {label: counts[label] for label in ordered},
        "natural_share": {label: counts[label] / total for label in ordered},
        "weights": {label: weights[label] for label in ordered},
        "effective_share": {label: counts[label] * weights[label] / total for label in ordered},
        "capped": [label for label in ordered if abs(weights[label] - cfg["share_cap"]) <= _TOL],
    }


def read_weights(root: Path, example_labels: list[str], config_file: str = CONFIG_FILE) -> dict[str, float]:
    """Weights for training, read from the data file. Refuses a file that is not the recomputation from these examples."""
    cfg = load_v7_config(root, config_file)
    stored = json.loads((root / cfg["weights_file"]).read_text(encoding="utf-8"))
    fresh = weights_payload(root, example_labels, config_file)
    if stored.get("counts") != fresh["counts"]:
        raise BalanceConfigError("class_weights.json counts do not match the training examples")
    for label, weight in fresh["weights"].items():
        if abs(float(stored["weights"].get(label, float("nan"))) - weight) > _TOL:
            raise BalanceConfigError("class_weights.json weights do not match the rule")
    if set(stored["weights"]) != set(fresh["weights"]):
        raise BalanceConfigError("class_weights.json classes do not match the training examples")
    return {label: float(w) for label, w in stored["weights"].items()}
