"""Router v7 class weights: tempered inverse frequency (exponent 0.5) with a 3x share cap, read from a data file."""

import inspect
import json
import random
from pathlib import Path

import pytest

import router.balance_v7 as balance_v7
from router.balance import SELECTION_METRIC, BalanceConfigError, class_weights
from router.balance_v7 import (
    WEIGHTS_FILE,
    load_v7_config,
    read_weights,
    tempered_class_weights,
    weights_payload,
)
from router.labels import load_labels
from router.train_router import fit_router_v7
from router.v7_pairs import v7_training_examples
from tests.router.test_v6_balance import _rows, _Sdk


def _labels_of(counts: dict[str, int]) -> list[str]:
    rows = [name for name, n in counts.items() for _ in range(n)]
    random.Random(1).shuffle(rows)
    return rows


def test_weights_depend_on_counts_only_and_keep_the_mean_at_one() -> None:
    counts = [448, 348, 96, 50, 24, 24]
    a = dict(zip([f"a{i}" for i in range(6)], counts))
    b = dict(zip([f"zz{i}" for i in reversed(range(6))], counts))
    wa = tempered_class_weights(_labels_of(a), 0.5, 3.0)
    wb = tempered_class_weights(_labels_of(b), 0.5, 3.0)
    assert [wa[k] for k in a] == pytest.approx([wb[k] for k in b])
    assert wa["a4"] == pytest.approx(wa["a5"])
    assert wa["a5"] > wa["a3"] > wa["a2"] > wa["a1"] > wa["a0"]
    assert sum(wa[k] * n for k, n in a.items()) / sum(counts) == pytest.approx(1.0)


def test_tempering_is_the_square_root_of_the_inverse_frequency_ratio() -> None:
    counts = {"x": 400, "y": 100, "z": 25}
    w = tempered_class_weights(_labels_of(counts), 0.5, 1e9)
    assert w["z"] / w["x"] == pytest.approx((400 / 25) ** 0.5)
    assert w["y"] / w["x"] == pytest.approx((400 / 100) ** 0.5)
    full = class_weights(_labels_of(counts), 1e9)
    assert tempered_class_weights(_labels_of(counts), 1.0, 1e9) == pytest.approx(full)
    assert all(v == pytest.approx(1.0) for v in tempered_class_weights(_labels_of(counts), 0.0, 3.0).values())


def test_share_cap_binds_at_three_times_the_natural_share() -> None:
    counts = {"big": 5000, "mid": 300, "tiny": 1}
    labels = _labels_of(counts)
    w = tempered_class_weights(labels, 0.5, 3.0)
    total = sum(counts.values())
    for label, n in counts.items():
        effective = n * w[label] / total
        assert effective <= 3.0 * n / total + 1e-12
    assert w["tiny"] == pytest.approx(3.0)
    assert sum(w[k] * n for k, n in counts.items()) / total == pytest.approx(1.0)
    with pytest.raises(BalanceConfigError):
        tempered_class_weights(labels, 0.5, 0.9)
    with pytest.raises(BalanceConfigError):
        tempered_class_weights(labels, 1.5, 3.0)


def test_no_label_name_in_the_weight_mechanism_or_config(repo_root: Path) -> None:
    labels = load_labels(repo_root / "configs" / "router" / "labels.json")
    sources = [inspect.getsource(balance_v7), inspect.getsource(fit_router_v7)]
    config = (repo_root / "configs" / "router" / "train_v7.json").read_text(encoding="utf-8")
    for source in sources + [config]:
        for label in labels:
            assert f'"{label}"' not in source and f"'{label}'" not in source, label


def test_committed_config_keeps_the_v6_recipe_and_points_to_the_data_file(repo_root: Path) -> None:
    v7 = load_v7_config(repo_root)
    v6 = json.loads((repo_root / "configs" / "router" / "train_v6.json").read_text(encoding="utf-8"))
    for key in ("base_model", "max_sequence_length", "epochs"):
        assert v7["lora"][key] == v6["overrides"][key]
    assert v7["config"]["lora_config"] == v6["lora_config"]
    assert v7["config"]["selection"] == v6["selection"]
    assert v7["exponent"] == 0.5 and v7["share_cap"] == 3.0
    assert v7["config"]["class_weighting"]["weights_file"] == WEIGHTS_FILE


@pytest.mark.private_data(
    'data/router/train.jsonl',
    'data/router/v7/class_weights.json',
)
def test_committed_weights_file_is_the_recomputation_from_the_v7_examples(repo_root: Path) -> None:
    examples, _ = v7_training_examples(repo_root)
    labels = [row["messages"][-1]["content"] for row in examples]
    stored = json.loads((repo_root / WEIGHTS_FILE).read_text(encoding="utf-8"))
    assert stored == json.loads(json.dumps(weights_payload(repo_root, labels)))
    weights = read_weights(repo_root, labels)
    assert set(weights) == set(labels)
    assert sum(weights[label] for label in labels) / len(labels) == pytest.approx(1.0)
    assert max(weights.values()) <= stored["share_cap"] + 1e-9
    with pytest.raises(BalanceConfigError):
        read_weights(repo_root, labels[:-1])


def _tmp_root(tmp_path: Path, repo_root: Path, labels_for_weights: list[str]) -> Path:
    for rel in ("configs/lora/qwen3-4b-router-r16.json", "configs/router/train_v7.json"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes((repo_root / rel).read_bytes())
    payload = weights_payload(tmp_path, labels_for_weights)
    (tmp_path / WEIGHTS_FILE).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / WEIGHTS_FILE).write_text(json.dumps(payload), encoding="utf-8")
    return tmp_path


def test_v7_fit_reads_the_weights_file_and_selects_on_val_macro(tmp_path: Path, repo_root: Path) -> None:
    labels = load_labels(repo_root / "configs" / "router" / "labels.json")
    train = _rows(labels[:3], [12, 4, 1], "t")
    val = _rows(labels[:3], [6, 2, 1], "v")
    train_labels = [row["messages"][-1]["content"] for row in train]
    root = _tmp_root(tmp_path, repo_root, train_labels)
    sdk = _Sdk(good={1}, fallback=labels[0])
    out = fit_router_v7(root, train, val, labels, sdk=sdk)
    stored = json.loads((root / WEIGHTS_FILE).read_text(encoding="utf-8"))["weights"]
    assert out["class_weights"] == pytest.approx(stored)
    for datum in sdk.trainer.batches:
        got = {w for w in datum.loss_fn_inputs["weights"] if w}
        assert len(got) == 1
        value = got.pop()
        assert any(abs(value - w) < 1e-9 for w in stored.values())
    assert [h["epoch"] for h in out["history"]] == [1, 2]
    assert out["selected_epoch"] == 1 and out["selection_metric"] == SELECTION_METRIC
    assert len(sdk.trainer.saved) <= 3


def test_v7_fit_refuses_a_weights_file_from_other_counts(tmp_path: Path, repo_root: Path) -> None:
    labels = load_labels(repo_root / "configs" / "router" / "labels.json")
    train = _rows(labels[:3], [12, 4, 1], "t")
    val = _rows(labels[:3], [6, 2, 1], "v")
    root = _tmp_root(tmp_path, repo_root, [row["messages"][-1]["content"] for row in train][:-1])
    with pytest.raises(BalanceConfigError):
        fit_router_v7(root, train, val, labels, sdk=_Sdk(good={1}, fallback=labels[0]))
