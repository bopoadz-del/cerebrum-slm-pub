"""pr11: v6 training mechanism. Class weights come from counts only; selection is validation macro recall; epochs <= 3."""

import inspect
import json
import random
from pathlib import Path

import pytest

import router.balance as balance
from router.balance import (
    SELECTION_METRIC,
    BalanceConfigError,
    class_weights,
    load_v6_config,
    macro_recall_present,
    select_epoch,
    should_stop,
)
from router.labels import load_labels
from router.train_router import fit_router_v6, training_examples


def test_weights_depend_on_counts_only_not_on_names() -> None:
    counts = [400, 300, 48, 10, 5, 48]
    names_a = [f"a{i}" for i in range(len(counts))]
    names_b = [f"zz{i}" for i in reversed(range(len(counts)))]
    rows_a = [name for name, n in zip(names_a, counts) for _ in range(n)]
    rows_b = [name for name, n in zip(names_b, counts) for _ in range(n)]
    random.Random(0).shuffle(rows_b)
    wa, wb = class_weights(rows_a, 5.0), class_weights(rows_b, 5.0)
    assert [wa[n] for n in names_a] == pytest.approx([wb[n] for n in names_b])
    assert wa["a2"] == pytest.approx(wa["a5"])  # same count, same weight
    assert wa["a4"] >= wa["a3"] > wa["a2"] > wa["a1"] > wa["a0"]  # 5 and 10 rows both reach the cap
    total = sum(counts)
    assert sum(wa[n] * c for n, c in zip(names_a, counts)) / total == pytest.approx(1.0)


def test_balanced_counts_give_unit_weights_and_the_cap_binds() -> None:
    assert all(abs(w - 1.0) < 1e-12 for w in class_weights(["x", "y", "z"] * 7, 5.0).values())
    rows = ["big"] * 1000 + ["tiny"] * 1
    capped = class_weights(rows, 5.0)
    uncapped = class_weights(rows, 1e9)
    assert capped["tiny"] / capped["big"] < uncapped["tiny"] / uncapped["big"]
    with pytest.raises(BalanceConfigError):
        class_weights(rows, 0.5)


def test_no_label_name_in_the_mechanism_code(repo_root: Path) -> None:
    labels = load_labels(repo_root / "configs" / "router" / "labels.json")
    sources = [inspect.getsource(balance), inspect.getsource(fit_router_v6)]
    config = (repo_root / "configs" / "router" / "train_v6.json").read_text(encoding="utf-8")
    for source in sources + [config]:
        for label in labels:
            assert f'"{label}"' not in source and f"'{label}'" not in source, label


def test_selection_metric_is_never_accuracy(repo_root: Path) -> None:
    assert "accuracy" not in SELECTION_METRIC
    for fn in (select_epoch, should_stop, macro_recall_present):
        assert "accuracy" not in inspect.getsource(fn)
    history = [
        {"epoch": 1, SELECTION_METRIC: 0.66, "accuracy": 0.86},
        {"epoch": 2, SELECTION_METRIC: 0.53, "accuracy": 0.87},
    ]
    assert select_epoch(history) == 1
    assert select_epoch([{"epoch": 1, SELECTION_METRIC: 0.5}, {"epoch": 2, SELECTION_METRIC: 0.5}]) == 1
    with pytest.raises(BalanceConfigError):
        select_epoch([{"epoch": 1, "accuracy": 0.9}])
    assert macro_recall_present(["a", "a", "a", "b"], ["a", "a", "a", "a"]) == pytest.approx(0.5)


def _config_root(tmp_path: Path, repo_root: Path, **edits) -> Path:
    (tmp_path / "configs" / "router").mkdir(parents=True)
    (tmp_path / "configs" / "lora").mkdir(parents=True)
    lora = "configs/lora/qwen3-4b-router-r16.json"
    (tmp_path / lora).write_bytes((repo_root / lora).read_bytes())
    data = json.loads((repo_root / "configs" / "router" / "train_v6.json").read_text(encoding="utf-8"))
    for path, value in edits.items():
        section, key = path.split(".")
        data[section][key] = value
    (tmp_path / "configs" / "router" / "train_v6.json").write_text(json.dumps(data), encoding="utf-8")
    return tmp_path


def test_committed_v6_config_keeps_the_rules(repo_root: Path) -> None:
    v6 = load_v6_config(repo_root)
    assert 1 <= v6["lora"]["epochs"] <= 3
    assert v6["max_epochs"] <= 3
    assert v6["lora"]["max_sequence_length"] == 6656
    assert v6["config"]["selection"]["metric"] == SELECTION_METRIC
    assert v6["cap"] >= 1


def test_v6_config_refuses_more_than_three_epochs_or_an_accuracy_metric(tmp_path: Path, repo_root: Path) -> None:
    with pytest.raises(BalanceConfigError):
        load_v6_config(_config_root(tmp_path / "a", repo_root, **{"overrides.epochs": 4}))
    with pytest.raises(BalanceConfigError):
        load_v6_config(_config_root(tmp_path / "b", repo_root, **{"selection.metric": "accuracy"}))


def test_early_stop_and_epoch_cap() -> None:
    h = [{"epoch": 1, SELECTION_METRIC: 0.6}]
    assert should_stop(h, 1, 3) is False
    h.append({"epoch": 2, SELECTION_METRIC: 0.7})
    assert should_stop(h, 1, 3) is False
    h.append({"epoch": 3, SELECTION_METRIC: 0.8})
    assert should_stop(h, 1, 3) is True
    assert should_stop([{"epoch": 1, SELECTION_METRIC: 0.6}, {"epoch": 2, SELECTION_METRIC: 0.6}], 1, 3) is True


class _Result:
    def __init__(self, value=None) -> None:
        self.value = value
        self.path = value

    def result(self):
        return self


class _Tok:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking=False):
        text = "".join(f"{m['role']}:{m['content']}|" for m in messages)
        if add_generation_prompt:
            text += "assistant:"
        return [ord(c) for c in text]

    def decode(self, tokens, skip_special_tokens=True):
        return "".join(chr(t) for t in tokens)


class _Seq:
    def __init__(self, tokens) -> None:
        self.tokens = tokens


class _Sampler:
    def __init__(self, path: str, good: set[int], fallback: str) -> None:
        self.epoch = int(path.rsplit("epoch", 1)[1])
        self.good = good
        self.fallback = fallback

    def get_tokenizer(self):
        return _Tok()

    def sample(self, prompt, num_samples, sampling_params):
        text = "".join(chr(t) for t in prompt)
        gold = text.split("gold=", 1)[1].split("|", 1)[0]
        out = gold if self.epoch in self.good else self.fallback
        result = _Result()
        result.sequences = [_Seq([ord(c) for c in out])]
        return _Result(result).value if False else _Wrap(result)


class _Wrap:
    def __init__(self, inner) -> None:
        self.inner = inner

    def result(self):
        return self.inner


class _Trainer:
    def __init__(self) -> None:
        self.saved: list[str] = []
        self.batches: list = []

    def get_tokenizer(self):
        return _Tok()

    def forward_backward(self, batch, loss_fn):
        self.batches.extend(batch)
        return _Result("ok")

    def optim_step(self, optim_params):
        return _Result("ok")

    def save_weights_for_sampler(self, name):
        self.saved.append(name)
        return _Result(f"tinker://fake/{name}")


class _Sdk:
    def __init__(self, good: set[int], fallback: str) -> None:
        self.trainer = _Trainer()
        sdk = self

        class Service:
            def create_lora_training_client(self, **kwargs):
                return sdk.trainer

            def create_sampling_client(self, model_path):
                return _Sampler(model_path, good, fallback)

            def close(self, status):
                return _Result(status)

        self.Service = Service

        class Types:
            class SamplingParams:
                def __init__(self, **kwargs):
                    pass

            class ModelInput:
                @staticmethod
                def from_ints(values):
                    return values

            class Datum:
                def __init__(self, model_input, loss_fn_inputs):
                    self.model_input = model_input
                    self.loss_fn_inputs = loss_fn_inputs

            class AdamParams:
                def __init__(self, **kwargs):
                    pass

            class DimuonParams:
                def __init__(self, **kwargs):
                    pass

            class AdamOptimizerConfig:
                def __init__(self, **kwargs):
                    pass

            class DimuonOptimizerConfig:
                def __init__(self, **kwargs):
                    pass

        self.types = Types

    def ServiceClient(self):
        return self.Service()


def _rows(labels: list[str], counts: list[int], tag: str) -> list[dict]:
    raw = [{"label": label, "excerpt": f"gold={label}|{tag}{i}", "project_id": "p", "doc_id": f"{tag}{label}{i}"} for label, n in zip(labels, counts) for i in range(n)]
    return training_examples(raw, labels)


def test_v6_fit_weights_the_loss_by_count_and_stops_on_val_macro(repo_root: Path) -> None:
    labels = load_labels(repo_root / "configs" / "router" / "labels.json")
    train = _rows(labels[:3], [12, 4, 1], "t")
    val = _rows(labels[:3], [6, 2, 1], "v")
    sdk = _Sdk(good={1}, fallback=labels[0])
    out = fit_router_v6(repo_root, train, val, labels, sdk=sdk)
    expected = class_weights([row["messages"][-1]["content"] for row in train], load_v6_config(repo_root)["cap"])
    assert out["class_weights"] == expected
    for datum in sdk.trainer.batches:
        label_weights = {w for w in datum.loss_fn_inputs["weights"] if w}
        assert len(label_weights) == 1
        got = label_weights.pop()
        assert any(abs(got - w) < 1e-9 for w in expected.values())
    assert [h["epoch"] for h in out["history"]] == [1, 2]  # epoch 2 did not improve: early stop
    assert out["selected_epoch"] == 1
    assert out["selection_metric"] == SELECTION_METRIC
    assert len(sdk.trainer.saved) == 2


def test_v6_fit_never_runs_past_three_epochs(repo_root: Path) -> None:
    labels = load_labels(repo_root / "configs" / "router" / "labels.json")
    train = _rows(labels[:2], [5, 2], "t")
    val = _rows(labels[:2], [3, 1], "v")
    sdk = _Sdk(good={1, 2, 3, 4, 5}, fallback=labels[0])
    out = fit_router_v6(repo_root, train, val, labels, sdk=sdk)
    assert len(sdk.trainer.saved) <= 3
    assert out["selected_epoch"] == 1  # every epoch ties at 1.0; the earlier one is kept
