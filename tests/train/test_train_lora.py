"""Dry-run and config checks for the posture LoRA entry point."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from train.train_lora import (
    TrainPair,
    TrainRefused,
    encode_chat,
    format_dry_run,
    load_lora_config,
    load_model_identity,
    load_train_pairs,
    main,
    messages_for,
    prepare_training_pairs,
    render_pairs,
    run_training,
    summarize,
    supervised_tokens,
    validate_lora_config,
)

ROOT = Path(__file__).resolve().parents[2]
EVAL_PROMPT = (ROOT / "eval" / "baselines" / "prompt_only_system_prompt.txt").read_text(encoding="utf-8")
_REGISTRY = {
    "count": 1,
    "formulas": [
        {
            "id": "example_formula",
            "definition": {
                "inputs": [
                    {"name": "span_m", "required": True},
                    {"name": "load_kn", "required": True},
                    {"name": "factor", "required": False},
                ]
            },
        }
    ],
}


def _lora(**overrides: object) -> dict:
    config = json.loads((ROOT / "configs" / "lora" / "qwen3-8b-posture.json").read_text(encoding="utf-8"))
    config.update(overrides)
    return config


def _model(**overrides: object) -> dict:
    config = {
        "model_id": "Qwen/Qwen3-8B",
        "tinker_base_model": "Qwen/Qwen3-8B",
        "sampling": {"enable_thinking": False},
    }
    config.update(overrides)
    return config


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _repo(tmp_path: Path, *, assistant: str = "Name the missing input.", user: str = "What is missing?", split: str = "train", count: int = 1, fact_count: int = 0, rows: list[dict] | None = None) -> Path:
    _write(tmp_path / "configs" / "model" / "qwen3-8b.json", json.dumps(_model()))
    _write(tmp_path / "configs" / "lora" / "qwen3-8b-posture.json", json.dumps(_lora()))
    _write(tmp_path / "eval" / "baselines" / "prompt_only_system_prompt.txt", EVAL_PROMPT)
    _write(tmp_path / "data" / "retrieval" / "formula_registry.json", json.dumps(_REGISTRY))
    if rows is None:
        rows = [
            {
                "system": "",
                "user": user,
                "assistant": assistant,
                "meta": {"split": split, "pair_type": "SCOPE_CHECK"},
            }
        ]
    body = "".join(json.dumps(row) + "\n" for row in rows)
    _write(tmp_path / "data" / "train" / "pairs.jsonl", body)
    _write(
        tmp_path / "data" / "train" / "manifest.json",
        json.dumps({"count": count, "fact_count": fact_count}),
    )
    return tmp_path


class _Tokenizer:
    def __init__(self, *, length: int = 0, stable: bool = True, thinking: dict | None = None) -> None:
        self.length = length
        self.stable = stable
        self.thinking = thinking if thinking is not None else {}

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, enable_thinking=True):
        self.thinking["enable_thinking"] = enable_thinking
        if self.length:
            base = [1, 2, 3]
            if add_generation_prompt:
                return base + [4]
            return base + [4] + [5] * self.length
        ids = [1]
        for message in messages:
            if message["role"] == "system":
                ids.append(2)
            elif message["role"] == "user":
                ids.append(3)
            elif message["role"] == "assistant":
                ids.extend([4, 8])
            else:
                raise AssertionError(message["role"])
            ids.extend((ord(char) % 17) + 10 for char in message["content"])
            if message["role"] == "assistant":
                ids.append(9)
            else:
                ids.append(7)
        if add_generation_prompt:
            ids.extend([4, 8])
        if not self.stable and not add_generation_prompt:
            ids = [0] + ids
        return ids


class _Future:
    def __init__(self, path: str | None = None) -> None:
        self.path = path

    def result(self) -> _Future:
        return self


class _Client:
    def __init__(self, tokenizer: _Tokenizer) -> None:
        self.tokenizer = tokenizer
        self.batches: list[tuple[str, list]] = []
        self.optims: list[object] = []
        self.saves: list[tuple[str, str]] = []

    def get_tokenizer(self) -> _Tokenizer:
        return self.tokenizer

    def forward_backward(self, data, loss_fn, loss_fn_config=None):
        self.batches.append((loss_fn, data))
        return _Future()

    def optim_step(self, optim_params=None, *, adam_params=None):
        self.optims.append(optim_params if optim_params is not None else adam_params)
        return _Future()

    def save_state(self, name, **kwargs):
        self.saves.append(("state", name))
        return _Future("tinker://run/weights/posture-qwen3-8b-lora")

    def save_weights_for_sampler(self, name, **kwargs):
        self.saves.append(("sampler", name))
        return _Future("tinker://run/sampler_weights/posture-qwen3-8b-lora")


class _Service:
    def __init__(self, client: _Client) -> None:
        self.client = client
        self.kwargs: dict | None = None
        self.closed: str | None = None

    def create_lora_training_client(self, **kwargs):
        self.kwargs = kwargs
        return self.client

    def close(self, status, detail=None):
        self.closed = status
        return _Future()


class _Types:
    class AdamParams:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class DimuonParams:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class AdamOptimizerConfig:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class DimuonOptimizerConfig:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class ModelInput:
        @staticmethod
        def from_ints(tokens):
            return list(tokens)

    class Datum:
        def __init__(self, model_input, loss_fn_inputs) -> None:
            self.model_input = model_input
            self.loss_fn_inputs = loss_fn_inputs


class _SDK:
    def __init__(self, tokenizer: _Tokenizer | None = None) -> None:
        self.client = _Client(tokenizer or _Tokenizer())
        self.service = _Service(self.client)
        self.types = _Types()

    def ServiceClient(self):
        return self.service


def test_committed_config_is_the_instruct_posture_adapter() -> None:
    config = load_lora_config(ROOT)
    validate_lora_config(config, load_model_identity(ROOT))
    assert config["base_model"] == "Qwen/Qwen3-8B"
    assert config["rank"] == 8
    assert config["alpha"] == 8
    assert config["target_modules"] == {"mlp": True, "attn": True, "unembed": False}
    assert config["learning_rate"] == pytest.approx(0.00005)
    assert config["epochs"] == 1
    assert config["batch_size"] == 8
    assert config["max_sequence_length"] == 1024
    assert config["seed"] == 0
    assert config["loss_fn"] == "cross_entropy"
    assert config["loss_on"] == "assistant_turns_only"
    assert config["enable_thinking"] is False
    assert config["train_file"] == "data/train/pairs.jsonl"
    assert config["system_prompt_file"] == "eval/baselines/prompt_only_system_prompt.txt"
    assert config["target_format"] == "posture_json_v1"


def test_rank16_config_is_written_down_and_valid() -> None:
    config = load_lora_config(ROOT, ROOT / "configs" / "lora" / "qwen3-8b-posture-r16.json")
    validate_lora_config(config, load_model_identity(ROOT))
    assert config["rank"] == 16
    assert config["alpha"] == 16
    assert 2 <= config["epochs"] <= 3
    # Tinker get_lr("Qwen/Qwen3-8B", is_lora=True): 5e-5 * 10 * (2000 / 4096) ** 0.0775
    assert config["learning_rate"] == pytest.approx(5e-5 * 10 * (2000 / 4096) ** 0.0775, rel=1e-3)
    assert config["system_prompt_file"] == "eval/baselines/prompt_only_system_prompt.txt"
    assert config["target_format"] == "posture_json_v1"


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"base_model": "Qwen/Qwen3-4B"}, "base_model"),
        ({"base_model": "Qwen/Qwen3-8B-Base"}, "base_model"),
        ({"alpha": 16}, "alpha must equal rank"),
        ({"rank": 64}, "rank"),
        ({"epochs": 4}, "epochs"),
        ({"learning_rate": 0.01}, "learning_rate must be > 0 and <= 0.0005"),
        ({"loss_on": "all_tokens"}, "loss_on"),
        ({"enable_thinking": True}, "enable_thinking"),
        ({"train_file": "data/held_out/pairs.jsonl"}, "train split only"),
        ({"api_key": "present"}, "unknown"),
        ({"system_prompt_file": "other_prompt.txt"}, "system_prompt_file"),
        ({"target_format": "prose"}, "target_format"),
    ],
)
def test_config_validation_rejects(overrides: dict, match: str) -> None:
    with pytest.raises(TrainRefused, match=match):
        validate_lora_config(_lora(**overrides), _model())


def test_identity_must_stay_on_the_instruct_checkpoint() -> None:
    with pytest.raises(TrainRefused, match="instruct checkpoint"):
        validate_lora_config(_lora(), _model(model_id="Qwen/Qwen3-4B", tinker_base_model="Qwen/Qwen3-4B"))


def test_dry_run_prints_counts_and_does_not_train(tmp_path: Path, monkeypatch, capsys) -> None:
    root = _repo(tmp_path)
    called = {"yes": False}

    def blocked(*args, **kwargs):
        called["yes"] = True
        raise AssertionError("dry-run contacted training")

    monkeypatch.setattr("train.train_lora.run_training", blocked)
    monkeypatch.setenv("TINKER_API_KEY", "present-but-not-used")
    assert main(["--dry-run", "--root", str(root)]) == 0
    assert called["yes"] is False
    out = capsys.readouterr().out
    assert "dry-run: ok" in out
    assert "pairs: 1" in out
    assert "assistant_turns: 1" in out
    assert "system_turns: 1" in out
    assert "target_format: posture_json_v1" in out
    assert "steps: 1" in out
    assert "tinker: not contacted" in out


@pytest.mark.private_data(
    'data/train/manifest.json',
    'data/train/pairs.jsonl',
)
def test_repo_dry_run_reports_the_train_split(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "train.train_lora.run_training",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("contacted")),
    )
    monkeypatch.setenv("TINKER_API_KEY", "present-but-not-used")
    assert main(["--dry-run", "--root", str(ROOT)]) == 0
    out = capsys.readouterr().out
    assert format_dry_run(load_lora_config(ROOT), summarize(prepare_training_pairs(ROOT, load_lora_config(ROOT)))) == out.strip()
    pairs = len(load_train_pairs(ROOT, load_lora_config(ROOT)))
    assert f"pairs: {pairs}" in out
    assert f"assistant_turns: {pairs}" in out
    assert f"system_turns: {pairs}" in out
    assert "tinker: not contacted" in out


def test_fact_in_an_assistant_turn_refuses(tmp_path: Path, capsys) -> None:
    root = _repo(tmp_path, assistant="The cover is 50 mm thick.")
    assert main(["--dry-run", "--root", str(root)]) == 1
    err = capsys.readouterr().err
    assert "number-unit" in err
    assert "50 mm" not in err


def test_fact_in_the_user_turn_is_not_a_training_fact(tmp_path: Path, capsys) -> None:
    root = _repo(tmp_path, user="The cover is 50 mm. What is missing?", assistant="Name the missing input.")
    assert main(["--dry-run", "--root", str(root)]) == 0
    assert "dry-run: ok" in capsys.readouterr().out


def test_wrong_split_and_manifest_mismatch_refuse(tmp_path: Path, capsys) -> None:
    root = _repo(tmp_path, split="held")
    assert main(["--dry-run", "--root", str(root)]) == 1
    assert "meta.split" in capsys.readouterr().err

    root = _repo(tmp_path, count=2)
    assert main(["--dry-run", "--root", str(root)]) == 1
    assert "count" in capsys.readouterr().err


def test_missing_key_blocks_a_real_run(tmp_path: Path, monkeypatch, capsys) -> None:
    root = _repo(tmp_path)
    monkeypatch.delenv("TINKER_API_KEY", raising=False)

    def blocked(*args, **kwargs):
        raise AssertionError("missing key still opened a client")

    monkeypatch.setattr("train.train_lora.run_training", blocked)
    assert main(["--root", str(root)]) == 2
    assert "TINKER_API_KEY" in capsys.readouterr().err


def test_assistant_tokens_carry_the_loss_and_the_prompt_does_not() -> None:
    pair = TrainPair(line=1, system="State the basis.", user="What is missing?", assistant="Name the missing input.")
    tokenizer = _Tokenizer()
    seen: dict = {}
    tokenizer.thinking = seen
    inputs, targets, weights = supervised_tokens(tokenizer, messages_for(pair), enable_thinking=False)
    prompt = encode_chat(tokenizer, messages_for(pair)[:-1], add_generation_prompt=True, enable_thinking=False)
    full = encode_chat(tokenizer, messages_for(pair), add_generation_prompt=False, enable_thinking=False)
    assert seen["enable_thinking"] is False
    assert inputs == full[:-1]
    assert targets == full[1:]
    assert weights[: len(prompt) - 1] == [0.0] * (len(prompt) - 1)
    assert weights[len(prompt) - 1 :] == [1.0] * (len(full) - len(prompt))
    assert 1.0 in weights and weights[0] == 0.0


def test_unstable_template_refuses() -> None:
    pair = TrainPair(line=1, system="", user="What is missing?", assistant="Name the missing input.")
    with pytest.raises(TrainRefused, match="prefix-stable"):
        supervised_tokens(_Tokenizer(stable=False), messages_for(pair), enable_thinking=False)


def test_run_training_uses_the_sdk_calls_and_saves_the_named_checkpoint() -> None:
    pair = TrainPair(line=1, system="", user="What is missing?", assistant="Name the missing input.")
    sdk = _SDK()
    result = run_training(_lora(), [pair], sdk=sdk)
    optim = sdk.service.kwargs["optimizer"]
    assert isinstance(optim, _Types.AdamOptimizerConfig)
    assert sdk.service.kwargs == {
        "base_model": "Qwen/Qwen3-8B",
        "rank": 8,
        "seed": 0,
        "train_mlp": True,
        "train_attn": True,
        "train_unembed": False,
        "user_metadata": {"checkpoint": "posture-qwen3-8b-lora"},
        "optimizer": optim,
    }
    assert len(sdk.client.batches) == 1
    loss_fn, batch = sdk.client.batches[0]
    assert loss_fn == "cross_entropy"
    assert len(batch) == 1
    weights = batch[0].loss_fn_inputs["weights"]
    from train.train_lora import ORIGINAL_LOSS_WEIGHT
    assert ORIGINAL_LOSS_WEIGHT in weights
    assert all(weight in {0.0, ORIGINAL_LOSS_WEIGHT} for weight in weights)
    assert sdk.client.optims[0].kwargs["learning_rate"] == pytest.approx(0.00005)
    assert sdk.client.saves == [
        ("state", "posture-qwen3-8b-lora"),
        ("sampler", "posture-qwen3-8b-lora"),
    ]
    assert result.steps == 1
    assert result.state_path.endswith("posture-qwen3-8b-lora")
    assert result.sampler_path.endswith("posture-qwen3-8b-lora")
    assert sdk.service.closed == "success"


def test_overlong_sequence_refuses_before_a_step() -> None:
    pair = TrainPair(line=1, system="", user="What is missing?", assistant="Name the missing input.")
    sdk = _SDK(_Tokenizer(length=2000))
    with pytest.raises(TrainRefused, match="sequence length"):
        run_training(_lora(), [pair], sdk=sdk)
    assert sdk.client.batches == []
    assert sdk.service.closed == "errored"


def test_training_script_does_not_name_a_sealed_tree() -> None:
    text = (ROOT / "scripts" / "train" / "train_lora.py").read_text(encoding="utf-8")
    assert "data/eval" not in text
    assert "held_out" not in text
    assert "TINKER_API_KEY=" not in text


def _graded(target: str, expected_block: str, formula_id: str, required: tuple[str, ...]):
    from eval.graders import grade_formula_routing, grade_refusal_or_qualification

    return (
        grade_refusal_or_qualification(target, expected_block=expected_block).passed,
        grade_formula_routing(target, formula_id=formula_id, required_inputs=required).passed,
    )


@pytest.mark.parametrize(
    ("pair_type", "formula_id", "block", "route", "required", "missing"),
    [
        ("why_formula_cannot_answer", "example_formula", "refuse", "example_formula", ["span_m", "load_kn"], []),
        ("which_formula_applies", "example_formula", "none", "example_formula", ["span_m", "load_kn"], []),
        ("what_must_be_known_first", "example_formula", "qualify", "example_formula", ["span_m", "load_kn"], ["span_m", "load_kn"]),
        ("which_standard_governs", "example_formula", "qualify", "", [], []),
        ("SCOPE_CHECK", "", "qualify", "", [], []),
        ("REFUSAL_SCOPE", "", "refuse", "", [], []),
    ],
)
def test_render_uses_the_eval_label_rule_and_the_registry(pair_type, formula_id, block, route, required, missing) -> None:
    from eval.labels import label_held_out_record

    pair = TrainPair(line=1, system="", user="Which applies?", assistant="Prose reply.", pair_type=pair_type, formula_id=formula_id)
    inputs = {"example_formula": ("span_m", "load_kn")}
    [rendered] = render_pairs([pair], EVAL_PROMPT, inputs)
    assert rendered.system == EVAL_PROMPT
    obj = json.loads(rendered.assistant)
    assert list(obj) == ["basis", "block", "missing_inputs", "formula_id", "required_inputs", "unsourced_figure"]
    assert obj["block"] == block
    assert obj["formula_id"] == route
    assert obj["required_inputs"] == required
    assert obj["missing_inputs"] == missing
    assert obj["unsourced_figure"] == ("refused" if block == "refuse" else "not_applicable")
    assert obj["basis"] == "unspecified"
    label = label_held_out_record(
        {"user": "Which applies?", "meta": {"pair_type": pair_type, "formula_id": formula_id or None}}
    )
    assert label.expected_block == block
    assert label.formula_id == route
    refusal_ok, _ = _graded(rendered.assistant, label.expected_block, label.formula_id, label.required_inputs)
    assert refusal_ok


def test_render_refuses_a_different_system_prompt_and_a_missing_pair_type() -> None:
    other = TrainPair(line=3, system="Another prompt.", user="Q?", assistant="A.", pair_type="SCOPE_CHECK")
    with pytest.raises(TrainRefused, match="system prompt differs"):
        render_pairs([other], EVAL_PROMPT, {})
    blank = TrainPair(line=4, system="", user="Q?", assistant="A.")
    with pytest.raises(TrainRefused, match="pair_type"):
        render_pairs([blank], EVAL_PROMPT, {})


@pytest.mark.private_data(
    'data/train/manifest.json',
    'data/train/pairs.jsonl',
)
def test_every_committed_train_pair_renders_with_the_eval_prompt() -> None:
    config = load_lora_config(ROOT)
    pairs = prepare_training_pairs(ROOT, config)
    assert pairs
    assert all(pair.system == EVAL_PROMPT for pair in pairs)
    for pair in pairs:
        obj = json.loads(pair.assistant)
        assert obj["block"] in {"refuse", "qualify", "none"}


def test_epoch_checkpoints_are_saved_when_there_is_more_than_one_epoch() -> None:
    pair = TrainPair(line=1, system="", user="What is missing?", assistant="Name the missing input.")
    sdk = _SDK()
    result = run_training(_lora(epochs=2, checkpoint_name="posture-test"), [pair], sdk=sdk)
    assert ("sampler", "posture-test-epoch1") in sdk.client.saves
    assert ("sampler", "posture-test-epoch2") in sdk.client.saves
    assert len(result.epoch_sampler_paths) == 2
    assert result.steps == 2
    assert [record["epoch"] for record in result.losses] == [1, 2]


class _TensorData:
    """Stands in for the SDK tensor wrapper: not iterable, like tinker TensorData."""

    def __init__(self, values) -> None:
        self._values = list(values)


class _TensorTypes(_Types):
    class Datum:
        def __init__(self, model_input, loss_fn_inputs) -> None:
            self.model_input = model_input
            self.loss_fn_inputs = {key: _TensorData(value) for key, value in loss_fn_inputs.items()}


class _MetricsFuture(_Future):
    metrics = {"loss:sum": 6.0}



def test_originals_use_triple_loss_weight_paraphrases_stay_unit() -> None:
    from train.train_lora import ORIGINAL_LOSS_WEIGHT

    original = TrainPair(line=1, system="", user="What is missing?", assistant="Name the missing input.")
    paraphrase = TrainPair(
        line=2, system="", user="What is missing?", assistant="Name the missing input.", aug="paraphrase"
    )
    tok = _Tokenizer()
    _, _, w_o = supervised_tokens(tok, messages_for(original), enable_thinking=False, example_weight=ORIGINAL_LOSS_WEIGHT)
    _, _, w_p = supervised_tokens(tok, messages_for(paraphrase), enable_thinking=False, example_weight=1.0)
    assert ORIGINAL_LOSS_WEIGHT in w_o and 1.0 not in w_o
    assert 1.0 in w_p and ORIGINAL_LOSS_WEIGHT not in w_p


def test_loss_log_does_not_iterate_sdk_tensors() -> None:
    pair = TrainPair(line=1, system="", user="What is missing?", assistant="Name the missing input.")
    sdk = _SDK()
    sdk.types = _TensorTypes()
    sdk.client.forward_backward = lambda data, loss_fn, loss_fn_config=None: _MetricsFuture()
    result = run_training(_lora(), [pair], sdk=sdk)
    [record] = result.losses
    assert record["loss_weight_tokens"] > 0
    assert record["loss_sum"] == 6.0
    assert record["loss_mean_per_token"] == pytest.approx(
        round(6.0 / record["loss_weight_tokens"], 6)
    )
