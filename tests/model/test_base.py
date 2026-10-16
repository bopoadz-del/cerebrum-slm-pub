import hashlib
import json
import os
import sys
import types

import pytest

from model.identity import API_KEY_ENV, MODEL_ID, licence_path, load_model_config, repo_root
from model.sampling import (
    TinkerKeyMissing,
    build_messages,
    default_client_factory,
    sample_messages,
    tinker_api_key_present,
)


def test_licence_text_is_the_apache_grant():
    text = licence_path().read_text(encoding="utf-8")
    assert text.lstrip().startswith("Apache License")
    assert "Apache License" in text
    assert "Version 2.0, January 2004" in text
    assert "Copyright 2024 Alibaba Cloud" in text
    assert "Grant of Copyright License" in text
    digest = hashlib.sha256(licence_path().read_bytes()).hexdigest()
    assert digest == load_model_config()["licence_sha256"]


def test_model_card_names_the_instruct_checkpoint_and_the_licence():
    card = (repo_root() / "configs" / "model" / "MODEL_CARD.md").read_text(encoding="utf-8")
    assert "Qwen/Qwen3-8B" in card
    assert "Qwen/Qwen3-8B-Base" in card
    assert "Apache License, Version 2.0" in card
    assert "LICENCE-Qwen3-8B.txt" in card
    assert "The Fork" in card
    assert "TINKER_API_KEY" in card
    assert "sampling call was not made" in card


def test_config_has_no_secret_and_points_at_the_fork():
    config = load_model_config()
    assert config["model_id"] == MODEL_ID
    assert config["role"] == "instruct"
    assert config["retrieval"] == "fork"
    assert config["api_key_env"] == API_KEY_ENV
    assert config["sampling"]["enable_thinking"] is False
    blob = json.dumps(config)
    assert "sk-" not in blob
    assert "tinker://" not in blob


def test_key_absent_blocks_sampling(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    called = {"yes": False}

    def factory():
        called["yes"] = True
        raise AssertionError("factory must not run without a key")

    assert tinker_api_key_present() is False
    with pytest.raises(TinkerKeyMissing, match="blocked"):
        sample_messages([{"role": "user", "content": "Reply with the single word ready."}], client_factory=factory)
    assert called["yes"] is False


def test_blank_key_blocks_sampling(monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, "   ")
    with pytest.raises(TinkerKeyMissing):
        sample_messages([{"role": "user", "content": "ready"}])


def test_sampling_uses_instruct_id_and_non_thinking_params(monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, "present-but-not-a-real-credential")
    seen = {}

    class Tokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=True):
            seen["messages"] = messages
            seen["enable_thinking"] = enable_thinking
            seen["tokenize"] = tokenize
            return "PROMPT"

        def encode(self, text):
            seen["encoded"] = text
            return [7, 8]

        def decode(self, tokens, skip_special_tokens=False):
            seen["decoded"] = (list(tokens), skip_special_tokens)
            return "ready"

    class SamplingClient:
        def get_tokenizer(self):
            return Tokenizer()

        def sample(self, prompt, num_samples, sampling_params):
            seen["prompt"] = prompt
            seen["num_samples"] = num_samples
            seen["params"] = sampling_params

            class Future:
                def result(self):
                    class Response:
                        sequences = [type("Seq", (), {"tokens": [9]})()]

                    return Response()

            return Future()

    class Types:
        class ModelInput:
            @staticmethod
            def from_ints(tokens):
                return ("ints", tuple(tokens))

        class SamplingParams:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

    def factory():
        return SamplingClient(), Types

    text = sample_messages(
        build_messages("State the basis.", system_prompt="system", fork_context="context line"),
        client_factory=factory,
    )
    assert text == "ready"
    assert seen["enable_thinking"] is False
    assert seen["messages"][0] == {"role": "system", "content": "system"}
    assert "Fork context:" in seen["messages"][1]["content"]
    assert "Store" not in seen["messages"][1]["content"]
    assert seen["prompt"] == ("ints", (7, 8))
    assert seen["num_samples"] == 1
    assert seen["params"].temperature == 0.7
    assert seen["params"].top_p == 0.8
    assert seen["params"].top_k == 20
    assert seen["params"].seed == 0
    assert seen["params"].max_tokens == 1024
    assert os.environ[API_KEY_ENV] not in json.dumps(seen["messages"])


def test_default_factory_asks_tinker_for_the_instruct_checkpoint(monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, "present-but-not-a-real-credential")
    created = {}

    class ServiceClient:
        def create_sampling_client(self, base_model):
            created["base_model"] = base_model
            return "client"

    module = types.ModuleType("tinker")
    module.ServiceClient = ServiceClient
    module.types = types.ModuleType("tinker.types")
    previous = sys.modules.get("tinker")
    sys.modules["tinker"] = module
    try:
        client, types_module = default_client_factory()
    finally:
        if previous is None:
            sys.modules.pop("tinker", None)
        else:
            sys.modules["tinker"] = previous
    assert client == "client"
    assert types_module is module.types
    assert created["base_model"] == MODEL_ID


def test_owned_sources_do_not_embed_a_key():
    root = repo_root()
    owned = [
        root / "src" / "model",
        root / "configs",
        root / "eval" / "baselines",
        root / "eval" / "graders",
        root / "eval",
        root / "scripts" / "eval",
    ]
    assignment = "TINKER_" + "API_KEY="
    secret_prefix = "s" + "k-"
    private_key = "begin " + "private key"
    inline_secret = "api_" + "key" + '":'
    for directory in owned:
        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".py", ".json", ".md", ".txt"}:
                continue
            text = path.read_text(encoding="utf-8")
            lowered = text.lower()
            assert assignment not in text
            assert secret_prefix not in text
            assert inline_secret not in lowered.replace(" ", "")
            assert private_key not in lowered
