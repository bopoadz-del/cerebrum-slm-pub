"""Sample Qwen3-8B instruct on Tinker.

The API key is read from the environment. If it is absent this module
raises and does not open a connection. It does not train.
"""

from __future__ import annotations

import os
from typing import Any, Callable

from model.identity import API_KEY_ENV, load_model_config


class TinkerKeyMissing(RuntimeError):
    """Sampling is blocked until TINKER_API_KEY is present in the environment."""


def tinker_api_key_present() -> bool:
    return bool(os.environ.get(API_KEY_ENV, "").strip())


def build_messages(
    user_text: str,
    *,
    system_prompt: str | None = None,
    fork_context: str | None = None,
) -> list[dict[str, str]]:
    """Build a chat. Attached facts are Fork context, never a store payload."""
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    user = user_text
    if fork_context:
        user = f"{user_text.rstrip()}\n\nFork context:\n{fork_context.strip()}"
    messages.append({"role": "user", "content": user})
    return messages


def default_client_factory() -> tuple[Any, Any]:
    import tinker
    from tinker import types

    config = load_model_config()
    service = tinker.ServiceClient()
    client = service.create_sampling_client(base_model=config["tinker_base_model"])
    return client, types


def sample_messages(
    messages: list[dict[str, str]],
    *,
    client_factory: Callable[[], tuple[Any, Any]] | None = None,
) -> str:
    """Sample one completion. Raises TinkerKeyMissing when the env key is absent."""
    if not tinker_api_key_present():
        raise TinkerKeyMissing(
            "Sampling is blocked: TINKER_API_KEY is not set in the environment. "
            "The key is not read from the repo."
        )
    if not messages:
        raise ValueError("messages must not be empty")

    config = load_model_config()
    sampling = config["sampling"]
    factory = client_factory or default_client_factory
    client, types = factory()
    tokenizer = client.get_tokenizer()
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=bool(sampling["enable_thinking"]),
    )
    encoded = tokenizer.encode(prompt_text)
    if hasattr(encoded, "ids"):
        encoded = encoded.ids
    token_ids = [int(token) for token in encoded]
    prompt = types.ModelInput.from_ints(token_ids)
    params = types.SamplingParams(
        max_tokens=int(sampling["max_tokens"]),
        temperature=float(sampling["temperature"]),
        top_p=float(sampling["top_p"]),
        top_k=int(sampling["top_k"]),
        seed=int(sampling["seed"]),
    )
    future = client.sample(prompt=prompt, num_samples=1, sampling_params=params)
    result = future.result()
    tokens = result.sequences[0].tokens
    try:
        return tokenizer.decode(tokens, skip_special_tokens=True)
    except TypeError:
        return tokenizer.decode(tokens)
