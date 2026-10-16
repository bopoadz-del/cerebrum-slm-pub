"""Load the committed Qwen3-8B identity. No secrets live here."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

API_KEY_ENV = "TINKER_API_KEY"
MODEL_ID = "Qwen/Qwen3-8B"
_CONFIG_NAME = Path("configs") / "model" / "qwen3-8b.json"


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / _CONFIG_NAME).is_file():
            return parent
    raise FileNotFoundError(f"{_CONFIG_NAME} not found from {here}")


def _read_text(path: Path) -> str:
    """Read through training_open so a computed path cannot skip the eval seal."""
    root = repo_root()
    scripts = str(root / "scripts")
    import sys

    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from gates.blind_seal import training_open

    with training_open(path, root, "r") as handle:
        return handle.read()


@lru_cache(maxsize=1)
def load_model_config() -> dict:
    path = repo_root() / _CONFIG_NAME
    config = json.loads(_read_text(path))
    if config.get("model_id") != MODEL_ID:
        raise ValueError(f"model config id {config.get('model_id')!r} != {MODEL_ID!r}")
    if config.get("tinker_base_model") != MODEL_ID:
        raise ValueError("tinker_base_model must be the instruct checkpoint Qwen/Qwen3-8B")
    if config.get("retrieval") != "fork":
        raise ValueError("retrieval is The Fork")
    if config.get("api_key_env") != API_KEY_ENV:
        raise ValueError("api key is read only from TINKER_API_KEY")
    if "api_key" in config or "token" in config or "secret" in config:
        raise ValueError("model config must not carry a secret")
    return config


def licence_path() -> Path:
    config = load_model_config()
    return repo_root() / config["licence_file"]
