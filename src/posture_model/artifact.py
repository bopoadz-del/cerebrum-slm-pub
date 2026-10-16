"""Load the pinned serve artifact (checkpoint path + system prompt)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from model.identity import API_KEY_ENV, MODEL_ID, repo_root

SERVE_CONFIG_RELATIVE = Path("configs") / "serve" / "posture-epoch2.json"
CHECKPOINT = (
    "tinker://d33e27e4-5533-5af4-9039-c9bf8254d4b7:train:0"
    "/sampler_weights/posture-qwen3-8b-lora-r16-epoch2"
)


@dataclass(frozen=True)
class ServeArtifact:
    """Pinned serve pair: LoRA sampler weights + committed system prompt."""

    artifact_id: str
    checkpoint: str
    base_model: str
    system_prompt: str
    system_prompt_file: str
    stop_gate: dict
    sampling: dict
    api_key_env: str


def _read_text(path: Path) -> str:
    root = repo_root()
    scripts = str(root / "scripts")
    import sys

    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from gates.blind_seal import training_open

    with training_open(path, root, "r") as handle:
        return handle.read()


@lru_cache(maxsize=1)
def load_serve_artifact() -> ServeArtifact:
    """Load the pinned epoch-2 serve config and its system prompt text."""
    root = repo_root()
    path = root / SERVE_CONFIG_RELATIVE
    config = json.loads(_read_text(path))
    checkpoint = str(config.get("checkpoint", "")).strip()
    if checkpoint != CHECKPOINT:
        raise ValueError(
            f"serve config checkpoint {checkpoint!r} does not match pinned {CHECKPOINT!r}"
        )
    if config.get("base_model") != MODEL_ID:
        raise ValueError(f"serve base_model must be {MODEL_ID}")
    if config.get("retrieval") != "fork":
        raise ValueError("retrieval is The Fork")
    if config.get("api_key_env") != API_KEY_ENV:
        raise ValueError("api key is read only from TINKER_API_KEY")
    for banned in ("api_key", "token", "secret"):
        if banned in config:
            raise ValueError("serve config must not carry a secret")
    prompt_rel = config.get("system_prompt_file")
    if not isinstance(prompt_rel, str) or not prompt_rel.strip():
        raise ValueError("system_prompt_file is required")
    prompt_path = root / prompt_rel
    system_prompt = _read_text(prompt_path)
    if not system_prompt.strip():
        raise ValueError("system prompt is empty")
    sampling = config.get("sampling")
    if not isinstance(sampling, dict):
        raise ValueError("sampling block is required")
    stop_gate = config.get("stop_gate")
    if not isinstance(stop_gate, dict):
        raise ValueError("stop_gate block is required")
    if stop_gate.get("result") != "PASS":
        raise ValueError("serve config stop_gate.result must be PASS")
    artifact_id = str(config.get("artifact_id", "")).strip()
    if not artifact_id:
        raise ValueError("artifact_id is required")
    return ServeArtifact(
        artifact_id=artifact_id,
        checkpoint=checkpoint,
        base_model=MODEL_ID,
        system_prompt=system_prompt,
        system_prompt_file=prompt_rel,
        stop_gate=dict(stop_gate),
        sampling=dict(sampling),
        api_key_env=API_KEY_ENV,
    )


def load_system_prompt() -> str:
    """Return the system prompt that ships with the served artifact."""
    return load_serve_artifact().system_prompt
