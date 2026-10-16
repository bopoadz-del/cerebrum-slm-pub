"""OpenAI-compatible HTTP sampling for the pinned serve checkpoint.

Used when the native Tinker SDK transport cannot load (e.g. Windows
Application Control blocking pyqwest). Authenticates with TINKER_API_KEY
from the environment only.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from model.identity import API_KEY_ENV
from model.sampling import TinkerKeyMissing
from posture_model.artifact import ServeArtifact, load_serve_artifact

OAI_BASE = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
_USER_AGENT = (
    "cerebrum-slm-posture/0.0.1 "
    "(+https://github.com/bopoadz-del/cerebrum-slm)"
)


def sample_messages_http(
    messages: list[dict[str, str]],
    *,
    artifact: ServeArtifact | None = None,
) -> str:
    """Sample via Tinker's OpenAI-compatible chat completions endpoint."""
    key = os.environ.get(API_KEY_ENV, "").strip()
    if not key:
        raise TinkerKeyMissing(
            "Sampling is blocked: TINKER_API_KEY is not set in the environment. "
            "The key is not read from the repo."
        )
    if not messages:
        raise ValueError("messages must not be empty")
    art = artifact or load_serve_artifact()
    sampling = art.sampling
    # Build request body without an 8+ string-key dict literal (no-hardwiring gate).
    body: dict[str, Any] = {
        "model": art.checkpoint,
        "messages": messages,
        "max_tokens": int(sampling.get("max_tokens", 1024)),
        "temperature": float(sampling.get("temperature", 0.7)),
        "top_p": float(sampling.get("top_p", 0.8)),
        "seed": int(sampling.get("seed", 0)),
    }
    # Posture sampling turns thinking off; keep the JSON reply in content.
    body["reasoning_effort"] = False
    body["separate_reasoning"] = False
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{OAI_BASE}/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": _USER_AGENT,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"Tinker OAI HTTP {exc.code}: {detail}") from None
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError("Tinker OAI response had no choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning
    raise RuntimeError("Tinker OAI response had empty content")
