"""Served construction posture model block.

Pins the stop-gate-passed checkpoint and ships the system prompt with the
artifact. Retrieval stays on The Fork; this block does not retrieve.
"""

from posture_model.artifact import (
    CHECKPOINT,
    SERVE_CONFIG_RELATIVE,
    ServeArtifact,
    load_serve_artifact,
    load_system_prompt,
)
from posture_model.http_sample import sample_messages_http
from posture_model.client import (
    ForkContextRequired,
    PostureSampleError,
    constrain_posture_response,
    sample_posture,
)

__all__ = [
    "CHECKPOINT",
    "SERVE_CONFIG_RELATIVE",
    "ForkContextRequired",
    "PostureSampleError",
    "ServeArtifact",
    "constrain_posture_response",
    "load_serve_artifact",
    "load_system_prompt",
    "sample_messages_http",
    "sample_posture",
]
