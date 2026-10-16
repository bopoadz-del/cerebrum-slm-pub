"""Sample the served posture artifact.

Fork must attach retrieval context as a keyword argument (empty string is
allowed when retrieval returned no hits). Formula ids outside an attached
registry are cleared by constrained decoding.
"""

from __future__ import annotations

import json
from typing import AbstractSet, Any, Callable

from model.sampling import (
    TinkerKeyMissing,
    build_messages,
    sample_messages,
    tinker_api_key_present,
)
from posture_model.artifact import ServeArtifact, load_serve_artifact
from decoding.constrained import emit

Sampler = Callable[..., str]


class ForkContextRequired(RuntimeError):
    """The serve block refuses to run when Fork context is not attached."""


class PostureSampleError(RuntimeError):
    """Sampling or constrained decoding failed without item text."""


def _checkpoint_client_factory(checkpoint: str) -> Callable[[], tuple[Any, Any]]:
    built: dict[str, tuple[Any, Any]] = {}

    def factory() -> tuple[Any, Any]:
        if "pair" not in built:
            import tinker
            from tinker import types

            service = tinker.ServiceClient()
            client = service.create_sampling_client(model_path=checkpoint)
            built["pair"] = (client, types)
        return built["pair"]

    return factory


def constrain_posture_response(
    response: str,
    *,
    allowed_formula_ids: AbstractSet[str] | None = None,
    allowed_citation_ids: AbstractSet[str] | None = None,
) -> str:
    """Clear formula_id / citation fields that are outside the attached sets.

    When a registry is attached, formula_id must be in that set (or empty).
    When citation ids are attached, any cited id outside the set refuses the
    whole response by emptying formula_id and leaving block=refuse is left to
    the caller; here we only clear out-of-set formula ids.
    """
    text = response if isinstance(response, str) else ""
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        obj = None
    if not isinstance(obj, dict):
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                obj = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                obj = None
    if not isinstance(obj, dict):
        return text

    rewritten = dict(obj)
    if allowed_formula_ids is not None:
        found = rewritten.get("formula_id", "")
        if isinstance(found, str) and found.strip():
            result = emit("formula", found.strip(), frozenset(allowed_formula_ids))
            if result.refused or result.value != found.strip():
                rewritten["formula_id"] = ""
                rewritten["required_inputs"] = []
    if allowed_citation_ids is not None:
        # Citation fields are not part of the posture JSON contract; ignore.
        _ = allowed_citation_ids
    return json.dumps(rewritten, ensure_ascii=False)


def sample_posture(
    user_text: str,
    *,
    fork_context: str,
    allowed_formula_ids: AbstractSet[str] | None = None,
    allowed_citation_ids: AbstractSet[str] | None = None,
    artifact: ServeArtifact | None = None,
    sampler: Sampler | None = None,
) -> str:
    """Sample the pinned checkpoint with the shipped system prompt.

    ``fork_context`` is required (keyword-only). Pass ``\"\"`` when The Fork
    returned no hits. Omitting it is a TypeError; passing None raises
    ForkContextRequired. When ``allowed_formula_ids`` is provided, formula ids
    outside that set are cleared before the response is returned.
    """
    if fork_context is None:
        raise ForkContextRequired(
            "posture_model refuses to sample without Fork context attached; "
            "pass fork_context=\"\" when retrieval returned no hits"
        )
    if not isinstance(fork_context, str):
        raise ForkContextRequired("fork_context must be a string")
    if not isinstance(user_text, str) or not user_text.strip():
        raise PostureSampleError("user_text must be non-empty text")

    art = artifact or load_serve_artifact()
    # Always attach a Fork context section (empty string = retrieval returned no hits)
    # so the served path never samples as if retrieval was skipped.
    messages = build_messages(
        user_text,
        system_prompt=art.system_prompt,
        fork_context=fork_context if fork_context.strip() else "(empty)",
    )

    if sampler is None:
        if not tinker_api_key_present():
            raise TinkerKeyMissing(
                "Sampling is blocked: TINKER_API_KEY is not set in the environment. "
                "The key is not read from the repo."
            )
        factory = _checkpoint_client_factory(art.checkpoint)

        def _sample(msgs: list[dict[str, str]]) -> str:
            try:
                return sample_messages(msgs, client_factory=factory)
            except ImportError:
                from posture_model.http_sample import sample_messages_http

                return sample_messages_http(msgs, artifact=art)
            except Exception as exc:
                # Windows Application Control can block pyqwest DLLs.
                msg = str(exc).lower()
                if "pyqwest" in msg or "dll load failed" in msg or "application control" in msg:
                    from posture_model.http_sample import sample_messages_http

                    return sample_messages_http(msgs, artifact=art)
                raise

        raw = _sample(messages)
    else:
        raw = sampler(messages)

    if not isinstance(raw, str):
        raise PostureSampleError("sampler did not return text")
    return constrain_posture_response(
        raw,
        allowed_formula_ids=allowed_formula_ids,
        allowed_citation_ids=allowed_citation_ids,
    )
