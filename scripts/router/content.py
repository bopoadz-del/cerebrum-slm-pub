"""Content fields the labeller may read. Names, ids, and Fork types are not content."""

from __future__ import annotations

import re

from router.strip_source_tags import strip_source_tags

# Encoded-bytes rule. An encoded run is an unbroken stretch of at least
# ENCODED_RUN_MIN characters drawn only from the base64 / hex / url-safe alphabet
# (ASCII letters, digits, "+", "/", "=", "_", "-") with no whitespace. Words in any
# language, numbers, codes and URLs are far shorter or contain other characters, and
# CJK text never matches. When encoded runs make up more than ENCODED_RATIO_MAX of
# the non-whitespace characters, the text is encoded bytes (an archive, image or
# binary dumped as text), not document content. The rule is structural: it uses no
# ids, names, labels or lists.
ENCODED_RUN_MIN = 64
ENCODED_RATIO_MAX = 0.5
_ENCODED_RUN = re.compile(r"[A-Za-z0-9+/=_\-]{%d,}" % ENCODED_RUN_MIN)
_WHITESPACE = re.compile(r"\s+")

# Excerpt columns, first match wins. `text` is accepted only when no excerpt
# column is present, because the raw pool's text column repeated the excerpt.
_EXCERPT_KEYS = ("excerpt", "first_512_tokens", "first_512")
_TEXT_KEY = "text"


def content_text(row: dict) -> str | None:
    """Return document text, or None when the row has no content to label.

    Source-path tags and file paths are stripped first, so a path can never reach
    the teacher prompt, a training excerpt, or a human sheet.
    """
    for key in _EXCERPT_KEYS:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return _clean(value)
    if any(key in row for key in _EXCERPT_KEYS):
        return None
    value = row.get(_TEXT_KEY)
    if isinstance(value, str) and value.strip():
        return _clean(value)
    return None


def _clean(value: str) -> str | None:
    cleaned = strip_source_tags(value)
    return cleaned if cleaned.strip() else None


def model_user_text(row: dict) -> str | None:
    """The only string placed in the teacher user message."""
    return content_text(row)


def encoded_ratio(text: str) -> float:
    """Share of non-whitespace characters that sit inside encoded runs (0.0 to 1.0)."""
    if not isinstance(text, str):
        return 0.0
    visible = len(_WHITESPACE.sub("", text))
    if visible == 0:
        return 0.0
    encoded = sum(len(run) for run in _ENCODED_RUN.findall(text))
    return encoded / visible


def is_mostly_encoded(text: str) -> bool:
    """True when the text is mostly encoded bytes and so carries no document content."""
    return encoded_ratio(text) > ENCODED_RATIO_MAX
