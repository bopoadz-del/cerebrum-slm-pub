"""Normalize a context attached by The Fork's retriever.

A chunk contributes the fields ``retrieve`` returns: ``chunk_id``,
``project_id``, ``doc_id``, ``text``, and ``filename``. Other keys are
ignored. ``filename`` is not evidence for a figure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

_CHUNK_ID = "chunk_id"
_PROJECT_ID = "project_id"
_DOC_ID = "doc_id"
_TEXT = "text"
_FILENAME = "filename"


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        dumped = dump()
        if isinstance(dumped, Mapping):
            return dumped
    return None


def _read(chunk: Any, name: str) -> Any:
    mapping = _as_mapping(chunk)
    if mapping is not None:
        return mapping.get(name)
    return getattr(chunk, name, None)


def _text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _chunks_of(raw: Any) -> list[Any]:
    if raw is None:
        return []
    mapping = _as_mapping(raw)
    if mapping is not None and "chunks" in mapping:
        chunks = mapping.get("chunks") or []
        return list(chunks)
    if isinstance(raw, (list, tuple)):
        return list(raw)
    return [raw]


@dataclass(frozen=True)
class AttachedContext:
    """Context attached by one call to The Fork's retriever."""

    chunks: tuple[Any, ...]
    source_ids: frozenset[str]
    project_ids: frozenset[str]
    support_text: str

    @property
    def present(self) -> bool:
        return len(self.chunks) > 0

    def text_contains(self, identifier: str) -> bool:
        if not identifier or not identifier.strip():
            return False
        return identifier in self.support_text

    def allows(self, kind: str, identifier: str) -> bool:
        """True when this id is present on a field ``retrieve`` returned."""

        if kind == "clause":
            return self.text_contains(identifier)
        if kind in {"citation", "source"}:
            return (
                identifier in self.source_ids
                or identifier in self.project_ids
                or self.text_contains(identifier)
            )
        return False

    def cites_figure(self, cited: Iterable[str]) -> bool:
        """A figure citation is a chunk id or a document id.

        ``project_id`` alone does not cite a figure.
        """

        return any(item in self.source_ids for item in cited)


def context_from_retrieve(raw: Any) -> AttachedContext:
    """Build attached context from whatever ``retrieve`` returned."""

    source_ids: set[str] = set()
    project_ids: set[str] = set()
    texts: list[str] = []
    chunks = tuple(_chunks_of(raw))
    for chunk in chunks:
        for name, sink in ((_CHUNK_ID, source_ids), (_DOC_ID, source_ids), (_PROJECT_ID, project_ids)):
            identifier = _text(_read(chunk, name))
            if identifier is not None:
                sink.add(identifier)
        body = _text(_read(chunk, _TEXT))
        if body is not None:
            texts.append(body)
        _read(chunk, _FILENAME)
    return AttachedContext(
        chunks=chunks,
        source_ids=frozenset(source_ids),
        project_ids=frozenset(project_ids),
        support_text="\n".join(texts),
    )


def context_from_chunks(chunks: Iterable[Any]) -> AttachedContext:
    return context_from_retrieve(list(chunks))
