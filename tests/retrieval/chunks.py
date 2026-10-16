"""A chunk shaped like the object The Fork retriever returns."""

from __future__ import annotations


def fork_chunk(
    *,
    chunk_id: str,
    project_id: str,
    doc_id: str,
    text: str,
    filename: str | None = None,
) -> dict[str, str]:
    chunk = {
        "chunk_id": chunk_id,
        "project_id": project_id,
        "doc_id": doc_id,
        "text": text,
    }
    if filename is not None:
        chunk["filename"] = filename
    return chunk
