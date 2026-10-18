"""thin-v9f guards: claim rows from public filled documents (content-labelled by the v9 teacher, internal only).

Isolation against EVERY sealed set under data/eval/router_unseen (found by glob, so a new addendum is covered
without editing this file) and against the pending sealed candidates kept in the repo; collection provenance;
split placement.
"""
import pytest

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlparse

from router.collection_provenance import validate_file
from router.labels import confidence_floor, load_labels, load_sealed_config
from router.strip_source_tags import row_has_path

V9F = "data/router/additions_thin_v9f.jsonl"
EARLIER = (
    "data/router/additions_thin_v9.jsonl",
    "data/router/additions_thin_v9b.jsonl",
    "data/router/additions_change_request_v9.jsonl",
)
PENDING = ("data/router/thin_v9_sealed_candidates_isolation.json", "data/router/cr_v9_sealed_candidates_isolation.json")


def _rows(root: Path, rel: str) -> list[dict]:
    return [json.loads(line) for line in (root / rel).read_text(encoding="utf-8").splitlines() if line.strip()]


def _sealed(root: Path) -> list[dict]:
    out: list[dict] = []
    for path in sorted((root / "data" / "eval" / "router_unseen").rglob("sealed_labels*.jsonl")):
        out.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return out


def _pending(root: Path) -> list[dict]:
    out: list[dict] = []
    for rel in PENDING:
        if (root / rel).is_file():
            out.extend(json.loads((root / rel).read_text(encoding="utf-8"))["candidates"])
    return out


def _grams(text: str, n: int = 12) -> set[str]:
    words = re.sub(r"\W+", " ", text.lower()).split()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


@pytest.mark.private_data(
    'data/eval/router_unseen/sealed_labels.jsonl',
    'data/eval/router_unseen/v8/sealed_labels.jsonl',
    'data/router/thin_v9_sealed_candidates_isolation.json',
    'data/router/cr_v9_sealed_candidates_isolation.json',
    'data/eval/router_unseen/v9_cr_addendum/sealed_labels.jsonl',
)
def test_v9f_rows_never_overlap_any_sealed_set(repo_root: Path) -> None:
    sealed = _sealed(repo_root)
    assert len(sealed) > 0
    shas = {str(r.get("file_sha256", "")).casefold() for r in sealed} - {""}
    projects = {r.get("project") for r in sealed} - {None, ""}
    grams = set().union(*(_grams(r.get("excerpt") or "") for r in sealed))
    first300 = {(r.get("excerpt") or "")[:300] for r in sealed} - {""}
    domains = {urlparse(r["source_url"]).netloc.lower() for r in sealed if r.get("source_url")}
    for row in _rows(repo_root, V9F):
        assert row["file_sha256"].casefold() not in shas, row["doc_id"]
        assert row["content_sha256"].casefold() not in shas, row["doc_id"]
        assert row["project_id"] not in projects, row["doc_id"]
        assert not (_grams(row["excerpt"]) & grams), row["doc_id"]
        assert row["excerpt"][:300] not in first300, row["doc_id"]
        assert urlparse(row["source_url"]).netloc.lower() not in domains, row["doc_id"]


@pytest.mark.private_data('data/router/additions_thin_v9f.jsonl')
def test_v9f_rows_are_isolated_from_pending_sources(repo_root: Path) -> None:
    pending = _pending(repo_root)
    pending_domains = {c["source_domain_sha256"] for c in pending}
    pending_shas = {c["file_sha256"] for c in pending} | {c["text_sha256"] for c in pending}
    for row in _rows(repo_root, V9F):
        assert row["source_domain_sha256"] not in pending_domains, row["doc_id"]
        assert row["file_sha256"] not in pending_shas and row["doc_id"] not in pending_shas
        assert row["source_domain_sha256"] == _sha(urlparse(row["source_url"]).netloc)


@pytest.mark.private_data('data/router/additions_thin_v9f.jsonl')
def test_v9f_rows_meet_the_floor_with_v9_labels_provenance_and_are_new(repo_root: Path) -> None:
    labels = load_labels(repo_root / "configs/router/labels.json")
    floor = confidence_floor(load_sealed_config(repo_root / "configs/router/sealed_eval.json"))
    rows = _rows(repo_root, V9F)
    assert rows
    assert all(r["label"] in labels and float(r["confidence"]) >= floor for r in rows)
    assert all(r.get("provenance_scope") == "internal only" for r in rows)
    assert all(re.fullmatch(r"th9[bdf]?_p\d{2}", r["project_id"]) for r in rows)
    assert not any(row_has_path(r) for r in rows)
    assert validate_file(repo_root / V9F).ok
    earlier = {r["doc_id"] for rel in EARLIER for r in _rows(repo_root, rel)}
    assert len({r["doc_id"] for r in rows}) == len(rows)
    assert not {r["doc_id"] for r in rows} & earlier


@pytest.mark.private_data(
    'data/router/additions_thin_v9f.jsonl',
    'data/router/train.jsonl',
    'data/router/val.jsonl',
)
def test_v9f_rows_are_placed_once_and_sources_never_span(repo_root: Path) -> None:
    train = _rows(repo_root, "data/router/train.jsonl")
    val = _rows(repo_root, "data/router/val.jsonl")
    train_ids = {r.get("doc_id") for r in train}
    val_ids = {r.get("doc_id") for r in val}
    rows = _rows(repo_root, V9F)
    for row in rows:
        assert (row["doc_id"] in train_ids) != (row["doc_id"] in val_ids), row["doc_id"]
    source = {r["doc_id"]: r["source_domain_sha256"] for rel in EARLIER for r in _rows(repo_root, rel)}
    source.update({r["doc_id"]: r["source_domain_sha256"] for r in rows})
    in_train = {source[d] for d in train_ids if d in source}
    in_val = {source[d] for d in val_ids if d in source}
    assert not in_train & in_val
    counts = json.loads((repo_root / "data" / "router" / "split_counts.json").read_text(encoding="utf-8"))["thin_v9f"]
    assert counts["rows"] == len(rows)
    assert counts["rows_in_train"] + counts["rows_in_val"] == len(rows)


@pytest.mark.private_data(
    'data/router/additions_thin_v9f.jsonl',
    'data/router/train.jsonl',
    'data/router/val.jsonl',
)
def test_train_and_val_hold_zero_sealed_rows(repo_root: Path) -> None:
    sealed = _sealed(repo_root)
    shas = {str(r.get("file_sha256", "")).casefold() for r in sealed} - {""}
    for split in ("train", "val"):
        for row in _rows(repo_root, f"data/router/{split}.jsonl"):
            assert str(row.get("file_sha256", "")).casefold() not in shas
