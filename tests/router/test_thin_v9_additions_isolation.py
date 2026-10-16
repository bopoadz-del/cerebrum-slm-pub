"""thin-class v9 guards: public filled-document train rows (content-labelled by the v9 teacher, internal only).

Isolation against every sealed set (v5, v8, v9 re-tick, v9 CR addendum), against the thin-class sealed
candidates awaiting owner ticks, and against the CR sealed candidates. Zero sealed overlap in train and val.
"""

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlparse

from router.labels import confidence_floor, load_labels, load_sealed_config

THIN = "data/router/additions_thin_v9.jsonl"
CR = "data/router/additions_change_request_v9.jsonl"
THIN_SEALED = "data/router/thin_v9_sealed_candidates_isolation.json"
CR_SEALED = "data/router/cr_v9_sealed_candidates_isolation.json"
SEALED = (
    "data/eval/router_unseen/sealed_labels.jsonl",
    "data/eval/router_unseen/v8/sealed_labels.jsonl",
    "data/eval/router_unseen/v9_retick/v5/sealed_labels.jsonl",
    "data/eval/router_unseen/v9_retick/v8/sealed_labels.jsonl",
    "data/eval/router_unseen/v9_cr_addendum/sealed_labels.jsonl",
)


def _rows(root: Path, rel: str) -> list[dict]:
    path = root / rel
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sealed(root: Path) -> list[dict]:
    out: list[dict] = []
    for rel in SEALED:
        if (root / rel).is_file():
            out.extend(_rows(root, rel))
    return out


def _pending(root: Path, rel: str) -> list[dict]:
    return json.loads((root / rel).read_text(encoding="utf-8"))["candidates"]


def _grams(text: str, n: int = 12) -> set[str]:
    words = re.sub(r"\W+", " ", text.lower()).split()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def test_thin_rows_never_overlap_any_sealed_set(repo_root: Path) -> None:
    sealed = _sealed(repo_root)
    shas = {str(r.get("file_sha256", "")).casefold() for r in sealed} - {""}
    projects = {r.get("project") for r in sealed} - {None, ""}
    grams = set().union(*(_grams(r.get("excerpt") or "") for r in sealed))
    first300 = {(r.get("excerpt") or "")[:300] for r in sealed} - {""}
    for row in _rows(repo_root, THIN):
        assert row["file_sha256"].casefold() not in shas, row["doc_id"]
        assert row["project_id"] not in projects, row["doc_id"]
        assert not (_grams(row["excerpt"]) & grams), row["doc_id"]
        assert row["excerpt"][:300] not in first300, row["doc_id"]


def test_thin_rows_are_isolated_from_sealed_and_pending_sources(repo_root: Path) -> None:
    rows = _rows(repo_root, THIN)
    pending = _pending(repo_root, THIN_SEALED) + _pending(repo_root, CR_SEALED)
    sealed_domains = {_sha(urlparse(r["source_url"]).netloc.lower()) for r in _sealed(repo_root) if r.get("source_url")}
    pending_domains = {c["source_domain_sha256"] for c in pending}
    pending_shas = {c["file_sha256"] for c in pending} | {c["text_sha256"] for c in pending}
    for row in rows:
        assert row["source_domain_sha256"] not in sealed_domains, row["doc_id"]
        assert row["source_domain_sha256"] not in pending_domains, row["doc_id"]
        assert row["file_sha256"] not in pending_shas and row["doc_id"] not in pending_shas
        assert re.fullmatch(r"th9_p\d{2}", row["project_id"])
        assert row.get("provenance_scope") == "internal only"
        assert "source_url" not in row and "source_domain" not in row


def test_thin_pending_candidates_share_no_source_with_any_train_batch(repo_root: Path) -> None:
    pending = _pending(repo_root, THIN_SEALED)
    pending_domains = {c["source_domain_sha256"] for c in pending}
    train_domains = {r["source_domain_sha256"] for r in _rows(repo_root, THIN) + _rows(repo_root, CR)}
    assert not pending_domains & train_domains
    assert len({c["id"] for c in pending}) == len(pending)
    assert all(set(c) == {"id", "file_sha256", "text_sha256", "source_domain_sha256"} for c in pending)


def test_thin_rows_meet_the_floor_with_v9_labels(repo_root: Path) -> None:
    labels = load_labels(repo_root / "configs/router/labels.json")
    floor = confidence_floor(load_sealed_config(repo_root / "configs/router/sealed_eval.json"))
    rows = _rows(repo_root, THIN)
    assert all(r["label"] in labels and float(r["confidence"]) >= floor for r in rows)
    projects: dict[str, set[str]] = {}
    for r in rows:
        projects.setdefault(r["label"], set()).add(r["project_id"])
    assert max(len(p) for p in projects.values()) >= 10
    assert len({r["doc_id"] for r in rows}) == len(rows)


def test_train_and_val_hold_zero_sealed_or_pending_rows(repo_root: Path) -> None:
    sealed = _sealed(repo_root)
    shas = {str(r.get("file_sha256", "")).casefold() for r in sealed} - {""}
    pending = _pending(repo_root, THIN_SEALED) + _pending(repo_root, CR_SEALED)
    pending_shas = {c["file_sha256"] for c in pending} | {c["text_sha256"] for c in pending}
    for split in ("train", "val"):
        for row in _rows(repo_root, f"data/router/{split}.jsonl"):
            sha = str(row.get("file_sha256", "")).casefold()
            assert sha not in shas, (split, row.get("doc_id"))
            assert sha not in pending_shas and row.get("doc_id") not in pending_shas


def test_thin_sources_never_span_train_and_val(repo_root: Path) -> None:
    source = {r["doc_id"]: r["source_domain_sha256"] for r in _rows(repo_root, CR) + _rows(repo_root, THIN)}
    train = _rows(repo_root, "data/router/train.jsonl")
    val = _rows(repo_root, "data/router/val.jsonl")
    in_train = {source[r["doc_id"]] for r in train if r.get("doc_id") in source}
    in_val = {source[r["doc_id"]] for r in val if r.get("doc_id") in source}
    assert not in_train & in_val
    thin_ids = {r["doc_id"] for r in _rows(repo_root, THIN)}
    assert thin_ids <= {r.get("doc_id") for r in train + val}
