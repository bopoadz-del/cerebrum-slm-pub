"""thin-class v9b guards: VO/CO vs change-request contrastive pairs from public packages (content-labelled by
the v9 teacher, internal only).

Isolation against every sealed set (v5, v8, v9 re-tick, v9 CR addendum) and against the pending sealed
candidates; pair records are audit metadata that never reach the prompt or the training text.
"""
import pytest

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlparse

from router.labels import confidence_floor, load_labels, load_sealed_config

V9B = "data/router/additions_thin_v9b.jsonl"
THIN = "data/router/additions_thin_v9.jsonl"
CR = "data/router/additions_change_request_v9.jsonl"
PAIRS = "data/router/vo_cr_pairs_v9b.json"
PENDING = ("data/router/thin_v9_sealed_candidates_isolation.json", "data/router/cr_v9_sealed_candidates_isolation.json")
SEALED = (
    "data/eval/router_unseen/sealed_labels.jsonl",
    "data/eval/router_unseen/v8/sealed_labels.jsonl",
    "data/eval/router_unseen/v9_retick/v5/sealed_labels.jsonl",
    "data/eval/router_unseen/v9_retick/v8/sealed_labels.jsonl",
    "data/eval/router_unseen/v9_cr_addendum/sealed_labels.jsonl",
)


def _rows(root: Path, rel: str) -> list[dict]:
    return [json.loads(line) for line in (root / rel).read_text(encoding="utf-8").splitlines() if line.strip()]


def _sealed(root: Path) -> list[dict]:
    out: list[dict] = []
    for rel in SEALED:
        if (root / rel).is_file():
            out.extend(_rows(root, rel))
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


@pytest.mark.private_data('data/router/additions_thin_v9b.jsonl')
def test_v9b_rows_never_overlap_any_sealed_set(repo_root: Path) -> None:
    sealed = _sealed(repo_root)
    shas = {str(r.get("file_sha256", "")).casefold() for r in sealed} - {""}
    projects = {r.get("project") for r in sealed} - {None, ""}
    grams = set().union(*(_grams(r.get("excerpt") or "") for r in sealed))
    first300 = {(r.get("excerpt") or "")[:300] for r in sealed} - {""}
    for row in _rows(repo_root, V9B):
        assert row["file_sha256"].casefold() not in shas, row["doc_id"]
        assert row["project_id"] not in projects, row["doc_id"]
        assert not (_grams(row["excerpt"]) & grams), row["doc_id"]
        assert row["excerpt"][:300] not in first300, row["doc_id"]


@pytest.mark.private_data('data/router/additions_thin_v9b.jsonl')
def test_v9b_rows_are_isolated_from_sealed_and_pending_sources(repo_root: Path) -> None:
    pending = _pending(repo_root)
    sealed_domains = {_sha(urlparse(r["source_url"]).netloc.lower()) for r in _sealed(repo_root) if r.get("source_url")}
    pending_domains = {c["source_domain_sha256"] for c in pending}
    pending_shas = {c["file_sha256"] for c in pending} | {c["text_sha256"] for c in pending}
    for row in _rows(repo_root, V9B):
        assert row["source_domain_sha256"] not in sealed_domains, row["doc_id"]
        assert row["source_domain_sha256"] not in pending_domains, row["doc_id"]
        assert row["file_sha256"] not in pending_shas and row["doc_id"] not in pending_shas
        assert re.fullmatch(r"th9b?_p\d{2}", row["project_id"])
        assert row.get("provenance_scope") == "internal only"
        assert "source_url" not in row and "source_domain" not in row


@pytest.mark.private_data('data/router/additions_thin_v9b.jsonl')
def test_v9b_rows_meet_the_floor_with_v9_labels_and_are_new(repo_root: Path) -> None:
    labels = load_labels(repo_root / "configs/router/labels.json")
    floor = confidence_floor(load_sealed_config(repo_root / "configs/router/sealed_eval.json"))
    rows = _rows(repo_root, V9B)
    assert all(r["label"] in labels and float(r["confidence"]) >= floor for r in rows)
    earlier = {r["doc_id"] for r in _rows(repo_root, THIN) + _rows(repo_root, CR)}
    assert len({r["doc_id"] for r in rows}) == len(rows)
    assert not {r["doc_id"] for r in rows} & earlier


@pytest.mark.private_data(
    'data/router/additions_thin_v9b.jsonl',
    'data/router/vo_cr_pairs_v9b.json',
)
def test_pairs_join_an_issued_instrument_to_its_request(repo_root: Path) -> None:
    pairs = json.loads((repo_root / PAIRS).read_text(encoding="utf-8"))["pairs"]
    rows = {r["doc_id"]: r for r in _rows(repo_root, V9B) + _rows(repo_root, THIN)}
    assert len(pairs) >= 10 and len({p["pair_id"] for p in pairs}) == len(pairs)
    for p in pairs:
        inst = rows[p["instrument"]]
        assert inst["label"] == "VO/CO", p["pair_id"]
        assert p["requests"], p["pair_id"]
        for d in p["requests"]:
            req = rows[d]
            assert req["label"] == "change request", p["pair_id"]
            assert req["project_id"] == inst["project_id"] and req["source_domain_sha256"] == inst["source_domain_sha256"]
            assert req.get("pair_id") == p["pair_id"] and req.get("pair_role") == "request"


@pytest.mark.private_data(
    'data/router/vo_cr_pairs_v9b.json',
    'data/router/additions_thin_v9b.jsonl',
)
def test_pair_metadata_never_reaches_the_prompt_or_text(repo_root: Path) -> None:
    pair_ids = {p["pair_id"] for p in json.loads((repo_root / PAIRS).read_text(encoding="utf-8"))["pairs"]}
    for split in ("train", "val"):
        for row in _rows(repo_root, f"data/router/{split}.jsonl"):
            assert "pair_id" not in row and "pair_role" not in row
            for message in row["messages"]:
                assert not any(pid in message["content"] for pid in pair_ids)
    for row in _rows(repo_root, V9B):
        assert not any(pid in row["excerpt"] for pid in pair_ids)


@pytest.mark.private_data(
    'data/router/additions_thin_v9b.jsonl',
    'data/router/train.jsonl',
    'data/router/val.jsonl',
)
def test_train_and_val_hold_zero_sealed_or_pending_rows_and_sources_never_span(repo_root: Path) -> None:
    sealed = _sealed(repo_root)
    shas = {str(r.get("file_sha256", "")).casefold() for r in sealed} - {""}
    pending = _pending(repo_root)
    pending_shas = {c["file_sha256"] for c in pending} | {c["text_sha256"] for c in pending}
    train = _rows(repo_root, "data/router/train.jsonl")
    val = _rows(repo_root, "data/router/val.jsonl")
    for row in train + val:
        sha = str(row.get("file_sha256", "")).casefold()
        assert sha not in shas and sha not in pending_shas and row.get("doc_id") not in pending_shas
    source = {r["doc_id"]: r["source_domain_sha256"] for r in _rows(repo_root, CR) + _rows(repo_root, THIN) + _rows(repo_root, V9B)}
    in_train = {source[r["doc_id"]] for r in train if r.get("doc_id") in source}
    in_val = {source[r["doc_id"]] for r in val if r.get("doc_id") in source}
    assert not in_train & in_val
    # every v9b row reaches train/val, except "other" rows cut by the fixed-seed "other" train cap (counted in split_counts)
    placed = {r.get("doc_id") for r in train + val}
    v9b = _rows(repo_root, V9B)
    assert {r["doc_id"] for r in v9b if r["label"] != "other"} <= placed
    counts = json.loads((repo_root / "data" / "router" / "split_counts.json").read_text(encoding="utf-8"))["thin_v9b"]
    assert sum(r["doc_id"] not in placed for r in v9b) == counts.get("rows_not_in_split", 0)
