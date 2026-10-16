"""pr5 guards: pr2 owner-source additions (unmasked, internal only) and v9 change-request train rows.

Isolation against every sealed set (v5, v8, v9 re-tick) and against the v9 change-request sealed
candidates awaiting owner ticks. Zero sealed overlap in train and val.
"""

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlparse

import pytest

from router.labels import confidence_floor, load_labels, load_sealed_config

PR2 = "data/router/additions_owner_sources_pr2.jsonl"
CR = "data/router/additions_change_request_v9.jsonl"
CR_SEALED = "data/router/cr_v9_sealed_candidates_isolation.json"
SEALED = (
    "data/eval/router_unseen/sealed_labels.jsonl",
    "data/eval/router_unseen/v8/sealed_labels.jsonl",
    "data/eval/router_unseen/v9_retick/v5/sealed_labels.jsonl",
    "data/eval/router_unseen/v9_retick/v8/sealed_labels.jsonl",
)
MASK = "\u2588"


def _rows(root: Path, rel: str) -> list[dict]:
    path = root / rel
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sealed(root: Path) -> list[dict]:
    out: list[dict] = []
    for rel in SEALED:
        if (root / rel).is_file():
            out.extend(_rows(root, rel))
    return out


def _grams(text: str, n: int = 12) -> set[str]:
    words = re.sub(r"\W+", " ", text.lower()).split()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def test_pr2_additions_are_the_unmasked_originals(repo_root: Path) -> None:
    rows = _rows(repo_root, PR2)
    assert len(rows) == 219
    assert sum(r.get("provenance") == "template" for r in rows) == 23
    assert sum(r.get("label_source") == "v8_definition_rule" for r in rows) == 19
    assert all(MASK not in r["excerpt"] for r in rows)
    assert all(r.get("provenance_scope") == "internal only" for r in rows)
    assert all(re.fullmatch(r"pr2_p\d{2}", r["project_id"]) for r in rows)


def test_additions_never_overlap_any_sealed_set(repo_root: Path) -> None:
    sealed = _sealed(repo_root)
    shas = {str(r.get("file_sha256", "")).casefold() for r in sealed} - {""}
    projects = {r.get("project") for r in sealed} - {None, ""}
    grams = set().union(*(_grams(r.get("excerpt") or "") for r in sealed))
    for rel in (PR2, CR):
        for row in _rows(repo_root, rel):
            assert row["file_sha256"].casefold() not in shas, (rel, row["doc_id"])
            assert row["project_id"] not in projects, (rel, row["doc_id"])
            assert not (_grams(row["excerpt"]) & grams), (rel, row["doc_id"])


def test_change_request_rows_are_isolated_from_sealed_sources(repo_root: Path) -> None:
    rows = _rows(repo_root, CR)
    pending = json.loads((repo_root / CR_SEALED).read_text(encoding="utf-8"))["candidates"]
    sealed_domains = {_sha(urlparse(r["source_url"]).netloc.lower()) for r in _sealed(repo_root) if r.get("source_url")}
    pending_domains = {c["source_domain_sha256"] for c in pending}
    pending_shas = {c["file_sha256"] for c in pending} | {c["text_sha256"] for c in pending}
    for row in rows:
        assert row["source_domain_sha256"] not in sealed_domains, row["doc_id"]
        assert row["source_domain_sha256"] not in pending_domains, row["doc_id"]
        assert row["file_sha256"] not in pending_shas and row["doc_id"] not in pending_shas
        assert re.fullmatch(r"cr9_p\d{2}", row["project_id"])
        assert row.get("provenance_scope") == "internal only"


def test_change_request_rows_meet_the_floor_with_v9_labels(repo_root: Path) -> None:
    labels = load_labels(repo_root / "configs/router/labels.json")
    floor = confidence_floor(load_sealed_config(repo_root / "configs/router/sealed_eval.json"))
    rows = _rows(repo_root, CR)
    assert "change request" in labels
    assert all(r["label"] in labels and float(r["confidence"]) >= floor for r in rows)
    projects = {r["project_id"] for r in rows if r["label"] == "change request"}
    assert len(projects) >= 12


def test_train_and_val_hold_zero_sealed_rows(repo_root: Path) -> None:
    sealed = _sealed(repo_root)
    shas = {str(r.get("file_sha256", "")).casefold() for r in sealed} - {""}
    pending = json.loads((repo_root / CR_SEALED).read_text(encoding="utf-8"))["candidates"]
    pending_shas = {c["file_sha256"] for c in pending} | {c["text_sha256"] for c in pending}
    for split in ("train", "val"):
        for row in _rows(repo_root, f"data/router/{split}.jsonl"):
            sha = str(row.get("file_sha256", "")).casefold()
            assert sha not in shas, (split, row.get("doc_id"))
            assert sha not in pending_shas and row.get("doc_id") not in pending_shas


def test_templates_train_only_and_projects_not_split(repo_root: Path) -> None:
    pr2 = _rows(repo_root, PR2)
    cr = _rows(repo_root, CR)
    templates = {r["doc_id"] for r in pr2 if r.get("provenance") == "template"}
    val = _rows(repo_root, "data/router/val.jsonl")
    train = _rows(repo_root, "data/router/train.jsonl")
    assert not templates & {r.get("doc_id") for r in val}
    cr_project = {r["doc_id"]: r["project_id"] for r in cr}
    in_train = {cr_project[r["doc_id"]] for r in train if r.get("doc_id") in cr_project}
    in_val = {cr_project[r["doc_id"]] for r in val if r.get("doc_id") in cr_project}
    assert not in_train & in_val
