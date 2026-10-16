"""The owner-ordered v9 web_g addendum: a new sealed file, held rows recorded and kept out, provenance on every sealed
row, the per-(project, class) cap, the exhaustion record recomputed from data, and the gap table. No score is made here."""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from router.labels import load_labels, load_sealed_config
from router.seal_addendum import exhausted_pairs, pair_counts
from router.sealed_gate import MIN_SEALED_PER_LABEL
from router.strip_source_tags import row_has_path

ADDENDUM_DIR = "data/eval/router_unseen/v9_web_g_addendum"
CANDIDATES = "data/router/web_g_sealed_candidates_isolation.json"
HEX = set("0123456789abcdef")


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _v9(root: Path) -> dict:
    return load_sealed_config(root / "configs" / "router" / "sealed_eval.json")["v9_sealed"]


def _mine(root: Path) -> dict:
    items = [s for s in _v9(root)["sets"] if s["sealed_labels"] == f"{ADDENDUM_DIR}/sealed_labels.jsonl"]
    assert len(items) == 1 and items[0].get("addendum") is True
    return items[0]


def _registered(root: Path) -> list[dict]:
    out: list[dict] = []
    for item in _v9(root)["sets"]:
        out.extend(_rows(root / item["sealed_labels"]))
    return out


def test_every_file_in_the_addendum_matches_its_seal(repo_root: Path) -> None:
    base = repo_root / ADDENDUM_DIR
    listed = (base / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    names = [line.split("  ", 1)[1] for line in listed]
    assert {"sealed_labels.jsonl", "held.jsonl", "isolation.json", "gap_table.json"} <= set(names)
    for line, name in zip(listed, names):
        raw = (base / name).read_bytes()
        assert b"\r\n" not in raw
        assert line == f"{hashlib.sha256(raw).hexdigest()}  {name}"
        assert (base / f"{name}.sha256").read_text(encoding="utf-8") == line + "\n"


def test_sealed_rows_are_owner_ticks_with_url_and_sha_provenance(repo_root: Path) -> None:
    labels = load_labels(repo_root / "configs" / "router" / "labels.json")
    rows = _rows(repo_root / _mine(repo_root)["sealed_labels"])
    assert len(rows) == _mine(repo_root)["rows"] == 7
    for row in rows:
        assert row["label"] in labels
        assert row["project"] == "web_public"
        assert row["source_url"].startswith(("https://", "http://"))
        assert len(row["file_sha256"]) == 64 and set(row["file_sha256"]) <= HEX
        assert not row_has_path(row)
    assert len({r["sheet_id"] for r in rows}) == len(rows)
    assert len({r["file_sha256"] for r in rows}) == len(rows)


def test_held_rows_are_recorded_and_never_sealed_or_trained(repo_root: Path) -> None:
    base = repo_root / ADDENDUM_DIR
    held = _rows(base / "held.jsonl")
    assert len(held) == 2 and all(r.get("held_reason") for r in held)
    held_ids = {r["sheet_id"] for r in held}
    held_shas = {r["file_sha256"] for r in held if r.get("file_sha256")}
    for path in (repo_root / "data" / "eval" / "router_unseen").rglob("sealed_labels*.jsonl"):
        rows = _rows(path)
        assert not held_ids & {r.get("sheet_id") for r in rows}, path
        assert not held_shas & {r.get("file_sha256") for r in rows}, path
    for path in sorted((repo_root / "data" / "router").glob("*.jsonl")):
        for row in _rows(path):
            keys = {str(row.get(k) or "") for k in ("file_sha256", "content_sha256", "doc_id")}
            assert not keys & held_shas, path.name


def test_candidate_hashes_cover_sealed_and_held_and_stay_out_of_training(repo_root: Path) -> None:
    candidates = json.loads((repo_root / CANDIDATES).read_text(encoding="utf-8"))["candidates"]
    sealed = _rows(repo_root / ADDENDUM_DIR / "sealed_labels.jsonl")
    held = _rows(repo_root / ADDENDUM_DIR / "held.jsonl")
    assert {c["id"] for c in candidates if c["status"] == "sealed"} == {r["sheet_id"] for r in sealed}
    assert {c["id"] for c in candidates if c["status"] == "held"} == {r["sheet_id"] for r in held}
    assert {c["file_sha256"] for c in candidates if c["status"] == "sealed"} == {r["file_sha256"] for r in sealed}
    shas = {c["file_sha256"] for c in candidates} | {c["text_sha256"] for c in candidates} | {c["source_sha256"] for c in candidates}
    for split in ("train", "val"):
        for row in _rows(repo_root / "data" / "router" / f"{split}.jsonl"):
            assert str(row.get("file_sha256") or "") not in shas and str(row.get("doc_id") or "") not in shas


def test_one_source_document_is_sealed_at_most_once(repo_root: Path) -> None:
    candidates = json.loads((repo_root / CANDIDATES).read_text(encoding="utf-8"))["candidates"]
    groups: dict[str, list[str]] = defaultdict(list)
    for cand in candidates:
        groups[cand["source_sha256"]].append(cand["status"])
    assert any(len(statuses) > 1 for statuses in groups.values())
    for statuses in groups.values():
        assert statuses.count("sealed") <= 1


def test_project_cap_holds_for_every_pair_this_addendum_touches(repo_root: Path) -> None:
    mine = _rows(repo_root / ADDENDUM_DIR / "sealed_labels.jsonl")
    combined = pair_counts(_registered(repo_root))
    for key in pair_counts(mine):
        assert combined[key] <= 3, key


def test_exhaustion_record_is_recomputed_from_data(repo_root: Path) -> None:
    v9 = _v9(repo_root)
    record = json.loads((repo_root / v9["cap_exhaustion"]).read_text(encoding="utf-8"))
    assert record["limit"] == 3
    assert record["pairs"] == exhausted_pairs(_registered(repo_root), record["limit"])


def test_no_document_of_this_addendum_is_sealed_elsewhere(repo_root: Path) -> None:
    mine = repo_root / ADDENDUM_DIR / "sealed_labels.jsonl"
    shas = {r["file_sha256"] for r in _rows(mine)}
    keys = {r["isolation_key"] for r in _rows(mine)}
    others = [p for p in (repo_root / "data" / "eval" / "router_unseen").rglob("sealed_labels.jsonl") if p != mine]
    assert len(others) >= 8
    for path in others:
        rows = _rows(path)
        assert not shas & {r.get("file_sha256") for r in rows}, path
        assert not keys & {r.get("isolation_key") for r in rows}, path
    iso = json.loads((repo_root / ADDENDUM_DIR / "isolation.json").read_text(encoding="utf-8"))
    assert iso["cross_sealed_duplicates_added_by_this_addendum"] == 0
    assert all(v == [v[0], 0] for v in iso["router_isolation_rejected"].values())


def test_gap_table_matches_the_registered_sets(repo_root: Path) -> None:
    v9 = _v9(repo_root)
    labels = load_labels(repo_root / v9["labels_file"])
    gap = json.loads((repo_root / ADDENDUM_DIR / "gap_table.json").read_text(encoding="utf-8"))
    assert gap["threshold"] == MIN_SEALED_PER_LABEL
    mine = f"{ADDENDUM_DIR}/sealed_labels.jsonl"
    before: Counter = Counter()
    seen = False
    for item in v9["sets"]:
        rows = _rows(repo_root / item["sealed_labels"])
        if item["sealed_labels"] == mine:
            seen = True
            break
        before.update(r["label"] for r in rows)
    assert seen
    assert gap["combined_before"] == sum(before.values()) == 312
    assert gap["combined_after"] == v9["combined"]["rows"] == sum(before.values()) + _mine(repo_root)["rows"]
    assert sum(item["rows"] for item in v9["sets"]) == v9["combined"]["rows"]
    assert [row["label"] for row in gap["per_label"]] == labels
    for row in gap["per_label"]:
        assert row["before"] == before.get(row["label"], 0)
        assert row["after"] == v9["combined"]["per_label_n"][row["label"]] == row["before"] + row["added"]
        assert row["passable_before"] is (row["before"] >= MIN_SEALED_PER_LABEL)
        assert row["passable_after"] is (row["after"] >= MIN_SEALED_PER_LABEL)
