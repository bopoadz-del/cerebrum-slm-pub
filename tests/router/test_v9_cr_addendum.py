"""pr9: the owner-ticked v9 change-request addendum is a new sealed file, isolated and wired into the v9 config.

No score is made here. Gold is compared with itself only to read the per-label n rule.
"""

import hashlib
import json
from collections import Counter
from pathlib import Path

from router.isolation import load_exclusions, rejected_rows
from router.labels import load_labels, load_sealed_config
from router.sealed_gate import MIN_SEALED_PER_LABEL, evaluate_sealed_gate_v8
from router.train_router import rows_from_pool

ADDENDUM_DIR = "data/eval/router_unseen/v9_cr_addendum"
CANDIDATES = "data/router/cr_v9_sealed_candidates_isolation.json"


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _v9(repo_root: Path) -> dict:
    return load_sealed_config(repo_root / "configs" / "router" / "sealed_eval.json")["v9_sealed"]


def _addendum(repo_root: Path) -> Path:
    """The v9 change-request addendum specifically (other addenda may exist)."""
    sets = [item for item in _v9(repo_root)["sets"] if item.get("addendum")]
    mine = [s for s in sets if s["sealed_labels"].endswith("v9_cr_addendum/sealed_labels.jsonl")]
    assert len(mine) == 1
    path = repo_root / mine[0]["sealed_labels"]
    assert path.parent == repo_root / ADDENDUM_DIR
    return path


def test_addendum_seal_matches_its_file(repo_root: Path) -> None:
    base = repo_root / ADDENDUM_DIR
    sums = []
    for name in ("sealed_labels.jsonl", "isolation.json"):
        raw = (base / name).read_bytes()
        assert b"\r\n" not in raw
        line = f"{hashlib.sha256(raw).hexdigest()}  {name}\n"
        assert (base / f"{name}.sha256").read_bytes() == line.encode()
        sums.append(line)
    assert (base / "SHA256SUMS").read_text(encoding="utf-8") == "".join(sums)


def test_addendum_rows_are_owner_ticks_on_preregistered_candidates(repo_root: Path) -> None:
    labels = load_labels(repo_root / "configs" / "router" / "labels.json")
    rows = _rows(_addendum(repo_root))
    assert len(rows) == 14
    assert all(row["label"] in labels for row in rows)
    assert len({row["label"] for row in rows}) == 1
    assert len({row["sheet_id"] for row in rows}) == len(rows)
    assert len({row["file_sha256"] for row in rows}) == len(rows)
    candidates = json.loads((repo_root / CANDIDATES).read_text(encoding="utf-8"))["candidates"]
    assert {row["file_sha256"] for row in rows} == {item["file_sha256"] for item in candidates}
    assert {row["sheet_id"] for row in rows} == {item["id"] for item in candidates}


def test_addendum_swaps_nothing_in_other_sealed_sets(repo_root: Path) -> None:
    addendum = _addendum(repo_root)
    shas = {row["file_sha256"] for row in _rows(addendum)}
    base = repo_root / "data" / "eval" / "router_unseen"
    others = [path for path in base.rglob("sealed_labels.jsonl") if path != addendum]
    assert len(others) >= 4
    for path in others:
        assert not shas & {row.get("file_sha256") for row in _rows(path)}, path
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert (path.parent / "sealed_labels.jsonl.sha256").read_text(encoding="utf-8") == f"{digest}  sealed_labels.jsonl\n"


def test_every_sealed_set_is_isolated_from_training_pools(repo_root: Path) -> None:
    """Generic: every sealed file under router_unseen against train, val and every pool file."""
    base = repo_root / "data" / "eval" / "router_unseen"
    pools = sorted((repo_root / "data" / "router").glob("*.jsonl"))
    assert any(path.name == "train.jsonl" for path in pools) and any(path.name == "val.jsonl" for path in pools)
    loaded = {path.name: rows_from_pool(path) for path in pools}
    for sealed in sorted(base.rglob("sealed_labels.jsonl")):
        exclusions = load_exclusions(sealed)
        for name, rows in loaded.items():
            assert not rejected_rows(rows, exclusions), (sealed, name)


def test_v9_config_lists_the_sets_and_counts_match_the_files(repo_root: Path) -> None:
    v9 = _v9(repo_root)
    labels = load_labels(repo_root / v9["labels_file"])
    combined: Counter = Counter()
    for item in v9["sets"]:
        rows = _rows(repo_root / item["sealed_labels"])
        assert len(rows) == item["rows"]
        combined.update(row["label"] for row in rows)
    assert v9["combined"]["rows"] == sum(combined.values())
    assert v9["combined"]["per_label_n"] == {label: combined.get(label, 0) for label in labels}
    assert v9["min_sealed_per_label"] == MIN_SEALED_PER_LABEL
    assert v9["score_once_per_adapter"] is True


def test_addendum_makes_its_label_passable_in_the_combined_set(repo_root: Path) -> None:
    v9 = _v9(repo_root)
    labels = load_labels(repo_root / v9["labels_file"])
    added = {row["label"] for row in _rows(_addendum(repo_root))}
    gold: list[str] = []
    for item in v9["sets"]:
        gold.extend(row["label"] for row in _rows(repo_root / item["sealed_labels"]))
    without = [label for item in v9["sets"] if not item.get("addendum") for label in (row["label"] for row in _rows(repo_root / item["sealed_labels"]))]
    result = {row["label"]: row for row in evaluate_sealed_gate_v8(gold, list(gold), labels, majority_class_fail=True)["per_label"]}
    before = {row["label"]: row for row in evaluate_sealed_gate_v8(without, list(without), labels, majority_class_fail=True)["per_label"]}
    for label in added:
        assert result[label]["passable"] is True
        assert before[label]["passable"] is False
    for label, row in result.items():
        assert row["passable"] is (row["n"] >= MIN_SEALED_PER_LABEL)
