"""slm-ingest-1015 guards: owner Drive documents (ORDERS_1838Z_SLM), redacted, placed in train under the 10% cap.

Isolation against every sealed set under data/eval/router_unseen (found by glob), every held-out shard and the
pending sealed candidates; generic residue checks for contact details and identifiers; neutral source ids and
hashed project groups; placement once in train, never val; 10% per-class project cap for the added rows.
"""
import pytest

import hashlib
import json
import math
import re
import subprocess
from collections import Counter
from pathlib import Path

from router.labels import confidence_floor, load_labels, load_sealed_config
from router.strip_source_tags import row_has_path

ADDS = "data/router/additions_slm_ingest_1015.jsonl"
RESIDUE = "data/router/slm_ingest_1015_box_side_residue.json"
PENDING_GLOB = "*sealed_candidates_isolation.json"
_RESIDUE_FORBIDDEN = frozenset(
    {"excerpt", "text", "body", "content", "raw", "document", "quote", "name"}
)


def _rows(root: Path, rel: str) -> list[dict]:
    return [
        json.loads(line)
        for line in (root / rel).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sealed_and_held(root: Path) -> list[dict]:
    out: list[dict] = []
    paths = sorted(
        (root / "data" / "eval" / "router_unseen").rglob("sealed_labels*.jsonl")
    )
    paths += (
        sorted((root / "data" / "held_out").rglob("*.jsonl"))
        if (root / "data" / "held_out").is_dir()
        else []
    )
    for path in paths:
        out.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return out


def _grams(text: str, n: int = 12) -> set[str]:
    words = re.sub(r"\W+", " ", text.lower()).split()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def _label(row: dict) -> str:
    return row["messages"][-1]["content"]


@pytest.mark.private_data(
    'data/eval/router_unseen/sealed_labels.jsonl',
    'data/eval/router_unseen/v8/sealed_labels.jsonl',
    'data/held_out/pairs.jsonl',
    'data/router/thin_v9_sealed_candidates_isolation.json',
    'data/router/cr_v9_sealed_candidates_isolation.json',
    'data/eval/router_unseen/v9_cr_addendum/sealed_labels.jsonl',
)
def test_rows_never_overlap_any_sealed_or_held_set(repo_root: Path) -> None:
    sealed = _sealed_and_held(repo_root)
    assert sealed
    shas = {str(r.get("file_sha256", "")).casefold() for r in sealed} - {""}
    projects = (
        {r.get("project") for r in sealed} | {r.get("project_id") for r in sealed}
    ) - {None, ""}
    grams = set().union(*(_grams(r.get("excerpt") or "") for r in sealed))
    first300 = {(r.get("excerpt") or "")[:300] for r in sealed} - {""}
    for row in _rows(repo_root, ADDS):
        assert row["file_sha256"].casefold() not in shas, row["doc_id"]
        assert row["project_id"] not in projects, row["doc_id"]
        assert not (_grams(row["excerpt"]) & grams), row["doc_id"]
        assert row["excerpt"][:300] not in first300, row["doc_id"]


@pytest.mark.private_data('data/router/additions_slm_ingest_1015.jsonl')
def test_rows_are_isolated_from_pending_sealed_candidates(repo_root: Path) -> None:
    pending: list[dict] = []
    for path in sorted((repo_root / "data" / "router").glob(PENDING_GLOB)):
        pending.extend(
            json.loads(path.read_text(encoding="utf-8")).get("candidates", [])
        )
    shas = {c.get("file_sha256") for c in pending} | {
        c.get("text_sha256") for c in pending
    }
    for row in _rows(repo_root, ADDS):
        assert row["file_sha256"] not in shas and row["doc_id"] not in shas, row[
            "doc_id"
        ]


@pytest.mark.private_data('data/router/additions_slm_ingest_1015.jsonl')
def test_ingest_excerpts_leave_no_written_or_exact_amounts(repo_root: Path) -> None:
    """LANE2: amounts are magnitude tokens, not number-words or exact digit strings."""
    from router.amount_buckets import bucket_amounts

    for row in _rows(repo_root, ADDS):
        rewritten, replacements = bucket_amounts(row["excerpt"])
        assert replacements == 0 and rewritten == row["excerpt"], row["doc_id"]


@pytest.mark.private_data('data/router/additions_slm_ingest_1015.jsonl')
def test_rows_carry_no_contact_details_or_identifiers(repo_root: Path) -> None:
    residue = (
        re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),
        re.compile(r"(?i)https?://|www\."),
        re.compile(r"(?i)\bP\.?\s?O\.?\s?Box\s*:?\s*\d"),
        re.compile(r"\+?\b971[\s)\-]*\d[\d\s\-]{6,}"),
        re.compile(r"\bAE\d{2}\s?\d{4}"),
        re.compile(
            r"\b[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12}\b",
            re.IGNORECASE,
        ),
        # exact clock times (timestamps must be date-only)
        re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:[APap][Mm])?\b"),
        # exact grouped currency-like amounts (must be bucket tokens)
        re.compile(r"(?<![\d])\d{1,3}(?:,\d{3})+(?:\.\d{2})?(?![\d])"),
        # legal-entity suffix attached to a surviving capitalised name run
        re.compile(
            r"(?i)(?<![A-Za-z0-9])(?:[A-Z][A-Za-z'&.-]{1,}(?:\s+(?:&|and|[A-Z][A-Za-z'&.-]{1,})){0,5})"
            r"\s*,?\s*(?:L\.?L\.?C\.?|Ltd\.?|Limited|Inc\.?|PLC|P\.?J\.?S\.?C\.?|FZE|FZ[- ]?LLC|Co\.|Company)\b"
        ),
    )
    for row in _rows(repo_root, ADDS):
        for pattern in residue:
            assert not pattern.search(row["excerpt"]), (row["doc_id"], pattern.pattern)
        assert row["redaction_count"] > 0, row["doc_id"]
        assert "\u2588" in row["excerpt"], row["doc_id"]


@pytest.mark.private_data('data/router/additions_slm_ingest_1015.jsonl')
def test_neutral_ids_hashed_projects_v9_labels_and_floor(repo_root: Path) -> None:
    labels = load_labels(repo_root / "configs/router/labels.json")
    floor = confidence_floor(
        load_sealed_config(repo_root / "configs/router/sealed_eval.json")
    )
    rows = _rows(repo_root, ADDS)
    assert rows
    for row in rows:
        assert row["label"] in labels and float(row["confidence"]) >= floor
        assert row["provenance_scope"] == "internal only"
        assert row["provenance_status"] == "owner-drive"
        assert row["provenance"] == (
            row["source_id"] + " p" + ",".join(str(p) for p in row["pages"])
        )
        assert isinstance(row["pages"], list) and row["pages"]
        assert row["second_pass"] == "agree"
        assert re.fullmatch(r"slm_src_[0-9a-f]{12}", row["source_id"])
        assert row["source_id"] == "slm_src_" + row["file_sha256"][:12]
        assert re.fullmatch(r"[0-9a-f]{64}", row["project_group_hash"])
        assert re.fullmatch(r"pr2_p01|slmg_[0-9a-f]{12}", row["project_id"])
        if row["project_id"].startswith("slmg_"):
            assert row["project_id"] == "slmg_" + row["project_group_hash"][:12]
        assert not row_has_path(row)
    assert len({r["doc_id"] for r in rows}) == len(rows)
    earlier = set()
    for path in sorted((repo_root / "data" / "router").glob("additions_*.jsonl")):
        if path.name != Path(ADDS).name:
            earlier |= {
                r.get("doc_id")
                for r in _rows(repo_root, str(path.relative_to(repo_root)))
            }
    assert not {r["doc_id"] for r in rows} & earlier


@pytest.mark.private_data(
    'data/router/additions_slm_ingest_1015.jsonl',
    'data/router/train.jsonl',
    'data/router/val.jsonl',
)
def test_rows_are_in_train_once_never_val_with_the_same_text_and_label(
    repo_root: Path,
) -> None:
    train = _rows(repo_root, "data/router/train.jsonl")
    val = _rows(repo_root, "data/router/val.jsonl")
    by_id = Counter(r.get("doc_id") for r in train)
    train_rows = {r.get("doc_id"): r for r in train}
    val_ids = {r.get("doc_id") for r in val}
    val_shas = {r.get("file_sha256") for r in val}
    for row in _rows(repo_root, ADDS):
        assert by_id[row["doc_id"]] == 1, row["doc_id"]
        assert row["doc_id"] not in val_ids and row["file_sha256"] not in val_shas
        chat = train_rows[row["doc_id"]]
        assert _label(chat) == row["label"]
        assert chat["messages"][1]["content"] == row["excerpt"]
        assert chat["project_id"] == row["project_id"]


@pytest.mark.private_data(
    'data/router/additions_slm_ingest_1015.jsonl',
    'data/router/train.jsonl',
)
def test_added_rows_respect_the_ten_percent_project_cap(repo_root: Path) -> None:
    train = _rows(repo_root, "data/router/train.jsonl")
    per_label = Counter(_label(r) for r in train)
    per_project = Counter((r["project_id"], _label(r)) for r in train)
    for row in _rows(repo_root, ADDS):
        key = (row["project_id"], row["label"])
        assert per_project[key] <= math.floor(0.10 * per_label[row["label"]]), key


def _reject_residue_text(obj: object) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            assert key.casefold() not in _RESIDUE_FORBIDDEN
            _reject_residue_text(value)
    elif isinstance(obj, list):
        for item in obj:
            _reject_residue_text(item)
    elif isinstance(obj, str):
        assert len(obj) <= 160
        assert "\n" not in obj


@pytest.mark.private_data('data/router/slm_ingest_1015_box_side_residue.json')
def test_box_side_residue_table_covers_eleven_rows_without_excerpts(
    repo_root: Path,
) -> None:
    """The 11 box-side rows are not stored. The table is ids, counts, and labels."""
    table = json.loads((repo_root / RESIDUE).read_text(encoding="utf-8"))
    _reject_residue_text(table)
    counts = json.loads(
        (repo_root / "data" / "router" / "split_counts.json").read_text(
            encoding="utf-8"
        )
    )["slm_ingest_1015"]
    assert table["spec_rows"] == counts["rows"] == 23
    assert table["repo_rows"] == counts["rows_in_train"] == len(_rows(repo_root, ADDS))
    assert table["box_side_rows"] == counts["rows_to_dev_box_side"] == 11
    assert table["rows_in_val"] == 0
    assert table["excerpt_stored"] is False
    assert table["amount_rerun"] == "rerun_done"
    rows = table["rows"]
    assert len(rows) == 11
    assert all(row["rows"] == 1 and row["doc_id"] is None for row in rows)
    by_label = Counter(row["label"] for row in rows)
    assert dict(by_label) == {
        label: rec["dev"] for label, rec in counts["by_label"].items() if rec["dev"]
    }
    assert dict(by_label) == table["by_label"]
    by_tag = Counter(row["dev_tag"] for row in rows)
    assert dict(by_tag) == counts["dev_tags"] == table["dev_tags"]
    by_source = Counter(row["source_id"] for row in rows)
    dev_sources = {
        source
        for source, rec in counts["by_source"].items()
        if rec["rows_dev"]
    }
    assert set(by_source) == dev_sources
    for source, produced in by_source.items():
        rec = counts["by_source"][source]
        assert produced == rec["rows_dev"]
        assert {row["label"] for row in rows if row["source_id"] == source} <= set(
            rec["labels"]
        )
    assert sum(by_label.values()) == 11
    assert sum(by_tag.values()) == 11


def _git_blob(root: Path, rel: str) -> str:
    return subprocess.check_output(
        ["git", "hash-object", "--", rel],
        cwd=root,
        text=True,
    ).strip()


@pytest.mark.private_data(
    'data/router/slm_ingest_1015_box_side_residue.json',
    'data/router/additions_slm_ingest_1015.jsonl',
)
def test_box_side_amount_rerun_matches_redactor_and_row_sums(
    repo_root: Path,
) -> None:
    """Box-side rerun totals are sums over the rows. The redactor blob is this tree."""
    table = json.loads((repo_root / RESIDUE).read_text(encoding="utf-8"))
    _reject_residue_text(table)
    rows = table["rows"]
    pairs = [(row["source_id"], row["label"]) for row in rows]
    assert table["amount_rerun"] == "rerun_done"
    assert table["excerpt_stored"] is False
    assert len(rows) == table["box_side_rows"] == table["rows_rerun"]
    assert len(pairs) == len(set(pairs))
    redactor = table["redactor"]
    rel = redactor["path"]
    assert rel == "scripts/router/amount_buckets.py"
    blob = _git_blob(repo_root, rel)
    assert redactor["git_blob"] == blob
    assert re.fullmatch(r"[0-9a-f]{40}", redactor["ref"])
    recorded = subprocess.check_output(
        ["git", "rev-parse", f"{redactor['ref']}:{rel}"],
        cwd=repo_root,
        text=True,
    ).strip()
    assert recorded == blob
    file_sha = hashlib.sha256((repo_root / rel).read_bytes()).hexdigest()
    assert redactor["file_sha256"] == file_sha
    assert re.fullmatch(r"[0-9a-f]{64}", redactor["file_sha256"])
    changed = 0
    replacements = 0
    residue_before = 0
    residue_after = 0
    for row in rows:
        for field in ("doc_sha256", "row_sha256_before", "row_sha256_after"):
            assert re.fullmatch(r"[0-9a-f]{64}", row[field])
        assert row["residue_after_total"] == 0
        assert row["idempotent"] is True
        assert isinstance(row["residue_before"], dict)
        assert row["residue_before_total"] == sum(row["residue_before"].values())
        assert all(
            isinstance(name, str) and isinstance(count, int) and count >= 0
            for name, count in row["residue_before"].items()
        )
        replacements += row["replacements"]
        residue_before += row["residue_before_total"]
        residue_after += row["residue_after_total"]
        if row["replacements"]:
            changed += 1
            assert row["row_sha256_before"] != row["row_sha256_after"]
        else:
            assert row["replacements"] == 0
            assert row["row_sha256_before"] == row["row_sha256_after"]
    assert table["rows_changed"] == changed
    assert table["replacements"] == replacements
    assert table["residue_before_total"] == residue_before
    assert table["residue_after_total"] == residue_after
    assert table["second_pass_replacements"] == 0
    assert table["idempotent"] is True
    assert table["covered"] == (
        f"{table['repo_rows'] + table['box_side_rows']}/{table['spec_rows']}"
    )
    assert table["repo_rows"] + table["box_side_rows"] == table["spec_rows"]
    repo_ids = {row["source_id"] for row in _rows(repo_root, ADDS)}
    ambiguous = table["ambiguous"]
    assert isinstance(ambiguous, list) and ambiguous
    for item in ambiguous:
        assert item["where"] == "repo"
        assert item["source_id"] in repo_ids
        assert isinstance(item["count"], int) and item["count"] > 0
        assert isinstance(item["kind"], str) and item["kind"]
        assert item["source_id"] not in {row["source_id"] for row in rows}


@pytest.mark.private_data(
    'data/router/additions_slm_ingest_1015.jsonl',
    'data/router/split_counts.json',
    'data/router/train.jsonl',
)
def test_split_counts_record_matches(repo_root: Path) -> None:
    rows = _rows(repo_root, ADDS)
    counts = json.loads(
        (repo_root / "data" / "router" / "split_counts.json").read_text(
            encoding="utf-8"
        )
    )
    rec = counts["slm_ingest_1015"]
    assert rec["rows_in_train"] == len(rows)
    assert rec["rows_in_val"] == 0
    assert rec["sealed_touched"] is False
    assert rec["rows"] == rec["rows_in_train"] + rec["rows_to_dev_box_side"]
    assert sum(v["rows_train"] for v in rec["by_source"].values()) == len(rows)
    assert set(rec["by_source"]) >= {r["source_id"] for r in rows}
    train = _rows(repo_root, "data/router/train.jsonl")
    assert rec["new_split"]["train"] == len(train)
    assert rec["new_split"]["train_counts"] == dict(Counter(_label(r) for r in train))
