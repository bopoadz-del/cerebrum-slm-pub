"""Router v7 pairs: contrastive (Is-not neighbours), combined-form (field state) and catch-all rows.

Every pair is generated from the v9 definitions file and the merged training rows. The committed pair files must equal a
regeneration, every pair must point at a training row with the same label, and nothing may overlap a sealed set.
"""
import pytest

import hashlib
import inspect
import json
import re
from collections import Counter
from pathlib import Path

import router.v7_pairs as v7_pairs
from collect.content_gate import gate
from router.isolation import load_exclusions, require_isolated
from router.labels import load_label_set, load_labels, load_sealed_config
from router.teacher_label import prompt_for
from router.v7_pairs import (
    PAIR_FILES,
    SUMMARY_FILE,
    AliasMatcher,
    build_pairs,
    class_budget,
    field_state,
    generate,
    is_not_neighbours,
    label_aliases,
    load_rows,
    load_rules,
    materialise,
    v7_training_examples,
)


def _setup(root: Path):
    rules = load_rules(root)
    label_set = load_label_set(root / rules["label_set"])
    labels = load_labels(root / "configs" / "router" / "labels.json")
    train = load_rows(root / "data" / "router" / "train.jsonl")
    return rules, label_set, labels, train


def _refs(root: Path) -> dict[str, list[dict]]:
    return {key: load_rows(root / rel) for key, rel in PAIR_FILES.items()}


@pytest.mark.private_data(
    'data/router/train.jsonl',
    'data/router/v7/pairs_contrastive.jsonl',
    'data/router/v7/pairs_combined_form.jsonl',
    'data/router/v7/pairs_catch_all.jsonl',
    'data/router/v7/pairs_summary.json',
)
def test_committed_pair_files_equal_a_regeneration(repo_root: Path) -> None:
    pairs, summary = generate(repo_root)
    assert _refs(repo_root) == json.loads(json.dumps(pairs))
    assert json.loads((repo_root / SUMMARY_FILE).read_text(encoding="utf-8")) == json.loads(json.dumps(summary))
    assert v7_pairs.main(["--root", str(repo_root), "--check"]) == 0


@pytest.mark.private_data('data/router/train.jsonl')
def test_no_label_project_or_document_string_in_the_generator(repo_root: Path) -> None:
    labels = load_labels(repo_root / "configs" / "router" / "labels.json")
    source = inspect.getsource(v7_pairs)
    for label in labels:
        assert f'"{label}"' not in source and f"'{label}'" not in source, label
    train = load_rows(repo_root / "data" / "router" / "train.jsonl")
    for project in {row.get("project_id") for row in train} - {None}:
        assert project not in source
    for name in ("filename", "file_name", "folder", "source_url", "path"):
        assert f'"{name}"' not in source and f"'{name}'" not in source


def test_definitions_are_the_v9_file_verbatim(repo_root: Path) -> None:
    md = (repo_root / "docs" / "router" / "LABEL_SET_v9.md").read_text(encoding="utf-8")
    label_set = load_label_set(repo_root / "configs" / "router" / "label_set_v9.json")
    for item in label_set["labels"]:
        assert f"Is not: {item['is_not']}" in md, item["label"]
        assert f"Definition: {item['definition']}" in md, item["label"]


@pytest.mark.private_data('data/router/train.jsonl')
def test_neighbours_are_named_in_an_is_not_line_and_symmetric(repo_root: Path) -> None:
    _, label_set, labels, _ = _setup(repo_root)
    neighbours = is_not_neighbours(label_set)
    matcher = AliasMatcher(label_aliases(label_set))
    is_not = {item["label"]: item["is_not"] for item in label_set["labels"]}
    for label, hood in neighbours.items():
        for other in hood:
            assert label in neighbours[other]
            assert other in matcher.labels_in(is_not[label]) or label in matcher.labels_in(is_not[other])


def test_neighbours_follow_the_definitions_not_the_names() -> None:
    def label_set(names):
        a, b, c = names
        return {"labels": [
            {"label": a, "definition": "d", "is_not": f"A {b} document.", "regional_names_text": ""},
            {"label": b, "definition": "d", "is_not": "Something else.", "regional_names_text": "alpha form, beta sheet."},
            {"label": c, "definition": "d", "is_not": "An alpha form.", "regional_names_text": ""},
        ]}
    one = is_not_neighbours(label_set(["x1", "y1", "z1"]))
    two = is_not_neighbours(label_set(["qq", "rr", "ss"]))
    assert one == {"x1": ["y1"], "y1": ["x1", "z1"], "z1": ["y1"]}
    assert two == {"qq": ["rr"], "rr": ["qq", "ss"], "ss": ["rr"]}


@pytest.mark.private_data(
    'data/router/train.jsonl',
    'data/router/val.jsonl',
    'data/router/v7/pairs_contrastive.jsonl',
    'data/router/v7/pairs_combined_form.jsonl',
    'data/router/v7/pairs_catch_all.jsonl',
)
def test_every_pair_points_at_a_training_row_with_its_label(repo_root: Path) -> None:
    rules, label_set, labels, train = _setup(repo_root)
    by_id = {row["doc_id"]: row for row in train}
    val_ids = {row["doc_id"] for row in load_rows(repo_root / "data" / "router" / "val.jsonl")}
    for key, refs in _refs(repo_root).items():
        for ref in refs:
            row = by_id[ref["doc_id"]]
            assert ref["doc_id"] not in val_ids
            assert row["messages"][-1]["content"] == ref["label"]
            assert ref["label"] in ref["menu"] and set(ref["menu"]) <= set(labels)
            assert gate(row["messages"][1]["content"])[0]
    materialised = materialise([r for refs in _refs(repo_root).values() for r in refs], train, label_set)
    for item in materialised:
        menu = [line[2:] for line in item["messages"][0]["content"].splitlines() if line.startswith("- ") and line[2:] in labels]
        assert item["messages"][0]["content"] == prompt_for(menu, label_set=label_set)
        assert item["messages"][1]["content"] == by_id[item["doc_id"]]["messages"][1]["content"]


@pytest.mark.private_data(
    'data/router/train.jsonl',
    'data/router/v7/pairs_contrastive.jsonl',
)
def test_contrastive_rows_respect_the_budget_and_the_combined_form_edge(repo_root: Path) -> None:
    rules, label_set, _, train = _setup(repo_root)
    budget = class_budget(train)
    neighbours = is_not_neighbours(label_set)
    refs = _refs(repo_root)["contrastive"]
    per_label = Counter(ref["label"] for ref in refs)
    assert max(per_label.values()) <= budget
    governed = set(rules["combined_form"]["labels"].values())
    for ref in refs:
        assert ref["neighbour"] in neighbours[ref["label"]]
        assert set(ref["menu"]) == {ref["label"], ref["neighbour"]}
        assert set(ref["menu"]) != governed
    assert len({ref["doc_id"] for ref in refs}) == len(refs)  # one contrastive row per document


def test_field_state_reads_content_only() -> None:
    rules = {"certified_cue": "certified", "certifier_cues": ["certif", "engineer"], "signature_cues": ["sign", "date"],
             "window_chars": 160, "value_window_chars": 60}
    params = inspect.signature(field_state).parameters
    assert list(params) == ["text", "rules"]
    filled = "Work done 12,500.00. Amount certified 11,875.00. Engineer: J. Smith signature date 03/04/2025"
    blank = "Work done 12,500.00. Amount certified ______. Engineer signature ______ date ______"
    neither = "Minutes of the meeting held on site."
    assert field_state(filled, rules)["state"] == "certificate"
    assert field_state(blank, rules)["state"] == "application"
    assert field_state(neither, rules)["state"] == "indeterminate"
    zero = "Work done 12,500.00. Amount certified 0.00. Engineer signature date 03/04/2025"
    assert field_state(zero, rules)["certified_amount"] is False


@pytest.mark.private_data('data/router/train.jsonl')
def test_combined_form_pairs_need_both_sides_in_one_family(repo_root: Path) -> None:
    rules, label_set, labels, _ = _setup(repo_root)
    app, cert = rules["combined_form"]["labels"]["application"], rules["combined_form"]["labels"]["certificate"]
    pad = "Interim statement for the period with the measured quantities listed below for review. " * 3

    def row(i, label, text, family):
        return {"doc_id": f"d{i}", "file_sha256": f"s{i}", "project_id": family,
                "messages": [{"role": "system", "content": "x"}, {"role": "user", "content": pad + text}, {"role": "assistant", "content": label}]}
    filled = "Work done 12,500.00. Amount certified 11,875.00. Engineer signature date 03/04/2025."
    blank = "Work done 12,500.00. Amount certified ____. Engineer signature ____ date ____."
    rows = [row(1, app, blank, "f1"), row(2, cert, filled, "f1"), row(3, app, blank, "f2"), row(4, cert, blank, "f1")]
    pairs, summary = build_pairs(rows, label_set, labels, rules)
    combined = pairs["combined_form"]
    assert sorted(r["doc_id"] for r in combined) == ["d1", "d2"]
    assert summary["combined_form"]["qualifying"] == {"application": 2, "certificate": 1}
    assert len({r["pair_id"] for r in combined}) == 1


@pytest.mark.private_data(
    'data/router/v7/pairs_summary.json',
    'data/router/v7/pairs_combined_form.jsonl',
)
def test_combined_form_counts_are_reported(repo_root: Path) -> None:
    summary = json.loads((repo_root / SUMMARY_FILE).read_text(encoding="utf-8"))["combined_form"]
    assert set(summary["qualifying"]) == {"application", "certificate"}
    assert summary["rows"] == len(_refs(repo_root)["combined_form"])
    assert summary["rows"] == 2 * sum(f["pairs"] for f in summary["families_with_both_sides"])


@pytest.mark.private_data('data/router/train.jsonl')
def test_combined_form_cues_come_from_the_definitions(repo_root: Path) -> None:
    rules, label_set, _, _ = _setup(repo_root)
    cf = rules["combined_form"]
    text = " ".join(
        " ".join([item["definition"], item["is_not"], *item.get("notes", [])])
        for item in label_set["labels"] if item["label"] in set(cf["labels"].values())
    ).casefold() + " " + cf["owner_rule"].casefold()
    for cue in [cf["certified_cue"], *cf["certifier_cues"], *cf["signature_cues"]]:
        assert cue.casefold() in text, cue


@pytest.mark.private_data(
    'data/router/train.jsonl',
    'data/router/v7/pairs_catch_all.jsonl',
)
def test_catch_all_rows_fail_every_instrument_test(repo_root: Path) -> None:
    rules, label_set, labels, train = _setup(repo_root)
    ca = rules["catch_all"]
    budget = class_budget(train)
    instruments = [label for label in labels if label not in ca["labels"]]
    matcher = AliasMatcher(label_aliases(label_set))
    by_id = {row["doc_id"]: row for row in train}
    refs = _refs(repo_root)["catch_all"]
    assert refs
    for ref in refs:
        assert ref["label"] in ca["labels"]
        assert set(ref["menu"]) == {ref["label"], *instruments}
        head = by_id[ref["doc_id"]]["messages"][1]["content"][: ca["head_chars"]]
        assert not matcher.labels_in(head, set(instruments))
    assert max(Counter(r["label"] for r in refs).values()) <= budget


def _grams(text: str, n: int = 12) -> set[str]:
    words = re.sub(r"\W+", " ", text.lower()).split()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


@pytest.mark.private_data(
    'data/eval/router_unseen/sealed_labels.jsonl',
    'data/eval/router_unseen/v8/sealed_labels.jsonl',
    'data/eval/router_unseen/v9_cr_addendum/sealed_labels.jsonl',
)
def test_v7_examples_never_overlap_any_sealed_set(repo_root: Path) -> None:
    examples, refs = v7_training_examples(repo_root)
    assert len(examples) == len(load_rows(repo_root / "data" / "router" / "train.jsonl")) + len(refs)
    cfg = load_sealed_config(repo_root / "configs" / "router" / "sealed_eval.json")
    files = [s["sealed_labels"] for s in cfg["v9_sealed"]["sets"]] + [cfg["sealed_labels"], "data/eval/router_unseen/v8/sealed_labels.jsonl"]
    sealed = []
    for rel in files:
        require_isolated(examples, load_exclusions(repo_root / rel))
        sealed.extend(load_rows(repo_root / rel))
    shas = {str(r.get("file_sha256", "")).casefold() for r in sealed} - {""}
    grams = set().union(*(_grams(r.get("excerpt") or "") for r in sealed))
    first300 = {(r.get("excerpt") or "")[:300] for r in sealed} - {""}
    seen_text = set()
    for item in examples:
        text = item["messages"][1]["content"]
        assert str(item.get("file_sha256", "")).casefold() not in shas
        assert hashlib.sha256(text.encode()).hexdigest() not in shas
        if text in seen_text:
            continue
        seen_text.add(text)
        assert text[:300] not in first300
        assert not (_grams(text) & grams)


@pytest.mark.private_data(
    'data/router/v7/isolation.json',
    'data/router/train.jsonl',
)
def test_box_isolation_report_is_clean(repo_root: Path) -> None:
    report = json.loads((repo_root / "data" / "router" / "v7" / "isolation.json").read_text(encoding="utf-8"))
    assert report["v7_examples"] == len(v7_training_examples(repo_root)[0])
    assert report["pair_texts_are_base_train_texts"] is True
    for name, entry in report["sets"].items():
        assert entry["rows_with_text"] == entry["rows"], name
        for key in ("file_sha256", "text_sha256", "first300", "project"):
            assert entry["overlaps"][key] == 0, (name, key)
        assert entry["shingle12_rows_v7"] == entry["shingle12_rows_base_train"], name  # v7 adds no new overlap
