"""Router v7b (repair attempt 1 of 2): project round-robin row order, content-named Is-not neighbours first
(with a 3x edge cap) and catch-all sibling pairs. All generic: options live in configs/router/pairs_v7b.json,
the generator still names no label, project or document, and v7 still regenerates byte for byte."""

import inspect
import json
from collections import Counter
from pathlib import Path

import pytest

import router.v7_pairs as v7_pairs
from router.balance import BalanceConfigError
from router.balance_v7 import load_v7_config, read_weights, weights_payload
from router.labels import load_label_set, load_labels
from router.train_router import fit_router_v7
from router.v7_pairs import (
    AliasMatcher,
    build_pairs,
    class_budget,
    is_not_neighbours,
    label_aliases,
    load_rows,
    load_rules,
    materialise,
    pair_files,
    summary_file,
    v7_training_examples,
)
from tests.router.test_v6_balance import _rows, _Sdk

RULES = "configs/router/pairs_v7b.json"
CONFIG = "configs/router/train_v7b.json"


def _setup(root: Path):
    rules = load_rules(root, RULES)
    label_set = load_label_set(root / rules["label_set"])
    labels = load_labels(root / "configs" / "router" / "labels.json")
    train = load_rows(root / "data" / "router" / "train.jsonl")
    return rules, label_set, labels, train


def _refs(root: Path) -> dict[str, list[dict]]:
    return {key: load_rows(root / rel) for key, rel in pair_files(load_rules(root, RULES)).items()}


def test_v7b_files_equal_a_regeneration_and_v7_is_unchanged(repo_root: Path) -> None:
    assert v7_pairs.main(["--root", str(repo_root), "--train-config", CONFIG, "--check"]) == 0
    assert v7_pairs.main(["--root", str(repo_root), "--check"]) == 0
    rules = load_rules(repo_root, RULES)
    assert list(pair_files(rules).values()) == json.loads((repo_root / CONFIG).read_text())["training_examples"]["pairs"]
    assert summary_file(rules).startswith("data/router/v7b/")


def test_v7b_recipe_is_v7_except_the_run_names(repo_root: Path) -> None:
    v7 = json.loads((repo_root / "configs/router/train_v7.json").read_text())
    v7b = json.loads((repo_root / CONFIG).read_text())
    assert v7b["class_weighting"]["method"] == v7["class_weighting"]["method"]
    assert v7b["class_weighting"]["exponent"] == v7["class_weighting"]["exponent"]
    assert v7b["class_weighting"]["share_cap"] == v7["class_weighting"]["share_cap"]
    assert v7b["selection"] == v7["selection"] and v7b["lora_config"] == v7["lora_config"]
    for key, value in v7["overrides"].items():
        if key != "checkpoint_name":
            assert v7b["overrides"][key] == value
    labels = load_labels(repo_root / "configs" / "router" / "labels.json")
    source = inspect.getsource(v7_pairs) + (repo_root / CONFIG).read_text()
    for label in labels:
        assert f'"{label}"' not in source and f"'{label}'" not in source, label


def test_v7b_pairs_point_at_training_rows_and_materialise(repo_root: Path) -> None:
    _, label_set, labels, train = _setup(repo_root)
    by_id = {row["doc_id"]: row for row in train}
    refs = [r for v in _refs(repo_root).values() for r in v]
    for ref in refs:
        assert by_id[ref["doc_id"]]["messages"][-1]["content"] == ref["label"]
        assert ref["label"] in ref["menu"] and set(ref["menu"]) <= set(labels)
    assert len(materialise(refs, train, label_set)) == len(refs)
    examples, got = v7_training_examples(repo_root, RULES)
    assert len(examples) == len(train) + len(got) == len(train) + len(refs)
    base_texts = {row["messages"][1]["content"] for row in train}
    assert all(item["messages"][1]["content"] in base_texts for item in examples)  # v7 isolation report covers v7b


def test_v7b_contrastive_named_neighbours_and_edge_cap(repo_root: Path) -> None:
    rules, label_set, _, train = _setup(repo_root)
    neighbours = is_not_neighbours(label_set)
    matcher = AliasMatcher(label_aliases(label_set))
    by_id = {row["doc_id"]: row for row in train}
    refs = _refs(repo_root)["contrastive"]
    budget = class_budget(train)
    head = rules["contrastive"]["head_chars"]
    per_label = Counter(r["label"] for r in refs)
    assert max(per_label.values()) <= budget
    assert len({r["doc_id"] for r in refs}) == len(refs)
    edges = Counter((r["label"], r["neighbour"]) for r in refs)
    for ref in refs:
        assert ref["neighbour"] in neighbours[ref["label"]]
        if ref["named_in_head"]:
            assert ref["neighbour"] in matcher.labels_in(by_id[ref["doc_id"]]["messages"][1]["content"][:head])
    governed = set(rules["combined_form"]["labels"].values())
    for label, n in per_label.items():
        hood = [x for x in neighbours[label] if {label, x} != governed]
        cap = rules["contrastive"]["edge_cap_factor"] * -(-n // len(hood))
        assert all(edges[(label, x)] <= cap for x in hood)


def test_v7b_sibling_pairs_are_catch_all_vs_catch_all_and_balanced(repo_root: Path) -> None:
    rules, _, _, _ = _setup(repo_root)
    refs = _refs(repo_root)
    catch_all = set(rules["catch_all"]["labels"])
    assert [r["doc_id"] for r in refs["catch_all_sibling"]] == [r["doc_id"] for r in refs["catch_all"]]
    per = {}
    for ref in refs["catch_all_sibling"]:
        assert ref["label"] in catch_all and ref["neighbour"] in catch_all - {ref["label"]}
        assert set(ref["menu"]) == {ref["label"], ref["neighbour"]}
        per.setdefault(ref["label"], Counter())[ref["neighbour"]] += 1
    for counts in per.values():
        assert max(counts.values()) - min(counts.values()) <= 1


def test_project_round_robin_spreads_a_class_budget_over_projects(repo_root: Path) -> None:
    rules, label_set, labels, _ = _setup(repo_root)
    hood_label = next(label for label, hood in is_not_neighbours(label_set).items() if hood and label in labels)
    pad = "Document body with a filled field: Date 01/02/2025 and reference text for the content gate. " * 4

    def row(i, label, project):
        return {"doc_id": f"d{i}", "file_sha256": f"s{i}", "project_id": project,
                "messages": [{"role": "system", "content": "x"}, {"role": "user", "content": pad}, {"role": "assistant", "content": label}]}
    filler = [x for x in labels if x != hood_label][:2]
    rows = [row(i, hood_label, "big") for i in range(10)] + [row(20, hood_label, "p2"), row(21, hood_label, "p3")]
    rows += [row(30 + i, filler[i % 2], "f") for i in range(6)]  # budget = median class count = 3
    pairs, _ = build_pairs(rows, label_set, labels, rules)
    chosen = [r for r in pairs["contrastive"] if r["label"] == hood_label]
    assert len(chosen) == 3 and {r["project_id"] for r in chosen} == {"big", "p2", "p3"}
    hashed, _ = build_pairs(rows, label_set, labels, load_rules(repo_root))
    assert len([r for r in hashed["contrastive"] if r["label"] == hood_label]) == 3


def test_v7b_weights_file_and_fit_use_the_v7b_config(tmp_path: Path, repo_root: Path) -> None:
    examples, _ = v7_training_examples(repo_root, RULES)
    names = [row["messages"][-1]["content"] for row in examples]
    cfg = load_v7_config(repo_root, CONFIG)
    assert cfg["weights_file"] == "data/router/v7b/class_weights.json"
    stored = json.loads((repo_root / cfg["weights_file"]).read_text())
    assert stored == json.loads(json.dumps(weights_payload(repo_root, names, CONFIG)))
    assert read_weights(repo_root, names, CONFIG) == pytest.approx(stored["weights"])
    with pytest.raises(BalanceConfigError):
        read_weights(repo_root, names)  # the v7 file does not match the v7b examples
    labels = load_labels(repo_root / "configs" / "router" / "labels.json")
    train = _rows(labels[:3], [12, 4, 1], "t")
    val = _rows(labels[:3], [6, 2, 1], "v")
    for rel in ("configs/lora/qwen3-4b-router-r16.json", CONFIG):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes((repo_root / rel).read_bytes())
    target = tmp_path / cfg["weights_file"]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(weights_payload(tmp_path, [r["messages"][-1]["content"] for r in train], CONFIG)))
    out = fit_router_v7(tmp_path, train, val, labels, sdk=_Sdk(good={1}, fallback=labels[0]), config_file=CONFIG)
    assert out["class_weights"] == pytest.approx(json.loads(target.read_text())["weights"])
    bad = json.loads((repo_root / CONFIG).read_text())
    bad["class_weighting"]["weights_file"] = "data/router/v7/class_weights.json"
    (tmp_path / CONFIG).write_text(json.dumps(bad))
    with pytest.raises(BalanceConfigError):
        load_v7_config(tmp_path, CONFIG)
