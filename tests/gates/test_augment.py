"""Augmentation gate: children hash to parents, splits stay grouped, no sealed overlap."""

from __future__ import annotations

import json
from pathlib import Path

from gates.augment import REFUSE_PARENT_TYPES, check_augmentation, jaccard, pair_content_hash, token_set


def _row(user: str, assistant: str, split: str, **meta) -> dict:
    body = {"split": split, "pair_type": meta.pop("pair_type", "SCOPE_CHECK"), **meta}
    body["content_hash"] = pair_content_hash("", user, assistant)
    return {"assistant": assistant, "meta": body, "system": "", "user": user}


def _repo(tmp_path: Path, train: list[dict], held: list[dict], sealed: list[dict] | None = None) -> Path:
    for rel, rows in (("data/train/pairs.jsonl", train), ("data/held_out/pairs.jsonl", held)):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    if sealed is not None:
        path = tmp_path / "data/eval/unseen/questions.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in sealed), encoding="utf-8")
    return tmp_path


PARENT = _row("Is the glazing installation complete?", "Name what was inspected first.", "train")
HELD = _row("Who signs the permit?", "State the authority first.", "held_out", holdout="stratified")


def _child(user: str, parent: dict = PARENT, **overrides) -> dict:
    meta = {k: v for k, v in parent["meta"].items() if k != "content_hash"}
    meta.update({"aug": "paraphrase", "parent_id": parent["meta"]["content_hash"], "split": "train"})
    meta.update(overrides)
    row = _row(user, parent["assistant"], "train", **{k: v for k, v in meta.items() if k != "split"})
    return row


def _messages(findings) -> str:
    return "\n".join(finding.format() for finding in findings)


def test_clean_child_passes(tmp_path):
    root = _repo(tmp_path, [PARENT, _child("Has the glazing been fully installed?")], [HELD], sealed=[])
    assert check_augmentation(root) == []


def test_child_must_hash_to_a_train_parent(tmp_path):
    orphan = _child("Glazing done?")
    orphan["meta"]["parent_id"] = "0" * 64
    root = _repo(tmp_path, [PARENT, orphan], [HELD], sealed=[])
    assert "does not hash to a train parent" in _messages(check_augmentation(root))


def test_child_of_a_held_out_parent_never_lands_in_train(tmp_path):
    child = _child("Who has to sign off the permit?", parent=HELD)
    root = _repo(tmp_path, [PARENT, child], [HELD], sealed=[])
    assert "child of a held-out parent" in _messages(check_augmentation(root))


def test_paraphrase_keeps_the_assistant_turn_verbatim(tmp_path):
    child = _child("Is glazing complete now?")
    child["assistant"] = PARENT["assistant"] + " "
    child["meta"]["content_hash"] = pair_content_hash("", child["user"], child["assistant"])
    root = _repo(tmp_path, [PARENT, child], [HELD], sealed=[])
    assert "changed the assistant turn" in _messages(check_augmentation(root))


def test_paraphrase_may_not_add_a_fact(tmp_path):
    for text in ("Is the 6 mm glazing installation complete?", "Is glazing worth USD 40 complete?", "Is panel 3 complete?"):
        root = _repo(tmp_path, [PARENT, _child(text)], [HELD], sealed=[])
        assert "paraphrase adds" in _messages(check_augmentation(root)), text


def test_sealed_overlap_fails(tmp_path):
    sealed = [{"id": "Q1", "question": "Is the glazing installation complete now?"}]
    root = _repo(tmp_path, [PARENT, _child("Is the glazing installation complete now?")], [HELD], sealed=sealed)
    assert "overlaps a sealed eval item" in _messages(check_augmentation(root))


def test_stale_content_hash_fails(tmp_path):
    child = _child("Glazing finished?")
    child["meta"]["content_hash"] = "f" * 64
    root = _repo(tmp_path, [PARENT, child], [HELD], sealed=[])
    assert "content_hash" in _messages(check_augmentation(root))


def test_parent_in_both_splits_and_augmented_held_out_fail(tmp_path):
    dup = dict(PARENT, meta=dict(PARENT["meta"], split="held_out"))
    held_child = _child("Glazing complete?")
    held_child["meta"]["split"] = "held_out"
    root = _repo(tmp_path, [PARENT], [HELD, dup, held_child], sealed=[])
    text = _messages(check_augmentation(root))
    assert "in both train and held-out" in text
    assert "held-out row carries augmentation" in text


def _hard_negative(parent: dict, context_id: str, route: str) -> dict:
    user = f"Which formula applies to example formula?\n\nFork context:\n[{context_id}]\nformula_id: example_formula"
    assistant = json.dumps({"block": "none", "formula_id": route, "required_inputs": ["span_m"]})
    return _row(
        user,
        assistant,
        "train",
        aug="hard_negative",
        parent_id=parent["meta"]["content_hash"],
        pair_type="which_formula_applies",
        formula_id="example_formula",
        context_ids=[context_id],
    )


def test_hard_negative_cites_only_context_ids(tmp_path):
    refuse = _row("Why can example formula not answer?", "It cannot answer outside its scope.", "train",
                  pair_type="why_formula_cannot_answer", formula_id="example_formula")
    good = _hard_negative(refuse, "formula:example_formula", "example_formula")
    root = _repo(tmp_path, [refuse, good], [HELD], sealed=[])
    assert check_augmentation(root) == []
    bad = _hard_negative(refuse, "formula:example_formula", "other_formula")
    root = _repo(tmp_path, [refuse, bad], [HELD], sealed=[])
    assert "not in its context" in _messages(check_augmentation(root))
    wrong_parent = _hard_negative(PARENT, "formula:example_formula", "example_formula")
    root = _repo(tmp_path, [PARENT, wrong_parent], [HELD], sealed=[])
    assert "not refuse-labelled" in _messages(check_augmentation(root))


def test_refuse_parent_types_match_the_eval_labels(repo_root):
    import sys

    for entry in (repo_root, repo_root / "src"):
        if str(entry) not in sys.path:
            sys.path.append(str(entry))
    from eval.labels import _REFUSE_PAIR_TYPES

    assert REFUSE_PARENT_TYPES == _REFUSE_PAIR_TYPES


def test_jaccard_is_token_overlap():
    assert jaccard(token_set("a b c d"), token_set("a b c d")) == 1.0
    assert jaccard(token_set("a b"), token_set("c d")) == 0.0


def test_committed_augmentation_is_green(repo_root):
    assert check_augmentation(repo_root) == []
