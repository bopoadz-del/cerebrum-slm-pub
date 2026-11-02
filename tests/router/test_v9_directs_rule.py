"""The v9 directs/asks-or-prices rule is one design-rule line; nothing else in the prompt changed."""

from __future__ import annotations
import pytest

import copy
import hashlib
import json
from pathlib import Path

from router.labels import load_labels
from router.teacher_label import prompt_for

RULE = "Directs the change = VO/CO, asks or prices the change = change request."
# sha256 of prompt_for(labels) on main 64b1df9, before this rule was added (also the v3/v5 sealed-run system_sha256).
PROMPT_BEFORE_SHA256 = "4bb6ee17041e680147008f87b8fc6faf5fa02cf36926b1e02e66a19231dac56b"


def _set(root: Path) -> dict:
    return json.loads((root / "configs" / "router" / "label_set_v9.json").read_text(encoding="utf-8"))


def test_rule_is_one_design_rule_in_the_approved_label_set(repo_root: Path) -> None:
    data = _set(repo_root)
    assert data["design_rules"].count(RULE) == 1
    assert data["design_rules"][-1] == RULE
    for item in data["labels"]:
        for field in ("definition", "is_not", "regional_names_text"):
            assert RULE not in (item.get(field) or "")
        assert RULE not in item.get("notes", [])


def test_prompt_carries_the_rule_once(repo_root: Path) -> None:
    labels = load_labels(repo_root / "configs" / "router" / "labels.json")
    prompt = prompt_for(labels)
    assert prompt.count(f"- {RULE}\n") == 1


def test_nothing_else_in_the_prompt_changed(repo_root: Path) -> None:
    labels = load_labels(repo_root / "configs" / "router" / "labels.json")
    prompt = prompt_for(labels)
    without = prompt.replace(f"- {RULE}\n", "", 1)
    assert hashlib.sha256(without.encode("utf-8")).hexdigest() == PROMPT_BEFORE_SHA256
    data = copy.deepcopy(_set(repo_root))
    data["design_rules"] = [rule for rule in data["design_rules"] if rule != RULE]
    assert prompt_for(labels, label_set=data) == without


@pytest.mark.private_data(
    'data/router/train.jsonl',
    'data/router/val.jsonl',
)
def test_train_and_val_rows_change_only_the_system_message(repo_root: Path) -> None:
    labels = load_labels(repo_root / "configs" / "router" / "labels.json")
    prompt = prompt_for(labels)
    for split in ("train", "val"):
        for line in (repo_root / "data" / "router" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                messages = json.loads(line)["messages"]
                assert messages[0] == {"role": "system", "content": prompt}
                assert RULE not in messages[1]["content"] and RULE not in messages[2]["content"]


def test_label_doc_mirrors_the_rule(repo_root: Path) -> None:
    text = (repo_root / "docs" / "router" / "LABEL_SET_v9.md").read_text(encoding="utf-8")
    assert text.count(f"- {RULE}\n") == 1
