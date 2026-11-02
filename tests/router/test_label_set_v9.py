import pytest
import hashlib
import json
from pathlib import Path

from router.labels import label_set_path, load_label_set, load_labels
from router.teacher_label import prompt_for


def _set(repo_root: Path) -> dict:
    return load_label_set(repo_root / "configs" / "router" / "label_set_v9.json")


def test_label_set_v9_is_approved_and_in_force(repo_root: Path) -> None:
    doc = (repo_root / "docs" / "router" / "LABEL_SET_v9.md").read_text(encoding="utf-8")
    data = _set(repo_root)
    labels = load_labels(repo_root / "configs" / "router" / "labels.json")
    names = [item["label"] for item in data["labels"]]
    assert data["label_set"] == "v9"
    assert data["status"] == "approved"
    assert len(names) == 22
    assert names == labels
    assert json.loads((repo_root / "configs" / "router" / "labels.json").read_text(encoding="utf-8"))["label_set"] == "v9"
    assert label_set_path(repo_root).name == "label_set_v9.json"
    for rule in data["design_rules"]:
        assert rule in doc
    for item in data["labels"]:
        assert item["label"] in doc
        assert item["definition"] in doc
        assert item["is_not"] in doc


def test_v9_adds_only_change_request_to_v8(repo_root: Path) -> None:
    v9 = load_labels(repo_root / "configs" / "router" / "labels.json")
    v8 = load_labels(repo_root / "configs" / "router" / "labels_v8.json")
    assert len(v8) == 21
    assert set(v9) - set(v8) == {"change request"}
    assert set(v8) <= set(v9)
    frozen = load_labels(repo_root / "configs" / "router" / "labels_v5.json")
    assert len(frozen) == 12
    assert "spec" in frozen


def test_owner_additions_are_in_the_definitions(repo_root: Path) -> None:
    data = _set(repo_root)
    by = {item["label"]: item for item in data["labels"]}
    change = by["change request"]
    assert "works" in change["definition"]
    assert "claim" in change["is_not"]
    assert "VO/CO" in change["is_not"]
    assert any("about the works" in note and "entitlement" in note for note in change["notes"])
    assert any("priced variation proposal is change request" in note for note in change["notes"])
    assert "change request" in by["claim"]["is_not"]
    assert "change request" in by["correspondence"]["is_not"]
    assert "change request" in by["VO/CO"]["is_not"]
    assert "construction change directive" in by["VO/CO"]["regional_names_text"]
    assert any("even when the price" in note for note in by["VO/CO"]["notes"])
    assert "variation settlement recommendation" in by["report"]["regional_names_text"]
    assert any("employer's instruction" in note and "field change notice" in note for note in by["site instruction"]["notes"])
    assert any("about the works" in rule and "entitlement" in rule for rule in data["design_rules"])
    assert len(data["owner_additions"]) == 3


def test_v9_prompt_carries_no_document_codes_or_form_numbers(repo_root: Path) -> None:
    """The prompt is in every training row. Codes and form numbers stay out of it."""
    import re

    prompt = prompt_for()
    assert re.search(r"\b(EMI|FCN|VAR|VSR|RFM|VAO|TPR|TPC|INR)\b", prompt) is None
    assert re.search(r"\bG7\d\d\b", prompt) is None


def test_teacher_prompt_carries_v9_by_default(repo_root: Path) -> None:
    data = _set(repo_root)
    prompt = prompt_for()
    assert "- change request\n" in prompt
    for rule in data["design_rules"]:
        assert rule in prompt
    for item in data["labels"]:
        assert item["definition"] in prompt
        assert item["is_not"] in prompt


@pytest.mark.private_data(
    'data/router/relabel_v9.json',
    'data/router/train.jsonl',
    'data/router/val.jsonl',
)
def test_train_and_val_rows_carry_the_v9_prompt_and_labels(repo_root: Path) -> None:
    labels = load_labels(repo_root / "configs" / "router" / "labels.json")
    prompt = prompt_for(labels)
    record = json.loads((repo_root / "data" / "router" / "relabel_v9.json").read_text(encoding="utf-8"))
    counts: dict[str, dict[str, int]] = {}
    for split in ("train", "val"):
        seen: dict[str, int] = {}
        for line in (repo_root / "data" / "router" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            assert row["messages"][0]["content"] == prompt
            label = row["messages"][2]["content"]
            assert label in labels
            seen[label] = seen.get(label, 0) + 1
        counts[split] = seen
    assert counts == record["counts"]
    assert record["change_request_train"] == counts["train"].get("change request", 0)
    for change in record["changes"]:
        assert change["new"] in labels


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.mark.private_data(
    'data/eval/router_unseen/sealed_labels.jsonl',
    'data/eval/router_unseen/v8/sealed_labels.jsonl',
)
def test_sealed_retick_adds_no_rows_and_records_every_change(repo_root: Path) -> None:
    base = repo_root / "data" / "eval" / "router_unseen"
    record = json.loads((base / "v9_retick" / "label_changes.json").read_text(encoding="utf-8"))
    labels = load_labels(repo_root / "configs" / "router" / "labels.json")
    recorded = {(item["set"], item["sheet_id"]): item for item in record["changes"]}
    found = set()
    for entry in record["sets"]:
        old_path = repo_root / entry["source"]
        new_path = repo_root / entry["path"]
        assert hashlib.sha256(old_path.read_bytes()).hexdigest() == entry["old_sha256"]
        digest = hashlib.sha256(new_path.read_bytes()).hexdigest()
        assert digest == entry["new_sha256"]
        assert digest != entry["old_sha256"]
        assert (new_path.parent / "SHA256SUMS").read_text(encoding="utf-8") == f"{digest}  sealed_labels.jsonl\n"
        old_rows, new_rows = _rows(old_path), _rows(new_path)
        assert len(old_rows) == len(new_rows) == entry["rows"]
        for old, new in zip(old_rows, new_rows):
            assert new["sheet_id"] == old["sheet_id"]
            assert new["label_set"] == "v9"
            assert new["label"] in labels
            assert {k: v for k, v in new.items() if k not in ("label", "label_set")} == {
                k: v for k, v in old.items() if k != "label"
            }
            if new["label"] != old["label"]:
                item = recorded[(entry["set"], old["sheet_id"])]
                assert (item["old"], item["new"]) == (old["label"], new["label"])
                assert item["reason"].strip()
                found.add((entry["set"], old["sheet_id"]))
    assert found == set(recorded)
