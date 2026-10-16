import json
from pathlib import Path

from router.labels import load_label_set, load_labels


def test_label_set_v8_strings_appear_verbatim_in_the_doc(repo_root: Path) -> None:
    doc = (repo_root / "docs" / "router" / "LABEL_SET_v8.md").read_text(encoding="utf-8")
    data = load_label_set(repo_root / "configs" / "router" / "label_set_v8.json")
    labels = load_labels(repo_root / "configs" / "router" / "labels_v8.json")
    names = [item["label"] for item in data["labels"]]
    assert data["label_set"] == "v8"
    assert data["status"] == "approved"
    assert len(names) == 21
    assert names == labels
    assert json.loads((repo_root / "configs" / "router" / "labels_v8.json").read_text(encoding="utf-8"))["label_set"] == "v8"
    for item in data["labels"]:
        assert item["label"] in doc
        assert item["definition"] in doc
        assert item["is_not"] in doc
    frozen = load_labels(repo_root / "configs" / "router" / "labels_v5.json")
    assert "spec" in frozen
    assert "specification" not in frozen
    assert len(frozen) == 12
