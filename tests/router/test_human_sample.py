import csv
import json

from router.human_sample import (
    agreement_by_label,
    labels_not_for_train,
    main,
    sample_sheet_rows,
    stratified_sample,
)
from router.labels import load_labels, load_sealed_config
from router.sheet import header


def test_sample_keeps_a_rare_label_once_and_does_not_clone(repo_root) -> None:
    labels = load_labels(repo_root / "configs" / "router" / "labels.json")
    rare = labels[-1]
    common = labels[0]
    rows = [{"label": common, "excerpt": f"body-{index}"} for index in range(8)]
    rows.append({"label": rare, "excerpt": "rare-body"})
    rows.append({"filename": "claim.pdf", "excerpt": "no assigned label"})
    sampled, plan = stratified_sample(rows, labels, per_label=5)
    by_label = {item.label: item for item in plan}
    assert by_label[common].available == 8
    assert by_label[common].taken == 5
    assert by_label[rare].available == 1
    assert by_label[rare].taken == 1
    assert by_label[labels[1]].taken == 0
    assert len(sampled) == 6
    excerpts = [row["excerpt"] for row in sampled]
    assert excerpts.count("rare-body") == 1


def test_fork_type_is_not_a_sample_label(repo_root) -> None:
    labels = load_labels(repo_root / "configs" / "router" / "labels.json")
    name = labels[0]
    rows = [
        {
            "filename": "from-the-filename.pdf",
            "fork_declared_type": name,
            "label": name,
            "first_512_tokens": "pool excerpt",
        }
    ]
    sampled, plan = stratified_sample(rows, labels, per_label=5)
    assert sampled == []
    assert all(item.taken == 0 for item in plan)


def test_sheet_leaves_filename_blank(repo_root) -> None:
    labels = load_labels(repo_root / "configs" / "router" / "labels.json")
    sampled = [{"label": labels[0], "filename": "secret-name.pdf", "excerpt": "content only"}]
    rows = sample_sheet_rows(sampled, labels)
    assert rows[0][1] == "content only"
    assert rows[0][2] == ""
    assert "secret-name.pdf" not in json.dumps(rows)


def test_agreement_flags_labels_under_the_config_threshold(repo_root) -> None:
    labels = load_labels(repo_root / "configs" / "router" / "labels.json")
    config = load_sealed_config(repo_root / "configs" / "router" / "sealed_eval.json")
    left, right = labels[0], labels[1]
    pairs = [(left, left), (left, left), (right, labels[3])]
    scores = agreement_by_label(pairs, labels)
    assert scores[left] == 1.0
    assert scores[right] == 0.0
    assert scores[labels[3]] == 0.0
    assert scores[labels[2]] is None
    flagged = labels_not_for_train(scores, config["agreement_threshold"], labels)
    assert left not in flagged
    assert right in flagged
    assert labels[3] in flagged
    assert labels[2] in flagged


def test_threshold_keeps_an_exact_match(repo_root) -> None:
    labels = load_labels(repo_root / "configs" / "router" / "labels.json")
    scores = {label: None for label in labels}
    scores[labels[0]] = 0.8
    flagged = labels_not_for_train(scores, 0.8, labels)
    assert labels[0] not in flagged


def test_cli_writes_a_sheet_outside_the_sealed_tree(tmp_path, repo_root) -> None:
    labels = load_labels(repo_root / "configs" / "router" / "labels.json")
    source = tmp_path / "labelled.jsonl"
    source.write_text(
        json.dumps({"label": labels[0], "excerpt": "one excerpt"}) + "\n",
        encoding="utf-8",
    )
    dest = tmp_path / "sheet.csv"
    assert main(["--input", str(source), "--output", str(dest), "--per-label", "5", "--root", str(repo_root)]) == 0
    with dest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == header(labels)
    assert rows[1][1] == "one excerpt"
    assert rows[1][2] == ""
    assert len(rows) == 2


def test_cli_refuses_sealed_output(repo_root, capsys) -> None:
    dest = repo_root / "data" / "sealed" / "router" / "labelling_sheet.csv"
    assert main(["--output", str(dest), "--root", str(repo_root)]) == 2
    assert "WALL: sealed corpus" in capsys.readouterr().out
