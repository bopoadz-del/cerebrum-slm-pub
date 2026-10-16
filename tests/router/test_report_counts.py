import json

from router.labels import load_labels
from router.report_counts import main, report_counts


def test_counts_do_not_synthesize_missing_labels(repo_root) -> None:
    labels = load_labels(repo_root / "configs" / "router" / "labels.json")
    rows = [
        {"label": labels[0], "excerpt": "a"},
        {"label": labels[0], "excerpt": "b"},
        {"label": labels[1], "excerpt": "c"},
        {"label": "unsure", "excerpt": "d"},
    ]
    report = report_counts(rows, labels)
    assert report["counts"][labels[0]] == 2
    assert report["counts"][labels[1]] == 1
    assert sum(report["counts"].values()) == 3
    assert all(report["counts"][label] == 0 for label in labels[2:])
    assert report["rejected"] == 1
    assert report["rows"] == 4
    assert list(report["counts"]) == labels


def test_fork_reference_rows_stay_unlabelled(repo_root) -> None:
    labels = load_labels(repo_root / "configs" / "router" / "labels.json")
    rows = [
        {
            "filename": f"{labels[0]}.pdf",
            "doc_id": "abc",
            "project_id": "p",
            "fork_declared_type": labels[0],
            "label": labels[0],
            "first_512_tokens": "text",
        }
    ]
    report = report_counts(rows, labels)
    assert report["labelled"] == 0
    assert report["unlabelled"] == 1
    assert all(count == 0 for count in report["counts"].values())


def test_cli_on_missing_input_counts_zero(tmp_path, repo_root, capsys) -> None:
    missing = tmp_path / "none.jsonl"
    assert main(["--input", str(missing), "--root", str(repo_root)]) == 0
    report = json.loads(capsys.readouterr().out)
    labels = load_labels(repo_root / "configs" / "router" / "labels.json")
    assert report["rows"] == 0
    assert report["labelled"] == 0
    assert list(report["counts"]) == labels
