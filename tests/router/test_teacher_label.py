import json
from pathlib import Path

import pytest

from router.guard import RouterWall
from router.teacher_label import (
    dry_run_report,
    eligible_for_train,
    filter_for_train,
    label_rows,
    main,
    messages_for,
    parse_teacher_response,
    prompt_for,
    with_model_version,
)


def test_prompt_uses_the_file_and_not_a_label_outside_it(tmp_path: Path) -> None:
    path = tmp_path / "label_set.json"
    path.write_text(
        json.dumps(
            {
                "design_rules": ["Use the document text only."],
                "labels": [
                    {
                        "label": "alpha-mark",
                        "definition": "Alpha means alpha.",
                        "is_not": "Not beta.",
                        "regional_names_text": None,
                        "notes": [],
                    },
                    {
                        "label": "beta-mark",
                        "definition": "Beta means beta.",
                        "is_not": "Not alpha.",
                        "regional_names_text": "beta form",
                        "notes": ["Combined forms: alpha stays alpha."],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    prompt = prompt_for(["alpha-mark", "beta-mark"], path=path)
    assert "alpha-mark" in prompt
    assert "beta-mark" in prompt
    assert "Alpha means alpha." in prompt
    assert "Not beta." in prompt
    assert "beta form" in prompt
    assert "Combined forms: alpha stays alpha." in prompt
    assert "Use the document text only." in prompt
    assert "RFI" not in prompt
    assert '{"label": "<one list entry or unsure>", "reason": "<one line>", "confidence": <your confidence, a number between 0 and 1>}' in prompt
    assert "Use only the document text in the next message." in prompt


def test_prompt_loads_the_approved_label_set_at_call_time(repo_root: Path, tmp_path: Path) -> None:
    data = json.loads((repo_root / "configs" / "router" / "label_set_v9.json").read_text(encoding="utf-8"))
    names = [item["label"] for item in data["labels"]]
    prompt = prompt_for(names)
    for rule in data["design_rules"]:
        assert rule in prompt
    for item in data["labels"]:
        assert item["definition"] in prompt
        assert item["is_not"] in prompt
        regional = item["regional_names_text"]
        if isinstance(regional, str):
            assert regional in prompt
        for note in item["notes"]:
            assert note in prompt
    mutated = json.loads(json.dumps(data))
    mutated["labels"][0]["definition"] = "UNIQUE-DEFINITION-TOKEN"
    mutated["design_rules"][0] = "UNIQUE-RULE-TOKEN"
    path = tmp_path / "label_set.json"
    path.write_text(json.dumps(mutated), encoding="utf-8")
    replacement = prompt_for(path=path)
    assert "UNIQUE-DEFINITION-TOKEN" in replacement
    assert "UNIQUE-RULE-TOKEN" in replacement
    assert data["labels"][0]["definition"] not in replacement


def test_messages_carry_excerpt_only() -> None:
    labels = ["alpha-mark"]
    row = {
        "doc_id": "doc-1",
        "project_id": "project-9",
        "filename": "rfi-from-the-name.pdf",
        "fork_declared_type": "document",
        "first_512_tokens": "excerpt body",
    }
    messages = messages_for(row, labels)
    assert messages is not None
    assert messages[1]["content"] == "excerpt body"
    blob = json.dumps(messages)
    assert "rfi-from-the-name.pdf" not in blob
    assert "project-9" not in blob
    assert "doc-1" not in blob
    assert "fork_declared_type" not in blob


def test_filename_alone_is_not_content() -> None:
    assert messages_for({"filename": "drawing.pdf", "project_id": "p"}, ["alpha-mark"]) is None


def test_unknown_model_label_becomes_unsure_and_stays_out_of_train() -> None:
    labels = ["alpha-mark", "beta-mark"]
    parsed = parse_teacher_response(
        '{"label": "invented-type", "reason": "no support", "confidence": 0.99}',
        labels,
    )
    assert parsed["label"] == "unsure"
    assert eligible_for_train(parsed, labels, 0.5) is False


def test_filter_drops_unsure_and_low_confidence() -> None:
    labels = ["alpha-mark"]
    rows = [
        {"label": "alpha-mark", "confidence": 0.9, "reason": "ok"},
        {"label": "alpha-mark", "confidence": 0.2, "reason": "low"},
        {"label": "unsure", "confidence": 0.99, "reason": "no"},
        {"label": "alpha-mark", "confidence": None, "reason": "missing"},
    ]
    kept = filter_for_train(rows, labels, 0.5)
    assert kept == [rows[0]]
    assert filter_for_train(rows, labels, None) == []


def test_mocked_teacher_does_not_copy_filename(tmp_path: Path) -> None:
    labels = ["alpha-mark"]

    def respond(messages):
        assert "drawing.pdf" not in messages[1]["content"]
        return '{"label": "alpha-mark", "reason": "from text", "confidence": 0.8}'

    rows = label_rows(
        [
            {
                "doc_id": "d1",
                "filename": "drawing.pdf",
                "excerpt": "body text",
                "project_id": "project-keep",
                "content_sha256": "abc123",
                "fork_declared_type": "document",
            }
        ],
        labels,
        respond,
    )
    assert rows == [
        {
            "label": "alpha-mark",
            "reason": "from text",
            "confidence": 0.8,
            "doc_id": "d1",
            "project_id": "project-keep",
            "file_sha256": "abc123",
            "excerpt": "body text",
        }
    ]
    assert "filename" not in rows[0]
    assert "fork_declared_type" not in rows[0]


def test_dry_run_prints_no_secret_and_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, repo_root: Path) -> None:
    monkeypatch.setenv("TINKER_API_KEY", "super-secret-value")
    card = repo_root / "docs" / "router" / "DATASET_CARD.md"
    before = card.read_text(encoding="utf-8")
    code = main(["--root", str(repo_root)])
    assert code == 0
    assert card.read_text(encoding="utf-8") == before
    captured = json.dumps(dry_run_report([], ["alpha-mark"]))
    assert "super-secret-value" not in captured
    assert captured.find("true") != -1 or "api_key_present" in captured


def test_dry_run_cli_hides_the_key(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], repo_root: Path) -> None:
    monkeypatch.setenv("ROUTER_TEACHER_API_KEY", "super-secret-value")
    assert main(["--root", str(repo_root)]) == 0
    out = capsys.readouterr().out
    assert "super-secret-value" not in out
    report = json.loads(out)
    assert report["labelled"] == 0
    assert report["wrote"] is False
    assert report["content_rows"] == 0
    assert report["api_key_present"] is True


def test_write_is_refused_while_sealed_file_is_empty(capsys: pytest.CaptureFixture[str], repo_root: Path) -> None:
    assert main(["--write", "--root", str(repo_root)]) == 2
    assert "WALL:" in capsys.readouterr().out
    assert not (repo_root / "data" / "train" / "router.jsonl").exists()


def test_sealed_input_is_refused(capsys: pytest.CaptureFixture[str], repo_root: Path) -> None:
    target = repo_root / "data" / "sealed" / "router" / "sealed_labels.jsonl"
    assert main(["--input", str(target), "--root", str(repo_root)]) == 2
    assert "WALL: sealed corpus" in capsys.readouterr().out


def test_model_version_updates_a_card_copy() -> None:
    card = "Status: no training labels.\n\nModel version: unset\n"
    updated = with_model_version(card, "teacher-model-stub")
    assert "Model version: teacher-model-stub" in updated
    with pytest.raises(RouterWall):
        with_model_version(card, "line\nbreak")
