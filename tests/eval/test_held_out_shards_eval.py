"""Eval harness scores every shard in data/held_out/shards.json, one line each."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from eval.labels import label_eval_sets
from eval.shards import ShardError, added_shards, label_shard, load_shards, original_shard, read_shard_records
from eval.stop_gate import load_tuned_scores
from eval.tuned import SHARD_SCHEMA, TunedEvalError, format_shard_scores, run_shard_eval, run_tuned_eval
from model.identity import API_KEY_ENV, repo_root
from tests.eval.test_labels import _write_tiny_split
from tests.eval.test_tuned_eval import _RESPONSE, _plant_routing_registries

SHARDS = "data/held_out/shards.json"


def _tiny_repo(root: Path) -> Path:
    _write_tiny_split(root, held_count=2, manifest_count=2)
    _plant_routing_registries(root)
    (root / "eval" / "baselines").mkdir(parents=True)
    (root / "eval" / "baselines" / "scores.md").write_text(
        "Prompt-only, held-out, n=2: refusal/qualification 0/2, fact-leak rate 0/2, "
        "formula-routing accuracy 0/2.\n"
        "Prompt-only, blind, n=1: refusal/qualification 0/1, fact-leak rate 0/1, "
        "formula-routing accuracy 0/1.\n",
        encoding="utf-8",
    )
    original = root / "data" / "held_out" / "pairs.jsonl"
    added = root / "data" / "held_out" / "extra.jsonl"
    rows = [
        {"instruction": "synthetic added user one", "input": "", "output": "synthetic answer", "meta": {"pair_type": "REFUSAL_VS_VACUUM"}},
        {"instruction": "synthetic added user two", "input": "", "output": "synthetic answer", "meta": {"pair_type": "AUTHORITY_SOURCE"}},
        {"instruction": "synthetic added user three", "input": "", "output": "synthetic answer", "meta": {"pair_type": "SCOPE_CHECK"}},
    ]
    added.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    shards = {
        "schema": "held_out_shards/v1",
        "shards": [
            _entry("held_out", "data/held_out/pairs.jsonl", original, 2, True, "system_user_assistant"),
            _entry("extra", "data/held_out/extra.jsonl", added, 3, False, "instruction_input_output"),
        ],
    }
    (root / SHARDS).write_text(json.dumps(shards) + "\n", encoding="utf-8")
    return root


def _entry(name, rel, path, count, original, schema):
    return {
        "name": name,
        "path": rel,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "count": count,
        "frozen": True,
        "created": "2026-01-01",
        "original": original,
        "record_schema": schema,
        "routing_registry": "held_out",
        "near_dup_at_freeze": 0,
    }


def test_real_shard_list_names_the_original_and_the_added_shards():
    shards = load_shards()
    original = original_shard(shards)
    assert original.path == "data/held_out/pairs.jsonl"
    assert original.count == 69
    assert all(shard.frozen for shard in shards)
    assert len(added_shards(shards)) >= 1


def test_original_shard_labels_are_the_held_out_labels():
    shards = load_shards()
    assert label_shard(None, original_shard(shards)) == dict(label_eval_sets())["held_out"]


def test_every_added_shard_labels_with_the_pair_type_rule():
    from eval.labels import label_held_out_record

    root = repo_root()
    for shard in added_shards(load_shards()):
        labels = label_shard(root, shard)
        assert len(labels) == shard.count
        for label, (line, record) in zip(labels, read_shard_records(root, shard)):
            expected = label_held_out_record(record, line=line)
            assert label.split == shard.name
            assert (label.expected_block, label.formula_id) == (expected.expected_block, expected.formula_id)


def test_a_changed_frozen_shard_is_refused(tmp_path: Path):
    root = _tiny_repo(tmp_path / "repo")
    path = root / "data" / "held_out" / "extra.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    path.write_bytes(path.read_bytes().replace(b"one", b"One"))
    with pytest.raises(ShardError, match="sha256"):
        load_shards(root)


def test_default_run_scores_every_shard_and_keeps_the_stop_gate_payload(tmp_path: Path, monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    root = _tiny_repo(tmp_path / "repo")
    seen = []

    def sampler(messages):
        seen.append(messages)
        return _RESPONSE

    output = tmp_path / "out" / "scores.json"
    payload = run_tuned_eval("tinker://example/weights/checkpoint", output, root=root, sampler=sampler)
    # Stop-gate sets first (2 held-out + 1 blind, two samples each), then the added shard.
    assert len(seen) == 2 * (2 + 1) + 2 * 3
    assert set(payload["sets"]) == {"held_out", "blind"}
    assert load_tuned_scores(output) == payload
    shard_payload = json.loads((tmp_path / "out" / "scores.shards.json").read_text(encoding="utf-8"))
    assert shard_payload["schema"] == SHARD_SCHEMA
    assert set(shard_payload["shards"]) == {"held_out", "extra"}
    held = shard_payload["shards"]["held_out"]
    assert {k: held[k] for k in payload["sets"]["held_out"]} == payload["sets"]["held_out"]
    assert held["original"] is True and shard_payload["shards"]["extra"]["original"] is False
    assert shard_payload["shards"]["extra"]["n"] == 3
    lines = format_shard_scores(shard_payload).splitlines()
    assert lines[0].startswith("shard held_out: n=2 ")
    assert lines[1].startswith("shard extra: n=3 ")


def test_default_run_without_added_shards_samples_exactly_as_before(tmp_path: Path, monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    root = _tiny_repo(tmp_path / "repo")
    seen = []
    run_tuned_eval(
        "tinker://example/weights/checkpoint",
        tmp_path / "scores.json",
        root=root,
        sampler=lambda messages: seen.append(messages) or _RESPONSE,
        score_added_shards=False,
    )
    assert len(seen) == 6
    assert not (tmp_path / "scores.shards.json").exists()


def test_shard_only_run_samples_only_that_shard_and_writes_no_text(tmp_path: Path, monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    root = _tiny_repo(tmp_path / "repo")
    seen = []
    output = tmp_path / "out" / "shard.json"
    items = tmp_path / "out" / "items.json"
    replies = tmp_path / "box" / "replies.jsonl"
    payload = run_shard_eval(
        "base://Qwen/Qwen3-8B",
        output,
        ["extra"],
        root=root,
        prompt_only_system_prompt=True,
        sampler=lambda messages: seen.append(messages) or _RESPONSE,
        item_results=items,
        replies=replies,
    )
    assert len(seen) == 6
    assert seen[0][0]["role"] == "system"
    assert list(payload["shards"]) == ["extra"]
    assert payload["system_prompt"] == "prompt_only"
    assert payload["fork_context"] is False
    blob = output.read_text(encoding="utf-8") + items.read_text(encoding="utf-8")
    assert "synthetic added user" not in blob
    assert "synthetic answer" not in blob
    rows = json.loads(items.read_text(encoding="utf-8"))["items"]
    assert [row["line"] for row in rows] == [1, 2, 3]
    assert [row["expected_block"] for row in rows] == ["refuse", "qualify", "qualify"]
    assert len(replies.read_text(encoding="utf-8").splitlines()) == 3


def test_shard_run_refuses_unknown_shards_and_replies_inside_the_repo(tmp_path: Path):
    root = _tiny_repo(tmp_path / "repo")
    never = lambda messages: (_ for _ in ()).throw(AssertionError("sampled"))  # noqa: E731
    with pytest.raises(TunedEvalError, match="not listed"):
        run_shard_eval("base://Qwen/Qwen3-8B", tmp_path / "o.json", ["missing"], root=root, sampler=never)
    with pytest.raises(TunedEvalError, match="raw replies"):
        run_shard_eval(
            "base://Qwen/Qwen3-8B",
            tmp_path / "o.json",
            ["extra"],
            root=root,
            sampler=never,
            replies=root / "replies.jsonl",
        )


def test_no_shard_name_or_pair_text_in_code():
    root = repo_root()
    shards = load_shards()
    code = [
        path
        for folder in ("eval", "scripts", "src")
        for path in (root / folder).rglob("*.py")
    ]
    source = "\n".join(path.read_text(encoding="utf-8") for path in code)
    for shard in added_shards(shards):
        assert shard.name not in source
        assert Path(shard.path).name not in source
    for shard in shards:
        for line, record in read_shard_records(root, shard):
            for text in (record["user"], record["assistant"]):
                if text in source:
                    raise AssertionError(f"shard {shard.name} line {line} text is in source")


def test_cli_shard_flag_parses(tmp_path: Path, monkeypatch):
    import importlib.util

    monkeypatch.delenv(API_KEY_ENV, raising=False)
    path = repo_root() / "scripts" / "eval" / "run_tuned_eval.py"
    spec = importlib.util.spec_from_file_location("run_tuned_eval_cli_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    name = added_shards(load_shards())[0].name
    output = tmp_path / "shard.json"
    # No key: refused before any sample, nothing written.
    assert module.main(["--checkpoint", "tinker://example/weights/checkpoint", "--output", str(output), "--shard", name]) == 1
    assert not output.exists()


def test_shard_code_does_not_call_the_training_reader():
    for path in (repo_root() / "eval" / "shards.py",):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {getattr(node, "id", None) or getattr(node, "attr", None) for node in ast.walk(tree)}
        assert "training_open" not in names
