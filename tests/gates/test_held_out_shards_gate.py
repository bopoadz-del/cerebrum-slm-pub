"""Held-out shards gate: frozen, sealed, listed, and never in train (every shard)."""

from __future__ import annotations
import pytest

import hashlib
import json
from pathlib import Path

from gates.augment import check_augmentation, pair_content_hash
from gates.held_out_shards import SHARDS, check_held_out_shards, near_dup_counts
from gates.run import evaluate


def _chat(user: str, assistant: str, **meta) -> dict:
    body = {"pair_type": meta.pop("pair_type", "SCOPE_CHECK"), **meta}
    body["content_hash"] = pair_content_hash("", user, assistant)
    return {"assistant": assistant, "meta": body, "system": "", "user": user}


def _iio(instruction: str, output: str, **meta) -> dict:
    return {"instruction": instruction, "input": "", "output": output, "meta": {"pair_type": "SCOPE_CHECK", **meta}}


def _write(path: Path, rows: list[dict]) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(row) + "\n" for row in rows).encode("utf-8")
    path.write_bytes(data)
    return data


ORIGINAL = [_chat("Who signs the permit to dig?", "State the authority first.")]
ADDED = [_iio("Which party issued the instruction to the subcontractor?", "Name the issuer and their authority basis.")]
TRAIN = [_chat("Is the glazing installation complete?", "Name what was inspected first.")]


def _repo(tmp_path: Path, *, train=TRAIN, original=ORIGINAL, added=ADDED, sidecar=True, near=(0, 0)) -> Path:
    _write(tmp_path / "data/train/pairs.jsonl", list(train))
    first = _write(tmp_path / "data/held_out/pairs.jsonl", list(original))
    second = _write(tmp_path / "data/held_out/added_shard.jsonl", list(added))
    if sidecar:
        digest = hashlib.sha256(second).hexdigest()
        (tmp_path / "data/held_out/added_shard.jsonl.sha256").write_text(
            f"{digest}  added_shard.jsonl\n", encoding="utf-8"
        )
    shards = {
        "schema": "held_out_shards/v1",
        "rule": "docs/eval/HELD_OUT.md",
        "shards": [
            _entry("held_out", "data/held_out/pairs.jsonl", first, len(original), True, "system_user_assistant", near[0]),
            _entry("added_shard", "data/held_out/added_shard.jsonl", second, len(added), False, "instruction_input_output", near[1]),
        ],
    }
    (tmp_path / SHARDS).write_text(json.dumps(shards, indent=2) + "\n", encoding="utf-8")
    return tmp_path


def _entry(name, path, data, count, original, schema, near) -> dict:
    return {
        "name": name,
        "path": path,
        "sha256": hashlib.sha256(data).hexdigest(),
        "count": count,
        "frozen": True,
        "created": "2026-01-01",
        "original": original,
        "record_schema": schema,
        "routing_registry": "held_out",
        "near_dup_at_freeze": near,
    }


def _messages(findings) -> list[str]:
    return [f"{finding.path} {finding.message}" for finding in findings]


def test_clean_synthetic_repo_is_green(tmp_path: Path) -> None:
    assert check_held_out_shards(_repo(tmp_path)) == []


def test_added_shard_pair_in_train_is_red(tmp_path: Path) -> None:
    leaked = _chat(ADDED[0]["instruction"], ADDED[0]["output"])
    found = _messages(check_held_out_shards(_repo(tmp_path, train=TRAIN + [leaked])))
    assert any("added_shard.jsonl" in m and "pair is in train" in m for m in found)
    assert any("added_shard.jsonl" in m and "user turn is in train" in m for m in found)


def test_original_shard_user_turn_in_train_is_red(tmp_path: Path) -> None:
    leaked = _chat("  WHO signs the permit   to dig? ", "A different answer.")
    found = _messages(check_held_out_shards(_repo(tmp_path, train=TRAIN + [leaked])))
    assert any("pairs.jsonl" in m and "user turn is in train" in m for m in found)


def test_near_duplicate_over_the_frozen_count_is_red(tmp_path: Path) -> None:
    near = _chat("Which party issued the instruction to the subcontractor today?", "Other answer.")
    found = _messages(check_held_out_shards(_repo(tmp_path, train=TRAIN + [near])))
    assert any("added_shard.jsonl" in m and "near-duplicate" in m for m in found)
    # Recorded at freeze: the same count is not a new finding.
    tmp2 = tmp_path / "frozen"
    assert check_held_out_shards(_repo(tmp2, train=TRAIN + [near], near=(0, 1))) == []
    assert near_dup_counts(tmp2) == {"held_out": 0, "added_shard": 1}


def test_cross_shard_duplicate_is_red(tmp_path: Path) -> None:
    dup = _iio(ORIGINAL[0]["user"], ORIGINAL[0]["assistant"])
    found = _messages(check_held_out_shards(_repo(tmp_path, added=[dup])))
    assert any("also in shard held_out" in m for m in found)


def test_changed_frozen_shard_is_red(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    path = root / "data/held_out/added_shard.jsonl"
    path.write_bytes(path.read_bytes().replace(b"authority", b"Authority"))
    found = _messages(check_held_out_shards(root))
    assert any("frozen shard sha256" in m for m in found)
    assert any("seal does not match" in m for m in found)


def test_added_shard_needs_its_sha256_seal(tmp_path: Path) -> None:
    found = _messages(check_held_out_shards(_repo(tmp_path, sidecar=False)))
    assert any("no .sha256 seal" in m for m in found)


def test_unlisted_held_out_file_is_red(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write(root / "data/held_out/stray.jsonl", ADDED)
    found = _messages(check_held_out_shards(root))
    assert any("stray.jsonl" in m and "not a listed shard" in m for m in found)


def test_missing_shard_list_is_red(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    (root / SHARDS).unlink()
    found = _messages(check_held_out_shards(root))
    assert any("shard list is missing" in m for m in found)


def test_augment_gate_sees_every_shard(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    added = ADDED[0]
    parent_hash = pair_content_hash("", added["instruction"], added["output"])
    child = _chat("Who issued the instruction to the subcontractor?", added["output"])
    child["meta"].update({"aug": "paraphrase", "parent_id": parent_hash})
    _write(root / "data/train/pairs.jsonl", TRAIN + [child])
    found = _messages(check_augmentation(root))
    assert any("child of a held-out parent is in train" in m for m in found)


def test_gate_runs_in_the_ci_entrypoint(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    (root / SHARDS).unlink()
    assert any(f.gate == "held-out-shards" for f in evaluate(root))


@pytest.mark.private_data(
    'data/held_out/shards.json',
    'data/held_out/pairs.jsonl',
    'data/held_out/manifest.json',
)
def test_real_repo_shards_are_green_and_match_freeze(repo_root: Path) -> None:
    assert check_held_out_shards(repo_root) == []
    listed = json.loads((repo_root / SHARDS).read_text(encoding="utf-8"))["shards"]
    assert near_dup_counts(repo_root) == {entry["name"]: entry["near_dup_at_freeze"] for entry in listed}
    for entry in listed:
        assert entry["frozen"] is True
        assert hashlib.sha256((repo_root / entry["path"]).read_bytes()).hexdigest() == entry["sha256"]
