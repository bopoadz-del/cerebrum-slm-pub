from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from gates.blind_seal import (
    BlindSealError,
    check_blind,
    opener_rejects_eval,
    path_is_sealed,
    seal_manifest,
    training_open,
)
from gates.records import sha256_file


def test_missing_eval_tree_hashes_empty_and_rejects_a_read(tmp_path: Path) -> None:
    assert seal_manifest(tmp_path) == []
    assert opener_rejects_eval(tmp_path)
    assert check_blind(tmp_path) == []
    probe = tmp_path / "data" / "eval" / "blind" / ".read-probe"
    with pytest.raises(BlindSealError):
        training_open(probe, tmp_path)
    assert not probe.exists()


def _write_sha256_seal(path: Path) -> None:
    digest = sha256_file(path)
    sidecar = path.with_name(path.name + ".sha256")
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="utf-8")


def test_read_attempt_fails_the_build_then_removed_is_green(tmp_path: Path) -> None:
    sealed = tmp_path / "data" / "eval" / "blind" / "case.jsonl"
    sealed.parent.mkdir(parents=True)
    sealed.write_text('{"instruction":"q","output":"a"}\n', encoding="utf-8")
    _write_sha256_seal(sealed)
    train = tmp_path / "src" / "train_step.py"
    train.parent.mkdir(parents=True)
    train.write_text('def load():\n    return open("data/eval/blind/case.jsonl").read()\n', encoding="utf-8")
    findings = check_blind(tmp_path)
    assert any("data/eval" in finding.message for finding in findings)

    joined = tmp_path / "scripts" / "train" / "fit.py"
    joined.parent.mkdir(parents=True)
    joined.write_text('from pathlib import Path\nroot = Path("data") / "eval"\n', encoding="utf-8")
    assert any(finding.path.endswith("fit.py") for finding in check_blind(tmp_path))

    train.unlink()
    joined.unlink()
    copied = tmp_path / "data" / "train" / "copied.jsonl"
    copied.parent.mkdir(parents=True)
    copied.write_bytes(sealed.read_bytes())
    byte_findings = check_blind(tmp_path)
    assert any("training manifest ∩ blind" in finding.message for finding in byte_findings)
    copied.unlink()
    assert check_blind(tmp_path) == []
    assert len(seal_manifest(tmp_path)) == 2


def test_training_can_read_outside_the_seal(tmp_path: Path) -> None:
    allowed = tmp_path / "data" / "train" / "ok.txt"
    allowed.parent.mkdir(parents=True)
    allowed.write_text("ok", encoding="utf-8")
    with training_open(allowed, tmp_path) as handle:
        assert handle.read() == b"ok"
    escaped = tmp_path / "data" / "eval" / ".." / "train" / "ok.txt"
    assert path_is_sealed(tmp_path / "data" / "eval" / "blind" / "x.jsonl", tmp_path)
    assert not path_is_sealed(escaped, tmp_path)
    with training_open(escaped, tmp_path) as handle:
        assert handle.read() == b"ok"


def test_sha256_sidecar_must_match_and_a_readme_is_not_sealed(tmp_path: Path) -> None:
    payload = tmp_path / "data" / "eval" / "unseen" / "questions.jsonl"
    payload.parent.mkdir(parents=True)
    payload.write_text('{"shape":"refusal"}\n', encoding="utf-8")
    missing = check_blind(tmp_path)
    assert any(finding.message == "missing seal" for finding in missing)

    sidecar = payload.with_name(payload.name + ".sha256")
    sidecar.write_text(f"{'ab' * 32}  {payload.name}\n", encoding="utf-8")
    mismatched = check_blind(tmp_path)
    assert any("sha256 mismatch" in finding.message for finding in mismatched)
    assert all(finding.message != "missing seal" for finding in mismatched)

    _write_sha256_seal(payload)
    (payload.parent / "README.md").write_text("Grade by shape.\n", encoding="utf-8")
    assert check_blind(tmp_path) == []
    extra = payload.parent / "extra.jsonl"
    extra.write_text("{}\n", encoding="utf-8")
    assert any(
        finding.path.endswith("extra.jsonl") and finding.message == "missing seal"
        for finding in check_blind(tmp_path)
    )
    extra.unlink()
    assert check_blind(tmp_path) == []
    with pytest.raises(BlindSealError):
        training_open(payload, tmp_path)


@pytest.mark.private_data('data/raw/slm-datasets.zip')
def test_seal_json_matches_then_count_hash_and_source_zip_fail(
    tmp_path: Path, repo_root: Path
) -> None:
    payload = tmp_path / "data" / "eval" / "regression" / "pairs.jsonl"
    payload.parent.mkdir(parents=True)
    payload.write_text('{"user":"q","assistant":"a"}\n{"user":"q2","assistant":"a2"}\n', encoding="utf-8")
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    shutil.copy(repo_root / "data" / "raw" / "slm-datasets.zip", raw / "slm-datasets.zip")
    digest = sha256_file(payload)
    zip_hash = sha256_file(raw / "slm-datasets.zip")
    seal = {
        "algorithm": "sha256",
        "artifact": "pairs.jsonl",
        "count": 2,
        "sha256": digest,
        "source_zip_sha256": zip_hash,
    }
    seal_path = payload.parent / "seal.json"
    seal_path.write_text(json.dumps(seal), encoding="utf-8")
    assert check_blind(tmp_path) == []

    seal["sha256"] = "0" * 64
    seal_path.write_text(json.dumps(seal), encoding="utf-8")
    assert any("sha256 mismatch" in finding.message for finding in check_blind(tmp_path))

    seal["sha256"] = digest
    seal["count"] = 1
    seal_path.write_text(json.dumps(seal), encoding="utf-8")
    counted = check_blind(tmp_path)
    assert any("seal count mismatch" in finding.message for finding in counted)

    seal["count"] = 2
    seal["source_zip_sha256"] = "0" * 64
    seal_path.write_text(json.dumps(seal), encoding="utf-8")
    sourced = check_blind(tmp_path)
    assert any("source_zip_sha256 mismatch" in finding.message for finding in sourced)

    seal_path.unlink()
    assert any(finding.message == "missing seal" for finding in check_blind(tmp_path))


def test_concatenated_eval_path_is_red_and_training_open_is_not(tmp_path: Path) -> None:
    train = tmp_path / "src" / "train_step.py"
    train.parent.mkdir(parents=True)
    train.write_text(
        'def load():\n    return open("data/" + "eval/blind/case.jsonl").read()\n',
        encoding="utf-8",
    )
    findings = check_blind(tmp_path)
    assert any("data/eval" in finding.message for finding in findings)

    joined = tmp_path / "scripts" / "train" / "fit.py"
    joined.parent.mkdir(parents=True)
    joined.write_text(
        'path = "".join(["dat", "a/eval/blind"])\n',
        encoding="utf-8",
    )
    assert any(finding.path.endswith("fit.py") and "data/eval" in finding.message for finding in check_blind(tmp_path))

    train.write_text(
        "def load(path, root):\n    return training_open(path, root).read()\n",
        encoding="utf-8",
    )
    joined.unlink()
    assert check_blind(tmp_path) == []


def test_unknown_path_requires_training_open_and_a_constant_outside_does_not(tmp_path: Path) -> None:
    train = tmp_path / "src" / "train_step.py"
    train.parent.mkdir(parents=True)
    train.write_text("def load(path):\n    return open(path).read()\n", encoding="utf-8")
    findings = check_blind(tmp_path)
    assert any("training_open" in finding.message for finding in findings)

    train.write_text(
        'def load():\n    return open("configs/model/qwen3-8b.json", encoding="utf-8").read()\n',
        encoding="utf-8",
    )
    assert check_blind(tmp_path) == []

