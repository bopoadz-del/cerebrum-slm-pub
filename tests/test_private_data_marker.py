"""private_data marker: public skip, private run, manifest membership."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from private_data import PRIVATE_ENV, is_offloaded, offloaded_paths, private_mode
from private_data_mark import private_data_skip_reason

ROOT = Path(__file__).resolve().parents[1]


def _clear(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in PRIVATE_ENV:
        monkeypatch.delenv(name, raising=False)


def _manifest(root: Path, paths: list[str]) -> None:
    path = root / "configs" / "private_data" / "offloaded_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    files = [{"path": rel, "bytes": 1, "sha256": "ab" * 32, "category": "owner_data"} for rel in paths]
    path.write_text(json.dumps({"schema": 1, "files": files}), encoding="utf-8")


def test_marker_skips_only_offloaded_absent_paths_in_public_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    listed = "data/router/train.jsonl"
    other = "data/public/kept.json"
    _manifest(tmp_path, [listed])
    assert not private_mode()
    reason = private_data_skip_reason(tmp_path, (listed,))
    assert reason is not None
    assert listed in reason
    assert "SLM_PRIVATE_DATA_DIR" in reason
    assert private_data_skip_reason(tmp_path, (other,)) is None
    assert private_data_skip_reason(tmp_path, (listed, other)) is None
    present = tmp_path / listed
    present.parent.mkdir(parents=True)
    present.write_text("{}\n", encoding="utf-8")
    assert private_data_skip_reason(tmp_path, (listed,)) is None


def test_marker_does_not_skip_in_private_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    listed = "data/router/train.jsonl"
    _manifest(tmp_path, [listed])
    monkeypatch.setenv("SLM_PRIVATE_DATA_DIR", str(tmp_path / "external"))
    assert private_mode()
    assert private_data_skip_reason(tmp_path, (listed,)) is None


def test_is_offloaded_is_exact_or_a_containing_directory(tmp_path: Path) -> None:
    _manifest(tmp_path, ["data/held_out/shards.json", "data/router/train.jsonl"])
    assert is_offloaded("data/held_out/shards.json", tmp_path)
    assert is_offloaded("data/held_out", tmp_path)
    assert is_offloaded("data/router/train.jsonl", tmp_path)
    assert not is_offloaded("data/held_out/extra.json", tmp_path)
    assert not is_offloaded("data/router/other.jsonl", tmp_path)
    assert not is_offloaded("data/held_out/shards.json", tmp_path / "empty")


def test_hook_skips_in_public_mode_and_runs_in_private_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "pytest.ini").write_text("[pytest]\nmarkers =\n    private_data: marker\n", encoding="utf-8")
    repo_conftest = ROOT / "tests" / "conftest.py"
    (suite / "conftest.py").write_text(
        "import importlib.util\n"
        f"spec = importlib.util.spec_from_file_location('repo_conftest', {str(repo_conftest)!r})\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)\n"
        "def pytest_runtest_setup(item):\n"
        "    return mod.pytest_runtest_setup(item)\n",
        encoding="utf-8",
    )
    (suite / "test_sample.py").write_text(
        "import pytest\n\n"
        "@pytest.mark.private_data('data/router/train.jsonl')\n"
        "def test_reads():\n"
        "    raise AssertionError('ran')\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    for name in PRIVATE_ENV:
        env.pop(name, None)
    public = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-rs", "--tb=line", str(suite / "test_sample.py")],
        cwd=suite,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert public.returncode == 0, public.stdout + public.stderr
    assert "private_data:" in (public.stdout + public.stderr)
    env["SLM_PRIVATE_DATA_DIR"] = str(tmp_path / "external")
    private = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-rs", "--tb=line", str(suite / "test_sample.py")],
        cwd=suite,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert private.returncode != 0
    assert "ran" in (private.stdout + private.stderr)


def test_private_data_marker_paths_are_manifest_entries(request: pytest.FixtureRequest) -> None:
    known = offloaded_paths(ROOT)
    bad: list[str] = []
    for item in request.session.items:
        marker = item.get_closest_marker("private_data")
        if marker is None:
            continue
        for arg in marker.args:
            if arg not in known:
                bad.append(f"{item.nodeid} {arg}")
    assert bad == []
