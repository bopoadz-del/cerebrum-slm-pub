from __future__ import annotations

from pathlib import Path

from gates.blind_seal import check_blind, seal_manifest
from gates.prove import run_proofs
from gates.run import evaluate, main


_RED_THEN_GREEN = (
    "probe-id",
    "domain",
    "word-list",
    "eval-hash",
    "number-unit",
    "currency",
    "clause-edition",
    "brand",
    "site",
    "blind-read",
    "blind-bytes",
)


def _tracked_paths(root: Path) -> tuple[str, ...]:
    """Paths the planting proof must leave untouched, present or absent."""
    found: list[str] = []
    for relative in ("src", "data/eval", "data/train"):
        base = root / relative
        if not base.exists():
            continue
        if base.is_file():
            found.append(relative)
            continue
        found.extend(path.relative_to(root).as_posix() for path in sorted(base.rglob("*")))
    return tuple(found)


def test_injection_is_red_then_removed_then_green(repo_root: Path) -> None:
    before = _tracked_paths(repo_root)
    lines = run_proofs()
    text = "\n".join(lines)
    cursor = 0
    for label in _RED_THEN_GREEN:
        red = text.find(f"PROOF {label} RED", cursor)
        green = text.find(f"PROOF {label} GREEN", cursor)
        assert red != -1 and green != -1 and red < green, label
        cursor = green
    posture = text.find("PROOF posture GREEN")
    number = text.find("PROOF number-unit RED")
    assert posture != -1 and posture < number
    assert "PROOF training-read REJECTED" in text
    assert text.endswith("PROOF sequence complete")
    assert _tracked_paths(repo_root) == before
    if (repo_root / "data" / "eval").is_dir():
        # The blind seal checks the raw archive. Proofs above do not.
        from private_paths import skip_if_missing

        skip_if_missing(repo_root, "data/raw/slm-datasets.zip", env="SLM_DATASETS_ZIP")
        assert check_blind(repo_root) == []


def test_repo_gates_are_green(repo_root: Path, capsys) -> None:
    assert evaluate(repo_root) == []
    assert main(["--root", str(repo_root)]) == 0
    captured = capsys.readouterr()
    sealed = seal_manifest(repo_root)
    assert f"blind files hashed: {len(sealed)}" in captured.out
    assert "training manifest ∩ blind = ∅" in captured.out
    assert "training read attempt: REJECTED" in captured.out
    assert "PROOF probe-id RED" in captured.out
    assert "PROOF probe-id GREEN" in captured.out
    assert "gates: green" in captured.out
    assert captured.err == ""
