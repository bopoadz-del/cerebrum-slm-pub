"""Plant a violation, watch the gate go red, remove it, and watch the gate go green."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from gates.blind_seal import BlindSealError, check_blind, training_open
from gates.findings import Finding
from gates.no_hardwiring import check_hardwiring
from gates.records import sha256_file
from gates.split import check_split

_EVAL_PAIR = {
    "instruction": "Gate probe question",
    "input": "",
    "output": "Gate probe answer that names no figure.",
    "meta": {
        "domain": "plant_domain_zz",
        "source_case": "probe_id_zz",
        "eval_only": True,
    },
}
_POSTURE = {
    "instruction": "What should the answer do?",
    "input": "",
    "output": (
        "Name the missing input and refuse an unsourced figure. "
        "Where a measured record exists it governs over a default."
    ),
}
_RATE = {
    "market": "North Wharf",
    "source": "AcmePlant Handbook 2024",
    "currency": "USD",
    "rate": 1,
    "unit": "hour",
}


class ProofError(RuntimeError):
    pass


def run_proofs() -> list[str]:
    """Red, then removed, then green. The planted files never touch the repo."""
    import tempfile

    lines: list[str] = []
    with tempfile.TemporaryDirectory(prefix="cerebrum-gates-") as temporary:
        root = Path(temporary)
        eval_path = root / "data" / "eval" / "blind" / "item.jsonl"
        _write_jsonl(eval_path, [_EVAL_PAIR])
        _write_sha256_seal(eval_path)
        _write_jsonl(root / "data" / "retrieval" / "rates.jsonl", [_RATE])
        posture_path = root / "data" / "train" / "posture.jsonl"
        _write_jsonl(posture_path, [_POSTURE])

        src = root / "src" / "hardwire.py"
        lines.append(
            _cycle(
                "probe-id",
                lambda: _write(src, 'token = "probe_id_zz"\n'),
                lambda: src.unlink(),
                lambda: check_hardwiring(root),
                "probe_id_zz",
            )
        )
        lines.append(
            _cycle(
                "domain",
                lambda: _write(src, 'token = "plant_domain_zz"\n'),
                lambda: src.unlink(),
                lambda: check_hardwiring(root),
                "plant_domain_zz",
            )
        )
        lines.append(
            _cycle(
                "word-list",
                lambda: _write(src, 'domains = ["alpha", "beta", "gamma"]\n'),
                lambda: src.unlink(),
                lambda: check_hardwiring(root),
                "word list",
            )
        )

        copied = root / "data" / "train" / "copied.jsonl"
        lines.append(
            _cycle(
                "eval-hash",
                lambda: shutil.copyfile(eval_path, copied),
                lambda: copied.unlink(),
                lambda: check_hardwiring(root),
                "matches a sealed eval item",
            )
        )

        lines.append(_green(check_split(root), "posture"))

        fact = root / "data" / "train" / "fact.jsonl"
        fact_turns = (
            ("number-unit", "Allow 12 mm cover."),
            ("currency", "The allowance is 6 USD."),
            ("clause-edition", "Apply ACI 318-19."),
            ("brand", "Take the AcmePlant value."),
            ("site", "Sequence the works at North Wharf."),
        )
        for label, assistant in fact_turns:
            lines.append(
                _cycle(
                    label,
                    lambda assistant=assistant: _write_jsonl(
                        fact,
                        [{"instruction": "Planted fact", "input": "", "output": assistant}],
                    ),
                    lambda: fact.unlink(),
                    lambda: check_split(root),
                    label,
                )
            )

        train_code = root / "src" / "train_step.py"
        lines.append(
            _cycle(
                "blind-read",
                lambda: _write(
                    train_code,
                    'def load():\n    return open("data/eval/blind/item.jsonl").read()\n',
                ),
                lambda: train_code.unlink(),
                lambda: check_blind(root),
                "data/eval",
            )
        )

        byte_copy = root / "data" / "train" / "copied_bytes.jsonl"
        lines.append(
            _cycle(
                "blind-bytes",
                lambda: shutil.copyfile(eval_path, byte_copy),
                lambda: byte_copy.unlink(),
                lambda: check_blind(root),
                "training manifest ∩ blind",
            )
        )

        try:
            training_open(root / "data" / "eval" / "blind" / "item.jsonl", root)
        except BlindSealError:
            lines.append("PROOF training-read REJECTED")
        else:
            raise ProofError("training read of data/eval was allowed")

        if check_hardwiring(root) or check_split(root) or check_blind(root):
            raise ProofError("tree was not green after the planted files were removed")
        lines.append("PROOF sequence complete")
    return _flatten(lines)


def _cycle(label, plant, remove, check, needle: str) -> list[str]:
    plant()
    red = _red(check(), label, needle)
    remove()
    green = _green(check(), label)
    return [red, green]


def _red(findings: list[Finding], label: str, needle: str) -> str:
    if not any(needle in finding.message for finding in findings):
        detail = findings[0].format() if findings else "no findings"
        raise ProofError(f"{label} expected RED ({needle}), saw: {detail}")
    return f"PROOF {label} RED"


def _green(findings: list[Finding], label: str) -> str:
    if findings:
        raise ProofError(f"{label} expected GREEN, saw: {findings[0].format()}")
    return f"PROOF {label} GREEN"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    _write(path, payload)


def _write_sha256_seal(path: Path) -> None:
    digest = sha256_file(path)
    _write(path.with_name(path.name + ".sha256"), f"{digest}  {path.name}\n")


def _flatten(items: list[str | list[str]]) -> list[str]:
    lines: list[str] = []
    for item in items:
        if isinstance(item, list):
            lines.extend(item)
        else:
            lines.append(item)
    return lines
