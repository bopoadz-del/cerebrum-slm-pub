"""Stratified practitioner sample and an agreement stub.

The sample takes existing content-labelled rows. It does not clone a rare
label up to the quota, and it does not read Fork types as labels.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from router.content import content_text
from router.guard import RouterWall, refuse_sealed_input
from router.labels import load_labels, load_sealed_config, repo_root
from router.sheet import blank_row, write_sheet


@dataclass(frozen=True)
class SampleCount:
    label: str
    available: int
    taken: int


def content_label(row: dict) -> str | None:
    """A label assigned from content. Fork reference columns block this read."""
    if "fork_declared_type" in row or "declared_type" in row:
        return None
    label = row.get("label")
    if isinstance(label, str) and label.strip():
        return label
    return None


def stratified_sample(
    rows: list[dict],
    labels: list[str],
    per_label: int,
) -> tuple[list[dict], list[SampleCount]]:
    """First N content rows of each declared label, in label-file order.

    A label with fewer than N rows contributes those rows once. A label with
    none contributes nothing. Rows are not synthesized.
    """
    if per_label < 1:
        raise ValueError("per_label must be at least 1")
    buckets: dict[str, list[dict]] = {label: [] for label in labels}
    allowed = set(labels)
    for row in rows:
        label = content_label(row)
        if label not in allowed:
            continue
        if content_text(row) is None:
            continue
        buckets[label].append(row)
    picked: list[dict] = []
    plan: list[SampleCount] = []
    for label in labels:
        available = buckets[label]
        taken = available[:per_label]
        picked.extend(taken)
        plan.append(SampleCount(label, len(available), len(taken)))
    return picked, plan


def sample_sheet_rows(sampled: list[dict], labels: list[str]) -> list[list[str]]:
    """Tick sheet. Excerpt is content. Filename stays blank."""
    rows: list[list[str]] = []
    for index, row in enumerate(sampled, 1):
        excerpt = content_text(row) or ""
        rows.append(blank_row(f"HR-{index:03d}", labels, excerpt=excerpt))
    return rows


def agreement_by_label(
    pairs: list[tuple[str, str]],
    labels: list[str],
) -> dict[str, float | None]:
    """Fraction of pairs that match, among pairs that name the label.

    None means the label never appears, so agreement is undefined.
    """
    scores: dict[str, float | None] = {}
    for label in labels:
        relevant = [pair for pair in pairs if label in pair]
        if not relevant:
            scores[label] = None
            continue
        matched = sum(1 for left, right in relevant if left == right)
        scores[label] = matched / len(relevant)
    return scores


def labels_not_for_train(
    scores: dict[str, float | None],
    threshold: float,
    labels: list[str],
) -> list[str]:
    """Labels under the config threshold, or with no paired judgements."""
    flagged: list[str] = []
    for label in labels:
        score = scores.get(label)
        if score is None or score < threshold:
            flagged.append(label)
    return flagged


def _load_rows(path: Path, root: Path) -> list[dict]:
    refuse_sealed_input(path, root)
    rows: list[dict] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RouterWall(f"WALL: {path}:{lineno}: {exc}") from exc
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stratified practitioner sample for router labels")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--per-label", type=int, default=None)
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args(argv)
    root = (args.root or repo_root()).resolve()
    try:
        labels = load_labels(root / "configs" / "router" / "labels.json")
        config = load_sealed_config(root / "configs" / "router" / "sealed_eval.json")
        per_label = args.per_label if args.per_label is not None else int(config["per_label"])
        rows = _load_rows(args.input, root) if args.input is not None else []
        sampled, plan = stratified_sample(rows, labels, per_label)
        if args.output is not None:
            refuse_sealed_input(args.output, root)
            if sampled:
                write_sheet(args.output, labels, sample_sheet_rows(sampled, labels))
    except RouterWall as exc:
        print(str(exc))
        return 2
    print(
        json.dumps(
            {
                "agreement_threshold": config["agreement_threshold"],
                "plan": [
                    {"available": item.available, "label": item.label, "taken": item.taken}
                    for item in plan
                ],
                "sampled": len(sampled),
                "wrote": bool(args.output is not None and sampled),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
