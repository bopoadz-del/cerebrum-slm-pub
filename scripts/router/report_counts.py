"""Per-label counts of content labels. Does not clone or invent rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from router.guard import RouterWall, refuse_sealed_input
from router.human_sample import content_label
from router.labels import load_labels, repo_root


def report_counts(rows: list[dict], labels: list[str]) -> dict:
    counts = {label: 0 for label in labels}
    allowed = set(labels)
    unlabelled = 0
    rejected = 0
    for row in rows:
        label = content_label(row)
        if label is None:
            unlabelled += 1
            continue
        if label in allowed:
            counts[label] += 1
        else:
            rejected += 1
    labelled = sum(counts.values())
    return {
        "counts": counts,
        "labelled": labelled,
        "rejected": rejected,
        "rows": len(rows),
        "unlabelled": unlabelled,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Count router content labels")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args(argv)
    root = (args.root or repo_root()).resolve()
    try:
        labels = load_labels(root / "configs" / "router" / "labels.json")
        rows: list[dict] = []
        if args.input is not None:
            refuse_sealed_input(args.input, root)
            if args.input.is_file():
                for line in args.input.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        rows.append(obj)
        report = report_counts(rows, labels)
    except (RouterWall, json.JSONDecodeError) as exc:
        print(str(exc))
        return 2
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
