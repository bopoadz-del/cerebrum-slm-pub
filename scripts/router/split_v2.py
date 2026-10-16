"""Group-held-out, label-stratified train/val split for the router (v2).

The v1 project split left almost every row in two project ids that share filenames.
This split holds out a document series instead. The pool has no folder path, so the
series is taken from the filename. The filename is a grouping key only. It is not
model input and it is not a label.

Rows that share a filename, an excerpt, or a file sha are one group. project_id is
not part of the key. Drawing rows in train are capped. Val is not capped. No row
is copied.

Sealed isolation runs on the examples before the split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from router.isolation import load_exclusions, require_isolated
from router.labels import load_labels, load_sealed_config, repo_root
from router.train_router import require_no_paths, rows_from_pool, training_examples

_SERIES = re.compile(
    r"^([A-Z]{2,}-[A-Z]{2,}-\d{3})-\d{4}-[A-Z]{2,}-([A-Z]{2,})-([A-Z]{2,})-(\d{3})-",
    re.I,
)
DRAWING = "drawing"


def series_key(filename: str) -> str:
    name = filename.strip()
    matched = _SERIES.match(name)
    if matched:
        return "S:" + "|".join(part.upper() for part in matched.groups())
    stem = re.sub(r"\.[A-Za-z0-9]{2,5}$", "", name).lower()
    stem = re.sub(r"\b(rev|r|v)\s*\d+\b", " ", stem)
    stem = re.sub(r"\b(draft|final|copy|caw|signed|part)\b", " ", stem)
    stem = re.sub(r"\d+", " ", stem)
    stem = re.sub(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b", " ", stem)
    stem = re.sub(r"[^a-z]+", " ", stem)
    stem = " ".join(stem.split())
    return "F:" + (stem or name.lower())


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: str, right: str) -> None:
        root_left, root_right = self.find(left), self.find(right)
        if root_left != root_right:
            self.parent[max(root_left, root_right)] = min(root_left, root_right)


def _excerpt_node(text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return "ex:" + digest


def assign_groups(rows: list[dict], filenames: dict[str, str]) -> list[str]:
    groups = _UnionFind()
    nodes: list[str] = []
    for index, row in enumerate(rows):
        fname = filenames.get(row.get("doc_id"), "") or ""
        node = f"row:{index}"
        groups.union(node, series_key(fname) if fname else f"nofile:{index}")
        if fname:
            groups.union(node, "file:" + fname.strip().lower())
        user = row["messages"][1]["content"]
        groups.union(node, _excerpt_node(user))
        sha = row.get("file_sha256")
        if isinstance(sha, str) and sha:
            groups.union(node, "sha:" + sha.lower())
        nodes.append(node)
    roots = [groups.find(node) for node in nodes]
    members: dict[str, list[int]] = defaultdict(list)
    for index, root in enumerate(roots):
        members[root].append(index)
    names: dict[str, str] = {}
    for root, indexes in members.items():
        counts = Counter(
            series_key(filenames.get(rows[index].get("doc_id"), "") or "") for index in indexes
        )
        label = counts.most_common(1)[0][0]
        if len(counts) > 1:
            label = f"{label} (+{len(counts) - 1} merged)"
        names[root] = label
    return [names[root] for root in roots]


def stratified_group_split(
    rows: list[dict],
    groups: list[str],
    labels: list[str],
    *,
    val_fraction: float,
    min_label_rows: int,
    seed: int,
) -> tuple[list[dict], list[dict], set[str], list[str]]:
    gold = [row["messages"][-1]["content"] for row in rows]
    total = Counter(gold)
    by_group: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        by_group[group].append(index)
    group_counts = {group: Counter(gold[index] for index in indexes) for group, indexes in by_group.items()}
    rng = random.Random(seed)
    order = sorted(by_group)
    rng.shuffle(order)
    target_rows = round(len(rows) * val_fraction)
    target = {label: val_fraction * total[label] for label in labels}
    val_groups: set[str] = set()
    val_counts: Counter[str] = Counter()
    notes: list[str] = []

    def can_take(group: str) -> bool:
        for label, count in group_counts[group].items():
            if total[label] >= 2 and total[label] - (val_counts[label] + count) < 1:
                return False
        return True

    need = sorted((label for label in labels if total[label] >= min_label_rows), key=lambda label: total[label])
    for label in need:
        if val_counts[label] >= 1:
            continue
        candidates = [
            group
            for group in order
            if group not in val_groups and group_counts[group][label] > 0 and can_take(group)
        ]
        if not candidates:
            notes.append(f"label {label}: no group can enter val without emptying train for a label")
            continue
        chosen = min(candidates, key=lambda group: (sum(group_counts[group].values()), -group_counts[group][label]))
        val_groups.add(chosen)
        val_counts.update(group_counts[chosen])

    def deficit(counter: Counter[str]) -> float:
        score = 0.0
        for label in labels:
            if total[label] < min_label_rows:
                continue
            goal = max(target[label], 1.0)
            score += ((target[label] - counter[label]) / goal) ** 2
        return score

    while sum(val_counts.values()) < target_rows:
        room = target_rows - sum(val_counts.values())
        base = deficit(val_counts)
        best: str | None = None
        best_gain = 0.0
        for group in order:
            if group in val_groups or not can_take(group):
                continue
            size = sum(group_counts[group].values())
            if size > max(room * 1.25, 1):
                continue
            gain = base - deficit(val_counts + group_counts[group])
            if gain > best_gain + 1e-12:
                best, best_gain = group, gain
        if best is None:
            break
        val_groups.add(best)
        val_counts.update(group_counts[best])

    train = [rows[index] for index, group in enumerate(groups) if group not in val_groups]
    val = [rows[index] for index, group in enumerate(groups) if group in val_groups]
    return train, val, val_groups, notes


def cap_label(train: list[dict], label: str, cap: int, seed: int) -> tuple[list[dict], int]:
    indexes = [index for index, row in enumerate(train) if row["messages"][-1]["content"] == label]
    if len(indexes) <= cap:
        return train, len(indexes)
    keep = set(random.Random(seed).sample(indexes, cap))
    capped = [
        row
        for index, row in enumerate(train)
        if row["messages"][-1]["content"] != label or index in keep
    ]
    return capped, len(indexes)


def filenames_for(pool_path: Path) -> dict[str, str]:
    """First non-empty content row per doc id. The name is a group key, not a label."""
    names: dict[str, str] = {}
    for line in pool_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        doc = row.get("doc_id")
        text = row.get("first_512_tokens")
        if doc in names or not isinstance(text, str) or not text.strip():
            continue
        name = row.get("filename") or ""
        names[doc] = name if isinstance(name, str) else ""
    return names


def split_examples(
    examples: list[dict],
    labels: list[str],
    filenames: dict[str, str],
    *,
    val_fraction: float = 0.15,
    min_label_rows: int = 5,
    split_seed: int = 20261006,
    drawing_cap: int = 600,
    cap_seed: int = 20261006,
) -> tuple[list[dict], list[dict], dict]:
    grouped = assign_groups(examples, filenames)
    for example, group in zip(examples, grouped):
        example["group"] = group
    train, val, val_groups, notes = stratified_group_split(
        examples,
        grouped,
        labels,
        val_fraction=val_fraction,
        min_label_rows=min_label_rows,
        seed=split_seed,
    )
    train_groups = {row["group"] for row in train}
    if train_groups & val_groups:
        raise RuntimeError("group in both train and val")
    train_before = len(train)
    train, drawing_before = cap_label(train, DRAWING, drawing_cap, cap_seed)
    random.Random(cap_seed).shuffle(train)

    def counts(rows: list[dict]) -> dict[str, int]:
        return {label: sum(1 for row in rows if row["messages"][-1]["content"] == label) for label in labels}

    kept = counts(examples)
    summary = {
        "split_version": "v2-series-grouped-stratified",
        "group_key": (
            "document series from filename numbering (PKG-AREA-NNN|TYPE|DISC|LVL) "
            "or normalised filename stem; merged on exact filename, exact excerpt, "
            "file_sha256; project_id not in key"
        ),
        "group_key_wall": (
            "WALL: pool has no folder path, so second-level source folder is not "
            "recoverable; document-series proxy used"
        ),
        "val_fraction_target": val_fraction,
        "min_label_rows_for_val_coverage": min_label_rows,
        "split_seed": split_seed,
        "drawing_cap_train": drawing_cap,
        "cap_seed": cap_seed,
        "drawing_train_before_cap": drawing_before,
        "examples": len(examples),
        "groups_total": len(set(grouped)),
        "groups_train": len(train_groups),
        "groups_val": len(val_groups),
        "group_overlap": 0,
        "sealed_isolation": (
            "router.isolation.require_isolated on all examples; sealed set not scored"
        ),
        "train_rows_before_cap": train_before,
        "train_rows": len(train),
        "val_rows": len(val),
        "val_share_of_examples": round(len(val) / len(examples), 4) if examples else 0,
        "label_counts_kept": kept,
        "train_label_counts": counts(train),
        "val_label_counts": counts(val),
        "labels_missing_from_val": [
            label for label in labels if kept[label] >= min_label_rows and counts(val)[label] == 0
        ],
        "notes": notes,
        "val_groups": sorted(val_groups),
        "no_duplication_or_synthesis": True,
    }
    return train, val, summary


def load_isolated_examples(root: Path, labels_path: Path) -> tuple[list[dict], list[str]]:
    """Replay the recorded split on the frozen v5 label list.

    The teacher file uses ``spec``. Pointing this replay at the current
    21-label file would drop those rows. v5 outputs stay byte-identical.
    """
    labels = load_labels(root / "configs" / "router" / "labels_v5.json")
    sealed = load_sealed_config(root / "configs" / "router" / "sealed_eval.json")
    examples = training_examples(rows_from_pool(labels_path), labels)
    isolated = require_isolated(examples, load_exclusions(root / sealed["sealed_labels"]))
    return isolated, labels


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Series-grouped router split. Does not score the sealed set.")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--pool", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--min-label-rows", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=20261006)
    parser.add_argument("--drawing-cap", type=int, default=600)
    parser.add_argument("--cap-seed", type=int, default=20261006)
    args = parser.parse_args(argv)
    root = (args.root or repo_root()).resolve()
    examples, labels = load_isolated_examples(root, args.labels)
    pool_path = args.pool or (root / "data" / "raw" / "router" / "labels.jsonl")
    train, val, summary = split_examples(
        examples,
        labels,
        filenames_for(pool_path),
        val_fraction=args.val_fraction,
        min_label_rows=args.min_label_rows,
        split_seed=args.split_seed,
        drawing_cap=args.drawing_cap,
        cap_seed=args.cap_seed,
    )
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    def write(path: Path, rows: list[dict]) -> None:
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    write(out / "train.jsonl", require_no_paths(train))
    write(out / "val.jsonl", require_no_paths(val))
    (out / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "examples": summary["examples"],
                "groups_total": summary["groups_total"],
                "groups_val": summary["groups_val"],
                "train_rows": summary["train_rows"],
                "val_rows": summary["val_rows"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
