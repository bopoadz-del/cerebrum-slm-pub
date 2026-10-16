"""Project-grouped train/val split.

A project id is entirely train or entirely val. Rows are not copied to fill a class.
"""

from __future__ import annotations


def project_grouped_split(
    rows: list[dict],
    *,
    val_fraction: float = 0.15,
) -> tuple[list[dict], list[dict]]:
    """Hold out whole projects until val reaches the fraction, without splitting one."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    groups: dict[str, list[dict]] = {}
    order: list[str] = []
    for row in rows:
        project = row.get("project_id")
        key = project if isinstance(project, str) and project.strip() else ""
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(row)
    # Stable: larger projects first would swallow the budget. Smallest projects
    # fill val, and a project bigger than the remaining budget stays in train.
    order.sort(key=lambda key: (len(groups[key]), key))
    target = int(round(len(rows) * val_fraction))
    val: list[dict] = []
    train: list[dict] = []
    val_ids: set[str] = set()
    for key in order:
        group = groups[key]
        if key and len(val) + len(group) <= max(target, 0) and len(val_ids) < len(order) - 1:
            val.extend(group)
            val_ids.add(key)
        else:
            train.extend(group)
    if not train and val:
        # Never leave train empty by moving every project to val.
        moved = max(val_ids, key=lambda key: len(groups[key]))
        train = list(groups[moved])
        val = [row for key, group in groups.items() if key != moved for row in group]
    return train, val
