"""Untuned baseline entry point.

Recorded scores stay in scores.md. This entry point labels held-out and
the blind set with the shared method and does not sample again.
"""

from __future__ import annotations

from eval.baselines.blocked import BaselineNotRun
from eval.labels import label_eval_sets

_BLOCKED = (
    "The untuned baseline is not run. Recorded scores stand. "
    "It does not resample the unseen set or the held-out split."
)


def run_untuned_baseline(*_args, **_kwargs):
    label_eval_sets()
    raise BaselineNotRun(_BLOCKED)
