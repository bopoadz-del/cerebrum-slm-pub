"""Fixed system prompt for the prompt-only baseline.

The prompt is committed. The recorded scores stay in scores.md.
This entry point labels held-out and the blind set with the shared method
and does not sample again.
"""

from __future__ import annotations

from pathlib import Path

from eval.baselines.blocked import BaselineNotRun
from eval.labels import label_eval_sets

_PROMPT_PATH = Path(__file__).resolve().parent / "prompt_only_system_prompt.txt"
_BLOCKED = (
    "The prompt-only baseline is not run. Recorded scores stand. "
    "It does not resample the unseen set or the held-out split."
)


def load_system_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def run_prompt_only_baseline(*_args, **_kwargs):
    label_eval_sets()
    raise BaselineNotRun(_BLOCKED)
