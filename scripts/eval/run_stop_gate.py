"""Compare tuned aggregate scores with the prompt-only baseline.

PASS is the only serving outcome. FAIL means no serving.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _entry in (_ROOT / "scripts", _ROOT / "src", _ROOT):
    _text = str(_entry)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from eval.stop_gate import RULE, StopGateError, decide, load_prompt_only_baselines, load_tuned_scores
from model.identity import repo_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stop-gate for a tuned checkpoint")
    parser.add_argument("--scores", type=Path, required=True, help="Aggregate tuned score JSON")
    parser.add_argument(
        "--baselines",
        type=Path,
        default=None,
        help="Published baseline scores. Defaults to eval/baselines/scores.md",
    )
    args = parser.parse_args(argv)
    baseline_path = args.baselines or (repo_root() / "eval" / "baselines" / "scores.md")
    try:
        decision = decide(load_tuned_scores(args.scores), load_prompt_only_baselines(baseline_path))
    except StopGateError as exc:
        print("FAIL")
        print("no serving")
        print(str(exc), file=sys.stderr)
        return 1
    print(decision.format())
    print(RULE)
    return 0 if decision.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
