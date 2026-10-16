"""Sample base or prompt-only under the routing-registry protocol.

Writes the same aggregate schema as ``run_tuned_eval.py``. Refusal/qualification
and fact-leak are context-free; routing attaches the split registry and uses
constrained decoding on formula ids. Uses one shared sampling client.
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

from eval.baselines.prompt_only import load_system_prompt
from eval.readers import EvalReadError
from eval.stop_gate import StopGateError
from eval.tuned import (
    TunedEvalError,
    format_scores,
    format_shard_scores,
    run_shard_eval,
    run_tuned_eval,
)
from model.sampling import TinkerKeyMissing, default_client_factory, sample_messages


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Grade base or prompt-only under the new protocol")
    parser.add_argument("--output", type=Path, required=True, help="Aggregate score JSON")
    parser.add_argument(
        "--prompt-only-system-prompt",
        action="store_true",
        help="Use the committed prompt-only system prompt",
    )
    _cli().add_shard_arguments(parser)
    args = parser.parse_args(argv)

    # One shared client (avoid per-sample ServiceClient / connection leak).
    built: dict = {}

    def factory():
        if "pair" not in built:
            built["pair"] = default_client_factory()
        return built["pair"]

    def sampler(messages):
        return sample_messages(messages, client_factory=factory)

    # Checkpoint field records the base model id for baseline runs.
    checkpoint = "base://Qwen/Qwen3-8B"
    try:
        if args.shard:
            payload = run_shard_eval(
                checkpoint,
                args.output,
                args.shard,
                prompt_only_system_prompt=args.prompt_only_system_prompt,
                sampler=sampler,
                item_results=args.item_results,
                replies=args.replies,
            )
            print(format_shard_scores(payload))
            print(f"wrote {args.output}")
            return 0
        payload = run_tuned_eval(
            checkpoint,
            args.output,
            prompt_only_system_prompt=args.prompt_only_system_prompt,
            sampler=sampler,
        )
    except (TunedEvalError, TinkerKeyMissing, EvalReadError, StopGateError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(format_scores(payload))
    print(f"wrote {args.output}")
    _cli().print_shard_lines(args.output)
    return 0


def _cli():
    """Shared --shard arguments and shard-line printing from run_tuned_eval.py."""
    import importlib.util

    path = Path(__file__).resolve().parent / "run_tuned_eval.py"
    spec = importlib.util.spec_from_file_location("run_tuned_eval_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    raise SystemExit(main())
