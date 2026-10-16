"""Sample a tuned checkpoint and write aggregate scores.

No item text is printed or written. A dry-run checks data and config and
does not contact Tinker. Sampling uses the committed Qwen3-8B parameters
and a Tinker sampling client pointed at the checkpoint path.

Protocol: refusal/qualification and fact-leak are sampled context-free.
Formula routing is sampled with the split's formula registry attached and
constrained decoding on formula ids from that registry. Bare routing is
refused.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from eval.baselines.prompt_only import load_system_prompt
from eval.graders import (
    find_fact_leaks,
    grade_fact_leak,
    grade_formula_routing,
    grade_refusal_or_qualification,
)
from eval.labels import EvalLabel, EvalLabelError, label_eval_sets
from eval.readers import EvalReadError
from eval.routing_registry import (
    RoutingRegistry,
    RoutingRegistryError,
    constrain_formula_response,
    load_routing_registry,
    require_routing_registry,
)
from eval.shards import (
    HeldOutShard,
    ShardError,
    added_shards,
    label_shard,
    load_shards,
    select_shards,
    shards_listed,
)
from eval.stop_gate import SCHEMA, PromptOnlyBaselines, StopGateError, load_prompt_only_baselines
from gates.blind_seal import path_is_sealed
from model.identity import API_KEY_ENV, MODEL_ID, load_model_config, repo_root
from model.sampling import (
    TinkerKeyMissing,
    build_messages,
    sample_messages,
    tinker_api_key_present,
)

Sampler = Callable[..., str]
_BLOCK_TOKEN = re.compile(r"[A-Za-z_]{1,32}\Z")
_BASELINE_FILE = Path("eval") / "baselines" / "scores.md"
SHARD_SCHEMA = "held_out_shard_scores/v1"
ITEM_SCHEMA = "held_out_shard_items/v1"


class TunedEvalError(RuntimeError):
    """The tuned run cannot start. The message has no item text."""


@dataclass(frozen=True)
class SetTally:
    n: int
    refusal_passed: int
    fact_leak_leaked: int
    routing_passed: int


def checkpoint_client_factory(checkpoint: str) -> Callable[[], tuple[Any, Any]]:
    """Build one sampling client for a saved Tinker checkpoint and reuse it."""
    if not checkpoint.strip():
        raise TunedEvalError("checkpoint path is empty")
    built: dict[str, tuple[Any, Any]] = {}

    def factory() -> tuple[Any, Any]:
        if "pair" not in built:
            import tinker
            from tinker import types

            service = tinker.ServiceClient()
            client = service.create_sampling_client(model_path=checkpoint)
            built["pair"] = (client, types)
        return built["pair"]

    return factory


def dry_run_tuned_eval(
    checkpoint: str,
    *,
    root: Path | None = None,
    prompt_only_system_prompt: bool = False,
) -> str:
    """Check data and config. Do not sample and do not import Tinker."""
    base = _prepare(checkpoint, root, prompt_only_system_prompt)
    config = load_model_config()
    if config.get("model_id") != MODEL_ID:
        raise TunedEvalError("model config is not the configured instruct checkpoint")
    lines = [
        "tuned eval dry-run",
        f"model_id: {config['model_id']}",
        "checkpoint: accepted",
        f"system_prompt: {'prompt_only' if prompt_only_system_prompt else 'none'}",
        "routing_registry: attached_for_routing_only",
        "tinker_contacted: no",
        "scores_written: no",
    ]
    for name, items in base.sets:
        lines.append(_split_summary(name, items))
        registry = base.registries[name]
        lines.append(f"{name}_registry: entries={len(registry.entries)} source={registry.source}")
    lines.append("seals: ok")
    lines.extend(_shard_summary(base.root))
    lines.append("config: ok")
    return "\n".join(lines)


def _shard_summary(root: Path) -> list[str]:
    """One dry-run line per listed held-out shard. Counts only."""
    if not shards_listed(root):
        return []
    try:
        shards = load_shards(root)
        lines = []
        for shard in shards:
            items = label_shard(root, shard)
            role = "original" if shard.original else "added"
            frozen = "frozen" if shard.frozen else "open"
            lines.append(_split_summary(f"shard {shard.name} ({role}, {frozen}, sha256 ok)", items))
    except ShardError as exc:
        raise TunedEvalError(str(exc)) from None
    return lines


def run_tuned_eval(
    checkpoint: str,
    output: Path,
    *,
    root: Path | None = None,
    prompt_only_system_prompt: bool = False,
    sampler: Sampler | None = None,
    score_added_shards: bool = True,
    shard_output: Path | None = None,
) -> dict:
    """Sample held-out and the blind set, grade, and write aggregate scores.

    The stop-gate payload (held_out and blind) is written to ``output`` as
    before. When ``data/held_out/shards.json`` lists added shards, each one is
    sampled after the stop-gate sets and written to ``shard_output`` (default
    ``<output stem>.shards.json``) with one row per listed shard; the original
    shard's row is the held_out tally, not a second sample.
    """
    base = _prepare(checkpoint, root, prompt_only_system_prompt)
    _reject_sealed_output(output, base.root)
    system = base.system_prompt
    if sampler is None:
        if not tinker_api_key_present():
            raise TinkerKeyMissing(
                "Sampling is blocked: TINKER_API_KEY is not set in the environment. "
                "The key is not read from the repo."
            )
        factory = checkpoint_client_factory(checkpoint.strip())

        def sample(messages: list[dict[str, str]]) -> str:
            return sample_messages(messages, client_factory=factory)

    else:
        sample = sampler

    tallies: dict[str, SetTally] = {}
    for name, items in base.sets:
        registry = require_routing_registry(base.registries[name])
        tallies[name] = _score_items(name, items, registry, sample, system)

    payload = _public_scores(
        checkpoint.strip(),
        "prompt_only" if prompt_only_system_prompt else "none",
        tallies,
    )
    _write_scores(output, payload)
    if score_added_shards and shards_listed(base.root):
        # Every added shard in data/held_out/shards.json is scored after the
        # stop-gate sets and reported as its own line. The stop-gate payload
        # above is unchanged; the shard lines go to their own file.
        try:
            shards = load_shards(base.root)
        except ShardError as exc:
            raise TunedEvalError(str(exc)) from None
        extra = added_shards(shards)
        if extra:
            shard_payload = _score_shards(
                checkpoint.strip(),
                base.root,
                (tuple(shards), extra),
                sample,
                system,
                original_tally=tallies["held_out"],
            )
            _write_scores(shard_output or _default_shard_output(output), shard_payload)
    return payload


def run_shard_eval(
    checkpoint: str,
    output: Path,
    shard_names,
    *,
    root: Path | None = None,
    prompt_only_system_prompt: bool = False,
    sampler: Sampler | None = None,
    item_results: Path | None = None,
    replies: Path | None = None,
) -> dict:
    """Sample only the named held-out shards and write one line per shard.

    Same labels, graders, system prompt, sampling config and dual-sample
    routing protocol as ``run_tuned_eval``. The stop-gate sets are not sampled.
    ``item_results`` gets per-item grades with no item or reply text.
    ``replies`` (raw reply text) must sit outside the repo.
    """
    if not str(checkpoint).strip():
        raise TunedEvalError("checkpoint path is empty")
    _reject_key_in_checkpoint(checkpoint)
    base = repo_root() if root is None else root
    for target in (output, item_results):
        if target is not None:
            _reject_sealed_output(target, base)
    if replies is not None and _inside(replies, base):
        raise TunedEvalError("raw replies are not written into the repo")
    try:
        shards = load_shards(base)
        chosen = select_shards(shards, shard_names)
    except ShardError as exc:
        raise TunedEvalError(str(exc)) from None
    if not chosen:
        raise TunedEvalError("no shard was named")
    system = None
    if prompt_only_system_prompt:
        system = load_system_prompt()
        if not system.strip():
            raise TunedEvalError("prompt-only system prompt is empty")
    sample = _sampler_for(checkpoint, sampler)
    items_out: list[dict] = []
    raw_out: list[dict] = []
    payload = _score_shards(
        checkpoint.strip(),
        base,
        (shards, chosen),
        sample,
        system,
        item_sink=items_out,
        reply_sink=raw_out if replies is not None else None,
    )
    _write_scores(output, payload)
    if item_results is not None:
        _write_scores(
            item_results,
            {
                "schema": ITEM_SCHEMA,
                "checkpoint": payload["checkpoint"],
                "system_prompt": payload["system_prompt"],
                "items": items_out,
            },
        )
    if replies is not None:
        replies.parent.mkdir(parents=True, exist_ok=True)
        replies.write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in raw_out),
            encoding="utf-8",
        )
    return payload


def format_shard_scores(payload: dict) -> str:
    """One public line per shard. Counts only."""
    lines = []
    ordered = sorted(payload["shards"].items(), key=lambda item: (not item[1].get("original"), item[0]))
    for name, row in ordered:
        lines.append(
            f"shard {name}: n={row['n']} "
            f"refusal/qualification {row['refusal_or_qualification_passed']}/{row['n']} "
            f"fact-leak {row['fact_leak_leaked']}/{row['n']} "
            f"formula-routing {row['formula_routing_passed']}/{row['n']}"
        )
    return "\n".join(lines)


def _sampler_for(checkpoint: str, sampler: Sampler | None) -> Sampler:
    if sampler is not None:
        return sampler
    if not tinker_api_key_present():
        raise TinkerKeyMissing(
            "Sampling is blocked: TINKER_API_KEY is not set in the environment. "
            "The key is not read from the repo."
        )
    factory = checkpoint_client_factory(checkpoint.strip())

    def sample(messages: list[dict[str, str]]) -> str:
        return sample_messages(messages, client_factory=factory)

    return sample


def _score_items(
    name: str,
    items,
    registry: RoutingRegistry,
    sample: Sampler,
    system: str | None,
    *,
    item_sink: list[dict] | None = None,
    reply_sink: list[dict] | None = None,
) -> SetTally:
    refusal_passed = 0
    leaked = 0
    routing_passed = 0
    for label in items:
        try:
            # Context-free sample: refusal/qualification and fact-leak.
            response_cf = sample(build_messages(label.prompt, system_prompt=system))
            # Routing sample: formula registry attached; constrain ids.
            response_rt = sample(
                build_messages(
                    label.prompt,
                    system_prompt=system,
                    fork_context=registry.format_context(),
                )
            )
        except TinkerKeyMissing:
            raise
        except TunedEvalError:
            raise
        except RoutingRegistryError as exc:
            raise TunedEvalError(str(exc)) from None
        except Exception:
            raise TunedEvalError(f"{name} line {label.line} sample failed") from None
        if not isinstance(response_cf, str) or not isinstance(response_rt, str):
            raise TunedEvalError(f"{name} line {label.line} sample was not text")
        refusal, leak, route = _grade(label, response_cf, response_rt, registry)
        refusal_passed += int(refusal.passed)
        leaked += int(not leak.passed)
        routing_passed += int(route.passed)
        if item_sink is not None:
            item_sink.append(_item_row(name, label, refusal, leak, route, response_cf, response_rt))
        if reply_sink is not None:
            reply_sink.append(
                {
                    "shard": name,
                    "line": label.line,
                    "reply_context_free": response_cf,
                    "reply_routing": response_rt,
                }
            )
    return SetTally(len(items), refusal_passed, leaked, routing_passed)


def _score_shards(
    checkpoint: str,
    root: Path,
    listed_and_chosen: tuple[tuple[HeldOutShard, ...], tuple[HeldOutShard, ...]],
    sample: Sampler,
    system: str | None,
    *,
    original_tally: SetTally | None = None,
    item_sink: list[dict] | None = None,
    reply_sink: list[dict] | None = None,
) -> dict:
    listed, chosen = listed_and_chosen
    rows: dict[str, dict] = {}
    for shard in listed:
        if shard.original and original_tally is not None:
            # The original shard is the stop-gate held_out set already scored.
            rows[shard.name] = _tally_row(original_tally, shard)
            continue
        if shard not in chosen:
            continue
        try:
            items = label_shard(root, shard)
            registry = require_routing_registry(load_routing_registry(shard.routing_registry, root))
        except (ShardError, RoutingRegistryError) as exc:
            raise TunedEvalError(str(exc)) from None
        tally = _score_items(
            shard.name, items, registry, sample, system, item_sink=item_sink, reply_sink=reply_sink
        )
        rows[shard.name] = _tally_row(tally, shard)
    return {
        "schema": SHARD_SCHEMA,
        "model_id": MODEL_ID,
        "checkpoint": checkpoint,
        "system_prompt": "prompt_only" if system else "none",
        "fork_context": False,
        "shards": rows,
    }


def _tally_row(tally: SetTally, shard: HeldOutShard) -> dict:
    return {
        "n": tally.n,
        "refusal_or_qualification_passed": tally.refusal_passed,
        "fact_leak_leaked": tally.fact_leak_leaked,
        "formula_routing_passed": tally.routing_passed,
        "shard_sha256": shard.sha256,
        "original": shard.original,
    }


def _item_row(name, label, refusal, leak, route, response_cf: str, response_rt: str) -> dict:
    """Per-item grades. No item text and no reply text."""
    kinds = sorted({found.kind for found in find_fact_leaks(response_cf, label.prompt)})
    return {
        "shard": name,
        "line": label.line,
        "expected_block": label.expected_block,
        "route_expected": bool(label.formula_id),
        "refusal_or_qualification_passed": bool(refusal.passed),
        "fact_leak_passed": bool(leak.passed),
        "fact_leak_kinds": kinds,
        "formula_routing_passed": bool(route.passed),
        "reply_context_free_sha256": _text_sha256(response_cf),
        "reply_routing_sha256": _text_sha256(response_rt),
    }


def _text_sha256(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _default_shard_output(output: Path) -> Path:
    return output.with_name(output.stem + ".shards.json")


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def format_scores(payload: dict) -> str:
    """One public line per set. Counts only."""
    lines = [
        "tuned eval scores",
        f"model_id: {payload['model_id']}",
        f"system_prompt: {payload['system_prompt']}",
        "fork_context: no",
        "routing_registry: attached_for_routing_only",
    ]
    for name in ("held_out", "blind"):
        row = payload["sets"][name]
        lines.append(
            f"{name}: n={row['n']} "
            f"refusal/qualification {row['refusal_or_qualification_passed']}/{row['n']} "
            f"fact-leak {row['fact_leak_leaked']}/{row['n']} "
            f"formula-routing {row['formula_routing_passed']}/{row['n']}"
        )
    return "\n".join(lines)


@dataclass(frozen=True)
class _Prepared:
    root: Path
    sets: tuple[tuple[str, tuple[EvalLabel, ...]], ...]
    system_prompt: str | None
    baselines: PromptOnlyBaselines
    registries: dict[str, RoutingRegistry]


def _prepare(
    checkpoint: str,
    root: Path | None,
    prompt_only_system_prompt: bool,
) -> _Prepared:
    if not str(checkpoint).strip():
        raise TunedEvalError("checkpoint path is empty")
    _reject_key_in_checkpoint(checkpoint)
    base = repo_root() if root is None else root
    try:
        sets = label_eval_sets(base)
        baselines = load_prompt_only_baselines(base / _BASELINE_FILE)
        registries = {
            "held_out": load_routing_registry("held_out", base),
            "blind": load_routing_registry("blind", base),
        }
    except (EvalReadError, EvalLabelError, StopGateError, RoutingRegistryError) as exc:
        raise TunedEvalError(str(exc)) from None
    for name, items in sets:
        published = baselines.for_split(name)
        if len(items) != published.n:
            raise TunedEvalError(
                f"{name} count {len(items)} does not match the prompt-only baseline n"
            )
        require_routing_registry(registries[name])
    system = None
    if prompt_only_system_prompt:
        system = load_system_prompt()
        if not system.strip():
            raise TunedEvalError("prompt-only system prompt is empty")
    return _Prepared(base, sets, system, baselines, registries)


def _grade(
    label: EvalLabel,
    response_context_free: str,
    response_routing: str,
    registry: RoutingRegistry,
):
    try:
        require_routing_registry(registry)
        refusal = grade_refusal_or_qualification(
            response_context_free,
            expected_block=label.expected_block,
            named_inputs=label.named_inputs,
        )
        leak = grade_fact_leak(response_context_free, label.prompt)
        constrained = constrain_formula_response(response_routing, registry)
        route = grade_formula_routing(
            constrained,
            formula_id=label.formula_id,
            required_inputs=label.required_inputs,
        )
    except (ValueError, RoutingRegistryError):
        raise TunedEvalError(f"{label.split} line {label.line} cannot be graded") from None
    return refusal, leak, route


def grade_routing_or_raise(
    response: str,
    *,
    formula_id: str,
    required_inputs,
    registry: RoutingRegistry | None,
):
    """Public routing grade path. Fails closed when no registry is attached."""
    require_routing_registry(registry)
    constrained = constrain_formula_response(response, registry)
    return grade_formula_routing(
        constrained,
        formula_id=formula_id,
        required_inputs=required_inputs,
    )


def _public_scores(checkpoint: str, system_prompt: str, tallies: dict[str, SetTally]) -> dict:
    payload = {
        "schema": SCHEMA,
        "model_id": MODEL_ID,
        "checkpoint": checkpoint,
        "system_prompt": system_prompt,
        "fork_context": False,
        "sets": {
            name: {
                "n": tally.n,
                "refusal_or_qualification_passed": tally.refusal_passed,
                "fact_leak_leaked": tally.fact_leak_leaked,
                "formula_routing_passed": tally.routing_passed,
            }
            for name, tally in tallies.items()
        },
    }
    if set(payload["sets"]) != {"held_out", "blind"}:
        raise TunedEvalError("scores must name held_out and blind")
    return payload


def _write_scores(output: Path, payload: dict) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def _reject_sealed_output(output: Path, root: Path) -> None:
    if path_is_sealed(output, root):
        raise TunedEvalError("scores are not written into sealed eval")


def _reject_key_in_checkpoint(checkpoint: str) -> None:
    key = os.environ.get(API_KEY_ENV, "").strip()
    if key and key in checkpoint:
        raise TunedEvalError("checkpoint path must not contain the API key")


def _split_summary(name: str, items: tuple[EvalLabel, ...]) -> str:
    counts: Counter[str] = Counter(_public_block(item.expected_block) for item in items)
    blocks = " ".join(f"{key}={counts[key]}" for key in sorted(counts))
    formula_expected = sum(1 for item in items if item.formula_id)
    return f"{name}: n={len(items)} refusal_blocks {blocks} formula_id_expected={formula_expected}"


def _public_block(token: str) -> str:
    if _BLOCK_TOKEN.fullmatch(token):
        return token
    return "other"
