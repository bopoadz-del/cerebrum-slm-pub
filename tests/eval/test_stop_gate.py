import importlib.util

import pytest

from eval.stop_gate import (
    RULE,
    SCHEMA,
    StopGateError,
    decide,
    load_prompt_only_baselines,
    load_tuned_scores,
)
from model.identity import MODEL_ID, repo_root


def test_gate_refuses_tuned_without_prompt():
    baselines = load_prompt_only_baselines(repo_root() / "eval" / "baselines" / "scores.md")
    payload = _tuned(
        baselines,
        held_refusal=baselines.held_out.refusal_passed + 1,
        held_route=baselines.held_out.routing_passed + 1,
        blind_refusal=baselines.blind.refusal_passed + 1,
        blind_route=20,
    )
    payload["system_prompt"] = "none"
    with pytest.raises(StopGateError, match="served artifact"):
        decide(payload, baselines)


def test_docs_state_the_stop_gate_rule():
    doc = (repo_root() / "eval" / "STOP_GATE.md").read_text(encoding="utf-8")
    assert RULE in doc
    assert "FAIL means no serving" in RULE


def test_published_prompt_only_counts():
    baselines = load_prompt_only_baselines(repo_root() / "eval" / "baselines" / "scores.md")
    assert baselines.held_out.n == 69
    assert baselines.held_out.refusal_passed == 51
    assert baselines.held_out.fact_leak == 0
    assert baselines.held_out.routing_passed == 61
    assert baselines.blind.n == 60
    assert baselines.blind.refusal_passed == 3
    assert baselines.blind.fact_leak == 0
    assert baselines.blind.routing_passed == 17


def test_pass_requires_zero_leaks_and_strict_improvement():
    baselines = load_prompt_only_baselines(repo_root() / "eval" / "baselines" / "scores.md")
    passing = _tuned(baselines, held_refusal=baselines.held_out.refusal_passed + 1, held_route=baselines.held_out.routing_passed + 1, blind_refusal=baselines.blind.refusal_passed + 1, blind_route=20)
    decision = decide(passing, baselines)
    assert decision.passed
    assert decision.format().splitlines()[0] == "PASS"
    assert "serve" in decision.format().splitlines()[1]

    leaks = _tuned(baselines, held_refusal=baselines.held_out.refusal_passed + 1, held_route=baselines.held_out.routing_passed + 1, blind_refusal=baselines.blind.refusal_passed + 1, blind_route=20, blind_leak=1)
    failed = decide(leaks, baselines)
    assert not failed.passed
    assert failed.format().splitlines()[0] == "FAIL"
    assert "no serving" in failed.format()

    tie = _tuned(baselines, held_refusal=baselines.held_out.refusal_passed, held_route=baselines.held_out.routing_passed + 1, blind_refusal=baselines.blind.refusal_passed + 1, blind_route=20)
    assert not decide(tie, baselines).passed

    short = _tuned(
        baselines,
        held_n=baselines.held_out.n - 1,
        held_refusal=baselines.held_out.refusal_passed + 1,
        held_route=baselines.held_out.routing_passed + 1,
        blind_refusal=baselines.blind.refusal_passed + 1,
        blind_route=20,
    )
    assert not decide(short, baselines).passed


def test_score_file_rejects_item_text(tmp_path):
    path = tmp_path / "scores.json"
    baselines = load_prompt_only_baselines(repo_root() / "eval" / "baselines" / "scores.md")
    payload = _tuned(
        baselines,
        held_refusal=baselines.held_out.refusal_passed + 1,
        held_route=baselines.held_out.routing_passed + 1,
        blind_refusal=baselines.blind.refusal_passed + 1,
        blind_route=20,
    )
    payload["question"] = "an item"
    path.write_text(__import__("json").dumps(payload), encoding="utf-8")
    with pytest.raises(StopGateError):
        load_tuned_scores(path)


def test_stop_gate_cli(tmp_path, capsys):
    baselines = load_prompt_only_baselines(repo_root() / "eval" / "baselines" / "scores.md")
    path = tmp_path / "scores.json"
    path.write_text(
        __import__("json").dumps(
            _tuned(baselines, held_refusal=baselines.held_out.refusal_passed + 1, held_route=baselines.held_out.routing_passed + 1, blind_refusal=baselines.blind.refusal_passed + 1, blind_route=20)
        ),
        encoding="utf-8",
    )
    script = _load_script("run_stop_gate.py")
    assert script.main(["--scores", str(path)]) == 0
    passed = capsys.readouterr().out
    assert passed.startswith("PASS\nserve\n")
    assert RULE in passed
    assert "an item" not in passed

    path.write_text(
        __import__("json").dumps(
            _tuned(baselines, held_refusal=baselines.held_out.refusal_passed, held_route=baselines.held_out.routing_passed + 1, blind_refusal=baselines.blind.refusal_passed + 1, blind_route=20)
        ),
        encoding="utf-8",
    )
    assert script.main(["--scores", str(path)]) == 1
    failed = capsys.readouterr().out
    assert failed.startswith("FAIL\nno serving\n")


def _tuned(
    baselines,
    *,
    held_n=None,
    blind_n=None,
    held_refusal=0,
    held_leak=0,
    held_route=0,
    blind_refusal=0,
    blind_leak=0,
    blind_route=0,
):
    return {
        "schema": SCHEMA,
        "model_id": MODEL_ID,
        "checkpoint": "tinker://example/weights/checkpoint",
        "system_prompt": "prompt_only",
        "fork_context": False,
        "sets": {
            "held_out": {
                "n": baselines.held_out.n if held_n is None else held_n,
                "refusal_or_qualification_passed": held_refusal,
                "fact_leak_leaked": held_leak,
                "formula_routing_passed": held_route,
            },
            "blind": {
                "n": baselines.blind.n if blind_n is None else blind_n,
                "refusal_or_qualification_passed": blind_refusal,
                "fact_leak_leaked": blind_leak,
                "formula_routing_passed": blind_route,
            },
        },
    }


def _load_script(name: str):
    path = repo_root() / "scripts" / "eval" / name
    spec = importlib.util.spec_from_file_location(f"cli_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
