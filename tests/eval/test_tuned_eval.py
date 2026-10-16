import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

from eval.baselines.blocked import BaselineNotRun
from eval.baselines.prompt_only import run_prompt_only_baseline
from eval.baselines.untuned import run_untuned_baseline
from eval.graders import grade_fact_leak, grade_formula_routing, grade_refusal_or_qualification
from eval.labels import label_eval_sets
from eval.stop_gate import SCHEMA
from eval.tuned import TunedEvalError, checkpoint_client_factory, dry_run_tuned_eval, run_tuned_eval
from model.identity import API_KEY_ENV, repo_root
from model.sampling import TinkerKeyMissing
from tests.eval.test_labels import _write_tiny_split

def _plant_routing_registries(root: Path) -> None:
    import json
    reg = root / "data" / "retrieval"
    reg.mkdir(parents=True, exist_ok=True)
    (reg / "formula_registry.json").write_text(
        json.dumps({
            "count": 1,
            "formulas": [{
                "id": "formula.synthetic",
                "definition": {
                    "computes": "synthetic",
                    "inputs": [{"name": "x", "required": True, "annotation": "float"}],
                },
            }],
        }) + "\n",
        encoding="utf-8",
    )
    eqs = root / "eval" / "graders" / "data"
    eqs.mkdir(parents=True, exist_ok=True)
    (eqs / "hds5_equations.json").write_text(
        json.dumps({
            "count": 1,
            "equations": [{
                "id": "HDS5-3.1",
                "number": "3.1",
                "name": "HL = He + Hf",
                "inputs": ["He", "Hf"],
                "pdf_page": 91,
            }],
            "source": {"pdf_pages": [91]},
            "extracted_by": "test",
        }) + "\n",
        encoding="utf-8",
    )



_RESPONSE = '{"block":"refuse","missing_inputs":[],"formula_id":"","required_inputs":[]}'


def test_dry_run_checks_real_data_without_tinker(monkeypatch, capsys):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    imported = []
    real_import = __import__

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        if name.split(".")[0] == "tinker":
            imported.append(name)
            raise AssertionError("tinker import")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", guarded)
    text = dry_run_tuned_eval("tinker://example/weights/checkpoint")
    assert imported == []
    assert "tinker_contacted: no" in text
    assert "scores_written: no" in text
    assert "model_id: Qwen/Qwen3-8B" in text
    assert "held_out: n=69" in text
    assert "blind: n=60" in text
    assert "seals: ok" in text
    assert "routing_registry: attached_for_routing_only" in text
    assert "config: ok" in text
    _assert_no_item_text(text)


def test_dry_run_can_stack_the_prompt_only_system_prompt(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    text = dry_run_tuned_eval(
        "tinker://example/weights/checkpoint",
        prompt_only_system_prompt=True,
    )
    assert "system_prompt: prompt_only" in text
    assert "tinker_contacted: no" in text


def test_missing_key_blocks_before_sampling(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, "   ")
    output = tmp_path / "scores.json"
    with pytest.raises(TinkerKeyMissing):
        run_tuned_eval("tinker://example/weights/checkpoint", output)
    assert not output.exists()


def test_fake_sampler_writes_aggregate_scores_only(tmp_path: Path, monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    root = tmp_path / "repo"
    _write_tiny_split(root, held_count=2, manifest_count=2)
    _plant_routing_registries(root)
    # The second held-out row expects a formula id. The first expects refusal.
    held = root / "data" / "held_out" / "pairs.jsonl"
    rows = [json.loads(line) for line in held.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows[1]["meta"] = {"pair_type": "which_formula_applies", "formula_id": "formula.synthetic"}
    held.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (root / "eval" / "baselines").mkdir(parents=True)
    (root / "eval" / "baselines" / "scores.md").write_text(
        "Prompt-only, held-out, n=2: refusal/qualification 0/2, fact-leak rate 0/2, "
        "formula-routing accuracy 0/2.\n"
        "Prompt-only, blind, n=1: refusal/qualification 0/1, fact-leak rate 0/1, "
        "formula-routing accuracy 0/1.\n",
        encoding="utf-8",
    )
    seen = []

    def sampler(messages):
        seen.append(messages)
        return _RESPONSE

    output = tmp_path / "out" / "scores.json"
    payload = run_tuned_eval(
        "tinker://example/weights/checkpoint",
        output,
        root=root,
        sampler=sampler,
    )
    assert len(seen) == 6
    assert all("role" in message for batch in seen for message in batch)
    assert payload["schema"] == SCHEMA
    assert payload["fork_context"] is False
    assert payload["system_prompt"] == "none"
    assert payload["sets"]["held_out"]["n"] == 2
    assert payload["sets"]["blind"]["n"] == 1
    blob = output.read_text(encoding="utf-8")
    assert "synthetic" not in blob
    assert "formula.synthetic" not in blob
    assert json.loads(blob) == payload
    _assert_matches_graders(root, payload, _RESPONSE)


def test_prompt_only_flag_is_a_system_message_and_not_a_score_field(tmp_path: Path):
    root = tmp_path / "repo"
    _write_tiny_split(root, held_count=1, manifest_count=1)
    _plant_routing_registries(root)
    (root / "eval" / "baselines").mkdir(parents=True)
    (root / "eval" / "baselines" / "scores.md").write_text(
        "Prompt-only, held-out, n=1: refusal/qualification 0, fact-leak rate 0, "
        "formula-routing accuracy 0.\n"
        "Prompt-only, blind, n=1: refusal/qualification 0, fact-leak rate 0, "
        "formula-routing accuracy 0.\n",
        encoding="utf-8",
    )
    seen = []

    def sampler(messages):
        seen.append(messages)
        return _RESPONSE

    payload = run_tuned_eval(
        "tinker://example/weights/checkpoint",
        tmp_path / "scores.json",
        root=root,
        prompt_only_system_prompt=True,
        sampler=sampler,
    )
    assert seen[0][0]["role"] == "system"
    assert "The Fork" in seen[0][0]["content"]
    assert payload["system_prompt"] == "prompt_only"
    assert "State the basis." not in (tmp_path / "scores.json").read_text(encoding="utf-8")


def test_scores_are_not_written_into_sealed_eval(tmp_path: Path):
    def sampler(messages):
        raise AssertionError("sampler called")

    sealed = repo_root() / "data" / "eval" / "unseen" / "tuned.json"
    with pytest.raises(TunedEvalError):
        run_tuned_eval(
            "tinker://example/weights/checkpoint",
            sealed,
            sampler=sampler,
        )
    assert not sealed.exists()


def test_checkpoint_must_not_carry_the_api_key(monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, "super-secret-value")
    with pytest.raises(TunedEvalError) as caught:
        dry_run_tuned_eval("tinker://super-secret-value/weights/checkpoint")
    assert "super-secret-value" not in str(caught.value)


def test_checkpoint_factory_loads_the_tinker_path_once(monkeypatch):
    created = {}

    class Service:
        def create_sampling_client(self, model_path=None, base_model=None):
            created["calls"] = created.get("calls", 0) + 1
            created["model_path"] = model_path
            created["base_model"] = base_model
            return "client"

    module = types.ModuleType("tinker")
    module.ServiceClient = Service
    module.types = types.ModuleType("tinker.types")
    monkeypatch.setitem(sys.modules, "tinker", module)
    factory = checkpoint_client_factory("tinker://example/weights/checkpoint")
    first = factory()
    second = factory()
    assert first == second
    assert first[0] == "client"
    assert created == {
        "calls": 1,
        "model_path": "tinker://example/weights/checkpoint",
        "base_model": None,
    }


def test_baselines_label_and_still_do_not_sample(monkeypatch):
    seen = {"labels": 0}

    def spy(*args, **kwargs):
        seen["labels"] += 1
        return label_eval_sets(*args, **kwargs)

    monkeypatch.setattr("eval.baselines.prompt_only.label_eval_sets", spy)
    monkeypatch.setattr("eval.baselines.untuned.label_eval_sets", spy)
    monkeypatch.setattr(
        "model.sampling.sample_messages",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("sampled")),
    )
    with pytest.raises(BaselineNotRun, match="unseen set"):
        run_prompt_only_baseline()
    with pytest.raises(BaselineNotRun, match="held-out"):
        run_untuned_baseline()
    assert seen["labels"] == 2


def test_cli_dry_run_and_missing_key(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    script = _load_script("run_tuned_eval.py")
    assert script.main(["--checkpoint", "tinker://example/weights/checkpoint", "--dry-run"]) == 0
    dry = capsys.readouterr().out
    assert "tinker_contacted: no" in dry
    _assert_no_item_text(dry)
    output = tmp_path / "scores.json"
    assert script.main(["--checkpoint", "tinker://example/weights/checkpoint", "--output", str(output)]) == 1
    assert not output.exists()
    with pytest.raises(SystemExit) as caught:
        script.main(["--checkpoint", "tinker://example/weights/checkpoint"])
    assert caught.value.code == 2


def _load_script(name: str):
    path = repo_root() / "scripts" / "eval" / name
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_no_item_text(blob: str) -> None:
    root = repo_root()
    for name, items in label_eval_sets(root):
        for item in items:
            if item.prompt and item.prompt in blob:
                raise AssertionError(f"{name} line {item.line} text leaked")


def _assert_matches_graders(root: Path, payload: dict, response: str) -> None:
    for name, items in label_eval_sets(root):
        refusal = 0
        leaked = 0
        routing = 0
        for item in items:
            refusal += int(
                grade_refusal_or_qualification(
                    response,
                    expected_block=item.expected_block,
                    named_inputs=item.named_inputs,
                ).passed
            )
            leaked += int(not grade_fact_leak(response, item.prompt).passed)
            routing += int(
                grade_formula_routing(
                    response,
                    formula_id=item.formula_id,
                    required_inputs=item.required_inputs,
                ).passed
            )
        row = payload["sets"][name]
        assert row["n"] == len(items)
        assert row["refusal_or_qualification_passed"] == refusal
        assert row["fact_leak_leaked"] == leaked
        assert row["formula_routing_passed"] == routing
