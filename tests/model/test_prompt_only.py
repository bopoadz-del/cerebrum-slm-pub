from eval.baselines.blocked import BaselineNotRun
from eval.baselines.prompt_only import load_system_prompt, run_prompt_only_baseline
from eval.baselines.untuned import run_untuned_baseline


def test_system_prompt_states_the_four_rules_and_uses_the_fork():
    prompt = load_system_prompt()
    assert "State the basis." in prompt
    assert "Name the missing input." in prompt
    assert "Refuse an unsourced figure." in prompt
    assert "Route to a formula." in prompt
    assert "The Fork" in prompt
    assert "The Store" in prompt
    assert "as-designed" in prompt
    assert "as-built" in prompt
    assert "as-operating" in prompt
    assert '"block"' in prompt
    assert '"missing_inputs"' in prompt
    assert '"formula_id"' in prompt
    assert '"required_inputs"' in prompt
    assert "sk-" not in prompt
    assert "TINKER_API_KEY=" not in prompt


def test_baselines_are_not_run():
    try:
        run_prompt_only_baseline()
    except BaselineNotRun as exc:
        assert "unseen set" in str(exc)
        assert "held-out" in str(exc)
    else:
        raise AssertionError("prompt-only baseline ran")
    try:
        run_untuned_baseline()
    except BaselineNotRun as exc:
        assert "unseen set" in str(exc)
        assert "held-out" in str(exc)
    else:
        raise AssertionError("untuned baseline ran")
