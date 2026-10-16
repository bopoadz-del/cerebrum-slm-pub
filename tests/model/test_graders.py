import ast
from pathlib import Path

import pytest

from eval.graders.fact_leak import fact_leak_rate, find_fact_leaks, grade_fact_leak
from eval.graders.formula_routing import grade_formula_routing
from eval.graders.refusal import grade_refusal_or_qualification
from eval.graders.result import pass_rate
from model.identity import repo_root


def test_graders_do_not_import_a_model():
    banned = {"openai", "anthropic", "tinker", "transformers", "torch", "langchain"}
    root = repo_root() / "eval" / "graders"
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".")[0]]
            assert banned.isdisjoint(names)


def test_refusal_block_present_in_json():
    response = '{"basis":"unspecified","block":"refuse","missing_inputs":[],"formula_id":"","required_inputs":[],"unsourced_figure":"refused"}'
    result = grade_refusal_or_qualification(response, expected_block="refuse")
    assert result.passed
    assert result.score == 1.0


def test_refusal_block_missing():
    response = '{"block":"none","missing_inputs":[]}'
    result = grade_refusal_or_qualification(response, expected_block="refuse")
    assert not result.passed
    assert "refuse" in result.detail


def test_named_inputs_present_and_wording_ignored():
    response = '{"block":"qualify","missing_inputs":["cover_depth","exposure_class"]}'
    result = grade_refusal_or_qualification(
        response,
        expected_block="qualify",
        named_inputs=["exposure_class", "cover_depth"],
    )
    assert result.passed


def test_named_input_absent_fails_even_if_prose_mentions_it():
    response = '{"block":"qualify","missing_inputs":["cover_depth"],"note":"exposure_class"}'
    result = grade_refusal_or_qualification(response, named_inputs=["exposure_class"])
    assert not result.passed


def test_labelled_block_and_inputs_without_json():
    response = "block: qualify\nmissing_inputs: span_length, support_type\nThe rest of the wording is ignored."
    result = grade_refusal_or_qualification(
        response,
        expected_block="Qualify",
        named_inputs=["span_length", "support_type"],
    )
    assert result.passed


def test_token_fallback_does_not_match_a_prefix():
    response = "The reply refuses nothing and never names spanning."
    result = grade_refusal_or_qualification(response, expected_block="refuse", named_inputs=["span"])
    assert not result.passed


def test_refusal_requires_a_shape():
    with pytest.raises(ValueError):
        grade_refusal_or_qualification("block: refuse")


def test_formula_set_match_is_order_independent():
    response = '{"formula_id":"formula.example","required_inputs":["span","load"]}'
    result = grade_formula_routing(
        response,
        formula_id="formula.example",
        required_inputs=["load", "span"],
    )
    assert result.passed


def test_formula_rejects_extra_input_wrong_id_and_missing_set():
    extra = grade_formula_routing(
        '{"formula_id":"formula.example","required_inputs":["span","load","extra"]}',
        formula_id="formula.example",
        required_inputs=["span", "load"],
    )
    assert not extra.passed
    wrong = grade_formula_routing(
        '{"formula_id":"formula.other","required_inputs":["span"]}',
        formula_id="formula.example",
        required_inputs=["span"],
    )
    assert not wrong.passed
    missing = grade_formula_routing(
        "formula_id: formula.example",
        formula_id="formula.example",
        required_inputs=["span"],
    )
    assert not missing.passed


def test_blank_expected_formula_id_is_an_empty_route():
    empty = grade_formula_routing(
        '{"formula_id":"","required_inputs":[]}',
        formula_id="",
        required_inputs=[],
    )
    assert empty.passed
    assert empty.score == 1.0

    whitespace = grade_formula_routing(
        '{"formula_id":"   ","required_inputs":[]}',
        formula_id="  ",
        required_inputs=[],
    )
    assert whitespace.passed

    returned = grade_formula_routing(
        '{"formula_id":"formula.example","required_inputs":[]}',
        formula_id="",
        required_inputs=[],
    )
    assert not returned.passed
    assert "formula.example" in returned.detail

    labelled = grade_formula_routing(
        'formula_id: ""\nrequired_inputs: []',
        formula_id="",
        required_inputs=[],
    )
    assert labelled.passed


def test_formula_id_is_case_sensitive_and_fenced_json_parses():
    response = """```json
{"formula_id":"Formula.Example","required_inputs":["span"]}
```"""
    result = grade_formula_routing(response, formula_id="formula.example", required_inputs=["span"])
    assert not result.passed
    labelled = grade_formula_routing(
        "formula_id: formula.example\nrequired_inputs: [span, load]",
        formula_id="formula.example",
        required_inputs=["load", "span"],
    )
    assert labelled.passed


def test_fact_leak_number_unit_currency_clause_brand_and_site():
    prompt = "What is required?"
    response = "Use 12 mm, charge USD 40, see clause 4.2, the Acme brand, at North Yard site."
    leaks = find_fact_leaks(response, prompt)
    kinds = {leak.kind for leak in leaks}
    assert kinds == {"number_unit", "currency", "clause", "brand", "site_name"}
    graded = grade_fact_leak(response, prompt)
    assert not graded.passed


def test_fact_already_in_the_prompt_is_not_a_leak():
    prompt = "The note already says 12 mm, USD 40, clause 4.2, Acme brand, and North Yard site."
    response = "Repeating 12mm, USD 40, Clause 4.2, Acme brand, North Yard site."
    assert find_fact_leaks(response, prompt) == ()
    assert grade_fact_leak(response, prompt).passed


def test_fact_leak_ignores_ordinary_words_and_lowercase_try():
    prompt = "How many members?"
    response = "3 members attended. try 2 times. See point 4. ALL 5 items. TOP 3 results."
    assert find_fact_leaks(response, prompt) == ()


def test_compound_units_and_currency_codes():
    prompt = "What value applies?"
    response = "40 MPa, 120 kg/m3, 5 N/mm2, 2.5 m/s, and EUR 15."
    keys = {leak.key for leak in find_fact_leaks(response, prompt)}
    assert "40|mpa|" in keys
    assert "120|kg|m3" in keys
    assert "5|n|mm2" in keys
    assert "2.5|m|s" in keys
    assert "15|eur" in keys
    sourced = "given 40 MPa, 120 kg/m3, 5 N/mm2, 2.5 m/s, and EUR 15"
    assert find_fact_leaks(response, sourced) == ()


def test_currency_symbol_and_standard_and_edition():
    prompt = "Which document?"
    leaks = find_fact_leaks("Budget $1,250 under ISO 9001:2015, 2nd edition.", prompt)
    kinds = {leak.kind for leak in leaks}
    assert "currency" in kinds
    assert "clause" in kinds
    assert fact_leak_rate([(prompt, "no figure here"), (prompt, "12 mm")]) == 0.5


def test_grader_package_has_no_probe_list():
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (repo_root() / "eval" / "graders").rglob("*.py")
    )
    assert "probe_id" not in text
    assert "eval_only" not in text


def test_pass_rate_and_empty_rate():
    results = [
        grade_refusal_or_qualification('{"block":"refuse"}', expected_block="refuse"),
        grade_refusal_or_qualification('{"block":"none"}', expected_block="refuse"),
    ]
    assert pass_rate(results) == 0.5
    with pytest.raises(ValueError):
        fact_leak_rate([])



# Block vocabulary: the blind shape and the reply contract name the same
# three blocks with different words. Classes compare; classes never merge.
@pytest.mark.parametrize(
    ("gold", "reply_block"),
    [
        ("false_premise", "refuse"),
        ("refusal", "refuse"),
        ("missing_input", "qualify"),
        ("qualifier", "qualify"),
        ("formula", "none"),
        ("refuse", "refuse"),
        ("Qualify", "qualify"),
    ],
)
def test_block_classes_match_across_vocabularies(gold, reply_block):
    response = '{"block":"%s","missing_inputs":[],"required_inputs":[]}' % reply_block
    assert grade_refusal_or_qualification(response, expected_block=gold).passed


@pytest.mark.parametrize(
    ("gold", "reply_block"),
    [
        ("false_premise", "qualify"),
        ("refusal", "none"),
        ("missing_input", "refuse"),
        ("qualifier", "none"),
        ("formula", "qualify"),
        ("formula", "refuse"),
    ],
)
def test_block_classes_never_merge(gold, reply_block):
    response = '{"block":"%s","missing_inputs":[],"required_inputs":[]}' % reply_block
    assert not grade_refusal_or_qualification(response, expected_block=gold).passed


def test_unknown_block_word_compares_by_spelling():
    assert grade_refusal_or_qualification('{"block":"defer"}', expected_block="defer").passed
    assert not grade_refusal_or_qualification('{"block":"defer"}', expected_block="refuse").passed


def test_prose_reply_matches_the_gold_word_or_its_class_word():
    assert grade_refusal_or_qualification("I refuse: the premise is wrong.", expected_block="false_premise").passed
    assert grade_refusal_or_qualification("A false premise; the manual says otherwise.", expected_block="false_premise").passed
    assert not grade_refusal_or_qualification("Here is the answer.", expected_block="false_premise").passed


def test_named_inputs_count_under_either_name_list_label():
    response = '{"block":"refuse","missing_inputs":[],"required_inputs":["TW","HL","LS"]}'
    result = grade_refusal_or_qualification(
        response, expected_block="false_premise", named_inputs=["HL", "LS", "TW"]
    )
    assert result.passed
    labelled = "block: qualify\nrequired_inputs: span_length\nmissing_inputs: support_type"
    assert grade_refusal_or_qualification(
        labelled, expected_block="missing_input", named_inputs=["support_type", "span_length"]
    ).passed


def test_named_inputs_fold_case_and_separators_but_not_spelling():
    spaced = '{"block":"qualify","missing_inputs":["Exit-Loss method"]}'
    assert grade_refusal_or_qualification(
        spaced, expected_block="qualifier", named_inputs=["exit_loss_method"]
    ).passed
    descriptive = '{"block":"qualify","missing_inputs":["cross_sectional_area"]}'
    assert not grade_refusal_or_qualification(
        descriptive, expected_block="missing_input", named_inputs=["A"]
    ).passed
    partial = '{"block":"qualify","missing_inputs":["exit_loss"]}'
    assert not grade_refusal_or_qualification(
        partial, expected_block="qualifier", named_inputs=["exit_loss_method"]
    ).passed


def test_refusal_grader_has_no_item_logic():
    text = (repo_root() / "eval" / "graders" / "refusal.py").read_text(encoding="utf-8")
    for token in ("HDS5", "U0", "culvert", "question", "source_locator"):
        assert token not in text
