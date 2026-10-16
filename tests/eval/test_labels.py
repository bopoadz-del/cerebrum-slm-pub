import ast
import json
from pathlib import Path

from gates.blind_seal import training_open
from gates.records import sha256_file

from eval.labels import (
    BLIND_RELATIVE,
    HELD_OUT_MANIFEST,
    HELD_OUT_RELATIVE,
    EvalLabelError,
    label_blind_record,
    label_eval_sets,
    label_held_out_record,
)
from eval.readers import EvalReadError, read_records
from eval.registry import REGISTRY_RELATIVE, load_registry_inputs
from model.identity import repo_root


def _mismatch(condition: bool, where: str) -> None:
    if not condition:
        raise AssertionError(where)


def test_held_out_pair_type_sets_block_and_route():
    refuse = label_held_out_record(_held("REFUSAL_SCOPE"))
    assert refuse.expected_block == "refuse"
    assert refuse.formula_id == ""
    assert refuse.required_inputs == ()
    assert refuse.named_inputs == ()

    none = label_held_out_record(
        _held("which_formula_applies", formula_id="formula.synthetic")
    )
    assert none.expected_block == "none"
    assert none.formula_id == "formula.synthetic"
    assert none.required_inputs == ()

    qualify = label_held_out_record(
        _held("what_must_be_known_first", formula_id="formula.other")
    )
    assert qualify.expected_block == "qualify"
    assert qualify.formula_id == "formula.other"

    standard = label_held_out_record(
        _held("which_standard_governs", formula_id="formula.ignored")
    )
    assert standard.expected_block == "qualify"
    assert standard.formula_id == ""

    other = label_held_out_record(_held("TRACEABILITY"))
    assert other.expected_block == "qualify"
    assert other.formula_id == ""


def test_held_out_formula_pair_without_an_id_is_rejected():
    try:
        label_held_out_record(_held("why_formula_cannot_answer"))
    except EvalLabelError as exc:
        assert "formula id" in str(exc)
        assert "synthetic held user" not in str(exc)
    else:
        raise AssertionError("missing formula id was accepted")


def test_blind_shape_labels_refusal_and_route():
    empty = label_blind_record(
        {
            "question": "synthetic blind question",
            "shape": {
                "expected_block": "qualifier",
                "named_inputs": ["switch_name"],
                "formula_id": None,
            },
        }
    )
    assert empty.expected_block == "qualifier"
    assert empty.named_inputs == ("switch_name",)
    assert empty.formula_id == ""
    assert empty.required_inputs == ()

    routed = label_blind_record(
        {
            "question": "synthetic routed question",
            "shape": {
                "expected_block": "formula",
                "named_inputs": ["width", "rise"],
                "formula_id": "eq-1",
            },
        }
    )
    assert routed.formula_id == "eq-1"
    assert routed.required_inputs == ("width", "rise")
    assert "synthetic" not in repr(routed)


def test_real_splits_use_the_same_label_method(repo_root: Path):
    sets = dict(label_eval_sets(repo_root))
    inputs = load_registry_inputs(repo_root)
    held_rows = read_records(repo_root, HELD_OUT_RELATIVE)
    blind_rows = read_records(repo_root, BLIND_RELATIVE)
    _mismatch(len(sets["held_out"]) == len(held_rows), "held-out count")
    _mismatch(len(sets["blind"]) == len(blind_rows), "blind count")
    refuse = {
        "REFUSAL_SCOPE",
        "REFUSAL_VS_VACUUM",
        "FORCING_THE_NUMBER",
        "bare_rate_not_answer",
        "why_formula_cannot_answer",
    }
    formula = {
        "which_formula_applies",
        "what_must_be_known_first",
        "why_formula_cannot_answer",
    }
    for label, (line, row) in zip(sets["held_out"], held_rows, strict=True):
        pair_type = row["meta"]["pair_type"]
        if pair_type in refuse:
            block = "refuse"
        elif pair_type == "which_formula_applies":
            block = "none"
        else:
            block = "qualify"
        _mismatch(label.expected_block == block, f"held-out line {line} block")
        _mismatch(label.named_inputs == (), f"held-out line {line} names")
        expected_inputs = inputs.get(row["meta"]["formula_id"].strip(), ()) if pair_type in formula else ()
        _mismatch(label.required_inputs == tuple(expected_inputs), f"held-out line {line} inputs")
        if pair_type in formula:
            _mismatch(label.formula_id == row["meta"]["formula_id"].strip(), f"held-out line {line} route")
        else:
            _mismatch(label.formula_id == "", f"held-out line {line} empty route")
        _mismatch(label.prompt == row["user"], f"held-out line {line} prompt")
        _mismatch(label.prompt not in repr(label), f"held-out line {line} repr")
    for label, (line, row) in zip(sets["blind"], blind_rows, strict=True):
        shape = row["shape"]
        _mismatch(label.expected_block == shape["expected_block"].strip(), f"blind line {line} block")
        _mismatch(
            label.named_inputs == tuple(item.strip() for item in shape["named_inputs"]),
            f"blind line {line} names",
        )
        raw_id = shape["formula_id"]
        if raw_id is None or (isinstance(raw_id, str) and not raw_id.strip()):
            _mismatch(label.formula_id == "" and label.required_inputs == (), f"blind line {line} empty route")
        else:
            _mismatch(label.formula_id == raw_id.strip(), f"blind line {line} formula")
            _mismatch(label.required_inputs == label.named_inputs, f"blind line {line} inputs")
        _mismatch(label.prompt == row["question"], f"blind line {line} prompt")
        _mismatch(row["question"] not in repr(label), f"blind line {line} repr")


def test_sealed_reader_rejects_a_bad_seal_and_training_open(tmp_path: Path):
    relative = "data/eval/unseen/questions.jsonl"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_text('{"question":"synthetic sealed question"}\n', encoding="utf-8")
    try:
        read_records(tmp_path, relative)
    except EvalReadError as exc:
        assert "synthetic sealed question" not in str(exc)
    else:
        raise AssertionError("unsealed eval was read")
    try:
        training_open(path, tmp_path)
    except Exception as exc:
        assert "synthetic sealed question" not in str(exc)
    else:
        raise AssertionError("training reader opened sealed eval")

    digest = sha256_file(path)
    path.with_name(path.name + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    rows = read_records(tmp_path, relative)
    assert len(rows) == 1
    assert rows[0][0] == 1


def test_eval_sources_do_not_call_the_training_reader():
    root = repo_root()
    files = list((root / "eval").rglob("*.py"))
    files.extend((root / "scripts" / "eval").rglob("*.py"))
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            name = None
            if isinstance(node, ast.Name):
                name = node.id
            elif isinstance(node, ast.Attribute):
                name = node.attr
            assert name != "training_open"


def test_label_sources_do_not_embed_eval_item_text():
    root = repo_root()
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in list((root / "eval").rglob("*.py")) + list((root / "scripts" / "eval").rglob("*.py"))
    )
    for relative in (HELD_OUT_RELATIVE, BLIND_RELATIVE):
        for line, row in read_records(root, relative):
            text = row.get("user") or row.get("question") or ""
            if text and text in source:
                raise AssertionError(f"item text from line {line} is in source")


def _held(pair_type: str, formula_id: str | None = None) -> dict:
    meta = {"pair_type": pair_type}
    if formula_id is not None:
        meta["formula_id"] = formula_id
    return {
        "system": "",
        "user": "synthetic held user",
        "assistant": "synthetic held assistant",
        "meta": meta,
    }


def test_manifest_count_must_match(tmp_path: Path):
    _write_tiny_split(tmp_path, held_count=1, manifest_count=2)
    try:
        label_eval_sets(tmp_path)
    except EvalLabelError as exc:
        assert "manifest" in str(exc)
        assert "synthetic" not in str(exc)
    else:
        raise AssertionError("manifest mismatch was accepted")


def _write_tiny_split(root: Path, held_count: int, manifest_count: int) -> None:
    held = root / HELD_OUT_RELATIVE
    held.parent.mkdir(parents=True)
    rows = [
        json.dumps(
            {
                "system": "",
                "user": f"synthetic held user {index}",
                "assistant": "synthetic assistant",
                "meta": {"pair_type": "REFUSAL_SCOPE"},
            }
        )
        for index in range(held_count)
    ]
    held.write_text("\n".join(rows) + "\n", encoding="utf-8")
    manifest = root / HELD_OUT_MANIFEST
    manifest.write_text(json.dumps({"count": manifest_count}) + "\n", encoding="utf-8")
    registry = root / REGISTRY_RELATIVE
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(json.dumps({"formulas": []}) + "\n", encoding="utf-8")
    blind = root / BLIND_RELATIVE
    blind.parent.mkdir(parents=True)
    blind.write_text(
        json.dumps(
            {
                "question": "synthetic blind question",
                "shape": {
                    "expected_block": "refusal",
                    "named_inputs": ["span"],
                    "formula_id": None,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    digest = sha256_file(blind)
    blind.with_name(blind.name + ".sha256").write_text(f"{digest}  {blind.name}\n", encoding="utf-8")


def test_held_out_route_expects_the_registry_inputs(tmp_path: Path):
    _write_tiny_split(tmp_path, held_count=3, manifest_count=3)
    held = tmp_path / HELD_OUT_RELATIVE
    rows = [json.loads(line) for line in held.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows[0]["meta"] = {"pair_type": "what_must_be_known_first", "formula_id": "formula.synthetic"}
    rows[1]["meta"] = {"pair_type": "which_formula_applies", "formula_id": "formula.unlisted"}
    held.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    registry = {
        "formulas": [
            {
                "id": "formula.synthetic",
                "definition": {
                    "inputs": [
                        {"name": "span", "required": True},
                        {"name": "optional_factor", "required": False},
                        {"name": "load", "required": True},
                    ]
                },
            }
        ]
    }
    (tmp_path / REGISTRY_RELATIVE).write_text(json.dumps(registry), encoding="utf-8")
    labels = dict(label_eval_sets(tmp_path))["held_out"]
    assert labels[0].required_inputs == ("span", "load")
    assert labels[1].required_inputs == ()
    assert labels[2].formula_id == "" and labels[2].required_inputs == ()


def test_training_targets_and_held_out_labels_share_the_registry_rule(repo_root: Path):
    import json as _json

    from train.posture_target import registry_inputs as target_inputs

    registry = _json.loads((repo_root / REGISTRY_RELATIVE).read_text(encoding="utf-8"))
    assert target_inputs(registry) == load_registry_inputs(repo_root)
