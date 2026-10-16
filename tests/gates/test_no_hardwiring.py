from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

from gates.catalog import build_catalog
from gates.no_hardwiring import check_hardwiring
from gates.records import content_hash, pair_view, record_hash


def _copy_zip(tmp_path: Path, repo_root: Path) -> None:
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    shutil.copy(repo_root / "data" / "raw" / "slm-datasets.zip", raw / "slm-datasets.zip")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _first_jsonl(repo_root: Path, suffix: str) -> dict:
    zip_path = repo_root / "data" / "raw" / "slm-datasets.zip"
    with zipfile.ZipFile(zip_path) as archive:
        name = next(info.filename for info in archive.infolist() if info.filename.endswith(suffix))
        line = archive.read(name).decode("utf-8").splitlines()[0]
    return json.loads(line)


def test_catalog_seals_eval_and_leaves_posture_unsealed(repo_root: Path) -> None:
    catalog = build_catalog(repo_root)
    probes = {probe.casefold() for probe in catalog.probe_ids}
    domains = {domain.casefold() for domain in catalog.domains}
    assert "striking_time" in probes
    assert "all" not in probes
    assert "case_1" not in probes
    assert "concrete_structures" in domains
    assert "data_centre" in domains
    assert "governance" not in domains
    assert "datacenter" not in domains

    adversarial = _first_jsonl(repo_root, "concrete/concrete_adversarial.jsonl")
    posture = _first_jsonl(repo_root, "concrete/concrete_pairs.jsonl")
    blind = _first_jsonl(repo_root, "governance/governance_blind.jsonl")
    governed = _first_jsonl(repo_root, "governance/governance.jsonl")
    datacenter = _first_jsonl(repo_root, "datacenter/datacenter_cases.jsonl")

    def content(obj: dict) -> str:
        view = pair_view(obj)
        assert view is not None
        return content_hash(view.user, view.assistant)

    assert content(adversarial) in catalog.content_hashes
    assert record_hash(adversarial) in catalog.record_hashes
    assert content(posture) not in catalog.content_hashes
    assert record_hash(posture) not in catalog.record_hashes
    assert content(blind) in catalog.content_hashes
    assert content(governed) not in catalog.content_hashes
    assert content(datacenter) in catalog.content_hashes
    assert len(catalog.content_hashes) == 74


def test_real_probe_and_domain_red_then_removed(tmp_path: Path, repo_root: Path) -> None:
    _copy_zip(tmp_path, repo_root)
    cheat = tmp_path / "src" / "cheat.py"
    _write(cheat, 'CASE = "striking_time"\n')
    assert any("striking_time" in finding.message for finding in check_hardwiring(tmp_path))
    _write(cheat, 'AREA = "concrete_structures"\n')
    assert any("concrete_structures" in finding.message for finding in check_hardwiring(tmp_path))
    cheat.unlink()
    assert check_hardwiring(tmp_path) == []


def test_real_eval_pair_in_training_red_then_removed(tmp_path: Path, repo_root: Path) -> None:
    _copy_zip(tmp_path, repo_root)
    adversarial = _first_jsonl(repo_root, "concrete/concrete_adversarial.jsonl")
    leaked = tmp_path / "data" / "train" / "leaked.jsonl"
    _write(leaked, json.dumps(adversarial, ensure_ascii=False) + "\n")
    findings = check_hardwiring(tmp_path)
    assert any("record hash matches a sealed eval item" in finding.message for finding in findings)
    leaked.unlink()
    assert check_hardwiring(tmp_path) == []


def test_schema_rewrite_matches_content_hash_then_a_different_answer_passes(tmp_path: Path) -> None:
    sealed = {
        "instruction": "What is the issue?",
        "input": "",
        "output": "Name the missing input.",
        "meta": {
            "domain": "gate_domain_zz",
            "source_case": "gate_probe_zz",
            "eval_only": True,
        },
    }
    _write(
        tmp_path / "data" / "eval" / "blind" / "item.jsonl",
        json.dumps(sealed) + "\n",
    )
    rewritten = {
        "messages": [
            {"role": "system", "content": "posture only"},
            {"role": "user", "content": "What is the issue?"},
            {"role": "assistant", "content": "Name the missing input."},
        ]
    }
    train = tmp_path / "data" / "train" / "rewritten.jsonl"
    _write(train, json.dumps(rewritten) + "\n")
    findings = check_hardwiring(tmp_path)
    assert any("content hash matches a sealed eval item" in finding.message for finding in findings)

    different = {
        "messages": [
            {"role": "user", "content": "What is the issue?"},
            {"role": "assistant", "content": "A different posture answer."},
        ]
    }
    _write(train, json.dumps(different) + "\n")
    assert check_hardwiring(tmp_path) == []


_RETRIEVAL_EXPORTS = (
    "AttachedContext",
    "GroundingVerdict",
    "TurnResult",
    "answer",
    "attach_context",
    "call_formula",
    "call_retrieve",
    "grounding_gate",
)


def _literal(names: tuple[str, ...] | list[str], opener: str, closer: str) -> str:
    body = ",\n".join(f"    {name!r}" for name in names)
    return f"__all__ = {opener}\n{body},\n{closer}\n"


def test_dunder_all_exports_are_not_a_word_list(tmp_path: Path) -> None:
    module = tmp_path / "src" / "retrieval" / "__init__.py"
    _write(module, _literal(_RETRIEVAL_EXPORTS, "[", "]"))
    assert check_hardwiring(tmp_path) == []
    _write(module, _literal(_RETRIEVAL_EXPORTS, "(", ")"))
    assert check_hardwiring(tmp_path) == []
    _write(module, "__all__: list[str] = [\n" + ",\n".join(f"    {name!r}" for name in _RETRIEVAL_EXPORTS) + ",\n]\n")
    assert check_hardwiring(tmp_path) == []


def test_word_lists_of_domains_brands_sites_and_probes_still_fail(tmp_path: Path) -> None:
    for name in ("domains", "brands", "sites", "probe_ids"):
        path = tmp_path / "src" / f"{name}.py"
        _write(path, f'{name} = ["alpha", "beta", "gamma"]\n')
        findings = check_hardwiring(tmp_path)
        assert any(
            f"word list assigned to {name}" in finding.message for finding in findings
        ), name
        path.unlink()

    terms = ", ".join(f'"term{i}"' for i in range(8))
    assigned = tmp_path / "src" / "terms.py"
    _write(assigned, f"values = [{terms}]\n")
    assert any(
        "word list literal with 8 strings" in finding.message
        for finding in check_hardwiring(tmp_path)
    )

    sites = ",\n".join(f'    "North Wharf {i}"' for i in range(8))
    _write(assigned, f"__all__ = [\n{sites},\n]\n")
    assert any(
        "word list literal with 8 strings" in finding.message
        for finding in check_hardwiring(tmp_path)
    )

    both = ",\n".join(f'    "token{i}"' for i in range(8))
    _write(assigned, f"__all__ = domains = [\n{both},\n]\n")
    findings = check_hardwiring(tmp_path)
    assert any("word list assigned to domains" in finding.message for finding in findings)


def test_word_list_in_python_json_and_yaml_then_removed(tmp_path: Path) -> None:
    py_file = tmp_path / "src" / "lexicon.py"
    _write(py_file, 'domains = ["alpha", "beta", "gamma"]\n')
    assert any("word list" in finding.message for finding in check_hardwiring(tmp_path))
    py_file.unlink()

    json_file = tmp_path / "src" / "domains.json"
    _write(json_file, json.dumps(["alpha", "beta", "gamma"]))
    assert any("word list" in finding.message for finding in check_hardwiring(tmp_path))
    json_file.unlink()

    yaml_file = tmp_path / "src" / "domains.yml"
    _write(yaml_file, "- alpha\n- beta\n- gamma\n")
    assert any("word list" in finding.message for finding in check_hardwiring(tmp_path))
    yaml_file.unlink()
    assert check_hardwiring(tmp_path) == []


def test_training_domain_words_and_a_different_case_stay_green(tmp_path: Path, repo_root: Path) -> None:
    _copy_zip(tmp_path, repo_root)
    note = tmp_path / "src" / "note.py"
    _write(note, "# governance of the datacenter stays in retrieval\n# DataCenter\nlabel = \"Case_1\"\n")
    assert check_hardwiring(tmp_path) == []
    _write(note, "# do not special-case striking_time\n")
    assert any("striking_time" in finding.message for finding in check_hardwiring(tmp_path))
    _write(note, "# Striking_Time is not the sealed token\n")
    assert check_hardwiring(tmp_path) == []


def test_plain_word_domain_is_not_sealed_and_an_identifier_domain_is(tmp_path: Path) -> None:
    _write(
        tmp_path / "data" / "eval" / "blind" / "item.jsonl",
        "\n".join(
            [
                json.dumps(
                    {
                        "instruction": "q",
                        "output": "a",
                        "meta": {"domain": "governance", "source_case": "striking_time", "eval_only": True},
                    }
                ),
                json.dumps(
                    {
                        "instruction": "q2",
                        "output": "a2",
                        "meta": {"domain": "concrete_structures", "source_case": "other_probe_zz", "eval_only": True},
                    }
                ),
            ]
        )
        + "\n",
    )
    note = tmp_path / "src" / "note.py"
    _write(note, 'label = "governance"\n')
    assert check_hardwiring(tmp_path) == []
    _write(note, 'label = "concrete_structures"\n')
    assert any("concrete_structures" in finding.message for finding in check_hardwiring(tmp_path))
    _write(note, 'label = "striking_time"\n')
    assert any("striking_time" in finding.message for finding in check_hardwiring(tmp_path))


def test_commented_probe_id_is_still_red(tmp_path: Path) -> None:
    _write(
        tmp_path / "data" / "eval" / "blind" / "item.jsonl",
        json.dumps(
            {
                "instruction": "q",
                "output": "a",
                "meta": {"domain": "gate_domain_zz", "source_case": "gate_probe_zz"},
            }
        )
        + "\n",
    )
    cheat = tmp_path / "src" / "note.py"
    _write(cheat, "# never special-case gate_probe_zz\n")
    assert any("gate_probe_zz" in finding.message for finding in check_hardwiring(tmp_path))
    cheat.unlink()
    assert check_hardwiring(tmp_path) == []
