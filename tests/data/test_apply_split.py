"""Invariants for the Split. Domain names live in data files, not in this module."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "data" / "apply_split.py"


def load_split():
    spec = importlib.util.spec_from_file_location("apply_split", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


split = load_split()


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_zip(path: Path, files: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, body in files.items():
            archive.writestr(f"slm-datasets/{name}", body)


class FactDetectorTests(unittest.TestCase):
    def test_unit_currency_and_edition_are_facts(self) -> None:
        lexicon = {"currencies": ["XYZ"], "sites": ["North Dock"]}
        self.assertTrue(split.assistant_has_fact("Cover is 40 mm.", lexicon))
        self.assertTrue(split.assistant_has_fact("Allow 12 hours.", lexicon))
        self.assertTrue(split.assistant_has_fact("Use 5%.", lexicon))
        self.assertTrue(split.assistant_has_fact("The source names QQ 1234.", lexicon))
        self.assertTrue(split.assistant_has_fact("Price is 10 XYZ.", lexicon))
        self.assertTrue(split.assistant_has_fact("See North Dock for the figure.", lexicon))
        self.assertTrue(split.assistant_has_fact("Cite clause 4 before answering.", lexicon))

    def test_posture_wording_is_not_a_fact(self) -> None:
        lexicon = {"currencies": ["XYZ"], "sites": ["North Dock"]}
        self.assertFalse(split.assistant_has_fact("State the missing input before answering.", lexicon))
        self.assertFalse(split.assistant_has_fact("Name the 2 missing inputs.", lexicon))
        self.assertFalse(split.assistant_has_fact("A figure with no source is not an answer.", lexicon))
        self.assertFalse(split.assistant_has_fact("It must name the specification clause.", lexicon))
        self.assertFalse(split.assistant_has_fact("Not necessarily the cause.", lexicon))


class SyntheticSplitTests(unittest.TestCase):
    def _files(self) -> dict[str, str]:
        def pair(domain: str, text: str, eval_only: bool = False) -> str:
            row = {
                "instruction": text,
                "input": "",
                "output": "State the missing input before answering.",
                "meta": {
                    "domain": domain,
                    "pair_type": "POSTURE",
                    "eval_only": eval_only,
                },
            }
            return json.dumps(row)

        def many(domain: str, count: int) -> str:
            return "\n".join(pair(domain, f"question {domain} {index}") for index in range(count)) + "\n"

        formula_a = "id: sample_calc\ncomputes: sample\ninputs: []\nexpression: 'return 1'\n"
        formula_b = "id: sample_calc\ncomputes: sample\ninputs: []\nexpression: 'return 2'\n"
        index = (
            "count: 1\nrepo: example/fork\nexcluded_helpers:\n  - helper_only\nformulas:\n"
            "  - id: sample_calc\n    file: sample_calc.yaml\n    citation: QQ 1234\n"
        )
        helper = "id: helper_only\ncomputes: helper\ninputs: []\nexpression: 'return 0'\n"
        selection_fork = "\n".join(
            [
                json.dumps(
                    {
                        "pair_type": "which_formula_applies",
                        "formula_id": "sample_calc",
                        "instruction": "Which formula applies to sample calc",
                        "response": "Use sample_calc when the task is sample calc",
                    }
                ),
                json.dumps(
                    {
                        "pair_type": "which_standard_governs",
                        "formula_id": "sample_calc",
                        "instruction": "Which standard does the source name",
                        "response": "The source names QQ 1234",
                    }
                ),
            ]
        ) + "\n"
        selection_other = json.dumps(
            {
                "pair_type": "which_formula_applies",
                "formula_id": "sample_calc",
                "instruction": "Which formula applies to sample calc",
                "response": "Use sample_calc when the load is 40 mm",
            }
        ) + "\n"
        return {
            "bucket/a.jsonl": many("bucket-a", 2),
            "bucket/b.jsonl": many("bucket-b", 4),
            "bucket/c.jsonl": many("bucket-c", 10),
            "bucket/sealed.jsonl": pair("bucket-a", "sealed question", eval_only=True) + "\n",
            "bucket/sample_adversarial.jsonl": pair("bucket-c", "adversarial question") + "\n",
            "bucket/sample_blind.jsonl": pair("bucket-b", "blind question") + "\n",
            "bucket/unroutable.jsonl": json.dumps(
                {"reason": "missing_field", "source_kind": "fixture", "record": {"parameter": "orphan"}}
            ) + "\n",
            "rates/market-year.jsonl": json.dumps(
                {
                    "market": "North Dock",
                    "year": 2020,
                    "currency": "XYZ",
                    "rate": 3,
                    "unit": "hour",
                    "description": "role",
                    "source": "handbook",
                }
            ) + "\n",
            "rates/handling.jsonl": json.dumps(
                {
                    "prompt": "Give me a rate.",
                    "completion": "Cite the source and the year.",
                    "type": "provenance_required",
                }
            ) + "\n",
            "practice/clauses.jsonl": json.dumps(
                {
                    "clause_number": "1",
                    "code_name": "Note",
                    "edition_year": "2020",
                    "page": 1,
                    "text": "Purpose only.",
                }
            ) + "\n",
            "formulas/sample_calc.yaml": formula_a,
            "formulas/from-source/sample_calc.yaml": formula_b,
            "formulas/helper_only.yaml": helper,
            "formulas/from-source/index.yaml": index,
            "formulas/formula_selection.jsonl": selection_other,
            "formulas/from-source/formula_selection.jsonl": selection_fork,
            "facilities/site_pack.yaml": "facility_id: Z\nvalues:\n- parameter: load\n  value: 1\n  unit: kW\n  facility_id: Z\n",
        }

    def test_split_is_idempotent_and_respects_the_rules(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            zip_path = root / "source.zip"
            _write_zip(zip_path, self._files())
            out_a = root / "a"
            out_b = root / "b"
            first = split.apply_split(zip_path, out_a)
            second = split.apply_split(zip_path, out_b)
            self.assertEqual(first, second)
            files_a = sorted(path.relative_to(out_a).as_posix() for path in out_a.rglob("*") if path.is_file())
            files_b = sorted(path.relative_to(out_b).as_posix() for path in out_b.rglob("*") if path.is_file())
            self.assertEqual(files_a, files_b)
            for relative in files_a:
                self.assertEqual((out_a / relative).read_bytes(), (out_b / relative).read_bytes())

            train = _read_jsonl(out_a / "data/train/pairs.jsonl")
            held = _read_jsonl(out_a / "data/held_out/pairs.jsonl")
            regression = _read_jsonl(out_a / "data/eval/regression/pairs.jsonl")
            facts = _read_jsonl(out_a / "data/retrieval/fact_pairs.jsonl")
            rates = _read_jsonl(out_a / "data/retrieval/rates/rates.jsonl")
            clauses = _read_jsonl(out_a / "data/retrieval/clauses/clauses.jsonl")
            held_manifest = json.loads((out_a / "data/held_out/manifest.json").read_text(encoding="utf-8"))
            seal = json.loads((out_a / "data/eval/regression/seal.json").read_text(encoding="utf-8"))
            registry = json.loads((out_a / "data/retrieval/formula_registry.json").read_text(encoding="utf-8"))

            for row in train + held + regression + facts:
                self.assertEqual(set(row), {"system", "user", "assistant", "meta"})
                self.assertIn("content_hash", row["meta"])

            domain = held_manifest["held_out_domain"]
            self.assertTrue(domain)
            self.assertEqual(domain, "bucket-b")
            self.assertFalse(any(row["meta"].get("domain") == domain for row in train))
            self.assertEqual(sum(1 for row in held if row["meta"].get("holdout") == "domain"), 4)
            self.assertEqual(sum(1 for row in held if row["meta"].get("holdout") == "stratified"), 2)

            self.assertTrue(all(row["market"] and row["year"] and row["source"] for row in rates))
            self.assertEqual(len(clauses), 1)
            self.assertEqual([entry["id"] for entry in registry["formulas"]], ["sample_calc"])
            self.assertEqual(registry["formulas"][0]["definition"]["expression"], "return 2")
            self.assertTrue(any("QQ 1234" in row["assistant"] for row in facts))
            self.assertFalse(any("40 mm" in row["assistant"] for row in train + held))

            blob = (out_a / "data/eval/regression/pairs.jsonl").read_bytes()
            self.assertEqual(seal["count"], len(regression))
            self.assertEqual(seal["sha256"], hashlib.sha256(blob).hexdigest())
            self.assertGreaterEqual(len(regression), 4)


class CorpusTests(unittest.TestCase):
    def test_committed_split(self) -> None:
        train_path = ROOT / "data/train/pairs.jsonl"
        if not train_path.exists():
            self.skipTest("split output is not in the tree")
        train = _read_jsonl(train_path)
        held = _read_jsonl(ROOT / "data/held_out/pairs.jsonl")
        regression = _read_jsonl(ROOT / "data/eval/regression/pairs.jsonl")
        facts = _read_jsonl(ROOT / "data/retrieval/fact_pairs.jsonl")
        rates = _read_jsonl(ROOT / "data/retrieval/rates/rates.jsonl")
        clauses = _read_jsonl(ROOT / "data/retrieval/clauses/clauses.jsonl")
        facilities = _read_jsonl(ROOT / "data/retrieval/facilities/records.jsonl")
        registry = json.loads((ROOT / "data/retrieval/formula_registry.json").read_text(encoding="utf-8"))
        held_manifest = json.loads((ROOT / "data/held_out/manifest.json").read_text(encoding="utf-8"))
        train_manifest = json.loads((ROOT / "data/train/manifest.json").read_text(encoding="utf-8"))
        seal = json.loads((ROOT / "data/eval/regression/seal.json").read_text(encoding="utf-8"))
        lexicon = split._build_lexicon(rates)

        domain = held_manifest["held_out_domain"]
        self.assertTrue(isinstance(domain, str) and domain)
        self.assertEqual(train_manifest["count"], len(train))
        self.assertEqual(train_manifest["fact_count"], 0)
        self.assertEqual(held_manifest["count"], len(held))
        self.assertFalse(any(row["meta"].get("domain") == domain for row in train))
        domain_rows = [row for row in held if row["meta"].get("holdout") == "domain"]
        self.assertTrue(domain_rows)
        self.assertTrue(all(row["meta"].get("domain") == domain for row in domain_rows))
        self.assertEqual(len(domain_rows), held_manifest["domain_pair_count"])

        hashes = []
        for row in train + held:
            self.assertEqual(list(row), ["assistant", "meta", "system", "user"])
            self.assertFalse(split.assistant_has_fact(row["assistant"], lexicon))
            self.assertFalse(row["meta"].get("eval_only") is True)
            hashes.append(row["meta"]["content_hash"])
        self.assertEqual(len(hashes), len(set(hashes)))

        regression_hashes = {row["meta"]["content_hash"] for row in regression}
        self.assertTrue(regression_hashes.isdisjoint(hashes))
        sealed = (ROOT / "data/eval/regression/pairs.jsonl").read_bytes()
        self.assertEqual(seal["count"], len(regression))
        self.assertEqual(seal["sha256"], hashlib.sha256(sealed).hexdigest())
        self.assertTrue(all(row["market"] not in (None, "") and row["year"] not in (None, "") and row["source"] for row in rates))
        self.assertGreater(len(clauses), 0)
        self.assertGreater(len(facilities), 0)
        isolation = [row for row in facilities if row.get("facility_id") == "B"]
        measured = [row for row in facilities if row.get("facility_id") == "A"]
        self.assertTrue(isolation)
        self.assertTrue(measured)
        self.assertTrue(all(row["source"] == "facility B is isolation data, not a measured example" for row in isolation))
        self.assertTrue(all("worked-example" not in row["source"] for row in isolation))
        self.assertTrue(all(row["source"] == "worked-example" for row in measured))
        self.assertTrue(
            all(row["value"] == "UNKNOWN" for row in isolation + measured if row["parameter"] == "cooling_redundancy_topology")
        )
        manifest = json.loads((ROOT / "data/retrieval/facilities/manifest.json").read_text(encoding="utf-8"))
        provenance = " ".join(
            text
            for note in manifest["notes"]
            for text in note["document"].get("provenance_notes", [])
            if isinstance(text, str)
        )
        self.assertIn("Facility B is isolation data, not a worked example.", provenance)
        self.assertIn("facility A, domain data_centre, source worked-example", provenance)
        self.assertNotIn("facility B, domain data_centre, source worked-example", provenance)
        self.assertNotIn("a second facility B, domain data_centre, source worked-example", provenance)
        orphan = [row for row in regression if row["meta"].get("source_relpath") == "facilities/unroutable.jsonl"]
        self.assertEqual(len(orphan), 1)
        self.assertIn("worked-example", orphan[0]["user"])
        ids = [entry["id"] for entry in registry["formulas"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(registry["count"], len(ids))
        self.assertGreater(len(facts), 0)
        self.assertTrue(all(split.assistant_has_fact(row["assistant"], lexicon) for row in facts))


if __name__ == "__main__":
    unittest.main()
