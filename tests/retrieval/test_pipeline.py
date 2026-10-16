"""Answers cite Fork source ids. Formulas are called. Outside cites are not emitted."""

from __future__ import annotations

import sys
import types
import unittest

from src.retrieval.fork import FORMULA_MODULE, RETRIEVER_MODULE
from src.retrieval.pipeline import answer
from tests.retrieval.chunks import fork_chunk

_CHUNK = fork_chunk(
    chunk_id="chunk-1",
    project_id="project-1",
    doc_id="doc-1",
    text="clause-marker applies. stored quantity 4. use formula-beta",
    filename="pack-2024-03-15.pdf",
)


def _install(module_name: str) -> types.ModuleType:
    parts = module_name.split(".")
    parent = ""
    for part in parts:
        name = part if not parent else f"{parent}.{part}"
        if name not in sys.modules:
            module = types.ModuleType(name)
            sys.modules[name] = module
            if parent:
                setattr(sys.modules[parent], part, module)
        parent = name
    return sys.modules[module_name]


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.formula_calls: list[tuple] = []
        retriever = _install(RETRIEVER_MODULE)
        retriever.available = lambda: True
        retriever.retrieve = lambda query, project_id, k=None, intent=None: {
            "chunks": [_CHUNK]
        }
        formulas = _install(FORMULA_MODULE)

        def run_formula(formula_id, params=None):
            self.formula_calls.append((formula_id, params))
            return {"result": 8, "width": 2, "note": "see 14"}

        formulas.run_formula = run_formula

    def tearDown(self) -> None:
        for name in list(sys.modules):
            if name == "app" or name.startswith("app."):
                sys.modules.pop(name, None)

    def test_answer_cites_only_source_ids_from_attached_context(self) -> None:
        result = answer(
            "quantity",
            "project-1",
            "stored quantity 4",
            sources=("chunk-1", "doc-1"),
            clauses=("clause-marker",),
            confidence=0.9,
            task_type="lookup",
        )
        self.assertEqual(result.answer, "stored quantity 4")
        self.assertEqual(result.citations, ("clause-marker", "chunk-1", "doc-1"))
        self.assertEqual(result.grounding_status, "PASS")
        self.assertFalse(result.escalated)
        self.assertEqual(self.formula_calls, [])

    def test_formula_is_called_from_text_and_only_its_result_is_a_figure(self) -> None:
        passed = answer(
            "compute",
            "project-1",
            "computed 8",
            sources=("chunk-1",),
            formula_id="formula-beta",
            formula_params={"width": 2},
            confidence=0.9,
            task_type="compute",
        )
        self.assertEqual(self.formula_calls, [("formula-beta", {"width": 2})])
        self.assertEqual(passed.answer, "computed 8")
        self.assertIn("formula-beta", passed.citations)
        self.assertIn("chunk-1", passed.citations)
        self.assertEqual(passed.formula_called, "formula-beta")

        blocked = answer(
            "compute",
            "project-1",
            "computed 2",
            sources=("chunk-1",),
            formula_id="formula-beta",
            formula_params={"width": 2},
            confidence=0.9,
            task_type="compute",
        )
        self.assertIsNone(blocked.answer)
        self.assertEqual(blocked.reason, "unsupported figure: 2")
        self.assertFalse(blocked.escalated)

    def test_formula_id_absent_from_text_is_not_called(self) -> None:
        result = answer(
            "compute",
            "project-1",
            "computed 8",
            sources=("chunk-1",),
            formula_id="formula-missing",
            confidence=0.9,
            task_type="compute",
        )
        self.assertEqual(self.formula_calls, [])
        self.assertIsNone(result.answer)
        self.assertEqual(result.reason, "unsupported figure: 8")

    def test_clause_absent_from_text_is_refused(self) -> None:
        result = answer(
            "quantity",
            "project-1",
            "stored quantity 4",
            sources=("chunk-1",),
            clauses=("clause-absent",),
            confidence=0.9,
            task_type="lookup",
        )
        self.assertIsNone(result.answer)
        self.assertEqual(result.citations, ())
        self.assertEqual(result.reason, "identifier is not in the attached context")

    def test_citation_outside_context_is_not_emitted(self) -> None:
        result = answer(
            "quantity",
            "project-1",
            "stored quantity 4",
            sources=("chunk-1", "src-not-attached"),
            confidence=0.9,
            task_type="lookup",
        )
        self.assertIsNone(result.answer)
        self.assertEqual(result.citations, ())
        self.assertNotIn("src-not-attached", result.citations)
        self.assertEqual(result.reason, "identifier is not in the attached context")

    def test_filename_date_stays_unsupported(self) -> None:
        result = answer(
            "when",
            "project-1",
            "filed 2024",
            sources=("doc-1",),
            confidence=0.9,
            task_type="lookup",
        )
        self.assertIsNone(result.answer)
        self.assertEqual(result.reason, "unsupported figure: 2024")

    def test_project_id_alone_does_not_cite_a_figure(self) -> None:
        result = answer(
            "quantity",
            "project-1",
            "stored quantity 4",
            sources=("project-1",),
            confidence=0.9,
            task_type="lookup",
        )
        self.assertIsNone(result.answer)
        self.assertEqual(result.citations, ())
        self.assertEqual(result.reason, "project_id is not a citation for a figure")

    def test_empty_retrieve_withholds_a_sentence_with_no_number(self) -> None:
        retriever = sys.modules[RETRIEVER_MODULE]
        retriever.retrieve = lambda *args, **kwargs: {"chunks": []}
        proposed = "nothing was recorded"
        with self.assertLogs("src.policy.escalation", level="INFO") as captured:
            result = answer(
                "quantity",
                "project-1",
                proposed,
                confidence=0.9,
                task_type="lookup",
            )
        self.assertIsNone(result.answer)
        self.assertNotEqual(result.answer, proposed)
        self.assertEqual(result.citations, ())
        self.assertEqual(result.reason, "empty context: no answer is possible")
        self.assertEqual(result.grounding_status, "BLOCKED")
        self.assertTrue(result.escalated)
        self.assertIn("context_absent", result.escalation_reasons)
        self.assertTrue(any("escalation" in line for line in captured.output))

    def test_blocked_grounding_withholds_the_answer_when_not_escalating(self) -> None:
        proposed = "stored quantity 9"
        with self.assertLogs("src.policy.escalation", level="INFO") as captured:
            result = answer(
                "quantity",
                "project-1",
                proposed,
                sources=("chunk-1",),
                confidence=0.9,
                task_type="lookup",
            )
        self.assertIsNone(result.answer)
        self.assertNotEqual(result.answer, proposed)
        self.assertEqual(result.citations, ())
        self.assertEqual(result.grounding_status, "BLOCKED")
        self.assertEqual(result.reason, "unsupported figure: 9")
        self.assertFalse(result.escalated)
        self.assertTrue(any("escalated=False" in line for line in captured.output))


if __name__ == "__main__":
    unittest.main()
