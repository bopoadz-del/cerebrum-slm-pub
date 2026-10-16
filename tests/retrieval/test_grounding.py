"""Unsupported figures are blocked. The verdict does not call a model."""

from __future__ import annotations

import ast
import inspect
import unittest

from src.retrieval.context import context_from_chunks
from src.retrieval.grounding import grounding_gate
from tests.retrieval.chunks import fork_chunk


def _context(text: str, filename: str | None = "pack-2024-03-15.pdf"):
    return context_from_chunks(
        [
            fork_chunk(
                chunk_id="chunk-1",
                project_id="project-1",
                doc_id="doc-1",
                text=text,
                filename=filename,
            )
        ]
    )


class GroundingGateTests(unittest.TestCase):
    def test_supported_figure_passes_and_is_logged(self) -> None:
        with self.assertLogs("src.retrieval.grounding", level="INFO") as captured:
            verdict = grounding_gate(
                "the recorded value is 4",
                _context("recorded value 4"),
                cited_source_ids=("chunk-1",),
            )
        self.assertEqual(verdict.status, "PASS")
        self.assertEqual(verdict.answer, "the recorded value is 4")
        self.assertIsNone(verdict.reason)
        self.assertTrue(any("status=PASS" in line for line in captured.output))

    def test_unsupported_figure_is_blocked_answer_null(self) -> None:
        with self.assertLogs("src.retrieval.grounding", level="INFO") as captured:
            verdict = grounding_gate(
                "the recorded value is 9",
                _context("recorded value 4"),
                cited_source_ids=("chunk-1",),
            )
        self.assertEqual(verdict.status, "BLOCKED")
        self.assertIsNone(verdict.answer)
        self.assertEqual(verdict.reason, "unsupported figure: 9")
        self.assertTrue(any("status=BLOCKED" in line for line in captured.output))

    def test_larger_number_does_not_support_a_shorter_one(self) -> None:
        verdict = grounding_gate(
            "use 10",
            _context("use 100"),
            cited_source_ids=("doc-1",),
        )
        self.assertEqual(verdict.status, "BLOCKED")
        self.assertIsNone(verdict.answer)

    def test_only_the_formula_result_supports_a_figure(self) -> None:
        context = _context("no digits here")
        formula_result = {"result": 8, "width": 2, "note": "see 14"}
        passed = grounding_gate(
            "result 8",
            context,
            formula_result,
            cited_source_ids=("chunk-1",),
        )
        self.assertEqual(passed.status, "PASS")
        self.assertEqual(passed.answer, "result 8")
        for stray in ("2", "14"):
            blocked = grounding_gate(
                f"result {stray}",
                context,
                formula_result,
                cited_source_ids=("chunk-1",),
            )
            self.assertEqual(blocked.status, "BLOCKED")
            self.assertIsNone(blocked.answer)
            self.assertEqual(blocked.reason, f"unsupported figure: {stray}")

    def test_filename_date_is_not_support(self) -> None:
        verdict = grounding_gate(
            "filed 2024",
            _context("the note has no date"),
            cited_source_ids=("chunk-1",),
        )
        self.assertEqual(verdict.status, "BLOCKED")
        self.assertIsNone(verdict.answer)
        self.assertEqual(verdict.reason, "unsupported figure: 2024")

    def test_project_id_alone_is_not_a_figure_citation(self) -> None:
        verdict = grounding_gate(
            "value 4",
            _context("value 4"),
            cited_source_ids=("project-1",),
        )
        self.assertEqual(verdict.status, "BLOCKED")
        self.assertIsNone(verdict.answer)
        self.assertEqual(verdict.reason, "project_id is not a citation for a figure")

    def test_figure_without_a_source_id_is_blocked(self) -> None:
        verdict = grounding_gate(
            "result 4",
            _context("result 4"),
            cited_source_ids=(),
        )
        self.assertEqual(verdict.status, "BLOCKED")
        self.assertIsNone(verdict.answer)
        self.assertIn("source id", verdict.reason or "")

    def test_same_inputs_are_deterministic(self) -> None:
        context = _context("value 4")
        first = grounding_gate("value 4", context, cited_source_ids=("chunk-1",))
        second = grounding_gate("value 4", context, cited_source_ids=("chunk-1",))
        self.assertEqual(first, second)

    def test_verdict_imports_no_model(self) -> None:
        source = inspect.getsource(inspect.getmodule(grounding_gate))
        tree = ast.parse(source)
        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        banned = ("openai", "anthropic", "transformers", "langchain")
        self.assertFalse(any(any(word in name for word in banned) for name in imports))
        for name in inspect.signature(grounding_gate).parameters:
            self.assertNotIn("model", name)


if __name__ == "__main__":
    unittest.main()
