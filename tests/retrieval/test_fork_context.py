"""Context attached by The Fork's retriever carries its real chunk fields."""

from __future__ import annotations

import sys
import types
import unittest

from tests.retrieval.chunks import fork_chunk
from src.retrieval.context import context_from_retrieve
from src.retrieval.fork import (
    FORMULA_MODULE,
    RETRIEVER_MODULE,
    attach_context,
    call_formula,
    call_retrieve,
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


class ForkRetrieverTests(unittest.TestCase):
    def tearDown(self) -> None:
        for name in list(sys.modules):
            if name == "app" or name.startswith("app."):
                sys.modules.pop(name, None)

    def test_attach_calls_fork_retrieve_and_keeps_source_ids(self) -> None:
        calls: dict[str, object] = {}
        chunk = fork_chunk(
            chunk_id="chunk-1",
            project_id="project-1",
            doc_id="doc-1",
            text="recorded value 4. see clause-marker. use formula-beta",
            filename="pack-2024-03-15.pdf",
        )

        def retrieve(query, project_id, k=None, intent=None):
            calls["args"] = (query, project_id, k, intent)
            return {"chunks": [chunk]}

        module = _install(RETRIEVER_MODULE)
        module.retrieve = retrieve
        module.available = lambda: True

        context = attach_context("what value", "project-1", k=3, intent="lookup")

        self.assertEqual(calls["args"], ("what value", "project-1", 3, "lookup"))
        self.assertEqual(context.source_ids, frozenset({"chunk-1", "doc-1"}))
        self.assertEqual(context.project_ids, frozenset({"project-1"}))
        self.assertNotIn("2024", context.support_text)
        self.assertTrue(context.text_contains("clause-marker"))
        self.assertTrue(context.text_contains("formula-beta"))
        self.assertTrue(context.allows("clause", "clause-marker"))
        self.assertFalse(context.allows("clause", "clause-absent"))
        self.assertFalse(context.cites_figure(("project-1",)))
        self.assertTrue(context.cites_figure(("chunk-1",)))
        self.assertTrue(context.present)

    def test_unavailable_retriever_attaches_empty_context(self) -> None:
        module = _install(RETRIEVER_MODULE)
        module.available = lambda: False
        module.retrieve = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("retrieve must not run when unavailable")
        )

        self.assertEqual(call_retrieve("q", "p"), [])
        context = attach_context("q", "p")
        self.assertFalse(context.present)
        self.assertEqual(context.source_ids, frozenset())

    def test_formula_is_called_through_fork_module(self) -> None:
        calls: dict[str, object] = {}

        def run_formula(formula_id, params=None):
            calls["args"] = (formula_id, params)
            return {"result": 8, "width": 2}

        module = _install(FORMULA_MODULE)
        module.run_formula = run_formula

        result = call_formula("formula-beta", {"left": 3})

        self.assertEqual(result["result"], 8)
        self.assertEqual(calls["args"], ("formula-beta", {"left": 3}))

    def test_payload_without_chunks_key_still_reads_ids(self) -> None:
        context = context_from_retrieve(
            [
                fork_chunk(
                    chunk_id="only-chunk",
                    project_id="project-1",
                    doc_id="doc-9",
                    text="bare 2",
                )
            ]
        )
        self.assertEqual(context.source_ids, frozenset({"only-chunk", "doc-9"}))
        self.assertIn("bare 2", context.support_text)


if __name__ == "__main__":
    unittest.main()
