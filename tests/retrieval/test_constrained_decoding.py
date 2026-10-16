"""A citation or formula id outside the attached context cannot be emitted."""

from __future__ import annotations

import unittest

from src.decoding.constrained import CitationToken, decode, emit


class ConstrainedDecodingTests(unittest.TestCase):
    def test_citation_outside_context_cannot_be_emitted(self) -> None:
        allowed = frozenset({"src-1", "clause-2"})
        outside = emit("citation", "src-not-attached", allowed)
        self.assertTrue(outside.refused)
        self.assertIsNone(outside.value)
        self.assertNotEqual(outside.value, "src-not-attached")

        decoded = decode(
            "see the record",
            (
                CitationToken("source", "src-1"),
                CitationToken("citation", "src-not-attached"),
            ),
            allowed,
        )
        self.assertTrue(decoded.refused)
        self.assertIsNone(decoded.text)
        self.assertEqual(decoded.emitted, ())
        self.assertNotIn("src-not-attached", decoded.emitted)
        rendered = "" if decoded.text is None else decoded.text
        self.assertNotIn("src-not-attached", rendered)

    def test_ids_present_in_context_can_be_emitted(self) -> None:
        allowed = frozenset({"src-1", "clause-2"})
        decoded = decode(
            "see the record",
            (
                CitationToken("source", "src-1"),
                CitationToken("clause", "clause-2"),
            ),
            allowed,
        )
        self.assertFalse(decoded.refused)
        self.assertEqual(decoded.text, "see the record")
        self.assertEqual(decoded.emitted, ("src-1", "clause-2"))

    def test_empty_context_refuses_every_citation(self) -> None:
        for kind in ("citation", "clause", "source", "formula"):
            result = emit(kind, "src-1", frozenset())
            self.assertTrue(result.refused)
            self.assertIsNone(result.value)
            self.assertEqual(result.reason, "empty context: no citation is possible")

        decoded = decode("nothing attached", (CitationToken("citation", "src-1"),), frozenset())
        self.assertTrue(decoded.refused)
        self.assertIsNone(decoded.text)
        self.assertEqual(decoded.emitted, ())
        self.assertIn("empty context", decoded.reason or "")

    def test_formula_id_outside_registry_cannot_be_emitted(self) -> None:
        allowed = frozenset({"HDS5-3.1", "beam_deflection_cantilever_udl"})
        outside = emit("formula", "not-in-registry", allowed)
        self.assertTrue(outside.refused)
        self.assertIsNone(outside.value)
        inside = emit("formula", "HDS5-3.1", allowed)
        self.assertFalse(inside.refused)
        self.assertEqual(inside.value, "HDS5-3.1")


if __name__ == "__main__":
    unittest.main()
