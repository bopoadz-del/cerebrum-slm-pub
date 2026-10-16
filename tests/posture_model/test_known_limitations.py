"""KNOWN_LIMITATIONS.md must lead with blind refusal 7/60."""

from __future__ import annotations

import unittest
from pathlib import Path


class KnownLimitationsTests(unittest.TestCase):
    def test_first_line_states_blind_refusal_7_of_60(self) -> None:
        root = Path(__file__).resolve().parents[2]
        text = (root / "KNOWN_LIMITATIONS.md").read_text(encoding="utf-8")
        first = text.splitlines()[0]
        self.assertIn("7/60", first)
        self.assertRegex(first.lower(), r"blind.*refusal|refusal.*blind")


if __name__ == "__main__":
    unittest.main()
