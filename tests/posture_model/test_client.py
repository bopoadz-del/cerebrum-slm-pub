"""Serve client requires Fork context and constrains formula ids."""

from __future__ import annotations

import json
import unittest

from posture_model.client import (
    ForkContextRequired,
    constrain_posture_response,
    sample_posture,
)


def _ok_json(**overrides: object) -> str:
    payload = {
        "basis": "unspecified",
        "block": "qualify",
        "missing_inputs": ["design_strength"],
        "formula_id": "",
        "required_inputs": [],
        "unsourced_figure": "not_applicable",
    }
    payload.update(overrides)
    return json.dumps(payload)


class PostureClientTests(unittest.TestCase):
    def test_fork_context_none_refuses(self) -> None:
        with self.assertRaises(ForkContextRequired):
            sample_posture("hello", fork_context=None, sampler=lambda m: _ok_json())  # type: ignore[arg-type]

    def test_empty_fork_context_allowed(self) -> None:
        seen: list[list[dict[str, str]]] = []

        def sampler(messages: list[dict[str, str]]) -> str:
            seen.append(messages)
            return _ok_json()

        text = sample_posture("Name the missing input for a pour strength check.", fork_context="", sampler=sampler)
        self.assertIn("qualify", text)
        self.assertEqual(len(seen), 1)
        roles = [m["role"] for m in seen[0]]
        self.assertEqual(roles[0], "system")
        self.assertEqual(roles[-1], "user")
        self.assertIn("Fork context:", seen[0][-1]["content"])
        self.assertIn("(empty)", seen[0][-1]["content"])

    def test_formula_id_outside_registry_cleared(self) -> None:
        raw = _ok_json(block="none", formula_id="not-in-registry", required_inputs=["L"], missing_inputs=[])
        out = constrain_posture_response(raw, allowed_formula_ids=frozenset({"beam_deflection_cantilever_udl"}))
        obj = json.loads(out)
        self.assertEqual(obj["formula_id"], "")
        self.assertEqual(obj["required_inputs"], [])

    def test_formula_id_in_registry_kept(self) -> None:
        raw = _ok_json(
            block="none",
            formula_id="beam_deflection_cantilever_udl",
            required_inputs=["L", "w", "E", "I"],
            missing_inputs=[],
        )
        out = constrain_posture_response(
            raw, allowed_formula_ids=frozenset({"beam_deflection_cantilever_udl"})
        )
        obj = json.loads(out)
        self.assertEqual(obj["formula_id"], "beam_deflection_cantilever_udl")

    def test_sample_applies_constraint(self) -> None:
        def sampler(_messages: list[dict[str, str]]) -> str:
            return _ok_json(block="none", formula_id="rogue-id", required_inputs=["x"], missing_inputs=[])

        text = sample_posture(
            "Route a deflection check.",
            fork_context="Formula registry\n- id: beam_deflection_cantilever_udl",
            allowed_formula_ids=frozenset({"beam_deflection_cantilever_udl"}),
            sampler=sampler,
        )
        obj = json.loads(text)
        self.assertEqual(obj["formula_id"], "")


if __name__ == "__main__":
    unittest.main()
