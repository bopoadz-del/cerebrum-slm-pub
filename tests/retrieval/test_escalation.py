"""Escalation is a pure function of confidence, task type, and context."""

from __future__ import annotations

import inspect
import unittest

from src.policy.escalation import CONFIDENCE_FLOOR, decide


class EscalationPolicyTests(unittest.TestCase):
    def test_decision_is_deterministic(self) -> None:
        first = decide(0.2, "lookup", False)
        second = decide(0.2, "lookup", False)
        self.assertEqual(first, second)

    def test_each_input_changes_the_decision(self) -> None:
        grounded = decide(CONFIDENCE_FLOOR, "lookup", True)
        self.assertFalse(grounded.escalate)

        low_confidence = decide(CONFIDENCE_FLOOR - 0.01, "lookup", True)
        self.assertTrue(low_confidence.escalate)
        self.assertIn("confidence_below_floor", low_confidence.reasons)

        missing_context = decide(CONFIDENCE_FLOOR, "lookup", False)
        self.assertTrue(missing_context.escalate)
        self.assertIn("context_absent", missing_context.reasons)

        missing_task = decide(CONFIDENCE_FLOOR, "  ", True)
        self.assertTrue(missing_task.escalate)
        self.assertIn("task_type_unspecified", missing_task.reasons)

    def test_every_escalation_is_logged(self) -> None:
        with self.assertLogs("src.policy.escalation", level="INFO") as captured:
            decide(0.1, "lookup", False)
        self.assertTrue(any("escalation" in line for line in captured.output))
        self.assertTrue(any("context_absent" in line for line in captured.output))
        self.assertTrue(any("confidence_below_floor" in line for line in captured.output))

    def test_a_pass_is_logged_too(self) -> None:
        with self.assertLogs("src.policy.escalation", level="INFO") as captured:
            decision = decide(0.9, "lookup", True)
        self.assertFalse(decision.escalate)
        self.assertEqual(decision.reasons, ())
        self.assertTrue(any("escalated=False" in line for line in captured.output))

    def test_out_of_range_confidence_escalates(self) -> None:
        decision = decide(1.4, "lookup", True)
        self.assertTrue(decision.escalate)
        self.assertIn("confidence_out_of_range", decision.reasons)

    def test_model_is_not_an_input(self) -> None:
        parameters = inspect.signature(decide).parameters
        self.assertEqual(list(parameters), ["confidence", "task_type", "context_present"])


if __name__ == "__main__":
    unittest.main()
