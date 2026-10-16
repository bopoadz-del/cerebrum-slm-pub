"""Pinned serve artifact loads without contacting Tinker."""

from __future__ import annotations

import unittest

from posture_model.artifact import CHECKPOINT, load_serve_artifact, load_system_prompt


class ServeArtifactTests(unittest.TestCase):
    def test_pinned_checkpoint_and_prompt(self) -> None:
        artifact = load_serve_artifact()
        self.assertEqual(artifact.checkpoint, CHECKPOINT)
        self.assertTrue(artifact.checkpoint.endswith("posture-qwen3-8b-lora-r16-epoch2"))
        self.assertEqual(artifact.base_model, "Qwen/Qwen3-8B")
        self.assertEqual(artifact.stop_gate.get("result"), "PASS")
        self.assertEqual(artifact.stop_gate.get("blind_refusal_or_qualification"), "7/60")
        self.assertIn("construction posture model", artifact.system_prompt.lower())
        self.assertEqual(load_system_prompt(), artifact.system_prompt)
        self.assertNotIn("api_key", artifact.sampling)
        self.assertEqual(artifact.api_key_env, "TINKER_API_KEY")


if __name__ == "__main__":
    unittest.main()
