"""HTTP sampler builds a chat request without contacting Tinker in unit tests."""

from __future__ import annotations

import json
import os
import unittest
from unittest import mock

from posture_model.http_sample import sample_messages_http


class HttpSampleTests(unittest.TestCase):
    def test_builds_chat_request_for_pinned_checkpoint(self) -> None:
        captured: dict = {}

        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps(
                    {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": '{"basis":"unspecified","block":"refuse","missing_inputs":[],"formula_id":"","required_inputs":[],"unsourced_figure":"refused"}',
                                }
                            }
                        ]
                    }
                ).encode()

        def fake_urlopen(req, timeout=0):
            captured["url"] = req.full_url
            captured["headers"] = {k.lower(): v for k, v in req.headers.items()}
            captured["body"] = json.loads(req.data.decode())
            return _Resp()

        with mock.patch.dict(os.environ, {"TINKER_API_KEY": "test-key-not-real"}):
            with mock.patch("urllib.request.urlopen", fake_urlopen):
                text = sample_messages_http(
                    [
                        {"role": "system", "content": "sys"},
                        {"role": "user", "content": "hello\n\nFork context:\n(empty)"},
                    ]
                )
        self.assertIn("refuse", text)
        self.assertIn("/chat/completions", captured["url"])
        self.assertTrue(captured["body"]["model"].endswith("posture-qwen3-8b-lora-r16-epoch2"))
        self.assertEqual(captured["body"]["reasoning_effort"], False)
        self.assertFalse(captured["body"]["separate_reasoning"])
        self.assertIn("authorization", captured["headers"])
        self.assertNotIn("test-key-not-real", json.dumps(captured["body"]))


if __name__ == "__main__":
    unittest.main()
