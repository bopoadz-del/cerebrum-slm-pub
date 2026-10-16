"""The teacher reply template must not show a literal confidence value.

With a literal example (`"confidence": 0`) the teacher copied the example
instead of reporting its own confidence, and the 0.7 floor then removed
almost every row. The template describes the value instead of giving one.
"""

from __future__ import annotations

import re

from router.labels import load_labels
from router.teacher_label import _PROMPT, prompt_for

_LITERAL_CONFIDENCE = re.compile(r'"confidence"\s*:\s*[-+]?(\d|\.\d)')


def test_template_has_no_literal_numeric_confidence() -> None:
    assert not _LITERAL_CONFIDENCE.search(_PROMPT)


def test_built_prompt_has_no_literal_numeric_confidence() -> None:
    prompt = prompt_for(load_labels())
    assert '"confidence":' in prompt
    assert not _LITERAL_CONFIDENCE.search(prompt)
    assert "a number between 0 and 1" in prompt


def test_detector_catches_the_old_placeholder() -> None:
    for old in ('"confidence": 0}', '"confidence":0.5}', '"confidence": .9}', '"confidence": 1}'):
        assert _LITERAL_CONFIDENCE.search(old)
