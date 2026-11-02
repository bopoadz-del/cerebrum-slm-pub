"""Test path setup — repo scripts and the collect tooling are importable."""

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_TESTS = Path(__file__).resolve().parent
for entry in (_ROOT / "scripts", _ROOT / "scripts" / "collect"):
    text = str(entry)
    if text not in sys.path:
        sys.path.insert(0, text)
if str(_TESTS) not in sys.path:
    sys.path.append(str(_TESTS))

from private_data_mark import private_data_skip_reason  # noqa: E402


def pytest_runtest_setup(item: pytest.Item) -> None:
    marker = item.get_closest_marker("private_data")
    if marker is None or not marker.args:
        return
    reason = private_data_skip_reason(_ROOT, tuple(str(arg) for arg in marker.args))
    if reason:
        pytest.skip(reason)
