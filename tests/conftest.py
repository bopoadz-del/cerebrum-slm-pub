"""Test path setup — repo scripts and the collect tooling are importable."""

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
for sub in ("scripts", "scripts/collect"):
    p = str(_ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from private_paths import EXTERNAL_DATA_TESTS, skip_if_missing  # noqa: E402


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip tests that read archive, sealed, or owner files when those files are absent.

    The skip is documented in tests/private_paths.py. Assertions are not changed.
    """
    paths = EXTERNAL_DATA_TESTS.get(item.nodeid)
    if paths:
        skip_if_missing(_ROOT, *paths)
