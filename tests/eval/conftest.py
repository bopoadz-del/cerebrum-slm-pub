import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


@pytest.fixture
def repo_root() -> Path:
    return ROOT
