"""Shared helpers for the router run driver. Never prints secret values."""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path("/tmp/cslm-0e81e95")  # read-only, blob-verified copy of main at 0e81e95
for entry in (REPO / "scripts", REPO / "src", REPO):
    text = str(entry)
    if text not in sys.path:
        sys.path.insert(0, text)


def load_env_file(path: Path) -> list[str]:
    """Load KEY=VALUE lines into os.environ. Returns names only."""
    names: list[str] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip().strip('"').strip("'")
        if name and value:
            os.environ[name] = value
            names.append(name)
    return names


def force_httpx() -> None:
    """Same transport workaround used by the 2026-10-05 posture runs on this laptop."""
    from tinker.types.client_config_response import ClientConfigResponse

    ClientConfigResponse.model_fields["use_pyqwest_transport"].default = False
    original = ClientConfigResponse.model_validate

    @classmethod
    def forced(cls, obj, *args, **kwargs):
        if isinstance(obj, dict):
            obj = dict(obj)
            obj["use_pyqwest_transport"] = False
        data = original(obj, *args, **kwargs)
        try:
            object.__setattr__(data, "use_pyqwest_transport", False)
        except Exception:
            pass
        return data

    ClientConfigResponse.model_validate = forced  # type: ignore[method-assign]
