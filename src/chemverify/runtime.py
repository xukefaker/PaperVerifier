from __future__ import annotations

import os
from pathlib import Path


def resolve_project_root(root: str | Path | None = None) -> Path:
    value = root or os.getenv("CHEMVERIFY_ROOT") or os.getcwd()
    return Path(value).expanduser().resolve()
