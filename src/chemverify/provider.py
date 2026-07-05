from __future__ import annotations

from .config import Settings


def require_openai_model(settings: Settings) -> str:
    model = (settings.openai_model or "").strip()
    if not model:
        raise RuntimeError("OPENAI_MODEL must be explicitly configured.")
    return model
