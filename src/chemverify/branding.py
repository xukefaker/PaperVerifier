from __future__ import annotations

import os


DEFAULT_APP_NAME = "ChemVerify"
DEFAULT_APP_TAGLINE = "Evidence-verified paper search"


def app_name() -> str:
    return os.getenv("CHEMVERIFY_APP_NAME") or DEFAULT_APP_NAME


def app_tagline() -> str:
    return os.getenv("CHEMVERIFY_APP_TAGLINE") or DEFAULT_APP_TAGLINE
