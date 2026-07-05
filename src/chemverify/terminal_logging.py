from __future__ import annotations

import logging
import os

from rich.console import Console
from rich.logging import RichHandler

_CONFIGURED = False

_NOISY_LOGGER_LEVELS = {
    "httpx": logging.WARNING,
    "httpcore": logging.WARNING,
    "urllib3": logging.WARNING,
    "sentence_transformers": logging.WARNING,
    "transformers": logging.WARNING,
    "huggingface_hub": logging.ERROR,
    "pypdfium2": logging.WARNING,
    "mineru": logging.WARNING,
}


def configure_terminal_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    level_name = os.getenv("CHEMVERIFY_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    console = Console(stderr=True, soft_wrap=True)
    handler = RichHandler(
        console=console,
        show_time=True,
        show_level=True,
        show_path=False,
        omit_repeated_times=False,
        rich_tracebacks=True,
        markup=True,
        log_time_format="%H:%M:%S",
    )
    handler.setFormatter(logging.Formatter("%(message)s"))

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    root.addHandler(handler)

    for logger_name, logger_level in _NOISY_LOGGER_LEVELS.items():
        logging.getLogger(logger_name).setLevel(logger_level)

    try:
        from loguru import logger as loguru_logger

        def _loguru_sink(message: object) -> None:
            record = message.record
            logging.getLogger(str(record["name"])).log(int(record["level"].no), str(record["message"]))

        loguru_logger.remove()
        loguru_logger.add(_loguru_sink, level=os.getenv("CHEMVERIFY_THIRD_PARTY_LOG_LEVEL", "WARNING").upper())
    except Exception:
        pass

    _CONFIGURED = True
