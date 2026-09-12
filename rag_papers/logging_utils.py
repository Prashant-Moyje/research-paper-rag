"""Logging setup, plus the Windows console encoding fix.

Why this exists: on Windows the default console encoding is cp1252, and this
corpus is full of characters it cannot encode (Greek letters, arrows, dashes,
authors' names). Printing extracted paper text raises UnicodeEncodeError and
kills the run. Reconfiguring stdout/stderr to UTF-8 here means every entry
point inherits the fix instead of each one rediscovering it.
"""

from __future__ import annotations

import logging
import sys

_NOISY = (
    "httpx",
    "httpcore",
    "urllib3",
    "sentence_transformers",
    "transformers",
    "filelock",
    "huggingface_hub",
)


def force_utf8_console() -> None:
    """Make stdout/stderr UTF-8 so paper text can be printed on Windows."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                # Non-reconfigurable stream (e.g. redirected pipe); safe to skip.
                pass


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging once, with third-party chatter suppressed."""
    force_utf8_console()
    logging.basicConfig(
        level=level,
        format="%(levelname)-7s %(name)-28s %(message)s",
        stream=sys.stderr,
        force=True,
    )
    for name in _NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)
