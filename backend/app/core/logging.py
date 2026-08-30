"""Structured application logging.

This module configures process-wide logging that emits one JSON object per
log record, which keeps log lines machine-parseable in the deployed
environment (Railway) while remaining human-readable. It is deliberately
free of any business logic: feature services and the data layer obtain a
logger through :func:`get_logger` and use it to record events such as the
rejection of an out-of-range external value before it reaches a feature
module.

Requirements: 15.4 (validation rejections are recorded in the application
log), 17.5 (logging is an isolated, separately-modifiable concern).

Notes
-----
* ``import logging`` below resolves to the Python standard library module,
  not this file: Python 3 uses absolute imports, so this module is only
  ever reachable as ``app.core.logging``.
* Any keyword passed through the standard ``extra=`` mechanism (for example
  ``logger.info("rejected", extra={"field": "price", "value": -1})``) is
  merged into the emitted JSON object, so callers can attach structured
  context without any bespoke API.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any

# Attributes that already exist on a standard ``LogRecord``. Anything on a
# record that is *not* in this set was supplied by the caller via ``extra=``
# and is therefore promoted into the structured payload. Building the set
# from a real record instance keeps it correct across Python versions.
_RESERVED_RECORD_ATTRS: frozenset[str] = frozenset(
    logging.LogRecord(
        name="", level=logging.INFO, pathname="", lineno=0, msg="", args=(), exc_info=None
    ).__dict__.keys()
) | {"message", "asctime", "taskName"}

# Marker used to recognise (and avoid duplicating) the handler this module
# installs, so that ``configure_logging`` is safe to call more than once.
_HANDLER_MARKER = "_price_truth_json_handler"


class JsonLogFormatter(logging.Formatter):
    """Format a :class:`logging.LogRecord` as a single-line JSON object.

    The object always carries a timestamp, severity, logger name, source
    location, and message. Exception information and any caller-supplied
    ``extra`` fields are included when present.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        # Promote structured context supplied via ``extra=``.
        for key, value in record.__dict__.items():
            if key not in _RESERVED_RECORD_ATTRS and not key.startswith("_"):
                payload[key] = value

        # ``default=str`` guarantees serialisation never raises on an
        # unexpected value type; ``ensure_ascii=False`` keeps unicode intact.
        return json.dumps(payload, default=str, ensure_ascii=False)


def _resolve_level(level: int | str | None) -> int:
    """Resolve an effective log level from an argument or the environment."""

    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO")
    if isinstance(level, str):
        return logging.getLevelName(level.upper())  # returns int for known names
    return level


def configure_logging(level: int | str | None = None) -> None:
    """Install the JSON handler on the root logger.

    Safe to call repeatedly: any previously installed handler from this
    module is removed first, so no duplicate log lines are produced. The
    level defaults to the ``LOG_LEVEL`` environment variable, or ``INFO``.
    """

    resolved_level = _resolve_level(level)
    root = logging.getLogger()
    root.setLevel(resolved_level)

    # Drop a handler we previously installed (idempotent reconfiguration).
    for existing in list(root.handlers):
        if getattr(existing, _HANDLER_MARKER, False):
            root.removeHandler(existing)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JsonLogFormatter())
    handler.setLevel(resolved_level)
    setattr(handler, _HANDLER_MARKER, True)
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a logger that emits structured JSON lines.

    Logging is configured on first use so that a logger obtained from this
    helper produces structured output even if :func:`configure_logging` was
    not called explicitly during application startup.
    """

    root = logging.getLogger()
    already_configured = any(
        getattr(handler, _HANDLER_MARKER, False) for handler in root.handlers
    )
    if not already_configured:
        configure_logging()
    return logging.getLogger(name)
