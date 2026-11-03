
"""Lightweight request tracing and structured logging utilities."""

import json
import logging
import uuid

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator


logger = logging.getLogger("crag")

_trace_id: ContextVar[str | None] = ContextVar(
    "crag_trace_id",
    default=None,
)


def new_trace_id() -> str:
    """Generate a unique trace ID."""

    return uuid.uuid4().hex


def get_trace_id() -> str | None:
    """Return the trace ID for the current execution context."""

    return _trace_id.get()


@contextmanager
def trace_context(
    trace_id: str | None = None,
) -> Iterator[str]:
    """Set a trace ID for a block of execution."""

    current_trace_id = trace_id or new_trace_id()

    token = _trace_id.set(current_trace_id)

    try:
        yield current_trace_id

    finally:
        _trace_id.reset(token)


def log_event(
    event: str,
    level: int = logging.INFO,
    **fields,
) -> None:
    """Write a structured JSON log event."""

    payload = {
        "event": event,
        "trace_id": get_trace_id(),
        **fields,
    }

    logger.log(
        level,
        json.dumps(payload, default=str),
    )


def configure_logging(
    level: int = logging.INFO,
) -> None:
    """Configure application logging."""

    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler()

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )

        handler.setFormatter(formatter)
        logger.addHandler(handler)
