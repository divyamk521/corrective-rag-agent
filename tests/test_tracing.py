
import json
import logging
from io import StringIO

from crag.tracing import (
    configure_logging,
    get_trace_id,
    log_event,
    new_trace_id,
    trace_context,
)


def test_new_trace_id_is_unique():
    first_id = new_trace_id()
    second_id = new_trace_id()

    assert first_id != second_id
    assert len(first_id) == 32
    assert len(second_id) == 32


def test_trace_context_sets_and_resets_trace_id():
    assert get_trace_id() is None

    with trace_context("test-trace-123") as trace_id:
        assert trace_id == "test-trace-123"
        assert get_trace_id() == "test-trace-123"

    assert get_trace_id() is None


def test_nested_trace_context_restores_parent():
    with trace_context("parent-trace"):
        assert get_trace_id() == "parent-trace"

        with trace_context("child-trace"):
            assert get_trace_id() == "child-trace"

        assert get_trace_id() == "parent-trace"

    assert get_trace_id() is None


def test_log_event_includes_trace_id():
    configure_logging()

    logger = logging.getLogger("crag")
    stream = StringIO()
    test_handler = logging.StreamHandler(stream)

    logger.addHandler(test_handler)

    try:
        with trace_context("trace-456"):
            log_event(
                "document_retrieved",
                document_count=5,
            )

        output = stream.getvalue().strip()
        payload = json.loads(output)

        assert payload["event"] == "document_retrieved"
        assert payload["trace_id"] == "trace-456"
        assert payload["document_count"] == 5

    finally:
        logger.removeHandler(test_handler)
        test_handler.close()
