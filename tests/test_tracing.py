"""Unit tests for tracing: Span, TraceCollector, structlog processors."""

import time

import pytest


class TestSpan:
    def test_measures_duration(self):
        from engram.tracing import Span

        with Span("test.op", component="test") as span:
            time.sleep(0.01)

        assert span.duration_ms > 0

    def test_does_not_suppress_exceptions(self):
        from engram.tracing import Span

        with pytest.raises(ValueError, match="boom"):
            with Span("test.error", component="test"):
                raise ValueError("boom")

    def test_records_histogram(self):
        from unittest.mock import MagicMock

        from engram.tracing import Span

        mock_hist = MagicMock()
        with Span("test.hist", component="test", histogram=mock_hist):
            pass

        mock_hist.observe.assert_called_once()
        assert mock_hist.observe.call_args[0][0] >= 0


class TestTraceCollector:
    def test_initial_state(self):
        from engram.tracing import TraceCollector

        tc = TraceCollector("test cue")
        assert tc.cue_preview == "test cue"
        assert tc.embedding_ms == 0.0
        assert tc.unique_candidates == 0

    def test_finish_returns_dict(self):
        from engram.tracing import TraceCollector

        tc = TraceCollector("test cue")
        result = tc.finish()
        assert isinstance(result, dict)
        assert "total_ms" in result
        assert "embedding_ms" in result
        assert "cue_preview" in result
        assert result["cue_preview"] == "test cue"

    def test_cue_preview_truncated(self):
        from engram.tracing import TraceCollector

        long_cue = "x" * 200
        tc = TraceCollector(long_cue)
        assert len(tc.cue_preview) == 100


class TestStructlogProcessors:
    def test_add_log_level(self):
        from engram.tracing import add_log_level

        event_dict = {"event": "test"}
        result = add_log_level(None, "info", event_dict)
        assert result["level"] == "INFO"

    def test_add_log_level_warning(self):
        from engram.tracing import add_log_level

        event_dict = {"event": "test"}
        result = add_log_level(None, "warning", event_dict)
        assert result["level"] == "WARNING"

    def test_inject_context(self):
        from engram.tracing import inject_context

        event_dict = {"event": "test"}
        result = inject_context(None, "info", event_dict)
        assert "correlation_id" in result
        assert "owner_id" in result

    def test_inject_context_preserves_existing(self):
        from engram.tracing import inject_context

        event_dict = {"event": "test", "correlation_id": "existing"}
        result = inject_context(None, "info", event_dict)
        assert result["correlation_id"] == "existing"


class TestContextVars:
    def test_set_and_get_correlation_id(self):
        from engram.core.tracing import _correlation_id
        from engram.tracing import set_correlation_id

        set_correlation_id("test_123")
        assert _correlation_id.get() == "test_123"
        set_correlation_id("")  # reset

    def test_set_and_get_owner_id(self):
        from engram.core.tracing import _owner_id
        from engram.tracing import set_owner_id

        set_owner_id("owner_abc")
        assert _owner_id.get() == "owner_abc"
        set_owner_id("")  # reset

    def test_set_and_get_trace(self):
        from engram.tracing import TraceCollector, get_trace, set_trace

        tc = TraceCollector("test")
        set_trace(tc)
        assert get_trace() is tc
        set_trace(None)  # reset
        assert get_trace() is None
