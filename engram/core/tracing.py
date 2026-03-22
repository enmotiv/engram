"""Observability: ContextVars, Span timing, TraceCollector, Prometheus metrics."""

from __future__ import annotations

import time
from contextvars import ContextVar

import structlog
from prometheus_client import Counter, Gauge, Histogram

logger = structlog.get_logger()

# --- ContextVars (set per-request) ---

_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")
_owner_id: ContextVar[str] = ContextVar("owner_id", default="")
_trace: ContextVar[TraceCollector | None] = ContextVar("trace", default=None)


def set_correlation_id(cid: str) -> None:
    _correlation_id.set(cid)


def set_owner_id(oid: str) -> None:
    _owner_id.set(oid)


def set_trace(tc: TraceCollector | None) -> None:
    _trace.set(tc)


def get_trace() -> TraceCollector | None:
    return _trace.get()


# --- structlog processors ---


def add_log_level(
    _logger: object, method_name: str, event_dict: dict
) -> dict:
    """Add 'level' field from the bound-logger method name."""
    event_dict["level"] = method_name.upper()
    return event_dict


def inject_context(
    _logger: object, _method_name: str, event_dict: dict
) -> dict:
    """Inject correlation_id and owner_id from ContextVars."""
    event_dict.setdefault("correlation_id", _correlation_id.get(""))
    event_dict.setdefault("owner_id", _owner_id.get(""))
    return event_dict


# --- Span ---


class Span:
    """Context manager for timed operations with structured logging."""

    def __init__(
        self,
        operation: str,
        component: str = "",
        expected_ms: float | None = None,
        histogram: Histogram | None = None,
        **extra: object,
    ) -> None:
        self.operation = operation
        self.component = component
        self.expected_ms = expected_ms
        self._histogram = histogram
        self.extra = extra
        self._start: float = 0.0
        self.duration_ms: float = 0.0

    def __enter__(self) -> Span:
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        _tb: object,
    ) -> bool:
        self.duration_ms = round((time.perf_counter() - self._start) * 1000, 2)

        if self._histogram:
            self._histogram.observe(self.duration_ms)

        log_fn = logger.info
        if exc_type is not None:
            log_fn = logger.error
        elif self.expected_ms and self.duration_ms > self.expected_ms * 2:
            log_fn = logger.warning
        elif self.operation.endswith(".total") and self.duration_ms > 1000:
            log_fn = logger.error

        extra = dict(self.extra)
        if exc_type is not None:
            extra["error"] = str(exc_val)

        log_fn(
            self.operation,
            component=self.component,
            duration_ms=self.duration_ms,
            **extra,
        )
        return False  # never suppress exceptions


# --- TraceCollector (?trace=true for recall) ---


class TraceCollector:
    """Accumulates retrieval diagnostics during a recall request."""

    def __init__(self, cue: str) -> None:
        self._start = time.perf_counter()
        self.cue_preview = cue[:100]
        self.embedding_ms: float = 0.0
        self.per_dimension: dict[str, dict] = {}
        self.unique_candidates: int = 0
        self.convergence_scores: list[dict] = []
        self.spreading: dict[str, int] = {
            "edges_loaded": 0,
            "excitatory_fired": 0,
            "inhibitory_fired": 0,
            "modulatory_fired": 0,
            "nodes_activated_by_spread": 0,
        }
        self.reconsolidation: dict[str, int] = {
            "nodes_boosted": 0,
            "edges_strengthened": 0,
        }
        self.post_filter: dict = {
            "nodes_excluded": 0,
            "exclude_tags": [],
        }
        self.attractor: dict | None = None
        self.stdp: dict | None = None

    def finish(self) -> dict:
        """Build the trace response object."""
        result = {
            "correlation_id": _correlation_id.get(""),
            "total_ms": round(
                (time.perf_counter() - self._start) * 1000, 1
            ),
            "embedding_ms": round(self.embedding_ms, 1),
            "cue_preview": self.cue_preview,
            "per_dimension_results": self.per_dimension,
            "unique_candidates": self.unique_candidates,
            "convergence_scores": self.convergence_scores,
            "spreading_activation": self.spreading,
            "attractor": self.attractor,
            "reconsolidation": self.reconsolidation,
            "post_filter": self.post_filter,
        }
        if self.stdp is not None:
            result["stdp"] = self.stdp
        return result


# --- Prometheus Metrics ---

RECALL_LATENCY = Histogram(
    "engram_recall_latency_ms",
    "Recall request latency in milliseconds",
    buckets=[50, 100, 200, 300, 500, 750, 1000, 2000],
)
WRITE_LATENCY = Histogram(
    "engram_write_latency_ms",
    "Write request latency in milliseconds",
    buckets=[50, 100, 200, 300, 500, 750, 1000, 2000],
)
EMBED_LATENCY = Histogram(
    "engram_embed_latency_ms",
    "OpenRouter embedding latency in milliseconds",
    buckets=[50, 100, 200, 300, 500, 750, 1000],
)
DREAMER_CYCLE = Histogram(
    "engram_dreamer_cycle_ms",
    "Dreamer cycle duration in milliseconds",
    buckets=[1000, 2000, 5000, 10000, 20000, 60000],
)
RECONSOLIDATION_FAILURES = Counter(
    "engram_reconsolidation_failures_total",
    "Reconsolidation failures during recall",
)
OPENROUTER_ERRORS = Counter(
    "engram_openrouter_errors_total",
    "OpenRouter API errors by status code",
    ["status_code"],
)
NODES_TOTAL = Gauge(
    "engram_nodes_total",
    "Non-deleted memory nodes",
    ["owner_id"],
)
EDGES_TOTAL = Gauge(
    "engram_edges_total",
    "Live edges with weight > 0",
    ["owner_id"],
)
AVG_ACTIVATION = Gauge(
    "engram_avg_activation",
    "Average activation level across nodes",
    ["owner_id"],
)
