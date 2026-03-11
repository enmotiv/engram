"""Backward compatibility — use engram.core.middleware instead."""

from engram.core.middleware import CorrelationMiddleware, register_error_handlers

__all__ = ["CorrelationMiddleware", "register_error_handlers"]
