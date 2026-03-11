"""Backward compatibility — use engram.core.db instead."""

from engram.core.db import close_pool, get_pool, tenant_connection, validate_vector_dimensions

__all__ = ["close_pool", "get_pool", "tenant_connection", "validate_vector_dimensions"]
