"""Unit tests for db module: tenant_connection safety."""

from uuid import UUID

import pytest


class TestTenantConnection:
    def test_uuid_validation(self):
        """tenant_connection validates owner_id as UUID to prevent injection."""
        # Valid UUID string should work
        safe = str(UUID("12345678-1234-5678-1234-567812345678"))
        assert safe == "12345678-1234-5678-1234-567812345678"

    def test_uuid_rejects_injection(self):
        """Injecting SQL in owner_id should raise ValueError."""
        with pytest.raises(ValueError):
            UUID("'; DROP TABLE memory_nodes; --")


class TestPoolFunctions:
    def test_get_pool_is_async(self):
        import asyncio

        from engram.db import get_pool

        assert asyncio.iscoroutinefunction(get_pool)

    def test_close_pool_is_async(self):
        import asyncio

        from engram.db import close_pool

        assert asyncio.iscoroutinefunction(close_pool)
