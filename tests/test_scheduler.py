"""Unit tests for scheduler: Redis settings parsing, worker config."""

import pytest


class TestParseRedisSettings:
    def test_default_url(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test")
        monkeypatch.setenv("DATABASE_URL", "postgresql://x")
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")

        # Force reimport with new env
        import importlib

        import engram.config
        importlib.reload(engram.config)

        from engram.scheduler import _parse_redis_settings
        rs = _parse_redis_settings()
        assert rs.host == "localhost"
        assert rs.port == 6379
        assert rs.database == 0
        assert rs.password is None

    def test_url_with_password(self, monkeypatch):
        """Test parsing a Redis URL with password, host, port, and database."""
        from urllib.parse import urlparse

        from arq.connections import RedisSettings

        url = "redis://:secret@myredis:6380/2"
        parsed = urlparse(url)
        rs = RedisSettings(
            host=parsed.hostname or "localhost",
            port=parsed.port or 6379,
            password=parsed.password,
            database=int(parsed.path.lstrip("/") or "0"),
        )
        assert rs.host == "myredis"
        assert rs.port == 6380
        assert rs.database == 2
        assert rs.password == "secret"


class TestWorkerSettings:
    def test_has_required_attributes(self):
        from engram.scheduler import WorkerSettings

        assert hasattr(WorkerSettings, "functions")
        assert hasattr(WorkerSettings, "cron_jobs")
        assert hasattr(WorkerSettings, "on_startup")
        assert hasattr(WorkerSettings, "on_shutdown")
        assert hasattr(WorkerSettings, "redis_settings")

    def test_function_count(self):
        from engram.scheduler import WorkerSettings

        # classify_new_memories is the only event-driven function
        assert len(WorkerSettings.functions) == 1

    def test_cron_job_count(self):
        from engram.scheduler import WorkerSettings

        # 5 cron jobs: classify unprocessed, decay activations, decay edges, dedup, prune
        assert len(WorkerSettings.cron_jobs) == 5

    def test_settings(self):
        from engram.scheduler import WorkerSettings

        assert WorkerSettings.max_jobs == 10
        assert WorkerSettings.job_timeout == 300
        assert WorkerSettings.retry_jobs is True
        assert WorkerSettings.max_tries == 3
