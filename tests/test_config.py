"""Unit tests for config: exclude_tags_set property."""

import pytest


class TestExcludeTagsSet:
    def test_empty_string(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test")
        monkeypatch.setenv("DATABASE_URL", "postgresql://x")
        monkeypatch.setenv("ENGRAM_RETRIEVAL_EXCLUDE_TAGS", "")

        import importlib

        import engram.config
        importlib.reload(engram.config)
        from engram.config import settings

        assert settings.exclude_tags_set == set()

    def test_single_tag(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test")
        monkeypatch.setenv("DATABASE_URL", "postgresql://x")
        monkeypatch.setenv("ENGRAM_RETRIEVAL_EXCLUDE_TAGS", "entity-summary")

        import importlib

        import engram.config
        importlib.reload(engram.config)
        from engram.config import settings

        assert settings.exclude_tags_set == {"entity-summary"}

    def test_multiple_tags(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test")
        monkeypatch.setenv("DATABASE_URL", "postgresql://x")
        monkeypatch.setenv("ENGRAM_RETRIEVAL_EXCLUDE_TAGS", "tag1, tag2, tag3")

        import importlib

        import engram.config
        importlib.reload(engram.config)
        from engram.config import settings

        assert settings.exclude_tags_set == {"tag1", "tag2", "tag3"}

    def test_strips_whitespace(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test")
        monkeypatch.setenv("DATABASE_URL", "postgresql://x")
        monkeypatch.setenv("ENGRAM_RETRIEVAL_EXCLUDE_TAGS", "  foo , bar  ")

        import importlib

        import engram.config
        importlib.reload(engram.config)
        from engram.config import settings

        assert settings.exclude_tags_set == {"foo", "bar"}
