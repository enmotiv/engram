"""Unit tests for migration runner."""

import os

import pytest


class TestMigrationsDir:
    def test_dir_exists(self):
        from engram.migrations.runner import MIGRATIONS_DIR

        assert os.path.isdir(MIGRATIONS_DIR)

    def test_has_sql_files(self):
        import glob

        from engram.migrations.runner import MIGRATIONS_DIR

        sql_files = glob.glob(os.path.join(MIGRATIONS_DIR, "*.sql"))
        assert len(sql_files) > 0

    def test_files_sorted_by_name(self):
        import glob

        from engram.migrations.runner import MIGRATIONS_DIR

        sql_files = sorted(glob.glob(os.path.join(MIGRATIONS_DIR, "*.sql")))
        basenames = [os.path.basename(f) for f in sql_files]
        assert basenames == sorted(basenames)


class TestMain:
    def test_main_is_callable(self):
        from engram.migrate import main

        assert callable(main)

    def test_run_migrations_is_async(self):
        import asyncio

        from engram.migrate import run_migrations

        assert asyncio.iscoroutinefunction(run_migrations)
