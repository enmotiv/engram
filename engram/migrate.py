"""Backward compatibility — use engram.migrations.runner instead."""

from engram.migrations.runner import main, run_migrations

__all__ = ["main", "run_migrations"]

if __name__ == "__main__":
    main()
