"""Run SQL migrations on startup. Idempotent — safe to run repeatedly."""

import asyncio
import glob
import os

import asyncpg
import structlog

from engram.config import settings

logger = structlog.get_logger()

# SQL migrations live in the top-level migrations/ directory
MIGRATIONS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "migrations"
)


async def run_migrations() -> None:
    """Execute all .sql files in migrations/ directory, in order.

    Each migration is wrapped in a transaction. Already-applied migrations
    are tracked in an _applied_migrations table.
    """
    conn = await asyncpg.connect(settings.database_url)
    try:
        # Create tracking table if it doesn't exist
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS _applied_migrations (
                filename VARCHAR(255) PRIMARY KEY,
                applied_at TIMESTAMPTZ DEFAULT now()
            )
        """)

        sql_files = sorted(glob.glob(os.path.join(MIGRATIONS_DIR, "*.sql")))
        for path in sql_files:
            filename = os.path.basename(path)
            already = await conn.fetchval(
                "SELECT 1 FROM _applied_migrations WHERE filename = $1",
                filename,
            )
            if already:
                logger.info("migration.skipped", filename=filename)
                continue

            with open(path) as f:
                sql = f.read()

            async with conn.transaction():
                await conn.execute(sql)
                await conn.execute(
                    "INSERT INTO _applied_migrations (filename) VALUES ($1)",
                    filename,
                )
            logger.info("migration.applied", filename=filename)
    finally:
        await conn.close()


def main() -> None:
    asyncio.run(run_migrations())


if __name__ == "__main__":
    main()
