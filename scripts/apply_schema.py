"""Apply every migrations/*.sql file, in filename order, to DATABASE_URL.

Usage:
    python -m scripts.apply_schema

This is a deliberately dumb migration runner - no versioning table, no
down-migrations. Every migration is written to be idempotent (IF NOT EXISTS),
so re-running the whole set is safe. A real project would use Alembic or
sqitch; that's overkill for a 4-day build.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import asyncpg

from backend.config import get_settings

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"


async def main() -> None:
    settings = get_settings()
    dsn = settings.database_url
    connect_kwargs: dict = {"statement_cache_size": 0}
    if "sslmode" not in dsn:
        connect_kwargs["ssl"] = "require"

    conn = await asyncpg.connect(dsn, **connect_kwargs)
    try:
        files = sorted(MIGRATIONS_DIR.glob("*.sql"))
        if not files:
            print(f"no .sql files in {MIGRATIONS_DIR}")
            return
        for path in files:
            print(f"applying {path.name} ...")
            await conn.execute(path.read_text("utf-8"))
        print(f"done - {len(files)} migration(s) applied")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
