"""asyncpg connection-pool lifecycle + pgvector type registration.

Why a pool: opening a Postgres connection is expensive (TCP + TLS + auth +
server-side backend process). A pool keeps a handful of connections warm and
hands them out per request. Everything in this project is async so we use
asyncpg's native pool rather than a thread-based one.

Why register_vector: pgvector's `vector` column arrives over the wire as a
string like "[0.1,0.2,...]". `pgvector.asyncpg.register_vector` installs a
codec on each connection so we can pass/receive plain Python lists (and numpy
arrays) instead of formatting that string by hand.
"""

from __future__ import annotations

import asyncpg
from pgvector.asyncpg import register_vector

from backend.config import get_settings

_pool: asyncpg.Pool | None = None


async def _init_connection(conn: asyncpg.Connection) -> None:
    """Run once for every physical connection the pool creates."""
    await register_vector(conn)


async def connect() -> asyncpg.Pool:
    """Create the global pool. Call once on app startup."""
    global _pool
    if _pool is not None:
        return _pool

    settings = get_settings()
    dsn = settings.database_url

    kwargs: dict = {
        "dsn": dsn,
        "init": _init_connection,
        "min_size": 1,
        "max_size": 10,
        # Supabase's pooler runs pgbouncer in transaction mode, which does not
        # support server-side prepared statements. Disabling the statement cache
        # keeps us compatible with both the pooler and a direct connection.
        "statement_cache_size": 0,
    }
    # Supabase requires TLS; honour an explicit sslmode in the DSN, otherwise
    # default to requiring it.
    if "sslmode" not in dsn:
        kwargs["ssl"] = "require"

    _pool = await asyncpg.create_pool(**kwargs)
    return _pool


async def disconnect() -> None:
    """Close the pool on app shutdown."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def get_pool() -> asyncpg.Pool:
    """Return the live pool; raise if connect() has not run yet."""
    if _pool is None:
        raise RuntimeError("DB pool not initialised - call connect() first")
    return _pool
