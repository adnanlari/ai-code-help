"""Raw-SQL access to the `repos` table (the cache + race guard).

No ORM on purpose - the point of the project is to be fluent in the actual SQL.
asyncpg uses $1, $2 ... positional params (never string interpolation) which is
both injection-safe and lets Postgres cache the plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from uuid import UUID

from backend.db import get_pool

# An 'indexing' row older (by claimed_at) than this many minutes is assumed to be
# from a crashed/interrupted run and may be retaken. Indexing a normal repo takes
# seconds to a couple of minutes, so 15 is comfortably past any real live job.
STALE_AFTER_MINUTES = 15


class ClaimOutcome(str, Enum):
    CREATED = "created"  # brand-new row - we own it, do the work
    RECLAIMED = "reclaimed"  # was failed or stale - we reset it, do the work
    IN_PROGRESS = "in_progress"  # a live job owns it - back off
    READY = "ready"  # already indexed - cache hit


@dataclass(frozen=True, slots=True)
class RepoRow:
    id: UUID
    repo_url: str
    commit_sha: str
    status: str  # 'indexing' | 'ready' | 'failed'


def _row(record) -> RepoRow:
    return RepoRow(
        id=record["id"],
        repo_url=record["repo_url"],
        commit_sha=record["commit_sha"],
        status=record["status"],
    )


async def get_by_id(repo_id: UUID) -> RepoRow | None:
    row = await get_pool().fetchrow(
        "select id, repo_url, commit_sha, status from repos where id = $1",
        repo_id,
    )
    return _row(row) if row else None


async def get_by_url_sha(repo_url: str, commit_sha: str) -> RepoRow | None:
    row = await get_pool().fetchrow(
        """
        select id, repo_url, commit_sha, status
        from repos
        where repo_url = $1 and commit_sha = $2
        """,
        repo_url,
        commit_sha,
    )
    return _row(row) if row else None


async def claim_for_indexing(
    repo_url: str,
    commit_sha: str,
    stale_after_minutes: int = STALE_AFTER_MINUTES,
) -> tuple[RepoRow, ClaimOutcome]:
    """Get-or-create the row for (repo_url, commit_sha) and report what to do.

    Three atomic SQL statements, each relying on Postgres for correctness under
    concurrency:

    1. INSERT ... ON CONFLICT DO NOTHING - the race guard. Two requests hitting
       a brand-new repo at once: exactly one INSERT wins (CREATED); the loser
       gets no row back and falls through.
    2. UPDATE ... WHERE status='failed' OR stale - the retry path. Whoever's
       WHERE clause still matches flips the row to 'indexing' and gets it back
       (RECLAIMED); a second racer's WHERE no longer matches, so it gets nothing.
    3. SELECT - the row is either already 'ready' (READY, cache hit) or being
       worked on by a live job (IN_PROGRESS).

    Only CREATED and RECLAIMED mean "you own this row, go index".
    """
    pool = get_pool()

    created = await pool.fetchrow(
        """
        insert into repos (repo_url, commit_sha, status, claimed_at)
        values ($1, $2, 'indexing', now())
        on conflict (repo_url, commit_sha) do nothing
        returning id, repo_url, commit_sha, status
        """,
        repo_url,
        commit_sha,
    )
    if created:
        return _row(created), ClaimOutcome.CREATED

    reclaimed = await pool.fetchrow(
        """
        update repos
           set status = 'indexing', claimed_at = now(), indexed_at = null
         where repo_url = $1 and commit_sha = $2
           and (
                status = 'failed'
                or (
                    status = 'indexing'
                    and (
                        claimed_at is null
                        or claimed_at < now() - make_interval(mins => $3)
                    )
                )
           )
        returning id, repo_url, commit_sha, status
        """,
        repo_url,
        commit_sha,
        stale_after_minutes,
    )
    if reclaimed:
        return _row(reclaimed), ClaimOutcome.RECLAIMED

    existing = await get_by_url_sha(repo_url, commit_sha)
    assert existing is not None  # the conflicting row must still exist
    if existing.status == "ready":
        return existing, ClaimOutcome.READY
    return existing, ClaimOutcome.IN_PROGRESS


async def mark_ready(repo_id: UUID) -> None:
    await get_pool().execute(
        "update repos set status = 'ready', indexed_at = now() where id = $1",
        repo_id,
    )


async def mark_failed(repo_id: UUID) -> None:
    await get_pool().execute(
        "update repos set status = 'failed' where id = $1",
        repo_id,
    )
