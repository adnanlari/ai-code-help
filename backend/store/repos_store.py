"""Raw-SQL access to the `repos` table (the cache + race guard).

No ORM on purpose - the point of the project is to be fluent in the actual SQL.
asyncpg uses $1, $2 ... positional params (never string interpolation) which is
both injection-safe and lets Postgres cache the plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from backend.db import get_pool


@dataclass(frozen=True, slots=True)
class RepoRow:
    id: UUID
    repo_url: str
    commit_sha: str
    status: str  # 'indexing' | 'ready' | 'failed'


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
    return RepoRow(**dict(row)) if row else None


async def claim_for_indexing(repo_url: str, commit_sha: str) -> tuple[RepoRow, bool]:
    """Atomically get-or-create the row for (repo_url, commit_sha).

    Returns (row, we_created_it). The INSERT ... ON CONFLICT DO NOTHING is the
    race guard: if two requests hit a brand-new repo at once, exactly one INSERT
    succeeds. The loser gets no row back from RETURNING and falls through to the
    SELECT, picking up the winner's row instead of starting a second job.

    A pre-existing 'failed' row is reset to 'indexing' here so a retry can run.
    """
    pool = get_pool()
    created = await pool.fetchrow(
        """
        insert into repos (repo_url, commit_sha, status)
        values ($1, $2, 'indexing')
        on conflict (repo_url, commit_sha) do nothing
        returning id, repo_url, commit_sha, status
        """,
        repo_url,
        commit_sha,
    )
    if created:
        return RepoRow(**dict(created)), True

    existing = await get_by_url_sha(repo_url, commit_sha)
    assert existing is not None  # the conflicting row must exist
    if existing.status == "failed":
        await pool.execute(
            "update repos set status = 'indexing', indexed_at = null where id = $1",
            existing.id,
        )
        existing = RepoRow(
            id=existing.id,
            repo_url=existing.repo_url,
            commit_sha=existing.commit_sha,
            status="indexing",
        )
    return existing, False


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
