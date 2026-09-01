"""The indexing pipeline: cache-check -> clone -> filter -> chunk -> embed -> store.

This is deliberately synchronous end-to-end (the HTTP request blocks until the
repo is indexed). For a personal-scale project that's fine and keeps the mental
model simple. Production would hand this to a job queue (Celery/RQ/Arq) and have
the client poll GET /repos/{id} - noted in the README.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid4

from backend.config import get_settings
from backend.indexing import clone
from backend.indexing.chunk import Chunk, chunk_file
from backend.indexing.embed import embed_documents
from backend.indexing.filter import iter_source_files
from backend.obslog import log_event, ms_since
from backend.store import chunks_store, repos_store

log = logging.getLogger("pipeline")


@dataclass(frozen=True, slots=True)
class IndexResult:
    repo_id: UUID
    commit_sha: str
    status: str
    cached: bool  # True => served entirely from the cache, no embedding spend
    chunk_count: int


async def index_repo(repo_url: str, ref: str = "HEAD") -> IndexResult:
    settings = get_settings()
    rid = uuid4().hex[:8]
    t_start = time.perf_counter()

    def _emit(outcome: str, chunk_count: int) -> None:
        log_event(
            "index",
            request_id=rid,
            repo_url=repo_url,
            commit_sha=commit_sha[:12],
            outcome=outcome,
            chunk_count=chunk_count,
            t_total_ms=ms_since(t_start),
        )

    # 1. Resolve the exact commit without downloading anything.
    commit_sha = clone.resolve_sha(repo_url, ref)

    # 2. Cache / race-guard: get-or-create the repos row and find out what to do.
    row, outcome = await repos_store.claim_for_indexing(repo_url, commit_sha)

    if outcome is repos_store.ClaimOutcome.READY:
        count = await chunks_store.count_for_repo(row.id)
        log.info("cache hit: %s @ %s (%d chunks)", repo_url, commit_sha[:8], count)
        _emit("cache_hit", count)
        return IndexResult(row.id, commit_sha, "ready", cached=True, chunk_count=count)

    if outcome is repos_store.ClaimOutcome.IN_PROGRESS:
        # A live job is already working on this exact repo+commit.
        log.info("index already in progress: %s @ %s", repo_url, commit_sha[:8])
        _emit("in_progress", 0)
        return IndexResult(row.id, commit_sha, "indexing", cached=False, chunk_count=0)

    # 3. outcome is CREATED or RECLAIMED -> we own the row, do the work.
    if outcome is repos_store.ClaimOutcome.RECLAIMED:
        removed = await chunks_store.delete_for_repo(row.id)
        log.info("reclaimed %s @ %s; cleared %d leftover chunks", repo_url, commit_sha[:8], removed)

    clone_path: Path | None = None
    try:
        clone_path = clone.shallow_clone(repo_url, ref, settings.workdir)

        all_chunks: list[Chunk] = []
        for abs_path, rel_path in iter_source_files(clone_path):
            text = abs_path.read_text("utf-8", errors="replace")
            all_chunks.extend(
                chunk_file(
                    rel_path,
                    text,
                    size=settings.chunk_lines,
                    overlap=settings.chunk_overlap_lines,
                )
            )

        log.info("chunked %s into %d chunks", repo_url, len(all_chunks))

        embeddings = await embed_documents([c.content for c in all_chunks])
        inserted = await chunks_store.bulk_insert(row.id, all_chunks, embeddings)

        await repos_store.mark_ready(row.id)
        log.info("indexed %s @ %s (%d chunks)", repo_url, commit_sha[:8], inserted)
        _emit("reclaimed" if outcome is repos_store.ClaimOutcome.RECLAIMED else "indexed", inserted)
        return IndexResult(row.id, commit_sha, "ready", cached=False, chunk_count=inserted)

    except BaseException as exc:
        # BaseException (not just Exception) so a Ctrl-C / cancellation also
        # leaves the row as 'failed' rather than stuck in 'indexing'. We re-raise
        # immediately - this is only a status-cleanup hook.
        await repos_store.mark_failed(row.id)
        log.exception("indexing failed for %s @ %s", repo_url, commit_sha[:8])
        log_event(
            "index_error",
            request_id=rid,
            repo_url=repo_url,
            commit_sha=commit_sha[:12],
            error_type=type(exc).__name__,
            error=str(exc)[:300],
            t_total_ms=ms_since(t_start),
        )
        raise
    finally:
        # Raw clone is disposable - drop it as soon as we've chunked + embedded.
        if clone_path is not None:
            clone.cleanup(clone_path)
