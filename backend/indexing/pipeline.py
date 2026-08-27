"""The indexing pipeline: cache-check -> clone -> filter -> chunk -> embed -> store.

This is deliberately synchronous end-to-end (the HTTP request blocks until the
repo is indexed). For a personal-scale project that's fine and keeps the mental
model simple. Production would hand this to a job queue (Celery/RQ/Arq) and have
the client poll GET /repos/{id} - noted in the README.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from backend.config import get_settings
from backend.indexing import clone
from backend.indexing.chunk import Chunk, chunk_file
from backend.indexing.embed import embed_documents
from backend.indexing.filter import iter_source_files
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

    # 1. Resolve the exact commit without downloading anything.
    commit_sha = clone.resolve_sha(repo_url, ref)

    # 2. Cache / race-guard: get-or-create the repos row.
    row, we_created = await repos_store.claim_for_indexing(repo_url, commit_sha)

    if not we_created and row.status == "ready":
        count = await chunks_store.count_for_repo(row.id)
        log.info("cache hit: %s @ %s (%d chunks)", repo_url, commit_sha[:8], count)
        return IndexResult(row.id, commit_sha, "ready", cached=True, chunk_count=count)

    if not we_created and row.status == "indexing":
        # Another job is already working on this exact repo+commit.
        log.info("index already in progress: %s @ %s", repo_url, commit_sha[:8])
        return IndexResult(row.id, commit_sha, "indexing", cached=False, chunk_count=0)

    # 3. Cache miss (we created the row, or reset a 'failed' one) -> do the work.
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
        return IndexResult(row.id, commit_sha, "ready", cached=False, chunk_count=inserted)

    except Exception:
        await repos_store.mark_failed(row.id)
        log.exception("indexing failed for %s @ %s", repo_url, commit_sha[:8])
        raise
    finally:
        # Raw clone is disposable - drop it as soon as we've chunked + embedded.
        if clone_path is not None:
            clone.cleanup(clone_path)
