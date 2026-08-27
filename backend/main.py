"""FastAPI application: health, index, repo-status.

Day 1 surface only. The agentic Q&A endpoint arrives on Day 2-3.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from uuid import UUID

from fastapi import FastAPI, HTTPException

from backend import db
from backend.indexing.clone import GitError
from backend.indexing.pipeline import index_repo
from backend.models import (
    HealthResponse,
    IndexRequest,
    IndexResponse,
    RepoStatusResponse,
)
from backend.store import chunks_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(_: FastAPI):
    await db.connect()
    yield
    await db.disconnect()


app = FastAPI(title="AI Coding Buddy", version="0.1.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    try:
        await db.get_pool().fetchval("select 1")
        db_status = "ok"
    except Exception:  # noqa: BLE001
        db_status = "error"
    return HealthResponse(status="ok", db=db_status)


@app.post("/index", response_model=IndexResponse)
async def index(req: IndexRequest) -> IndexResponse:
    try:
        result = await index_repo(req.repo_url, req.ref)
    except GitError as exc:
        raise HTTPException(status_code=400, detail=f"git error: {exc}") from exc
    return IndexResponse(
        repo_id=result.repo_id,
        commit_sha=result.commit_sha,
        status=result.status,
        cached=result.cached,
        chunk_count=result.chunk_count,
    )


@app.get("/repos/{repo_id}", response_model=RepoStatusResponse)
async def repo_status(repo_id: UUID) -> RepoStatusResponse:
    row = await db.get_pool().fetchrow(
        "select id, repo_url, commit_sha, status, indexed_at from repos where id = $1",
        repo_id,
    )
    if row is None:
        raise HTTPException(status_code=404, detail="repo not found")
    count = await chunks_store.count_for_repo(repo_id)
    return RepoStatusResponse(
        repo_id=row["id"],
        repo_url=row["repo_url"],
        commit_sha=row["commit_sha"],
        status=row["status"],
        indexed_at=row["indexed_at"],
        chunk_count=count,
    )
