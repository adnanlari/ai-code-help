"""FastAPI application: health, index, repo-status, ask."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException

from backend import db
from backend.agent.service import RepoNotFound, RepoNotReady, run_qa
from backend.indexing.clone import GitError
from backend.indexing.pipeline import index_repo
from backend.llm import LLMError
from backend.models import (
    AskRequest,
    AskResponse,
    CitationModel,
    HealthResponse,
    IndexRequest,
    IndexResponse,
    RepoStatusResponse,
    RetrievedChunk,
    TraceStepModel,
    UsageModel,
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


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    """Answer a question about an already-indexed repo via the RAG-seeded agent loop.

    Runs synchronously (like /index): the request blocks for the whole tool loop.
    Production would stream the trace / move this to a task + websocket.
    """
    rid = uuid4().hex[:8]
    try:
        outcome = await run_qa(
            req.repo_id,
            req.question,
            top_k=req.top_k,
            max_iterations=req.max_iterations,
            request_id=rid,
        )
    except RepoNotFound as exc:
        raise HTTPException(status_code=404, detail=f"repo not found (request_id={rid})") from exc
    except RepoNotReady as exc:
        raise HTTPException(status_code=409, detail=f"{exc} (request_id={rid})") from exc
    except GitError as exc:
        raise HTTPException(
            status_code=502, detail=f"could not check out repo: {exc} (request_id={rid})"
        ) from exc
    except LLMError as exc:
        raise HTTPException(
            status_code=502, detail=f"LLM provider error: {exc} (request_id={rid})"
        ) from exc

    r = outcome.result
    return AskResponse(
        request_id=outcome.request_id,
        repo_id=outcome.repo_id,
        question=outcome.question,
        answer=r.answer,
        model=r.model,
        stop_reason=r.stop_reason,
        iterations=r.iterations,
        usage=UsageModel(
            input_tokens=r.usage.input_tokens,
            output_tokens=r.usage.output_tokens,
            total_tokens=r.usage.total_tokens,
        ),
        retrieved=[
            RetrievedChunk(
                file_path=h.file_path,
                start_line=h.start_line,
                end_line=h.end_line,
                score=h.score,
            )
            for h in outcome.hits
        ],
        trace=[
            TraceStepModel(
                index=s.index,
                tool=s.tool,
                arguments=s.arguments,
                ok=s.ok,
                result_preview=s.result_preview,
                result_chars=s.result_chars,
            )
            for s in r.trace
        ],
        grounded=outcome.grounding.grounded,
        citations=[
            CitationModel(
                raw=c.raw,
                file_path=c.file_path,
                start_line=c.start_line,
                end_line=c.end_line,
                status=c.status,
            )
            for c in outcome.grounding.citations
        ],
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
