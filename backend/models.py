"""Pydantic request/response models for the HTTP API.

FastAPI uses these for validation, OpenAPI docs, and serialisation. Keeping them
in one module makes the API surface easy to scan.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class IndexRequest(BaseModel):
    repo_url: str = Field(..., description="HTTPS git URL of a public repo")
    ref: str = Field("HEAD", description="branch, tag, or commit SHA to index")


class IndexResponse(BaseModel):
    repo_id: UUID
    commit_sha: str
    status: str  # 'indexing' | 'ready' | 'failed'
    cached: bool
    chunk_count: int


class RepoStatusResponse(BaseModel):
    repo_id: UUID
    repo_url: str
    commit_sha: str
    status: str
    indexed_at: datetime | None
    chunk_count: int


class HealthResponse(BaseModel):
    status: str
    db: str


# --- /ask ------------------------------------------------------------------


class AskRequest(BaseModel):
    repo_id: UUID
    question: str = Field(..., min_length=1)
    top_k: int | None = Field(None, ge=1, le=20, description="override retrieval size")
    max_iterations: int | None = Field(None, ge=1, le=12, description="override tool-loop cap")


class RetrievedChunk(BaseModel):
    file_path: str
    start_line: int
    end_line: int
    score: float


class TraceStepModel(BaseModel):
    index: int
    tool: str
    arguments: dict
    ok: bool
    result_preview: str
    result_chars: int


class UsageModel(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class AskResponse(BaseModel):
    repo_id: UUID
    question: str
    answer: str
    stop_reason: str  # answered | max_iterations | empty_response
    iterations: int
    usage: UsageModel
    retrieved: list[RetrievedChunk]  # the RAG seed
    trace: list[TraceStepModel]  # the tool-call sequence
