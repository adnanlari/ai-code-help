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
