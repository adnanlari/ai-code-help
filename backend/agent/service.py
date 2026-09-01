"""Q&A orchestration: the glue between retrieval, the worktree, and the agent loop.

    repo_id
      -> repos_store.get_by_id            (must be status='ready')
      -> embed_query(question)            (asymmetric: input_type="query")
      -> chunks_store.similarity_search   (RAG seed - top-K chunks)
      -> workspace.ensure_worktree        (files for the tools; blocking git -> thread)
      -> ToolBox(repo_root) + get_llm_client()
      -> loop.run_agent                   (the tool-use loop)
      -> QAOutcome

Everything provider/DB-specific lives here so `loop.run_agent` stays a pure
"given an LLM, tools, question and hits, produce an answer" function that tests
can drive with a fake LLM.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from uuid import UUID

from backend.agent.loop import AgentResult, run_agent
from backend.agent.tools import ToolBox
from backend.agent.workspace import ensure_worktree
from backend.config import get_settings
from backend.indexing.embed import embed_query
from backend.llm import get_llm_client
from backend.store import chunks_store, repos_store
from backend.store.chunks_store import Hit

log = logging.getLogger("agent.service")


class RepoNotFound(LookupError):
    pass


class RepoNotReady(RuntimeError):
    """Repo exists but is not 'ready' (still indexing, or failed)."""

    def __init__(self, status: str) -> None:
        super().__init__(f"repo is not ready (status={status})")
        self.status = status


@dataclass(slots=True)
class QAOutcome:
    repo_id: UUID
    question: str
    hits: list[Hit]
    result: AgentResult


async def run_qa(
    repo_id: UUID,
    question: str,
    *,
    top_k: int | None = None,
    max_iterations: int | None = None,
) -> QAOutcome:
    settings = get_settings()
    k = top_k or settings.top_k
    max_iter = max_iterations or settings.agent_max_iterations

    repo = await repos_store.get_by_id(repo_id)
    if repo is None:
        raise RepoNotFound(str(repo_id))
    if repo.status != "ready":
        raise RepoNotReady(repo.status)

    query_vec = await embed_query(question)
    hits = await chunks_store.similarity_search(repo_id, query_vec, k=k)

    # git operations are blocking - keep them off the event loop.
    repo_root = await asyncio.to_thread(ensure_worktree, repo.repo_url, repo.commit_sha)

    toolbox = ToolBox(repo_root)
    llm = get_llm_client()

    log.info(
        "qa: repo=%s q=%r seeded %d chunks, model=%s, max_iter=%d",
        repo_id,
        question[:80],
        len(hits),
        llm.model,
        max_iter,
    )
    result = await run_agent(
        question=question,
        hits=hits,
        toolbox=toolbox,
        llm=llm,
        max_iterations=max_iter,
    )
    return QAOutcome(repo_id=repo_id, question=question, hits=hits, result=result)
