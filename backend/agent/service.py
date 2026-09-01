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
import time
from collections import Counter
from dataclasses import dataclass
from uuid import UUID, uuid4

from backend.agent.citations import GroundingReport, verify_answer
from backend.agent.loop import AgentResult, run_agent
from backend.agent.tools import ToolBox
from backend.agent.workspace import ensure_worktree
from backend.config import get_settings
from backend.indexing.embed import embed_query
from backend.llm import get_llm_client
from backend.llm.pricing import estimate_cost_usd
from backend.obslog import log_event, ms_since
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
    request_id: str
    repo_id: UUID
    question: str
    hits: list[Hit]
    result: AgentResult
    grounding: GroundingReport  # citations in the answer, verified vs. fabricated


async def run_qa(
    repo_id: UUID,
    question: str,
    *,
    top_k: int | None = None,
    max_iterations: int | None = None,
    request_id: str | None = None,
) -> QAOutcome:
    settings = get_settings()
    k = top_k or settings.top_k
    max_iter = max_iterations or settings.agent_max_iterations
    rid = request_id or uuid4().hex[:8]
    t_start = time.perf_counter()

    repo = await repos_store.get_by_id(repo_id)
    if repo is None:
        log_event("ask_error", request_id=rid, repo_id=str(repo_id), error_type="RepoNotFound")
        raise RepoNotFound(str(repo_id))
    if repo.status != "ready":
        log_event(
            "ask_error",
            request_id=rid,
            repo_id=str(repo_id),
            error_type="RepoNotReady",
            repo_status=repo.status,
        )
        raise RepoNotReady(repo.status)

    t = time.perf_counter()
    query_vec = await embed_query(question)
    t_embed = ms_since(t)

    t = time.perf_counter()
    hits = await chunks_store.similarity_search(repo_id, query_vec, k=k)
    t_search = ms_since(t)

    # git operations are blocking - keep them off the event loop.
    t = time.perf_counter()
    try:
        repo_root = await asyncio.to_thread(ensure_worktree, repo.repo_url, repo.commit_sha)
    except Exception as exc:
        log_event(
            "ask_error",
            request_id=rid,
            repo_id=str(repo_id),
            error_type=type(exc).__name__,
            error=str(exc)[:300],
            t_total_ms=ms_since(t_start),
        )
        raise
    t_worktree = ms_since(t)

    toolbox = ToolBox(repo_root)
    llm = get_llm_client()

    t = time.perf_counter()
    try:
        result = await run_agent(
            question=question,
            hits=hits,
            toolbox=toolbox,
            llm=llm,
            max_iterations=max_iter,
        )
    except Exception as exc:
        log_event(
            "ask_error",
            request_id=rid,
            repo_id=str(repo_id),
            error_type=type(exc).__name__,
            error=str(exc)[:300],
            t_agent_ms=ms_since(t),
            t_total_ms=ms_since(t_start),
        )
        raise
    t_agent = ms_since(t)

    # Grounding check: are the answer's path:line citations backed by code the
    # agent was actually shown? Flag-only - see backend/agent/citations.py.
    grounding = verify_answer(result.answer, hits, result.messages)

    log_event(
        "ask",
        request_id=rid,
        repo_id=str(repo_id),
        question=question[:200],
        model=llm.model,
        top_k=k,
        max_iterations=max_iter,
        n_retrieved=len(hits),
        iterations=result.iterations,
        stop_reason=result.stop_reason,
        tool_calls=dict(Counter(s.tool for s in result.trace)),
        n_tool_calls=len(result.trace),
        input_tokens=result.usage.input_tokens,
        output_tokens=result.usage.output_tokens,
        total_tokens=result.usage.total_tokens,
        est_cost_usd=estimate_cost_usd(
            llm.model, result.usage.input_tokens, result.usage.output_tokens
        ),
        grounded=grounding.grounded,
        n_citations=len(grounding.citations),
        n_unverified=grounding.unverified_count,
        t_embed_ms=t_embed,
        t_search_ms=t_search,
        t_worktree_ms=t_worktree,
        t_agent_ms=t_agent,
        t_total_ms=ms_since(t_start),
    )

    return QAOutcome(
        request_id=rid,
        repo_id=repo_id,
        question=question,
        hits=hits,
        result=result,
        grounding=grounding,
    )
