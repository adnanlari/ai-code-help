"""Orchestration tests for run_qa - every external dependency is monkeypatched,
so this stays offline (no DB, no Voyage, no git, no LLM API)."""

from __future__ import annotations

from collections.abc import Sequence
from uuid import uuid4

import pytest

from backend.agent import service
from backend.agent.service import QAOutcome, RepoNotFound, RepoNotReady, run_qa
from backend.llm.base import LLMClient, LLMResponse, Message, ToolSpec, Usage
from backend.store.chunks_store import Hit
from backend.store.repos_store import RepoRow

REPO_ID = uuid4()


class _StubLLM(LLMClient):
    @property
    def model(self) -> str:
        return "stub"

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSpec] = (),
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        return LLMResponse(
            text="the answer (a.py:1-2)",
            tool_calls=(),
            finish_reason="stop",
            usage=Usage(11, 7),
            model="stub",
        )


@pytest.fixture
def wired(monkeypatch, tmp_path):
    """Patch run_qa's collaborators; return the knobs a test may want to tweak."""
    row = RepoRow(id=REPO_ID, repo_url="https://x/y", commit_sha="deadbeef", status="ready")
    hits = [Hit("a.py", 1, 2, "code", 0.9)]

    async def fake_get_by_id(rid):
        return row if rid == REPO_ID else None

    async def fake_embed_query(text):
        return [0.1] * 8

    async def fake_similarity_search(rid, vec, k):
        return hits

    def fake_ensure_worktree(url, sha):
        (tmp_path / "a.py").write_text("x = 1\ny = 2\n")
        return tmp_path

    monkeypatch.setattr(service.repos_store, "get_by_id", fake_get_by_id)
    monkeypatch.setattr(service, "embed_query", fake_embed_query)
    monkeypatch.setattr(service.chunks_store, "similarity_search", fake_similarity_search)
    monkeypatch.setattr(service, "ensure_worktree", fake_ensure_worktree)
    monkeypatch.setattr(service, "get_llm_client", lambda: _StubLLM())
    return {"row": row, "hits": hits}


async def test_happy_path_returns_outcome(wired):
    out = await run_qa(REPO_ID, "what does it do?")
    assert isinstance(out, QAOutcome)
    assert out.repo_id == REPO_ID
    assert out.hits == wired["hits"]
    assert out.result.answer == "the answer (a.py:1-2)"
    assert out.result.stop_reason == "answered"
    assert out.result.usage.total_tokens == 18
    # "a.py:1-2" is cited and matches the retrieved chunk span -> grounded
    assert out.grounding.grounded is True
    assert [c.status for c in out.grounding.citations] == ["verified"]


async def test_unknown_repo_raises(wired):
    with pytest.raises(RepoNotFound):
        await run_qa(uuid4(), "q")


async def test_repo_not_ready_raises(wired, monkeypatch):
    async def indexing_row(rid):
        return RepoRow(id=REPO_ID, repo_url="u", commit_sha="s", status="indexing")

    monkeypatch.setattr(service.repos_store, "get_by_id", indexing_row)
    with pytest.raises(RepoNotReady) as ei:
        await run_qa(REPO_ID, "q")
    assert ei.value.status == "indexing"


async def test_top_k_and_max_iterations_are_passed_through(wired, monkeypatch):
    seen = {}

    async def spy_similarity(rid, vec, k):
        seen["k"] = k
        return wired["hits"]

    async def spy_run_agent(*, question, hits, toolbox, llm, max_iterations):
        seen["max_iterations"] = max_iterations
        from backend.agent.loop import AgentResult

        return AgentResult(answer="ok")

    monkeypatch.setattr(service.chunks_store, "similarity_search", spy_similarity)
    monkeypatch.setattr(service, "run_agent", spy_run_agent)

    await run_qa(REPO_ID, "q", top_k=3, max_iterations=9)
    assert seen == {"k": 3, "max_iterations": 9}
