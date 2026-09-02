"""Run the golden set through the real system and score each case.

Per case:  index the repo (cached after the first case for that repo) ->
run_qa (retrieval + agent loop + grounding) -> score_retrieval (mechanical) ->
judge_answer (LLM-as-judge, unless disabled).

Needs the same credentials as `/ask`: DATABASE_URL, VOYAGE_API_KEY, KIMI_API_KEY.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from backend import db
from backend.agent.service import run_qa
from backend.indexing.pipeline import index_repo
from backend.llm import get_llm_client
from backend.llm.pricing import estimate_cost_usd
from evals.dataset import EvalCase, RepoSpec
from evals.scorers import JudgeScore, RetrievalScore, judge_answer, score_retrieval

_PATH_TOOLS = {"read_file", "grep", "list_dir"}


@dataclass(slots=True)
class CaseResult:
    id: str
    kind: str
    question: str
    answer: str = ""
    stop_reason: str = ""
    iterations: int = 0
    grounded: bool = False
    n_unverified: int = 0
    retrieval: RetrievalScore | None = None
    judge: JudgeScore | None = None
    agent_tokens: int = 0
    judge_tokens: int = 0
    est_cost_usd: float = 0.0
    latency_ms: float = 0.0
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.judge is not None and self.judge.passed


@dataclass(slots=True)
class SuiteReport:
    results: list[CaseResult] = field(default_factory=list)
    judged: bool = True
    dataset_path: str = ""


def _tool_files(trace) -> list[str]:
    out = []
    for step in trace:
        if step.tool in _PATH_TOOLS:
            p = step.arguments.get("path")
            if p:
                out.append(p)
    return out


async def run_case(case: EvalCase, repo_id, *, judge_llm, do_judge: bool = True) -> CaseResult:
    res = CaseResult(id=case.id, kind=case.kind, question=case.question)
    t0 = time.perf_counter()
    try:
        outcome = await run_qa(repo_id, case.question)
    except Exception as exc:  # noqa: BLE001 - one bad case shouldn't abort the suite
        res.error = f"{type(exc).__name__}: {exc}"
        res.latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        return res

    r = outcome.result
    res.answer = r.answer
    res.stop_reason = r.stop_reason
    res.iterations = r.iterations
    res.grounded = outcome.grounding.grounded
    res.n_unverified = outcome.grounding.unverified_count
    res.agent_tokens = r.usage.total_tokens
    res.retrieval = score_retrieval(
        case.expected_files,
        [h.file_path for h in outcome.hits],
        _tool_files(r.trace),
    )

    if do_judge:
        res.judge = await judge_answer(case.question, case.expected_facts, r.answer, judge_llm)
        res.judge_tokens = res.judge.usage_total

    agent_model = r.model or judge_llm.model
    res.est_cost_usd = round(
        estimate_cost_usd(agent_model, r.usage.input_tokens, r.usage.output_tokens)
        # judge output is a few tokens of JSON; approximate its cost as all-input
        + estimate_cost_usd(judge_llm.model, res.judge_tokens, 0),
        6,
    )
    res.latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    return res


async def run_suite(cases: list[EvalCase], *, do_judge: bool = True) -> SuiteReport:
    report = SuiteReport(judged=do_judge)
    await db.connect()
    try:
        # Index each distinct repo once; run_qa re-uses the cached index after.
        repo_ids: dict[str, object] = {}
        specs: dict[str, RepoSpec] = {c.repo.slug: c.repo for c in cases}
        for slug, spec in specs.items():
            idx = await index_repo(spec.url, spec.ref)
            repo_ids[slug] = idx.repo_id

        judge_llm = get_llm_client()
        for case in cases:
            r = await run_case(
                case, repo_ids[case.repo.slug], judge_llm=judge_llm, do_judge=do_judge
            )
            report.results.append(r)
    finally:
        await db.disconnect()
    return report
