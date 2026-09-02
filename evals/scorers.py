"""Two scorers for an eval case.

1. score_retrieval - mechanical. Did the answer draw on the right files? Pure set
   overlap of expected files vs. files actually used (RAG hits + files the agent
   read/grepped). No model call.

2. judge_answer - LLM-as-judge. A separate, low-temperature model call that reads
   (question, required facts, candidate answer) and decides, per fact, whether
   the answer conveys it (paraphrase allowed), then PASS iff all facts are
   present. Grading is far easier than answering, which is why this works - but
   in production you'd validate the judge against human labels first and watch
   for position/verbosity bias. Here it's one call, same provider as the agent.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from backend.llm.base import LLMClient, Message


def _norm(path: str) -> str:
    return path.strip().lstrip("./").lstrip("/")


# --- retrieval -----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RetrievalScore:
    expected: tuple[str, ...]
    used: tuple[str, ...]  # RAG hits + files read/grepped by the agent
    matched: tuple[str, ...]  # expected ∩ used
    recall: float  # matched / expected  (did we reach the right files at all?)
    precision: float  # matched / used      (how much of what we touched was relevant?)
    seed_recall: float  # matched-in-RAG-hits / expected (was RAG alone enough?)


def score_retrieval(
    expected_files: list[str] | tuple[str, ...],
    hit_files: list[str] | tuple[str, ...],
    tool_files: list[str] | tuple[str, ...] = (),
) -> RetrievalScore:
    exp = {_norm(f) for f in expected_files if f}
    hits = {_norm(f) for f in hit_files if f}
    used = hits | {_norm(f) for f in tool_files if f}

    matched = exp & used
    recall = len(matched) / len(exp) if exp else 0.0
    precision = len(matched) / len(used) if used else 0.0
    seed_recall = len(exp & hits) / len(exp) if exp else 0.0

    return RetrievalScore(
        expected=tuple(sorted(exp)),
        used=tuple(sorted(used)),
        matched=tuple(sorted(matched)),
        recall=round(recall, 3),
        precision=round(precision, 3),
        seed_recall=round(seed_recall, 3),
    )


# --- LLM-as-judge --------------------------------------------------------

_JUDGE_SYSTEM = """\
You are a strict grader for answers about a codebase. You are given a QUESTION, a
numbered list of REQUIRED FACTS, and a CANDIDATE ANSWER.

For each required fact, decide whether the candidate answer conveys that idea
(exact wording is not required; a clear paraphrase counts). Then the answer PASSES
only if every required fact is conveyed and nothing in the answer flatly
contradicts them.

Respond with ONLY a JSON object, no prose:
{"facts": [true, false, ...], "pass": true|false, "reason": "<= 25 words"}
The "facts" array must have one boolean per required fact, in order."""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True, slots=True)
class JudgeScore:
    passed: bool
    reason: str
    facts_found: tuple[bool, ...]
    raw: str
    usage_total: int = 0  # tokens spent on the judge call


def _build_judge_messages(question: str, facts: tuple[str, ...], answer: str) -> list[Message]:
    facts_block = "\n".join(f"{i}. {f}" for i, f in enumerate(facts, 1))
    user = (
        f"QUESTION:\n{question}\n\n"
        f"REQUIRED FACTS:\n{facts_block}\n\n"
        f"CANDIDATE ANSWER:\n{answer or '(empty answer)'}"
    )
    return [Message(role="system", content=_JUDGE_SYSTEM), Message(role="user", content=user)]


def _parse_judge(text: str, n_facts: int) -> JudgeScore:
    m = _JSON_RE.search(text or "")
    if not m:
        return JudgeScore(False, "judge output had no JSON", (False,) * n_facts, text or "")
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return JudgeScore(False, "judge JSON did not parse", (False,) * n_facts, text or "")

    facts = data.get("facts", [])
    facts_found = tuple(bool(x) for x in facts)[:n_facts]
    if len(facts_found) < n_facts:
        facts_found = facts_found + (False,) * (n_facts - len(facts_found))
    passed = bool(data.get("pass")) and all(facts_found)
    reason = str(data.get("reason", "")).strip()[:300]
    return JudgeScore(passed=passed, reason=reason, facts_found=facts_found, raw=text or "")


async def judge_answer(
    question: str,
    expected_facts: tuple[str, ...],
    answer: str,
    llm: LLMClient,
) -> JudgeScore:
    messages = _build_judge_messages(question, expected_facts, answer)
    resp = await llm.complete(messages, tools=(), temperature=0.0, max_tokens=400)
    score = _parse_judge(resp.text or "", len(expected_facts))
    return JudgeScore(
        passed=score.passed,
        reason=score.reason,
        facts_found=score.facts_found,
        raw=score.raw,
        usage_total=resp.usage.total_tokens,
    )
