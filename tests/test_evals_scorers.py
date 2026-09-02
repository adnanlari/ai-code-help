from __future__ import annotations

from collections.abc import Sequence

from backend.llm.base import LLMClient, LLMResponse, Message, ToolSpec, Usage
from evals.scorers import _parse_judge, judge_answer, score_retrieval

# --- score_retrieval -----------------------------------------------------


def test_retrieval_recall_precision_and_seed():
    s = score_retrieval(
        expected_files=["src/a.py", "src/b.py"],
        hit_files=["src/a.py", "src/z.py"],
        tool_files=["src/b.py"],
    )
    assert s.recall == 1.0  # both expected files reached (a via RAG, b via tool)
    assert s.seed_recall == 0.5  # only a.py came from RAG
    assert s.precision == round(2 / 3, 3)  # a, b relevant out of {a, b, z}
    assert s.matched == ("src/a.py", "src/b.py")


def test_retrieval_all_missed():
    s = score_retrieval(["a.py"], ["x.py", "y.py"], [])
    assert s.recall == 0.0
    assert s.precision == 0.0


def test_retrieval_normalises_paths():
    s = score_retrieval(["./a.py"], ["a.py"], [])
    assert s.recall == 1.0


def test_retrieval_empty_expected_is_zero_not_crash():
    s = score_retrieval([], ["a.py"], [])
    assert (s.recall, s.precision, s.seed_recall) == (0.0, 0.0, 0.0)


# --- _parse_judge ------------------------------------------------------


def test_parse_judge_pass():
    j = _parse_judge('{"facts":[true,true],"pass":true,"reason":"looks right"}', 2)
    assert j.passed is True
    assert j.facts_found == (True, True)
    assert j.reason == "looks right"


def test_parse_judge_fail_when_not_all_facts():
    j = _parse_judge('{"facts":[true,false],"pass":true,"reason":"x"}', 2)
    assert j.passed is False  # pass:true but a fact is missing -> overall fail


def test_parse_judge_extracts_json_from_prose():
    j = _parse_judge('Sure! {"facts":[true],"pass":true,"reason":"ok"} done', 1)
    assert j.passed is True


def test_parse_judge_unparseable():
    j = _parse_judge("no json at all", 3)
    assert j.passed is False
    assert j.facts_found == (False, False, False)
    assert "JSON" in j.reason


def test_parse_judge_pads_missing_facts():
    j = _parse_judge('{"facts":[true],"pass":true,"reason":"r"}', 3)
    assert j.facts_found == (True, False, False)
    assert j.passed is False


# --- judge_answer (fake LLM) -----------------------------------------


class _CannedLLM(LLMClient):
    def __init__(self, text: str, tokens: int = 120):
        self._text = text
        self._tokens = tokens

    @property
    def model(self) -> str:
        return "fake-judge"

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSpec] = (),
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        assert tools == ()  # judge must not be given tools
        return LLMResponse(
            text=self._text,
            tool_calls=(),
            finish_reason="stop",
            usage=Usage(self._tokens, 20),
            model="fake-judge",
        )


async def test_judge_answer_roundtrip():
    llm = _CannedLLM('{"facts":[true,true],"pass":true,"reason":"good"}')
    j = await judge_answer("q", ("fact one", "fact two"), "the answer", llm)
    assert j.passed is True
    assert j.facts_found == (True, True)
    assert j.usage_total == 140  # 120 + 20
