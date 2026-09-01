"""Loop tests driven by a scripted fake LLM - no network, no API key."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from backend.agent.loop import run_agent
from backend.agent.tools import ToolBox
from backend.llm.base import LLMClient, LLMResponse, Message, ToolCall, ToolSpec, Usage
from backend.store.chunks_store import Hit

# --- scaffolding -----------------------------------------------------------


class FakeLLM(LLMClient):
    """Replays a script. Each entry is either an LLMResponse or a callable
    (messages, tools) -> LLMResponse so a step can react to the transcript."""

    def __init__(self, script):
        self._script = list(script)
        self.calls: list[dict] = []

    @property
    def model(self) -> str:
        return "fake"

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSpec] = (),
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self.calls.append({"messages": list(messages), "tools": list(tools)})
        step = self._script[min(len(self.calls) - 1, len(self._script) - 1)]
        return step(list(messages), list(tools)) if callable(step) else step


def text_resp(text: str, *, usage=(3, 5)) -> LLMResponse:
    return LLMResponse(
        text=text,
        tool_calls=(),
        finish_reason="stop",
        usage=Usage(*usage),
        model="fake",
    )


def tool_resp(*calls: ToolCall, usage=(7, 2)) -> LLMResponse:
    return LLMResponse(
        text=None,
        tool_calls=tuple(calls),
        finish_reason="tool_calls",
        usage=Usage(*usage),
        model="fake",
    )


def call(name: str, **args) -> ToolCall:
    import json

    return ToolCall(id=f"c_{name}", name=name, arguments=args, arguments_raw=json.dumps(args))


@pytest.fixture
def toolbox(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("def main():\n    return 42\n")
    (tmp_path / "README.md").write_text("# Demo\n")
    return ToolBox(tmp_path)


HITS = [Hit("src/app.py", 1, 2, "def main():\n    return 42", 0.71)]


# --- tests --------------------------------------------------------------


async def test_answers_immediately_without_tools(toolbox):
    llm = FakeLLM([text_resp("main() returns 42 (src/app.py:1-2)")])
    res = await run_agent(question="what does main do?", hits=HITS, toolbox=toolbox, llm=llm)

    assert res.stop_reason == "answered"
    assert res.iterations == 1
    assert res.trace == []
    assert "42" in res.answer
    assert res.usage.total_tokens == 8  # 3 + 5
    # transcript: system, user, assistant
    assert [m.role for m in res.messages] == ["system", "user", "assistant"]


async def test_excerpts_are_in_the_seed_prompt(toolbox):
    llm = FakeLLM([text_resp("ok")])
    await run_agent(question="q", hits=HITS, toolbox=toolbox, llm=llm)
    seed_user = llm.calls[0]["messages"][1].content
    assert "src/app.py:1-2" in seed_user
    assert "return 42" in seed_user
    # tools were offered
    assert {s.name for s in llm.calls[0]["tools"]} == {"read_file", "grep", "list_dir"}


async def test_runs_a_tool_then_answers(toolbox):
    llm = FakeLLM(
        [
            tool_resp(call("read_file", path="src/app.py")),
            text_resp("returns 42 (src/app.py:1-2)"),
        ]
    )
    res = await run_agent(question="q", hits=HITS, toolbox=toolbox, llm=llm)

    assert res.stop_reason == "answered"
    assert res.iterations == 2
    assert len(res.trace) == 1
    step = res.trace[0]
    assert step.tool == "read_file" and step.ok is True
    assert "def main" in step.result_preview
    # a tool-result message was inserted for the model's second turn
    roles = [m.role for m in res.messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    assert res.usage.total_tokens == 9 + 8  # tool_resp(7+2) + text_resp(3+5)


async def test_parallel_tool_calls_all_execute(toolbox):
    llm = FakeLLM(
        [
            tool_resp(call("list_dir", path="."), call("read_file", path="README.md")),
            text_resp("done"),
        ]
    )
    res = await run_agent(question="q", hits=HITS, toolbox=toolbox, llm=llm)
    assert [s.tool for s in res.trace] == ["list_dir", "read_file"]
    assert (
        res.messages.count(  # two tool messages appended
            next(m for m in res.messages if m.role == "tool")
        )
        >= 1
    )
    assert sum(1 for m in res.messages if m.role == "tool") == 2


async def test_tool_error_is_fed_back_and_recorded(toolbox):
    def step2(messages, tools):
        last_tool = [m for m in messages if m.role == "tool"][-1]
        assert "escapes the repository root" in last_tool.content
        return text_resp("cannot access that path")

    llm = FakeLLM(
        [
            tool_resp(call("read_file", path="../../../../etc/passwd")),
            step2,
        ]
    )
    res = await run_agent(question="q", hits=HITS, toolbox=toolbox, llm=llm)
    assert res.trace[0].ok is False
    assert "escapes" in res.trace[0].result_preview


async def test_hits_iteration_cap_then_forced_answer(toolbox):
    def always_tool(messages, tools):
        if not tools:  # the forced final call withholds tools
            return text_resp("final answer under duress")
        return tool_resp(call("list_dir", path="."))

    llm = FakeLLM([always_tool])
    res = await run_agent(question="q", hits=HITS, toolbox=toolbox, llm=llm, max_iterations=3)

    assert res.stop_reason == "max_iterations"
    assert res.iterations == 3
    assert len(res.trace) == 3  # one tool call per capped iteration
    assert res.answer == "final answer under duress"
    # 3 tool-calls made a request each + 1 forced final call
    assert len(llm.calls) == 4
    assert llm.calls[-1]["tools"] == []


async def test_empty_response_stops_cleanly(toolbox):
    empty = LLMResponse(
        text=None, tool_calls=(), finish_reason="stop", usage=Usage(1, 0), model="fake"
    )
    llm = FakeLLM([empty])
    res = await run_agent(question="q", hits=HITS, toolbox=toolbox, llm=llm)
    assert res.stop_reason == "empty_response"
    assert res.answer == ""


async def test_no_hits_still_runs(toolbox):
    llm = FakeLLM([text_resp("no relevant code found")])
    res = await run_agent(question="q", hits=[], toolbox=toolbox, llm=llm)
    assert "no relevant code" in res.answer
    assert "no excerpts" in llm.calls[0]["messages"][1].content
