"""The RAG-seeded agentic tool-use loop - the heart of the project.

Mechanism (this is the thing to be able to explain):

    seed the conversation with:
        [system]  who you are, the tools, how to cite
        [user]    the question + the top-K chunks from vector search

    repeat up to `max_iterations` times:
        resp = llm.complete(messages, tools=<read_file/grep/list_dir>)
        append resp as an [assistant] message
        if resp asked for tools:
            for each tool call:
                run it through ToolBox (which applies the path guardrail)
                append the result as a [tool] message
                record a trace step
            loop again
        else:
            resp.text is the answer -> stop

    if the cap is hit with no answer:
        make ONE more call WITHOUT tools ("answer now with what you have")

Why a hand-rolled loop instead of a framework:
  * the control flow *is* the lesson - `stop_reason`/`tool_calls` drives it,
    nothing hidden.
  * the iteration cap is a real cost lever: each pass is a paid LLM call, so an
    agent that keeps saying "let me check one more file" must be stoppable.
  * every tool call is our code, so the guardrail and output caps are enforced
    by construction, not hope.

The loop is provider-agnostic (depends only on `LLMClient`) and repo-agnostic
(depends only on `ToolBox`). It does not touch the DB or the network directly -
the caller does retrieval + worktree setup and hands in `hits` and `toolbox`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from backend.agent.tools import ToolBox
from backend.llm.base import LLMClient, Message, ToolCall, Usage
from backend.store.chunks_store import Hit

log = logging.getLogger("agent.loop")

DEFAULT_SYSTEM_PROMPT = """\
You are a precise assistant answering questions about ONE code repository.

Tools available to you:
- list_dir(path): list a directory's entries
- grep(pattern, path?): find where a Python regex matches (optionally scoped to a file/dir)
- read_file(path, start_line?, end_line?): read exact lines of a file

You are given some excerpts retrieved by vector search as a starting point. They
may be incomplete, out of order, or slightly off-target. Before answering, use
the tools to read the surrounding code and confirm details - do NOT guess about
code you have not actually seen.

When you have enough information, answer concisely. Support every concrete claim
with a `path:start-end` citation to code that appeared in the excerpts or that
you read with a tool. If the repository does not contain the answer, say so
plainly."""


@dataclass(frozen=True, slots=True)
class TraceStep:
    """One tool invocation, for the reasoning trace shown in the UI."""

    index: int
    tool: str
    arguments: dict
    ok: bool
    result_preview: str  # first slice of the tool output
    result_chars: int


@dataclass(slots=True)
class AgentResult:
    answer: str
    trace: list[TraceStep] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    iterations: int = 0
    stop_reason: str = "answered"  # answered | max_iterations | empty_response
    messages: list[Message] = field(default_factory=list)  # full transcript


_PREVIEW_CHARS = 240


def _format_excerpts(hits: list[Hit]) -> str:
    if not hits:
        return "(vector search returned no excerpts)"
    blocks = []
    for h in hits:
        blocks.append(
            f"--- {h.file_path}:{h.start_line}-{h.end_line} (similarity {h.score:.2f}) ---\n"
            f"{h.content}"
        )
    return "\n\n".join(blocks)


def _accumulate(total: Usage, add: Usage) -> Usage:
    return Usage(
        input_tokens=total.input_tokens + add.input_tokens,
        output_tokens=total.output_tokens + add.output_tokens,
    )


async def run_agent(
    *,
    question: str,
    hits: list[Hit],
    toolbox: ToolBox,
    llm: LLMClient,
    max_iterations: int = 6,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> AgentResult:
    messages: list[Message] = [
        Message(role="system", content=system_prompt),
        Message(
            role="user",
            content=(
                f"Question: {question}\n\n"
                f"Excerpts retrieved from the repository:\n\n{_format_excerpts(hits)}"
            ),
        ),
    ]
    result = AgentResult(answer="", messages=messages)
    specs = toolbox.specs

    for i in range(1, max_iterations + 1):
        result.iterations = i
        resp = await llm.complete(messages, tools=specs)
        result.usage = _accumulate(result.usage, resp.usage)

        # Record the assistant turn (may be text, tool calls, or both).
        messages.append(Message(role="assistant", content=resp.text, tool_calls=resp.tool_calls))

        if not resp.wants_tools:
            if resp.text:
                result.answer = resp.text
                result.stop_reason = "answered"
            else:
                result.stop_reason = "empty_response"
            log.info("agent finished after %d iteration(s): %s", i, result.stop_reason)
            return result

        # Execute every requested tool call before the next model turn.
        for tc in resp.tool_calls:
            await _run_one_tool(tc, toolbox, messages, result)

    # Cap reached with the model still wanting tools. Force a final answer with
    # tools withheld so it cannot ask for more.
    messages.append(
        Message(
            role="user",
            content=(
                "You have reached the tool-call limit. Answer the question now "
                "using only what you have already gathered. Keep the citations."
            ),
        )
    )
    final = await llm.complete(messages, tools=())
    result.usage = _accumulate(result.usage, final.usage)
    messages.append(Message(role="assistant", content=final.text))
    result.answer = final.text or ""
    result.stop_reason = "max_iterations"
    log.info("agent hit iteration cap (%d); forced final answer", max_iterations)
    return result


async def _run_one_tool(
    tc: ToolCall,
    toolbox: ToolBox,
    messages: list[Message],
    result: AgentResult,
) -> None:
    tool_result = await toolbox.run(tc.name, tc.arguments)
    messages.append(
        Message(
            role="tool",
            tool_call_id=tc.id,
            name=tc.name,
            content=tool_result.content,
        )
    )
    result.trace.append(
        TraceStep(
            index=len(result.trace) + 1,
            tool=tc.name,
            arguments=tc.arguments,
            ok=tool_result.ok,
            result_preview=tool_result.content[:_PREVIEW_CHARS],
            result_chars=len(tool_result.content),
        )
    )
    log.info(
        "tool %s(%s) -> ok=%s (%d chars)",
        tc.name,
        tc.arguments,
        tool_result.ok,
        len(tool_result.content),
    )
