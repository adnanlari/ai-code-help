"""Offline tests for the LLM Strategy layer - no network, no API key needed."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from backend.config import Settings
from backend.llm import available_providers, get_llm_client
from backend.llm.base import LLMError, Message, ToolCall, ToolSpec, parse_tool_arguments
from backend.llm.kimi import KimiClient, _from_choice, _to_openai_messages, _to_openai_tools


def _settings(**over) -> Settings:
    base = {"database_url": "postgresql://x", "kimi_api_key": "test-key"}
    base.update(over)
    return Settings(**base)  # type: ignore[arg-type]


# --- parse_tool_arguments -------------------------------------------------------


def test_parse_tool_arguments_ok():
    assert parse_tool_arguments('{"path": "a.py", "n": 3}') == {"path": "a.py", "n": 3}


def test_parse_tool_arguments_malformed_returns_empty():
    assert parse_tool_arguments("{not json") == {}
    assert parse_tool_arguments("") == {}
    assert parse_tool_arguments("[1,2,3]") == {}  # not an object


# --- Message -> OpenAI shape --------------------------------------------------


def test_to_openai_messages_roles():
    msgs = [
        Message(role="system", content="be terse"),
        Message(role="user", content="hi"),
    ]
    out = _to_openai_messages(msgs)
    assert out == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hi"},
    ]


def test_to_openai_messages_assistant_tool_call_and_result_roundtrip():
    tc = ToolCall(
        id="call_1",
        name="read_file",
        arguments={"path": "a.py"},
        arguments_raw='{"path": "a.py"}',
    )
    msgs = [
        Message(role="assistant", content=None, tool_calls=(tc,)),
        Message(role="tool", tool_call_id="call_1", name="read_file", content="file body"),
    ]
    out = _to_openai_messages(msgs)

    assert out[0]["role"] == "assistant"
    assert out[0]["content"] is None
    fn = out[0]["tool_calls"][0]
    assert fn["id"] == "call_1"
    assert fn["type"] == "function"
    assert fn["function"]["name"] == "read_file"
    assert json.loads(fn["function"]["arguments"]) == {"path": "a.py"}

    assert out[1] == {"role": "tool", "tool_call_id": "call_1", "content": "file body"}


def test_to_openai_tools_shape_or_none():
    assert _to_openai_tools([]) is None
    spec = ToolSpec(
        name="grep",
        description="search",
        parameters={"type": "object", "properties": {"q": {"type": "string"}}},
    )
    (out,) = _to_openai_tools([spec])
    assert out["type"] == "function"
    assert out["function"]["name"] == "grep"
    assert out["function"]["parameters"]["properties"]["q"]["type"] == "string"


# --- OpenAI choice -> LLMResponse -------------------------------------------


def _fake_choice(*, content=None, tool_calls=None, finish_reason="stop"):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(message=msg, finish_reason=finish_reason)


def test_from_choice_plain_text():
    choice = _fake_choice(content="the answer", finish_reason="stop")
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=4)
    resp = _from_choice(choice, "kimi-k2.6", usage, raw="R")

    assert resp.text == "the answer"
    assert resp.tool_calls == ()
    assert resp.wants_tools is False
    assert resp.finish_reason == "stop"
    assert resp.usage.input_tokens == 10
    assert resp.usage.total_tokens == 14
    assert resp.model == "kimi-k2.6"
    assert resp.raw == "R"


def test_from_choice_with_tool_calls():
    fn = SimpleNamespace(name="list_dir", arguments='{"path": "src"}')
    tc = SimpleNamespace(id="call_9", function=fn)
    choice = _fake_choice(tool_calls=[tc], finish_reason="tool_calls")
    resp = _from_choice(choice, "kimi-k2.6", None, raw=None)

    assert resp.wants_tools is True
    assert resp.finish_reason == "tool_calls"
    (call,) = resp.tool_calls
    assert call.id == "call_9"
    assert call.name == "list_dir"
    assert call.arguments == {"path": "src"}
    assert resp.usage.total_tokens == 0  # no usage object -> zeros


def test_from_choice_malformed_tool_args_do_not_crash():
    fn = SimpleNamespace(name="grep", arguments="{oops")
    tc = SimpleNamespace(id="c1", function=fn)
    resp = _from_choice(_fake_choice(tool_calls=[tc], finish_reason="tool_calls"), "m", None, None)
    (call,) = resp.tool_calls
    assert call.arguments == {}
    assert call.arguments_raw == "{oops"


# --- factory / Strategy selection ------------------------------------------


def test_get_llm_client_returns_kimi():
    client = get_llm_client("kimi", settings=_settings())
    assert isinstance(client, KimiClient)
    assert client.model == "kimi-k2.6"


def test_get_llm_client_uses_settings_default():
    client = get_llm_client(settings=_settings(llm_provider="kimi", kimi_model="kimi-k2.7-code"))
    assert client.model == "kimi-k2.7-code"


def test_get_llm_client_unknown_provider():
    with pytest.raises(ValueError, match="unknown LLM provider"):
        get_llm_client("gpt5", settings=_settings())
    assert "kimi" in available_providers()


@pytest.mark.asyncio
async def test_kimi_client_without_key_raises_llmerror_lazily():
    # Construction must not need a key; only the first call does.
    client = KimiClient(api_key="", model="kimi-k2.6")
    with pytest.raises(LLMError, match="KIMI_API_KEY"):
        await client.complete([Message(role="user", content="hi")])
