"""Kimi K2.6 (Moonshot) concrete Strategy, via the OpenAI-compatible API.

Moonshot speaks the exact OpenAI Chat Completions request/response shape, so we
use the official `openai` AsyncOpenAI client pointed at Kimi's base_url. All the
work here is translation:

    our Message[]  --_to_openai_messages-->  OpenAI messages[]
    our ToolSpec[] --_to_openai_tools----->  OpenAI tools[]
    OpenAI choice  --_from_choice---------> our LLMResponse

The translation helpers are module-level pure functions so they can be unit
tested without any network or API key.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from backend.llm.base import (
    LLMClient,
    LLMError,
    LLMResponse,
    Message,
    ToolCall,
    ToolSpec,
    Usage,
    parse_tool_arguments,
)

# OpenAI's finish_reason values map almost 1:1 onto ours; normalise the couple
# that differ in spelling so loop code never has to special-case a provider.
_FINISH_REASON_MAP = {
    "stop": "stop",
    "tool_calls": "tool_calls",
    "function_call": "tool_calls",  # legacy spelling
    "length": "length",
    "content_filter": "content_filter",
}


def _to_openai_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "tool":
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": m.tool_call_id,
                    "content": m.content or "",
                }
            )
        elif m.role == "assistant" and m.tool_calls:
            out.append(
                {
                    "role": "assistant",
                    "content": m.content,  # may be None when the turn is only tool calls
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": tc.arguments_raw
                                or json.dumps(tc.arguments, separators=(",", ":")),
                            },
                        }
                        for tc in m.tool_calls
                    ],
                }
            )
        else:
            out.append({"role": m.role, "content": m.content or ""})
    return out


def _to_openai_tools(tools: Sequence[ToolSpec]) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in tools
    ]


def _from_choice(choice: Any, model: str, usage_obj: Any, raw: Any) -> LLMResponse:
    msg = choice.message
    tool_calls: tuple[ToolCall, ...] = ()
    if getattr(msg, "tool_calls", None):
        tool_calls = tuple(
            ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=parse_tool_arguments(tc.function.arguments),
                arguments_raw=tc.function.arguments or "{}",
            )
            for tc in msg.tool_calls
        )

    usage = Usage()
    if usage_obj is not None:
        usage = Usage(
            input_tokens=getattr(usage_obj, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage_obj, "completion_tokens", 0) or 0,
        )

    return LLMResponse(
        text=msg.content,
        tool_calls=tool_calls,
        finish_reason=_FINISH_REASON_MAP.get(choice.finish_reason, choice.finish_reason or "stop"),
        usage=usage,
        model=model,
        raw=raw,
    )


class KimiClient(LLMClient):
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.moonshot.ai/v1",
        model: str = "kimi-k2.6",
        *,
        default_temperature: float = 0.0,
        default_max_tokens: int = 2048,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._model = model
        self._default_temperature = default_temperature
        self._default_max_tokens = default_max_tokens
        self._client: Any = None  # lazily created - constructing it needs the key

    @property
    def model(self) -> str:
        return self._model

    def _get_client(self) -> Any:
        if self._client is None:
            if not self._api_key:
                raise LLMError("KIMI_API_KEY is not set")
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:  # pragma: no cover - dependency is declared
                raise LLMError("the `openai` package is required for KimiClient") from exc
            self._client = AsyncOpenAI(api_key=self._api_key, base_url=self._base_url)
        return self._client

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSpec] = (),
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        client = self._get_client()
        payload_tools = _to_openai_tools(tools)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": _to_openai_messages(messages),
            "temperature": (self._default_temperature if temperature is None else temperature),
            "max_tokens": self._default_max_tokens if max_tokens is None else max_tokens,
        }
        if payload_tools is not None:
            kwargs["tools"] = payload_tools
            kwargs["tool_choice"] = "auto"

        try:
            resp = await client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - normalise every provider error
            raise LLMError(f"Kimi request failed: {exc}") from exc

        if not resp.choices:
            raise LLMError("Kimi returned no choices")
        return _from_choice(resp.choices[0], self._model, resp.usage, resp)
