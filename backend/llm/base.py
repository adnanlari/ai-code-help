"""Provider-agnostic LLM interface (the Strategy) + the value types it speaks.

Why this layer exists
---------------------
The agent's tool-use loop only needs one capability: "given the conversation so
far and the tools available, tell me what the model wants to do next - emit text,
or call these tools." Every provider expresses that differently:

  * OpenAI / Kimi / DeepSeek: `choices[0].message` with a `tool_calls` array;
    stop signal is `finish_reason` ("stop" | "tool_calls" | "length").
  * Anthropic: a list of content blocks (`text` / `tool_use`); stop signal is
    `stop_reason` ("end_turn" | "tool_use" | "max_tokens").

`LLMClient` is the abstract Strategy. Each concrete subclass (KimiClient today,
a ClaudeClient later) translates our neutral `Message` / `ToolSpec` types into
its provider's request shape and the provider's response back into `LLMResponse`.
The loop code depends only on this module - swapping providers is a config change,
not a code change.

Everything here is a frozen dataclass: these are immutable values passed between
the loop and the adapter, not entities with behaviour.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A single request by the model to invoke one tool."""

    id: str  # provider-assigned; must be echoed back on the matching tool result
    name: str
    arguments: dict[str, Any]  # parsed JSON args; {} if the model sent malformed JSON
    arguments_raw: str = "{}"  # the original string, kept for debugging / error messages


@dataclass(frozen=True, slots=True)
class Message:
    """One turn in the conversation, in our neutral shape.

    - system/user/assistant text -> `content`
    - assistant asking for tools  -> `content` (often None) + `tool_calls`
    - a tool's result            -> role="tool" + `tool_call_id` + `name` + `content`
    """

    role: Role
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A tool offered to the model. `parameters` is a JSON Schema object."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Normalised result of one model call."""

    text: str | None
    tool_calls: tuple[ToolCall, ...]
    finish_reason: str  # normalised: "stop" | "tool_calls" | "length" | "content_filter" | other
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    raw: Any = None  # provider-native response object, for logging/debugging

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class LLMError(RuntimeError):
    """Wraps any provider-side failure so callers catch one exception type."""


def parse_tool_arguments(raw: str) -> dict[str, Any]:
    """Best-effort JSON-decode of a tool-call argument string.

    Models occasionally emit invalid JSON. Rather than crash the whole loop we
    return {} and let the caller surface a tool error back to the model, which it
    can then correct on the next turn.
    """
    try:
        val = json.loads(raw or "{}")
    except (json.JSONDecodeError, TypeError):
        return {}
    return val if isinstance(val, dict) else {}


class LLMClient(ABC):
    """Strategy interface. One method: run a single model turn."""

    @property
    @abstractmethod
    def model(self) -> str:
        """The concrete model id this client will call."""

    @abstractmethod
    async def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSpec] = (),
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Send `messages` (+ optional `tools`) and return one normalised response.

        Implementations must:
          * translate Message/ToolSpec into the provider request shape,
          * map the provider's stop signal onto our `finish_reason` vocabulary,
          * never raise provider-native exceptions - wrap them in LLMError.
        """
