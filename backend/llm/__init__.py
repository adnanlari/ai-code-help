"""LLM provider selection - the Strategy pattern's context/factory.

`LLMClient` (in .base) is the strategy interface; `KimiClient` (in .kimi) is a
concrete strategy. `_PROVIDERS` maps a config string to a builder, and
`get_llm_client()` returns the one named by `settings.llm_provider`.

Adding a provider = write a `SomeClient(LLMClient)` and add one line to
`_PROVIDERS`. No caller changes: the agent loop only ever imports `LLMClient`
and `get_llm_client`.
"""

from __future__ import annotations

from collections.abc import Callable

from backend.config import Settings, get_settings
from backend.llm.base import (
    LLMClient,
    LLMError,
    LLMResponse,
    Message,
    ToolCall,
    ToolSpec,
    Usage,
)
from backend.llm.kimi import KimiClient

__all__ = [
    "LLMClient",
    "LLMError",
    "LLMResponse",
    "Message",
    "ToolCall",
    "ToolSpec",
    "Usage",
    "get_llm_client",
    "available_providers",
]


def _build_kimi(s: Settings) -> LLMClient:
    return KimiClient(
        api_key=s.kimi_api_key,
        base_url=s.kimi_base_url,
        model=s.kimi_model,
        default_temperature=s.llm_temperature,
        default_max_tokens=s.llm_max_tokens,
    )


# name -> builder. Keys are what you put in LLM_PROVIDER / settings.llm_provider.
_PROVIDERS: dict[str, Callable[[Settings], LLMClient]] = {
    "kimi": _build_kimi,
}


def available_providers() -> list[str]:
    return sorted(_PROVIDERS)


def get_llm_client(provider: str | None = None, *, settings: Settings | None = None) -> LLMClient:
    """Return the configured LLM strategy.

    `provider` overrides `settings.llm_provider` (handy for tests / one-offs).
    `settings` lets tests inject a Settings without a real .env.
    """
    s = settings or get_settings()
    name = (provider or s.llm_provider).strip().lower()
    try:
        builder = _PROVIDERS[name]
    except KeyError:
        raise ValueError(
            f"unknown LLM provider {name!r}; available: {available_providers()}"
        ) from None
    return builder(s)
