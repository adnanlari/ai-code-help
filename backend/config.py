"""Typed application settings, loaded from environment / .env.

pydantic-settings reads each field from an env var of the same (case-insensitive)
name. Keeping this in one typed object means the rest of the code never touches
os.environ directly and a missing/misspelled var fails loudly at startup.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- external services ---
    database_url: str
    voyage_api_key: str = ""
    anthropic_api_key: str = ""

    # --- agent LLM (Strategy-selected; see backend/llm) ---
    llm_provider: str = "kimi"  # key into backend.llm._PROVIDERS
    llm_temperature: float = 0.0
    llm_max_tokens: int = 2048
    # Hard cap on tool-call rounds before the agent is forced to answer.
    agent_max_iterations: int = 6
    # Kimi K2.6 (Moonshot) - OpenAI-compatible endpoint.
    kimi_api_key: str = ""
    kimi_base_url: str = "https://api.moonshot.ai/v1"
    kimi_model: str = "kimi-k2.6"

    # --- local scratch space for clones ---
    workdir: str = "./workdir"

    # --- chunking ---
    chunk_lines: int = 60
    chunk_overlap_lines: int = 15

    # --- embeddings ---
    embed_model: str = "voyage-code-3"
    embed_dim: int = 1024
    # Per-request batch budget (estimated tokens). Voyage's hard ceiling is ~120k.
    embed_max_batch_tokens: int = 100_000
    # Client-side account-limit throttle. 0 = off. Set to your Voyage tier's
    # limits to avoid RateLimitError (no-payment-method free tier: rpm=3, tpm=10000).
    embed_rpm: int = 0
    embed_tpm: int = 0

    # --- retrieval ---
    top_k: int = 5

    # --- structured event log (one JSON object per line) ---
    events_log_path: str = "logs/events.jsonl"
    events_log_stderr: bool = True


@lru_cache
def get_settings() -> Settings:
    """Cached accessor so the .env file is parsed once per process."""
    return Settings()  # type: ignore[call-arg]  # values come from env/.env
