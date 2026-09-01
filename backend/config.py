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


@lru_cache
def get_settings() -> Settings:
    """Cached accessor so the .env file is parsed once per process."""
    return Settings()  # type: ignore[call-arg]  # values come from env/.env
