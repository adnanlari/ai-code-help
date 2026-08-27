"""Turn chunk text into vectors with Voyage AI.

Why a code-specific model (voyage-code-3): it is trained on code + technical
text, so identifiers, syntax and API usage land closer together in vector space
than they would with a general-purpose text model. Retrieval quality on code
queries is noticeably better.

Why input_type matters:
  Voyage embeds documents and queries into the *same* space but prepends a
  different instruction to each ("represent this document..." vs "represent
  this query..."). Using "document" at index time and "query" at search time -
  asymmetric embedding - measurably improves retrieval versus treating both
  sides identically. embed.py always uses "document"; the query side lives in
  scripts/query_vectors.py (Day 2 moves it into the agent).

Batching: Voyage caps a request at 128 inputs and a total token budget. We
chunk the work into <=128-item batches with a char-based safety cap and retry
with exponential backoff on rate limits (the free tier's RPM is low).
"""

from __future__ import annotations

import asyncio
import time

import voyageai

from backend.config import get_settings

_MAX_BATCH_ITEMS = 128
_MAX_BATCH_CHARS = 400_000  # ~100k tokens, well under the per-request ceiling
_MAX_RETRIES = 5

_client: voyageai.Client | None = None


def _get_client() -> voyageai.Client:
    global _client
    if _client is None:
        _client = voyageai.Client(api_key=get_settings().voyage_api_key or None)
    return _client


def _batches(texts: list[str]) -> list[list[str]]:
    batches: list[list[str]] = []
    cur: list[str] = []
    cur_chars = 0
    for t in texts:
        if cur and (len(cur) >= _MAX_BATCH_ITEMS or cur_chars + len(t) > _MAX_BATCH_CHARS):
            batches.append(cur)
            cur, cur_chars = [], 0
        cur.append(t)
        cur_chars += len(t)
    if cur:
        batches.append(cur)
    return batches


def _embed_batch_sync(batch: list[str]) -> list[list[float]]:
    settings = get_settings()
    client = _get_client()
    delay = 2.0
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = client.embed(
                batch,
                model=settings.embed_model,
                input_type="document",
                output_dimension=settings.embed_dim,
            )
            return resp.embeddings
        except Exception as exc:  # noqa: BLE001 - Voyage raises several rate/5xx types
            if attempt == _MAX_RETRIES:
                raise
            msg = str(exc).lower()
            transient = ("rate", "429", "timeout", "timed out", "503", "502", "overloaded")
            if any(m in msg for m in transient):
                time.sleep(delay)
                delay *= 2
                continue
            raise
    raise RuntimeError("unreachable")


async def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed chunk texts, preserving input order. Runs the blocking SDK calls
    in a worker thread so the event loop stays free."""
    if not texts:
        return []
    out: list[list[float]] = []
    for batch in _batches(texts):
        vectors = await asyncio.to_thread(_embed_batch_sync, batch)
        out.extend(vectors)
    return out
