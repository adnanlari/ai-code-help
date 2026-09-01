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

Rate limiting: Voyage's per-request ceiling is 128 inputs / ~120k tokens. On top
of that an *account* has RPM (requests/min) and TPM (tokens/min) limits - and the
no-payment-method free tier is a punishing 3 RPM / 10k TPM. So we:
  * split work into batches under EMBED_MAX_BATCH_TOKENS (token count estimated
    as chars/4 - good enough for budgeting),
  * optionally self-throttle to stay under EMBED_RPM / EMBED_TPM using a rolling
    60-second window (set them to 0 to disable, the default),
  * retry a RateLimitError with long minute-scale backoff, since the free tier's
    window is per-minute.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque

import voyageai
import voyageai.error

from backend.config import get_settings

# Voyage hard per-request limits (not account limits).
_MAX_BATCH_ITEMS = 128
_CHARS_PER_TOKEN = 4  # rough estimate for budgeting only

_MAX_RETRIES = 6
_RATELIMIT_BACKOFF_S = (15, 30, 60, 60, 90)  # per retry; free-tier window is 60s

_client: voyageai.Client | None = None


def _get_client() -> voyageai.Client:
    global _client
    if _client is None:
        _client = voyageai.Client(api_key=get_settings().voyage_api_key or None)
    return _client


def _est_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _batches(texts: list[str], max_batch_tokens: int) -> list[list[str]]:
    """Greedy pack into batches under both the item cap and the token budget."""
    batches: list[list[str]] = []
    cur: list[str] = []
    cur_tokens = 0
    for t in texts:
        tok = _est_tokens(t)
        too_big = cur and (len(cur) >= _MAX_BATCH_ITEMS or cur_tokens + tok > max_batch_tokens)
        if too_big:
            batches.append(cur)
            cur, cur_tokens = [], 0
        cur.append(t)
        cur_tokens += tok
    if cur:
        batches.append(cur)
    return batches


class _RollingLimiter:
    """Blocks until sending `tokens` more in one request keeps the last 60s under
    both `rpm` requests and `tpm` tokens. rpm/tpm <= 0 means 'no limit'."""

    def __init__(self, rpm: int, tpm: int) -> None:
        self.rpm = rpm
        self.tpm = tpm
        self._events: deque[tuple[float, int]] = deque()  # (timestamp, tokens)

    def _trim(self, now: float) -> None:
        while self._events and now - self._events[0][0] >= 60.0:
            self._events.popleft()

    def acquire(self, tokens: int) -> None:
        if self.rpm <= 0 and self.tpm <= 0:
            return
        while True:
            now = time.monotonic()
            self._trim(now)
            reqs = len(self._events)
            toks = sum(t for _, t in self._events)
            over_rpm = self.rpm > 0 and reqs + 1 > self.rpm
            over_tpm = self.tpm > 0 and toks + tokens > self.tpm
            if not over_rpm and not over_tpm:
                self._events.append((now, tokens))
                return
            # Sleep until the oldest event ages out of the window.
            wait = 60.0 - (now - self._events[0][0]) + 0.25
            time.sleep(max(wait, 0.25))


_limiter: _RollingLimiter | None = None


def _get_limiter() -> _RollingLimiter:
    global _limiter
    if _limiter is None:
        s = get_settings()
        _limiter = _RollingLimiter(s.embed_rpm, s.embed_tpm)
    return _limiter


def _embed_batch_sync(batch: list[str]) -> list[list[float]]:
    settings = get_settings()
    client = _get_client()
    limiter = _get_limiter()
    batch_tokens = sum(_est_tokens(t) for t in batch)

    for attempt in range(1, _MAX_RETRIES + 1):
        limiter.acquire(batch_tokens)
        try:
            resp = client.embed(
                batch,
                model=settings.embed_model,
                input_type="document",
                output_dimension=settings.embed_dim,
            )
            return resp.embeddings
        except voyageai.error.RateLimitError:
            if attempt == _MAX_RETRIES:
                raise
            back = _RATELIMIT_BACKOFF_S[min(attempt - 1, len(_RATELIMIT_BACKOFF_S) - 1)]
            time.sleep(back)
        except Exception as exc:  # noqa: BLE001 - Voyage raises several 5xx/timeout types
            if attempt == _MAX_RETRIES:
                raise
            msg = str(exc).lower()
            if any(m in msg for m in ("timeout", "timed out", "503", "502", "overloaded")):
                time.sleep(2**attempt)
                continue
            raise
    raise RuntimeError("unreachable")


async def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed chunk texts, preserving input order. Blocking SDK calls run in a
    worker thread so the event loop stays free."""
    if not texts:
        return []
    max_batch_tokens = get_settings().embed_max_batch_tokens
    out: list[list[float]] = []
    for batch in _batches(texts, max_batch_tokens):
        vectors = await asyncio.to_thread(_embed_batch_sync, batch)
        out.extend(vectors)
    return out
