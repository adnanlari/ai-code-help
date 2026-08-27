"""Manually query the vector store - the Day 1 acceptance demo.

Usage:
    python -m scripts.query_vectors <repo_id> "<your question>"

Example:
    python -m scripts.query_vectors 3f2b...  "how are options parsed?"

This embeds the question with input_type="query" (the asymmetric counterpart to
the "document" embeddings we stored), runs a cosine-distance search scoped to
the repo, and prints the top-K chunks with their similarity scores.
"""

from __future__ import annotations

import asyncio
import sys
from uuid import UUID

import voyageai

from backend import db
from backend.config import get_settings
from backend.store import chunks_store


def embed_query(text: str) -> list[float]:
    settings = get_settings()
    client = voyageai.Client(api_key=settings.voyage_api_key or None)
    resp = client.embed(
        [text],
        model=settings.embed_model,
        input_type="query",  # <-- query side of the asymmetric pair
        output_dimension=settings.embed_dim,
    )
    return resp.embeddings[0]


async def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    repo_id = UUID(argv[0])
    question = argv[1]
    k = get_settings().top_k

    query_vec = embed_query(question)

    await db.connect()
    try:
        hits = await chunks_store.similarity_search(repo_id, query_vec, k=k)
    finally:
        await db.disconnect()

    if not hits:
        print("no results - is the repo_id correct and indexed?")
        return 1

    print(f'\ntop-{len(hits)} for: "{question}"\n')
    for i, h in enumerate(hits, 1):
        print(f"[{i}] {h.score:.3f}  {h.file_path}:{h.start_line}-{h.end_line}")
        snippet = "\n".join(f"      {ln}" for ln in h.content.splitlines()[:6])
        print(snippet)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(sys.argv[1:])))
