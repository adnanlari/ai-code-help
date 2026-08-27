"""Raw-SQL access to the `chunks` table: bulk insert + similarity search."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from backend.db import get_pool
from backend.indexing.chunk import Chunk


@dataclass(frozen=True, slots=True)
class Hit:
    file_path: str
    start_line: int
    end_line: int
    content: str
    score: float  # cosine similarity in [-1, 1]; higher = closer


async def bulk_insert(
    repo_id: UUID,
    chunks: list[Chunk],
    embeddings: list[list[float]],
) -> int:
    """Insert all chunks for a repo. `chunks` and `embeddings` are row-aligned.

    executemany is plenty fast at this scale. The faster path for large repos is
    asyncpg's binary COPY (copy_records_to_table) - noted as a future tweak.
    """
    if len(chunks) != len(embeddings):
        raise ValueError("chunks / embeddings length mismatch")
    if not chunks:
        return 0

    records = [
        (repo_id, c.file_path, c.start_line, c.end_line, c.content, emb)
        for c, emb in zip(chunks, embeddings, strict=True)
    ]
    await get_pool().executemany(
        """
        insert into chunks
            (repo_id, file_path, start_line, end_line, content, embedding)
        values ($1, $2, $3, $4, $5, $6)
        """,
        records,
    )
    return len(records)


async def count_for_repo(repo_id: UUID) -> int:
    val = await get_pool().fetchval("select count(*) from chunks where repo_id = $1", repo_id)
    return int(val or 0)


async def similarity_search(
    repo_id: UUID,
    query_embedding: list[float],
    k: int = 5,
) -> list[Hit]:
    """Return the k chunks in `repo_id` closest to `query_embedding`.

    `<=>` is pgvector's cosine *distance* (0 = identical, 2 = opposite); we
    convert to similarity as 1 - distance and order ascending by distance so the
    HNSW index is used.
    """
    rows = await get_pool().fetch(
        """
        select file_path, start_line, end_line, content,
               1 - (embedding <=> $1::vector) as score
        from chunks
        where repo_id = $2
        order by embedding <=> $1::vector
        limit $3
        """,
        query_embedding,
        repo_id,
        k,
    )
    return [
        Hit(
            file_path=r["file_path"],
            start_line=r["start_line"],
            end_line=r["end_line"],
            content=r["content"],
            score=float(r["score"]),
        )
        for r in rows
    ]
