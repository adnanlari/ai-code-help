"""Line-window chunking.

Why line-based (for Day 1):
  * Cheap and deterministic - no tokenizer, no parser, same output every run.
  * Never splits mid-line, so a chunk always reads as whole lines of code.
  * start_line/end_line give us exact citation anchors ("src/foo.py:60-119")
    which Day 4's grounding check needs.

Why overlap:
  A function or block that straddles a window boundary would otherwise be cut in
  half in *both* neighbouring chunks. With `overlap` lines shared between
  consecutive windows, it stays intact in at least one - better retrieval recall.

Trade-offs to know:
  * Fixed windows ignore code structure: a chunk can start mid-function. AST /
    tree-sitter chunking aligns to function/class boundaries (better precision)
    at the cost of per-language parsers - a stretch goal.
  * Token-based chunking uses the embedding context window more fully but needs
    a tokenizer and loses the clean line-number mapping.
  * Bigger chunks blur meaning (the embedding averages more concepts); smaller
    chunks lose surrounding context. 40-80 lines is a reasonable middle; revisit
    with real eval numbers on Day 4.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Chunk:
    file_path: str  # repo-relative
    start_line: int  # 1-based, inclusive
    end_line: int  # 1-based, inclusive
    content: str


def chunk_file(
    file_path: str,
    text: str,
    *,
    size: int = 60,
    overlap: int = 15,
) -> list[Chunk]:
    """Split `text` into overlapping line windows.

    - An empty / whitespace-only file yields no chunks (nothing to embed).
    - A file shorter than `size` yields exactly one chunk.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")

    if not text.strip():
        return []

    lines = text.splitlines()
    step = size - overlap
    chunks: list[Chunk] = []

    for start in range(0, len(lines), step):
        window = lines[start : start + size]
        if not window:
            break
        chunks.append(
            Chunk(
                file_path=file_path,
                start_line=start + 1,
                end_line=start + len(window),
                content="\n".join(window),
            )
        )
        # Last window reached the end of the file - stop.
        if start + size >= len(lines):
            break

    return chunks
