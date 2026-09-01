"""Index one repo from the command line (no HTTP server needed).

Usage:
    python -m scripts.index_repo <repo_url> [ref]

Example:
    python -m scripts.index_repo https://github.com/pallets/click
    python -m scripts.index_repo https://github.com/pallets/click 8.1.7

Run it twice on the same URL to see the cache kick in: the second run returns
almost instantly with cached=True and makes zero embedding calls.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time

from backend import db
from backend.indexing.pipeline import index_repo

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


async def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    repo_url = argv[0]
    ref = argv[1] if len(argv) > 1 else "HEAD"

    await db.connect()
    try:
        started = time.perf_counter()
        result = await index_repo(repo_url, ref)
        elapsed = time.perf_counter() - started
    finally:
        await db.disconnect()

    print()
    print(f"  repo_id     : {result.repo_id}")
    print(f"  commit_sha  : {result.commit_sha}")
    print(f"  status      : {result.status}")
    print(f"  cached      : {result.cached}")
    print(f"  chunk_count : {result.chunk_count}")
    print(f"  elapsed     : {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(sys.argv[1:])))
