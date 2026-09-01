"""Delete a repo's index so it can be re-indexed from scratch (dev helper).

Usage:
    python -m scripts.reset_repo <repo_url | repo_id>
    python -m scripts.reset_repo --all-failed        # clear every failed/stuck row

Deleting the repos row cascades to its chunks (ON DELETE CASCADE). This is a
convenience for development - in production you would never hand-delete cache
entries like this.
"""

from __future__ import annotations

import asyncio
import sys
from uuid import UUID

from backend import db


def _looks_like_uuid(s: str) -> bool:
    try:
        UUID(s)
        return True
    except ValueError:
        return False


async def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2

    await db.connect()
    try:
        pool = db.get_pool()
        if argv[0] == "--all-failed":
            status = await pool.execute("delete from repos where status in ('failed', 'indexing')")
            print(f"cleared rows: {status}")
            return 0

        target = argv[0]
        if _looks_like_uuid(target):
            status = await pool.execute("delete from repos where id = $1", UUID(target))
        else:
            status = await pool.execute("delete from repos where repo_url = $1", target)
        print(f"{status}  (chunks cascade-deleted)")
        if status.endswith(" 0"):
            print("nothing matched - check the url/id")
            return 1
        return 0
    finally:
        await db.disconnect()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(sys.argv[1:])))
