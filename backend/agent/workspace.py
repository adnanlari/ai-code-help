"""Materialise a repo's working tree on demand for the agent's file tools.

Day 1's indexing pipeline deletes the clone right after embedding - the raw files
are disposable there. But the agent needs real files to `read_file` / `grep` /
`list_dir` at question time. So we re-create the working tree for the *exact
indexed commit* when a Q&A session needs it, and keep it around under
`WORKDIR/worktrees/<sha>` so repeated questions on the same repo reuse it.

Getting one specific commit cheaply: `git init` + `git fetch --depth 1 origin
<sha>` + `git checkout FETCH_HEAD`. GitHub allows fetching a commit by SHA
(`uploadpack.allowReachableSHA1InWant`), so this pulls just that one commit's
tree - no history, no branch guessing.

Cleanup of stale worktrees (LRU / TTL) is a follow-up; for now they persist in
WORKDIR, which is gitignored and safe to wipe by hand.
"""

from __future__ import annotations

import logging
from pathlib import Path

from backend.config import get_settings
from backend.indexing.clone import GitError, _run

log = logging.getLogger("workspace")


def worktree_root() -> Path:
    return Path(get_settings().workdir) / "worktrees"


def worktree_path(commit_sha: str) -> Path:
    return worktree_root() / commit_sha


def ensure_worktree(repo_url: str, commit_sha: str) -> Path:
    """Return a path to `repo_url`'s tree at `commit_sha`, cloning it if needed."""
    dest = worktree_path(commit_sha)
    if (dest / ".git").exists():
        return dest

    dest.mkdir(parents=True, exist_ok=True)
    try:
        _run(["git", "init", "--quiet"], cwd=str(dest))
        _run(["git", "remote", "add", "origin", repo_url], cwd=str(dest))
        _run(["git", "fetch", "--depth", "1", "--quiet", "origin", commit_sha], cwd=str(dest))
        _run(["git", "checkout", "--quiet", "FETCH_HEAD"], cwd=str(dest))
    except GitError:
        # leave nothing half-materialised behind
        import shutil

        shutil.rmtree(dest, ignore_errors=True)
        raise

    log.info("materialised worktree for %s @ %s", repo_url, commit_sha[:8])
    return dest
