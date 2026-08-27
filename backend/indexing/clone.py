"""Git operations: resolve a commit SHA cheaply, then shallow-clone.

Key idea: we want to check the cache *before* paying for a clone. `git
ls-remote` asks the remote for its ref -> SHA mapping over a single HTTPS
round-trip without downloading any objects. So the flow is:

    sha = resolve_sha(url, ref)      # cheap, no download
    ... check DB for (url, sha) ...   # cache hit -> stop here
    path = shallow_clone(url, ref)    # only on cache miss

`--depth 1` fetches just the tree at the tip of `ref`, not the full history.
History (every past version of every file) is large and useless for indexing
the current state of the code.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class GitError(RuntimeError):
    pass


def _run(args: list[str], cwd: str | None = None) -> str:
    proc = subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if proc.returncode != 0:
        raise GitError(f"`{' '.join(args)}` failed: {proc.stderr.strip()}")
    return proc.stdout


def resolve_sha(repo_url: str, ref: str = "HEAD") -> str:
    """Return the 40-char commit SHA that `ref` points to on the remote.

    No repository data is downloaded - this is one metadata round-trip.
    """
    out = _run(["git", "ls-remote", repo_url, ref])
    if not out.strip():
        # `ref` might already be a raw commit SHA (ls-remote returns nothing for
        # those). Accept it if it looks like one; we'll verify at clone time.
        if _SHA_RE.match(ref):
            return ref
        raise GitError(f"ref {ref!r} not found on {repo_url}")
    sha = out.split()[0]
    if not _SHA_RE.match(sha):
        raise GitError(f"unexpected ls-remote output: {out!r}")
    return sha


def shallow_clone(repo_url: str, ref: str, workdir: str) -> Path:
    """Clone `repo_url` at `ref` into a fresh temp dir under `workdir`.

    Returns the path to the working tree. Caller is responsible for deleting it
    (see cleanup()) once chunking + embedding is done.
    """
    Path(workdir).mkdir(parents=True, exist_ok=True)
    dest = Path(tempfile.mkdtemp(prefix="repo_", dir=workdir))

    args = ["git", "clone", "--depth", "1", "--quiet"]
    # A branch/tag name can be passed to --branch; a raw commit SHA cannot, so
    # for SHAs we clone the default branch shallowly (good enough for Day 1;
    # fetching an arbitrary historical SHA needs a deeper fetch).
    if ref not in ("HEAD",) and not _SHA_RE.match(ref):
        args += ["--branch", ref]
    args += [repo_url, str(dest)]

    _run(args)
    return dest


def cleanup(path: Path) -> None:
    """Delete a cloned working tree. Best-effort; never raises."""
    import shutil

    shutil.rmtree(path, ignore_errors=True)
