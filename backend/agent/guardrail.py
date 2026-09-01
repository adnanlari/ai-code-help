"""Path-traversal guardrail for agent file tools.

The model chooses the paths passed to `read_file` / `grep` / `list_dir`. That
makes path handling a real security boundary: a compromised or confused model
could ask for `../../../../etc/passwd`, an absolute `/etc/shadow`, or a path that
tunnels out through a symlink. Every tool must resolve a model-supplied path
through `safe_path()` before it touches the filesystem.

The check, in order:
  1. reject NUL bytes (path smuggling / C-string truncation tricks),
  2. treat the input as *always* relative to the repo root - a leading "/" is
     stripped, so "/etc/passwd" becomes "<root>/etc/passwd", not the real one,
  3. `Path.resolve()` the result: this collapses ".." segments AND expands every
     symlink to its real target, so a symlink inside the repo that points outside
     is caught,
  4. require the resolved path to be the root itself or sit underneath it.

`resolve(strict=False)` is used for the candidate so that "file does not exist"
is a normal tool result, not a guardrail failure - only the root must exist.
"""

from __future__ import annotations

from pathlib import Path


class PathError(ValueError):
    """Raised when a model-supplied path is unsafe or malformed."""


def safe_path(repo_root: Path | str, user_path: str) -> Path:
    """Resolve `user_path` inside `repo_root`, or raise PathError.

    Returns an absolute, symlink-resolved Path guaranteed to be within the root.
    """
    root = Path(repo_root).resolve(strict=True)

    if user_path is None or "\x00" in user_path:
        raise PathError("path contains a NUL byte or is empty")

    cleaned = user_path.strip()
    if cleaned in ("", "."):
        return root

    # Force relativity: drop any leading slashes / drive so the join can't jump
    # to a filesystem-absolute location.
    rel = cleaned.lstrip("/\\")
    candidate = (root / rel).resolve(strict=False)

    if candidate != root and root not in candidate.parents:
        raise PathError(f"path escapes the repository root: {user_path!r}")
    return candidate
