"""Decide which files in a cloned repo are worth embedding.

Strategy: denylist + binary sniff, NOT an extension allowlist.

An allowlist ("only .py .js .ts ...") is safe but silently drops unusual-but-
valid source: .astro, .zig, .sql, Dockerfile, Makefile, plain README, shell
scripts with no extension. A denylist plus a cheap binary check is more
permissive and generalises across arbitrary GitHub repos. The cost of letting
an odd text file through is one wasted (tiny) embedding - low harm.

Three filters, cheapest first:
  1. path-based   - skip vendored/build dirs, lockfiles, known binary extensions
  2. size cap     - skip files > MAX_FILE_BYTES (generated bundles, data dumps)
  3. content sniff - read first 8 KB; skip if NUL byte present or not valid UTF-8
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

MAX_FILE_BYTES = 1_000_000  # ~1 MB
_SNIFF_BYTES = 8192

SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "bower_components",
    "vendor",
    "venv",
    ".venv",
    "env",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    "dist",
    "build",
    "out",
    "target",
    ".next",
    ".nuxt",
    ".svelte-kit",
    "coverage",
    ".idea",
    ".vscode",
    ".gradle",
}

SKIP_FILENAMES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "npm-shrinkwrap.json",
    "poetry.lock",
    "Pipfile.lock",
    "uv.lock",
    "Cargo.lock",
    "composer.lock",
    "Gemfile.lock",
    "go.sum",
    "flake.lock",
}

SKIP_SUFFIXES = {
    # images / fonts / media
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".ico",
    ".webp",
    ".svg",
    ".woff",
    ".woff2",
    ".ttf",
    ".otf",
    ".eot",
    ".mp3",
    ".wav",
    ".ogg",
    ".mp4",
    ".mov",
    ".avi",
    ".webm",
    # archives / binaries
    ".zip",
    ".tar",
    ".gz",
    ".tgz",
    ".bz2",
    ".7z",
    ".rar",
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".so",
    ".dylib",
    ".dll",
    ".a",
    ".o",
    ".class",
    ".pyc",
    ".pyo",
    ".wasm",
    ".exe",
    ".bin",
    ".dat",
    ".db",
    ".sqlite",
    ".sqlite3",
    # generated / minified
    ".min.js",
    ".min.css",
    ".map",
    # misc noise
    ".lock",
    ".log",
}


def _has_skipped_suffix(name: str) -> bool:
    lower = name.lower()
    return any(lower.endswith(suffix) for suffix in SKIP_SUFFIXES)


def _looks_binary(path: Path) -> bool:
    try:
        head = path.read_bytes()[:_SNIFF_BYTES]
    except OSError:
        return True
    if b"\x00" in head:
        return True
    try:
        head.decode("utf-8")
    except UnicodeDecodeError:
        return True
    return False


def iter_source_files(root: Path) -> Iterator[tuple[Path, str]]:
    """Yield (absolute_path, repo_relative_posix_path) for each keepable file."""
    root = root.resolve()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        rel_parts = path.relative_to(root).parts
        if any(part in SKIP_DIRS for part in rel_parts[:-1]):
            continue
        if path.name in SKIP_FILENAMES or _has_skipped_suffix(path.name):
            continue
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        if _looks_binary(path):
            continue
        yield path, "/".join(rel_parts)
