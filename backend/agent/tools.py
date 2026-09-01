"""The three file tools the agent can call, bound to one repo working tree.

Design notes
------------
* Every tool returns a **string** - exactly what gets fed back to the model as
  the tool result. Errors are returned as `"error: ..."` strings, never raised,
  so a bad path or missing file becomes something the model can read and correct
  on its next turn instead of crashing the loop.
* Outputs are **bounded** (line/byte/match caps). Tool results go straight into
  the next prompt, so an un-capped `read_file` on a 5k-line file would blow up
  both the context window and the bill. When output is truncated we say so and
  tell the model how to narrow the request.
* Path arguments always go through `guardrail.safe_path` before any filesystem
  access - see that module for the traversal defence.
* `grep` / `list_dir` reuse `indexing.filter` (SKIP_DIRS + is_source_file) so the
  agent sees the same slice of the repo that was indexed - no surprises where a
  chunk exists for a file the agent "can't see", or vice versa.
* `ToolBox.run` is async and offloads the blocking fs/regex work to a thread so
  it doesn't stall the event loop during the agent loop.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path

from backend.agent.guardrail import PathError, safe_path
from backend.indexing.filter import SKIP_DIRS, SKIP_FILENAMES, is_source_file
from backend.llm.base import ToolSpec

# --- output caps -------------------------------------------------------------
READ_FILE_MAX_LINES = 400
READ_FILE_MAX_BYTES = 32_000
GREP_MAX_RESULTS = 60
GREP_MAX_LINE_LEN = 300
LIST_DIR_MAX_ENTRIES = 300


# --- tool schemas (JSON Schema, provider-neutral via ToolSpec) --------------

TOOL_SPECS: list[ToolSpec] = [
    ToolSpec(
        name="read_file",
        description=(
            "Read a text file from the repository. Optionally restrict to a line "
            "range with start_line/end_line (1-based, inclusive). Output is line-"
            "numbered and truncated if very large."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "repo-relative file path"},
                "start_line": {"type": "integer", "minimum": 1},
                "end_line": {"type": "integer", "minimum": 1},
            },
            "required": ["path"],
        },
    ),
    ToolSpec(
        name="grep",
        description=(
            "Search the repository for a Python regular expression. Returns "
            "'path:line: text' hits. Scope with `path` (a file or directory); "
            "omit it to search the whole repo."
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Python regex"},
                "path": {
                    "type": "string",
                    "description": "optional repo-relative file or dir to scope the search",
                },
                "ignore_case": {"type": "boolean", "default": False},
                "max_results": {"type": "integer", "minimum": 1, "maximum": GREP_MAX_RESULTS},
            },
            "required": ["pattern"],
        },
    ),
    ToolSpec(
        name="list_dir",
        description=(
            "List the entries of a directory in the repository. Directories are "
            "shown with a trailing '/'. Vendored/build dirs are omitted."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "repo-relative directory (default: repo root)",
                },
            },
            "required": [],
        },
    ),
]

TOOL_NAMES = frozenset(spec.name for spec in TOOL_SPECS)


# --- implementations -------------------------------------------------------


def _read_file(
    root: Path,
    *,
    path: str,
    start_line: int | None = None,
    end_line: int | None = None,
) -> str:
    target = safe_path(root, path)
    if not target.exists():
        return f"error: no such file: {path}"
    if target.is_dir():
        return f"error: {path} is a directory (use list_dir)"

    raw = target.read_bytes()
    truncated_bytes = False
    if len(raw) > READ_FILE_MAX_BYTES:
        raw = raw[:READ_FILE_MAX_BYTES]
        truncated_bytes = True
    lines = raw.decode("utf-8", errors="replace").splitlines()
    total = len(lines)

    lo = 1 if start_line is None else max(1, start_line)
    hi = total if end_line is None else min(total, end_line)
    if lo > total:
        return f"error: start_line {lo} is past end of file ({total} lines)"
    window = lines[lo - 1 : hi]

    notes: list[str] = []
    if len(window) > READ_FILE_MAX_LINES:
        window = window[:READ_FILE_MAX_LINES]
        hi = lo + READ_FILE_MAX_LINES - 1
        notes.append(f"showing lines {lo}-{hi} of {total}; pass start_line/end_line to see more")
    if truncated_bytes:
        notes.append(f"file truncated at {READ_FILE_MAX_BYTES} bytes")

    width = len(str(hi))
    body = "\n".join(f"{n:>{width}}| {ln}" for n, ln in enumerate(window, start=lo))
    header = f"{path} ({total} lines)"
    if notes:
        header += "  [" + "; ".join(notes) + "]"
    return f"{header}\n{body}"


def _iter_files(search_root: Path, repo_root: Path):
    """Yield (abs_path, repo_relative_posix) under search_root, pruning SKIP_DIRS
    and non-source files, paths reported relative to the repo root."""
    if search_root.is_file():
        if is_source_file(search_root):
            yield search_root, search_root.relative_to(repo_root).as_posix()
        return
    for p in sorted(search_root.rglob("*")):
        rel_parts = p.relative_to(repo_root).parts
        if any(part in SKIP_DIRS for part in rel_parts[:-1]):
            continue
        if is_source_file(p):
            yield p, "/".join(rel_parts)


def _grep(
    root: Path,
    *,
    pattern: str,
    path: str | None = None,
    ignore_case: bool = False,
    max_results: int | None = None,
) -> str:
    try:
        rx = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
    except re.error as exc:
        return f"error: invalid regex: {exc}"

    cap = min(max_results or GREP_MAX_RESULTS, GREP_MAX_RESULTS)
    search_root = safe_path(root, path) if path else root
    if not search_root.exists():
        return f"error: no such path: {path}"

    hits: list[str] = []
    truncated = False
    for abs_path, rel in _iter_files(search_root, root):
        try:
            text = abs_path.read_text("utf-8", errors="replace")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if rx.search(line):
                snippet = line.strip()
                if len(snippet) > GREP_MAX_LINE_LEN:
                    snippet = snippet[:GREP_MAX_LINE_LEN] + " ..."
                hits.append(f"{rel}:{lineno}: {snippet}")
                if len(hits) >= cap:
                    truncated = True
                    break
        if truncated:
            break

    if not hits:
        return f"no matches for /{pattern}/"
    out = "\n".join(hits)
    if truncated:
        out += f"\n... stopped at {cap} matches; narrow the pattern or scope with `path`"
    return out


def _list_dir(root: Path, *, path: str | None = None) -> str:
    target = safe_path(root, path or ".")
    if not target.exists():
        return f"error: no such directory: {path}"
    if not target.is_dir():
        return f"error: {path} is not a directory (use read_file)"

    entries: list[str] = []
    for child in sorted(target.iterdir(), key=lambda c: c.name.lower()):
        if child.name in SKIP_DIRS or child.name in SKIP_FILENAMES:
            continue
        entries.append(child.name + "/" if child.is_dir() else child.name)

    shown = entries[:LIST_DIR_MAX_ENTRIES]
    rel = target.relative_to(root).as_posix() or "."
    head = f"{rel}/  ({len(entries)} entries)"
    if len(entries) > len(shown):
        head += f"  [showing first {LIST_DIR_MAX_ENTRIES}]"
    return head + "\n" + "\n".join(shown) if shown else head + "\n(empty)"


# --- dispatcher ----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ToolResult:
    name: str
    ok: bool
    content: str


class ToolBox:
    """All file tools bound to one repo working tree."""

    def __init__(self, repo_root: Path | str) -> None:
        self.repo_root = Path(repo_root).resolve(strict=True)

    @property
    def specs(self) -> list[ToolSpec]:
        return TOOL_SPECS

    async def run(self, name: str, arguments: dict) -> ToolResult:
        if name not in TOOL_NAMES:
            return ToolResult(name, False, f"error: unknown tool {name!r}")
        try:
            content = await asyncio.to_thread(self._run_sync, name, arguments)
        except PathError as exc:
            return ToolResult(name, False, f"error: {exc}")
        except TypeError as exc:
            # bad / missing arguments from the model
            return ToolResult(name, False, f"error: bad arguments for {name}: {exc}")
        except Exception as exc:  # noqa: BLE001 - surface as a tool error, don't kill the loop
            return ToolResult(name, False, f"error: {name} failed: {exc}")
        return ToolResult(name, not content.startswith("error:"), content)

    def _run_sync(self, name: str, arguments: dict) -> str:
        args = dict(arguments or {})
        if name == "read_file":
            return _read_file(self.repo_root, **args)
        if name == "grep":
            return _grep(self.repo_root, **args)
        if name == "list_dir":
            return _list_dir(self.repo_root, **args)
        raise AssertionError(name)  # guarded by TOOL_NAMES above
