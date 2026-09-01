"""Programmatic citation grounding for agent answers.

The agent is told to cite claims as `path:start-end`. Nothing stops it from
emitting a citation to code it was never shown - a fabricated `path:line` looks
exactly like a real one, which is worse than no citation at all (it manufactures
false confidence). So after the loop finishes we check every citation in the
answer against the **evidence set**: the union of

  * the chunks vector search retrieved (file + line span), and
  * the exact lines the agent actually saw via `read_file` / `grep`
    (parsed back out of the tool-result messages, so we trust what was *shown*,
    not what was *requested*), plus
  * file names revealed by `list_dir` (existence-only, weak evidence).

Each citation comes back tagged:
  * "verified"          - file in evidence AND cited lines overlap lines seen
  * "unverified_lines"  - file was seen, but not those lines
  * "unverified_file"   - that file was never retrieved or read (likely fabricated)

## Why flag-only, and NOT auto-correction

When a citation fails we annotate it and set `grounded=False`. We deliberately do
NOT bounce the failure back into the loop for the model to "fix":

  * Flagging is deterministic, unit-testable offline, and costs zero extra tokens
    or latency. A retry adds a paid round-trip per bad citation and can oscillate,
    needing its own cap and backoff.
  * This is a read-only code-Q&A tool. A flagged-but-present citation costs the
    reader one manual double-check - acceptable. (A legal-filing product would
    choose differently: reject, don't ship an unverified cite.)
  * The hook stays open: `verify_answer` returns structured results, so a
    retry/self-correction policy can be layered in `service`/`loop` later without
    touching this module.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from backend.llm.base import Message
from backend.store.chunks_store import Hit

# A citation: an optional dir path + filename WITH an extension, optionally
# followed by :line or :start-end. Requiring an extension keeps us from matching
# ordinary prose words. Surrounding backticks/parens are handled by the search.
_CITATION_RE = re.compile(
    r"(?P<path>(?:[\w\-.]+/)*[\w\-.]+\.[A-Za-z0-9]+)"
    r"(?::(?P<start>\d+)(?:\s*-\s*(?P<end>\d+))?)?"
)

# read_file output: header "path (N lines) [..]" then body lines "  12| code"
_READ_HEADER_RE = re.compile(r"^(?P<path>\S.*?) \(\d+ lines\)")
_READ_LINE_RE = re.compile(r"^\s*(?P<n>\d+)\| ")
# grep output lines: "path:123: matched text"
_GREP_LINE_RE = re.compile(r"^(?P<path>[\w\-./]+):(?P<n>\d+): ")
# list_dir header: "some/dir/  (N entries)"
_LISTDIR_HEADER_RE = re.compile(r"^(?P<dir>.*?)/  \(\d+ entries\)$")


def _norm(path: str) -> str:
    return path.strip().lstrip("./").lstrip("/")


@dataclass(frozen=True, slots=True)
class Citation:
    raw: str
    file_path: str
    start_line: int  # 0 => file-level citation, no line given
    end_line: int


@dataclass(frozen=True, slots=True)
class VerifiedCitation:
    raw: str
    file_path: str
    start_line: int
    end_line: int
    status: str  # verified | unverified_lines | unverified_file


@dataclass(slots=True)
class Evidence:
    """What the agent was actually shown."""

    lines_by_file: dict[str, set[int]] = field(default_factory=dict)  # precise line evidence
    seen_files: set[str] = field(default_factory=set)  # existence-only (incl. list_dir)

    def add_span(self, path: str, start: int, end: int) -> None:
        p = _norm(path)
        self.seen_files.add(p)
        if start and end:
            self.lines_by_file.setdefault(p, set()).update(range(start, end + 1))

    def add_line(self, path: str, n: int) -> None:
        p = _norm(path)
        self.seen_files.add(p)
        self.lines_by_file.setdefault(p, set()).add(n)

    def add_file(self, path: str) -> None:
        self.seen_files.add(_norm(path))


# --- extraction -----------------------------------------------------------


def extract_citations(text: str) -> list[Citation]:
    """Pull every `path[:line[-line]]` reference out of answer text (dedup, ordered)."""
    out: list[Citation] = []
    seen: set[tuple[str, int, int]] = set()
    for m in _CITATION_RE.finditer(text or ""):
        start = int(m.group("start")) if m.group("start") else 0
        end = int(m.group("end")) if m.group("end") else start
        if end < start:
            start, end = end, start
        path = _norm(m.group("path"))
        key = (path, start, end)
        if key in seen:
            continue
        seen.add(key)
        out.append(Citation(raw=m.group(0), file_path=path, start_line=start, end_line=end))
    return out


def build_evidence(hits: list[Hit], messages: list[Message]) -> Evidence:
    """Union of retrieved chunk spans + lines actually shown by the tools."""
    ev = Evidence()
    for h in hits:
        ev.add_span(h.file_path, h.start_line, h.end_line)

    for msg in messages:
        if msg.role != "tool" or not msg.content:
            continue
        if msg.name == "read_file":
            _absorb_read_file(msg.content, ev)
        elif msg.name == "grep":
            for line in msg.content.splitlines():
                gm = _GREP_LINE_RE.match(line)
                if gm:
                    ev.add_line(gm.group("path"), int(gm.group("n")))
        elif msg.name == "list_dir":
            _absorb_list_dir(msg.content, ev)
    return ev


def _absorb_read_file(content: str, ev: Evidence) -> None:
    lines = content.splitlines()
    if not lines:
        return
    hm = _READ_HEADER_RE.match(lines[0])
    if not hm:
        return
    path = hm.group("path")
    for line in lines[1:]:
        lm = _READ_LINE_RE.match(line)
        if lm:
            ev.add_line(path, int(lm.group("n")))


def _absorb_list_dir(content: str, ev: Evidence) -> None:
    lines = content.splitlines()
    if not lines:
        return
    hm = _LISTDIR_HEADER_RE.match(lines[0])
    base = "" if not hm or hm.group("dir") in (".", "") else hm.group("dir") + "/"
    for entry in lines[1:]:
        entry = entry.strip()
        if not entry or entry == "(empty)" or entry.endswith("/"):
            continue
        ev.add_file(base + entry)


# --- verification ---------------------------------------------------------


def verify_citations(citations: list[Citation], ev: Evidence) -> list[VerifiedCitation]:
    result: list[VerifiedCitation] = []
    for c in citations:
        result.append(
            VerifiedCitation(
                raw=c.raw,
                file_path=c.file_path,
                start_line=c.start_line,
                end_line=c.end_line,
                status=_status(c, ev),
            )
        )
    return result


def _status(c: Citation, ev: Evidence) -> str:
    known_lines = ev.lines_by_file.get(c.file_path)
    if c.file_path not in ev.seen_files and known_lines is None:
        return "unverified_file"
    if c.start_line == 0:  # file-level citation, no lines to check
        return "verified"
    if known_lines and any(n in known_lines for n in range(c.start_line, c.end_line + 1)):
        return "verified"
    return "unverified_lines"


@dataclass(slots=True)
class GroundingReport:
    citations: list[VerifiedCitation]
    grounded: bool  # >=1 citation and none unverified

    @property
    def unverified_count(self) -> int:
        return sum(1 for c in self.citations if c.status != "verified")


def verify_answer(answer: str, hits: list[Hit], messages: list[Message]) -> GroundingReport:
    cites = verify_citations(extract_citations(answer), build_evidence(hits, messages))
    grounded = bool(cites) and all(c.status == "verified" for c in cites)
    return GroundingReport(citations=cites, grounded=grounded)
