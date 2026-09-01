"""Offline tests for citation extraction + grounding verification."""

from __future__ import annotations

from backend.agent.citations import (
    build_evidence,
    extract_citations,
    verify_answer,
    verify_citations,
)
from backend.llm.base import Message
from backend.store.chunks_store import Hit

# --- extraction --------------------------------------------------------------


def test_extract_various_formats():
    text = (
        "See `src/app.py:42`, and index.html:660-720, plus (pkg/mod.py:5). "
        "The file README.md is also relevant. No cite here: just words."
    )
    cites = extract_citations(text)
    got = {(c.file_path, c.start_line, c.end_line) for c in cites}
    assert ("src/app.py", 42, 42) in got
    assert ("index.html", 660, 720) in got
    assert ("pkg/mod.py", 5, 5) in got
    assert ("README.md", 0, 0) in got  # file-level, no lines


def test_extract_dedups_and_normalises_paths():
    cites = extract_citations("./a.py:1 and a.py:1 and ./a.py:1")
    assert len(cites) == 1
    assert cites[0].file_path == "a.py"


def test_extract_ignores_bare_words():
    assert extract_citations("this mentions main() and options but no paths") == []


def test_extract_flips_reversed_range():
    (c,) = extract_citations("x.py:90-10")
    assert (c.start_line, c.end_line) == (10, 90)


# --- evidence building ------------------------------------------------------


def _tool_msg(name: str, content: str) -> Message:
    return Message(role="tool", name=name, tool_call_id=f"c_{name}", content=content)


def test_evidence_from_hits():
    ev = build_evidence([Hit("src/x.py", 60, 62, "code", 0.5)], [])
    assert ev.lines_by_file["src/x.py"] == {60, 61, 62}
    assert "src/x.py" in ev.seen_files


def test_evidence_from_read_file_message():
    content = "src/app.py (200 lines)  [showing lines 10-12 of 200]\n 10| a\n 11| b\n 12| c"
    ev = build_evidence([], [_tool_msg("read_file", content)])
    assert ev.lines_by_file["src/app.py"] == {10, 11, 12}


def test_evidence_from_grep_message():
    content = "src/a.py:4: def main():\nsrc/b.py:9: x = 1\n... stopped at 2 matches"
    ev = build_evidence([], [_tool_msg("grep", content)])
    assert ev.lines_by_file["src/a.py"] == {4}
    assert ev.lines_by_file["src/b.py"] == {9}


def test_evidence_from_list_dir_message_is_existence_only():
    content = "src/  (2 entries)\napp.py\nutil.py"
    ev = build_evidence([], [_tool_msg("list_dir", content)])
    assert "src/app.py" in ev.seen_files
    assert "src/app.py" not in ev.lines_by_file  # name seen, no lines


# --- verification --------------------------------------------------------


def test_verified_when_lines_overlap_a_hit():
    ev = build_evidence([Hit("a.py", 1, 60, "…", 0.9)], [])
    (v,) = verify_citations(extract_citations("a.py:10-20"), ev)
    assert v.status == "verified"


def test_unverified_lines_when_file_seen_but_wrong_lines():
    ev = build_evidence([Hit("a.py", 1, 60, "…", 0.9)], [])
    (v,) = verify_citations(extract_citations("a.py:400-420"), ev)
    assert v.status == "unverified_lines"


def test_unverified_file_when_never_seen():
    ev = build_evidence([Hit("a.py", 1, 60, "…", 0.9)], [])
    (v,) = verify_citations(extract_citations("secrets.py:1"), ev)
    assert v.status == "unverified_file"


def test_file_level_citation_verified_by_presence():
    ev = build_evidence([], [_tool_msg("list_dir", "./  (1 entries)\nconfig.py")])
    (v,) = verify_citations(extract_citations("see config.py for details"), ev)
    assert v.status == "verified"


# --- verify_answer roll-up --------------------------------------------------


def test_verify_answer_grounded_true_when_all_verified():
    hits = [Hit("a.py", 1, 10, "…", 0.8)]
    rep = verify_answer("the logic is in a.py:3-5", hits, [])
    assert rep.grounded is True
    assert rep.unverified_count == 0


def test_verify_answer_grounded_false_with_a_fabricated_cite():
    hits = [Hit("a.py", 1, 10, "…", 0.8)]
    rep = verify_answer("see a.py:3 and also ghost.py:99", hits, [])
    assert rep.grounded is False
    assert rep.unverified_count == 1
    assert {c.status for c in rep.citations} == {"verified", "unverified_file"}


def test_verify_answer_no_citations_is_not_grounded():
    rep = verify_answer("the repository does not contain authentication code", [], [])
    assert rep.citations == []
    assert rep.grounded is False
