from frontend.render import (
    citation_line,
    grounding_summary,
    meta_line,
    retrieved_line,
    trace_line,
)


def test_grounding_summary_states():
    assert "no citations" in grounding_summary({"citations": []})
    assert grounding_summary({"grounded": True, "citations": [{"status": "verified"}]}).startswith(
        "✅"
    )
    s = grounding_summary(
        {
            "grounded": False,
            "citations": [{"status": "verified"}, {"status": "unverified_file"}],
        }
    )
    assert s.startswith("⚠️") and "1 of 2" in s


def test_citation_line_icons():
    assert citation_line({"raw": "a.py:1", "status": "verified"}).startswith("🟢")
    assert citation_line({"raw": "a.py:9", "status": "unverified_lines"}).startswith("🟡")
    assert citation_line({"raw": "x.py:1", "status": "unverified_file"}).startswith("🔴")
    assert "`a.py:1`" in citation_line({"raw": "a.py:1", "status": "verified"})


def test_trace_line_formats_args_and_status():
    line = trace_line(
        {"index": 2, "tool": "grep", "arguments": {"pattern": "x"}, "ok": True, "result_chars": 42}
    )
    assert "2. grep" in line
    assert "pattern='x'" in line
    assert "✅" in line and "42 chars" in line
    assert "❌" in trace_line({"index": 1, "tool": "read_file", "ok": False, "arguments": {}})


def test_retrieved_line():
    assert (
        retrieved_line({"file_path": "src/a.py", "start_line": 1, "end_line": 9, "score": 0.7123})
        == "`src/a.py:1-9` · score 0.71"
    )


def test_meta_line_has_key_fields():
    line = meta_line(
        {
            "model": "kimi-k2.6",
            "iterations": 3,
            "stop_reason": "answered",
            "usage": {"total_tokens": 8535},
            "request_id": "a1b2c3d4",
        }
    )
    for bit in ("kimi-k2.6", "3 iteration", "answered", "8535 tokens", "a1b2c3d4"):
        assert bit in line


def test_helpers_tolerate_missing_keys():
    # partial / malformed payloads must not raise
    trace_line({})
    retrieved_line({})
    meta_line({})
    citation_line({})
