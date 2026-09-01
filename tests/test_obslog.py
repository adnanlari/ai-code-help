import json
from pathlib import Path

from backend.llm.pricing import estimate_cost_usd
from backend.obslog import log_event, ms_since


def test_writes_one_json_line_with_ts_and_fields(tmp_path):
    p = tmp_path / "e.jsonl"
    rec = log_event("ask", path=str(p), request_id="abc123", iterations=3, grounded=True)
    parsed = json.loads(p.read_text().strip())
    assert parsed == rec
    assert parsed["event"] == "ask"
    assert parsed["request_id"] == "abc123"
    assert parsed["iterations"] == 3
    assert parsed["grounded"] is True
    assert parsed["ts"].endswith("Z")


def test_appends_line_per_call(tmp_path):
    p = tmp_path / "e.jsonl"
    log_event("a", path=str(p))
    log_event("b", path=str(p))
    events = [json.loads(line)["event"] for line in p.read_text().splitlines()]
    assert events == ["a", "b"]


def test_non_serialisable_values_are_stringified(tmp_path):
    p = tmp_path / "e.jsonl"
    log_event("x", path=str(p), where=Path("/tmp/z"), tool_calls={"grep": 1})
    parsed = json.loads(p.read_text())
    assert parsed["where"] == "/tmp/z"
    assert parsed["tool_calls"] == {"grep": 1}


def test_never_raises_on_bad_path():
    # parent is a file, not a directory -> mkdir raises OSError, swallowed
    log_event("x", path="/dev/null/nope.jsonl")


def test_ms_since_is_non_negative():
    import time

    t = time.perf_counter()
    assert ms_since(t) >= 0.0


def test_estimate_cost():
    assert estimate_cost_usd("kimi-k2.6", 1_000_000, 1_000_000) == 3.2
    assert estimate_cost_usd("kimi-k2.6", 500_000, 0) == 0.275
    assert estimate_cost_usd("unknown", 999, 999) == 0.0
