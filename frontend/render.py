"""Pure formatting helpers for the Streamlit UI - no Streamlit, no I/O.

Kept separate from app.py so the display logic is unit-testable and the app file
stays a thin sequence of `st.*` calls.
"""

from __future__ import annotations

_CIT_ICON = {"verified": "🟢", "unverified_lines": "🟡", "unverified_file": "🔴"}
_CIT_NOTE = {
    "verified": "verified against retrieved / read code",
    "unverified_lines": "file was seen, but not these lines",
    "unverified_file": "file never retrieved or read — likely fabricated",
}


def grounding_summary(resp: dict) -> str:
    cites = resp.get("citations") or []
    if not cites:
        return "⚪ no citations in this answer"
    bad = sum(1 for c in cites if c.get("status") != "verified")
    if resp.get("grounded"):
        return f"✅ grounded — {len(cites)} citation(s), all verified"
    return f"⚠️ {bad} of {len(cites)} citation(s) unverified"


def citation_line(c: dict) -> str:
    icon = _CIT_ICON.get(c.get("status", ""), "⚪")
    note = _CIT_NOTE.get(c.get("status", ""), c.get("status", "?"))
    return f"{icon} `{c.get('raw', '?')}` — {note}"


def trace_line(step: dict) -> str:
    args = step.get("arguments") or {}
    arg_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
    mark = "✅" if step.get("ok") else "❌"
    idx, tool, chars = step.get("index", "?"), step.get("tool", "?"), step.get("result_chars", 0)
    return f"**{idx}. {tool}**({arg_str}) → {mark} {chars} chars"


def retrieved_line(ch: dict) -> str:
    score = ch.get("score")
    score_str = f"{score:.2f}" if isinstance(score, int | float) else "?"
    loc = f"{ch.get('file_path', '?')}:{ch.get('start_line', '?')}-{ch.get('end_line', '?')}"
    return f"`{loc}` · score {score_str}"


def meta_line(resp: dict) -> str:
    u = resp.get("usage") or {}
    return (
        f"model `{resp.get('model', '?')}` · "
        f"{resp.get('iterations', '?')} iteration(s) · "
        f"stop: {resp.get('stop_reason', '?')} · "
        f"{u.get('total_tokens', '?')} tokens · "
        f"req `{resp.get('request_id', '?')}`"
    )
