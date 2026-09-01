"""Structured event logging — one JSON object per line, append-only.

Why JSON lines (not prose `log.info`): each request emits a single self-contained
record that `jq` / pandas / DuckDB can aggregate directly — cost totals, latency
percentiles, grounding rate, `stop_reason` distribution for tuning the iteration
cap. A prose log can be read but not summed.

`log_event` must never break a request: any write failure is swallowed with a
warning. The file write is a few bytes; it's called from async handlers without
`to_thread` on purpose (negligible) — revisit if volume ever grows.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.config import get_settings

_fallback_log = logging.getLogger("events")
_write_lock = threading.Lock()


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def log_event(event: str, *, path: str | None = None, **fields: Any) -> dict:
    """Append one `{ts, event, **fields}` JSON line. Returns the record."""
    record: dict[str, Any] = {"ts": _now_iso(), "event": event, **fields}
    line = json.dumps(record, default=str, separators=(",", ":"))

    settings = get_settings()
    target = Path(path or settings.events_log_path)
    try:
        with _write_lock:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    except OSError as exc:  # disk full, bad path, permissions
        _fallback_log.warning("event log write failed (%s): %s", target, exc)

    if settings.events_log_stderr:
        print(line, file=sys.stderr)

    return record


def ms_since(start: float) -> float:
    """Milliseconds elapsed since a `time.perf_counter()` reading, 1 dp."""
    import time

    return round((time.perf_counter() - start) * 1000, 1)
