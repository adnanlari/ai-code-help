import pytest

from backend.config import get_settings


@pytest.fixture(autouse=True)
def _isolate_event_log(tmp_path, monkeypatch):
    """Point the structured event log at a per-test temp file and silence its
    stderr echo, so tests never touch the real logs/events.jsonl or spam output."""
    monkeypatch.setenv("EVENTS_LOG_PATH", str(tmp_path / "events.jsonl"))
    monkeypatch.setenv("EVENTS_LOG_STDERR", "false")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
