from backend.indexing.chunk import chunk_file


def _lines(n: int) -> str:
    return "\n".join(f"line{i}" for i in range(1, n + 1))


def test_empty_file_yields_nothing():
    assert chunk_file("a.py", "") == []
    assert chunk_file("a.py", "   \n  \n") == []


def test_short_file_is_one_chunk():
    text = _lines(10)
    chunks = chunk_file("a.py", text, size=60, overlap=15)
    assert len(chunks) == 1
    assert (chunks[0].start_line, chunks[0].end_line) == (1, 10)
    assert chunks[0].content == text
    assert chunks[0].file_path == "a.py"


def test_exact_window_is_one_chunk():
    chunks = chunk_file("a.py", _lines(60), size=60, overlap=15)
    assert len(chunks) == 1
    assert (chunks[0].start_line, chunks[0].end_line) == (1, 60)


def test_windows_and_overlap_boundaries():
    # 100 lines, size 60, overlap 15 -> step 45
    # window 1: lines 1-60
    # window 2: lines 46-100  (starts at index 45 => line 46)
    chunks = chunk_file("a.py", _lines(100), size=60, overlap=15)
    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 60), (46, 100)]
    # overlap region really is shared
    assert "line46" in chunks[0].content
    assert "line46" in chunks[1].content
    assert "line60" in chunks[0].content
    assert "line60" in chunks[1].content


def test_no_trailing_empty_chunk():
    # 90 lines, step 45 -> starts at 0, 45; start 90 would be past the end
    chunks = chunk_file("a.py", _lines(90), size=60, overlap=15)
    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 60), (46, 90)]


def test_invalid_params():
    import pytest

    with pytest.raises(ValueError):
        chunk_file("a.py", "x", size=0)
    with pytest.raises(ValueError):
        chunk_file("a.py", "x", size=10, overlap=10)
