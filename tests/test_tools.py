import pytest

from backend.agent.tools import ToolBox


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("import os\n\n\ndef main():\n    return os.getcwd()\n")
    (tmp_path / "README.md").write_text("# Demo\nhello world\n")
    (tmp_path / "node_modules" / "pkg").mkdir(parents=True)
    (tmp_path / "node_modules" / "pkg" / "index.js").write_text("TODO nope\n")
    (tmp_path / "big.py").write_text("\n".join(f"line{i}" for i in range(1, 1001)) + "\n")
    return tmp_path


@pytest.fixture
def box(repo):
    return ToolBox(repo)


# --- read_file --------------------------------------------------------------


async def test_read_file_line_numbered(box):
    r = await box.run("read_file", {"path": "src/app.py"})
    assert r.ok
    assert "1| import os" in r.content
    assert "src/app.py (5 lines)" in r.content


async def test_read_file_line_range(box):
    r = await box.run("read_file", {"path": "src/app.py", "start_line": 4, "end_line": 5})
    assert "4| def main():" in r.content
    assert "import os" not in r.content


async def test_read_file_missing(box):
    r = await box.run("read_file", {"path": "src/nope.py"})
    assert not r.ok
    assert "no such file" in r.content


async def test_read_file_on_directory(box):
    r = await box.run("read_file", {"path": "src"})
    assert not r.ok
    assert "directory" in r.content


async def test_read_file_truncates_large(box):
    r = await box.run("read_file", {"path": "big.py"})
    assert r.ok
    assert "big.py (1000 lines)" in r.content
    assert "showing lines 1-400 of 1000" in r.content
    assert "401|" not in r.content


# --- grep ----------------------------------------------------------------


async def test_grep_finds_match_with_location(box):
    r = await box.run("grep", {"pattern": r"def \w+\("})
    assert r.ok
    assert "src/app.py:4: def main():" in r.content


async def test_grep_skips_vendored_dirs(box):
    r = await box.run("grep", {"pattern": "TODO"})
    assert "node_modules" not in r.content
    assert "no matches" in r.content


async def test_grep_scoped_to_path(box):
    r = await box.run("grep", {"pattern": "hello", "path": "README.md"})
    assert "README.md:2: hello world" in r.content


async def test_grep_invalid_regex(box):
    r = await box.run("grep", {"pattern": "("})
    assert not r.ok
    assert "invalid regex" in r.content


async def test_grep_ignore_case(box):
    r = await box.run("grep", {"pattern": "DEMO", "ignore_case": True})
    assert "README.md:1" in r.content


# --- list_dir ----------------------------------------------------------


async def test_list_dir_root(box):
    r = await box.run("list_dir", {})
    assert r.ok
    assert "src/" in r.content
    assert "README.md" in r.content
    assert "node_modules" not in r.content  # pruned


async def test_list_dir_subdir(box):
    r = await box.run("list_dir", {"path": "src"})
    assert "app.py" in r.content


async def test_list_dir_on_file(box):
    r = await box.run("list_dir", {"path": "README.md"})
    assert not r.ok
    assert "not a directory" in r.content


# --- dispatcher / safety --------------------------------------------------


async def test_unknown_tool(box):
    r = await box.run("rm_rf", {})
    assert not r.ok
    assert "unknown tool" in r.content


async def test_bad_arguments_do_not_raise(box):
    r = await box.run("read_file", {})  # missing required 'path'
    assert not r.ok
    assert "bad arguments" in r.content


async def test_path_traversal_through_tool_is_blocked(box):
    r = await box.run("read_file", {"path": "../../../../etc/passwd"})
    assert not r.ok
    assert "escapes the repository root" in r.content
