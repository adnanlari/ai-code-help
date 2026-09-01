import sys

import pytest

from backend.agent.guardrail import PathError, safe_path


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("x = 1\n")
    return tmp_path


def test_dot_and_empty_map_to_root(repo):
    assert safe_path(repo, ".") == repo.resolve()
    assert safe_path(repo, "") == repo.resolve()


def test_normal_relative_path_stays_inside(repo):
    p = safe_path(repo, "src/app.py")
    assert p == (repo / "src" / "app.py").resolve()


def test_missing_file_is_not_an_error(repo):
    # guardrail only checks containment; "does it exist" is the tool's problem
    p = safe_path(repo, "src/does_not_exist.py")
    assert p.parent == (repo / "src").resolve()


def test_dotdot_escape_is_blocked(repo):
    with pytest.raises(PathError, match="escapes"):
        safe_path(repo, "../../../../etc/passwd")


def test_absolute_path_is_forced_relative(repo):
    # "/etc/passwd" must become "<root>/etc/passwd", never the real one
    p = safe_path(repo, "/etc/passwd")
    assert str(p).startswith(str(repo.resolve()))


def test_nul_byte_rejected(repo):
    with pytest.raises(PathError):
        safe_path(repo, "src/app\x00.py")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlink semantics")
def test_symlink_pointing_outside_is_blocked(repo, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside")
    (outside / "secret.txt").write_text("token")
    (repo / "link").symlink_to(outside)
    with pytest.raises(PathError, match="escapes"):
        safe_path(repo, "link/secret.txt")
