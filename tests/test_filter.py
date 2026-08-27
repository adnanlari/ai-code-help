from pathlib import Path

from backend.indexing.filter import iter_source_files


def _build_tree(root: Path) -> None:
    (root / "src").mkdir(parents=True)
    (root / "src" / "main.py").write_text("print('hi')\n")
    (root / "README.md").write_text("# hello\n")
    (root / "Makefile").write_text("all:\n\techo hi\n")  # no extension, keep

    # vendored dir -> skipped
    (root / "node_modules" / "leftpad").mkdir(parents=True)
    (root / "node_modules" / "leftpad" / "index.js").write_text("module.exports=1\n")

    # lockfile -> skipped
    (root / "package-lock.json").write_text('{"lockfileVersion": 3}\n')

    # binary extension -> skipped
    (root / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)

    # binary content, innocuous extension -> skipped by the NUL sniff
    (root / "src" / "blob.txt").write_bytes(b"abc\x00def")

    # oversized -> skipped
    (root / "huge.csv").write_text("x\n" * 600_000)


def test_filter_keeps_only_real_source(tmp_path: Path):
    _build_tree(tmp_path)
    rels = sorted(rel for _abs, rel in iter_source_files(tmp_path))
    assert rels == ["Makefile", "README.md", "src/main.py"]


def test_paths_are_repo_relative_posix(tmp_path: Path):
    _build_tree(tmp_path)
    for abs_path, rel in iter_source_files(tmp_path):
        assert abs_path.is_absolute()
        assert "\\" not in rel
        assert not rel.startswith("/")
