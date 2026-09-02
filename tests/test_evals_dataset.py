import textwrap

import pytest

from evals.dataset import load_golden


def test_real_golden_set_loads_and_is_sane():
    cases = load_golden()
    assert len(cases) >= 10
    assert len({c.id for c in cases}) == len(cases)  # unique ids
    for c in cases:
        assert c.kind in {"localized", "multihop"}
        assert c.question.strip()
        assert len(c.expected_facts) >= 1
        assert c.repo.url.startswith("http")
    # brief wants both flavours represented
    kinds = {c.kind for c in cases}
    assert kinds == {"localized", "multihop"}


def _write(tmp_path, body: str):
    p = tmp_path / "g.toml"
    p.write_text(textwrap.dedent(body))
    return p


_BASE_REPO = """
[repos.demo]
url = "https://example.com/x"
ref = "main"
"""


def test_rejects_duplicate_ids(tmp_path):
    p = _write(
        tmp_path,
        _BASE_REPO
        + """
        [[cases]]
        id = "dup"
        repo = "demo"
        kind = "localized"
        question = "q1"
        expected_facts = ["f"]

        [[cases]]
        id = "dup"
        repo = "demo"
        kind = "localized"
        question = "q2"
        expected_facts = ["f"]
        """,
    )
    with pytest.raises(ValueError, match="duplicate case id"):
        load_golden(p)


def test_rejects_unknown_repo(tmp_path):
    p = _write(
        tmp_path,
        _BASE_REPO
        + """
        [[cases]]
        id = "a"
        repo = "nope"
        kind = "localized"
        question = "q"
        expected_facts = ["f"]
        """,
    )
    with pytest.raises(ValueError, match="unknown repo"):
        load_golden(p)


def test_rejects_bad_kind(tmp_path):
    p = _write(
        tmp_path,
        _BASE_REPO
        + """
        [[cases]]
        id = "a"
        repo = "demo"
        kind = "wat"
        question = "q"
        expected_facts = ["f"]
        """,
    )
    with pytest.raises(ValueError, match="kind must be"):
        load_golden(p)


def test_rejects_no_facts(tmp_path):
    p = _write(
        tmp_path,
        _BASE_REPO
        + """
        [[cases]]
        id = "a"
        repo = "demo"
        kind = "localized"
        question = "q"
        expected_facts = []
        """,
    )
    with pytest.raises(ValueError, match="expected_fact"):
        load_golden(p)
