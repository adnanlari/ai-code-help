"""Load and validate the golden eval set (evals/golden.toml)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

GOLDEN_PATH = Path(__file__).with_name("golden.toml")
_KINDS = {"localized", "multihop"}


@dataclass(frozen=True, slots=True)
class RepoSpec:
    slug: str
    url: str
    ref: str


@dataclass(frozen=True, slots=True)
class EvalCase:
    id: str
    repo: RepoSpec
    kind: str  # localized | multihop
    question: str
    expected_files: tuple[str, ...]
    expected_facts: tuple[str, ...]


def load_golden(path: Path | None = None) -> list[EvalCase]:
    raw = tomllib.loads((path or GOLDEN_PATH).read_text("utf-8"))

    repos_raw = raw.get("repos", {})
    if not repos_raw:
        raise ValueError("golden set: no [repos.*] defined")
    repos = {
        slug: RepoSpec(slug=slug, url=r["url"], ref=str(r.get("ref", "HEAD")))
        for slug, r in repos_raw.items()
    }

    cases: list[EvalCase] = []
    seen_ids: set[str] = set()
    for i, c in enumerate(raw.get("cases", [])):
        cid = c.get("id") or f"case-{i}"
        if cid in seen_ids:
            raise ValueError(f"golden set: duplicate case id {cid!r}")
        seen_ids.add(cid)

        repo_slug = c.get("repo")
        if repo_slug not in repos:
            raise ValueError(f"{cid}: unknown repo {repo_slug!r} (have {sorted(repos)})")
        kind = c.get("kind")
        if kind not in _KINDS:
            raise ValueError(f"{cid}: kind must be one of {sorted(_KINDS)}, got {kind!r}")
        if not c.get("question", "").strip():
            raise ValueError(f"{cid}: empty question")
        facts = tuple(f for f in c.get("expected_facts", []) if f.strip())
        if not facts:
            raise ValueError(f"{cid}: needs at least one expected_fact")

        cases.append(
            EvalCase(
                id=cid,
                repo=repos[repo_slug],
                kind=kind,
                question=c["question"].strip(),
                expected_files=tuple(c.get("expected_files", [])),
                expected_facts=facts,
            )
        )

    if not cases:
        raise ValueError("golden set: no [[cases]] defined")
    return cases
