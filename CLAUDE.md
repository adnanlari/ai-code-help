# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A learning + portfolio project (see `README.md` for the full brief). Point it at a
public GitHub repo; an agent answers questions about that codebase by planning,
retrieving via vector search, reading across files with tools, and citing sources.
The codebase stands in for a document corpus — the transferable skills are
multi-document reasoning, hand-built tool-use loops, grounded answers, and evals.

**This is built to be understood, not just to work.** Prefer hand-rolled
mechanisms over frameworks that hide them (no LangChain/LlamaIndex; the
`anthropic` SDK is used directly). When adding to the tool-use loop, RAG, chunking,
or the eval harness, keep the reasoning explicit in comments/docstrings — the
owner needs to defend every design choice in an interview.

## Environment

- **Python 3.12** (Homebrew: `/opt/homebrew/bin/python3.12`). The system Python 3.9
  is too old — 3.10+ syntax is used (`@dataclass(slots=True)`, `zip(strict=True)`).
- Virtualenv at `.venv/`. Activate with `source .venv/bin/activate`.
- Secrets in `.env` (gitignored), template in `.env.example`. `DATABASE_URL` must be
  a plain `postgresql://` URI (not SQLAlchemy's `postgresql+asyncpg://`).

## Commands

```bash
# setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt        # runtime + dev deps (plain requirements.txt is runtime only)

# lint / format / test — run before finishing any change
ruff check .
ruff format .
pytest -q
pytest -q tests/test_chunk.py::test_windows_and_overlap_boundaries   # single test

# database schema (idempotent; needs DATABASE_URL)
python -m scripts.apply_schema

# index a repo without the HTTP server (needs DATABASE_URL + VOYAGE_API_KEY)
python -m scripts.index_repo https://github.com/pallets/click [ref]
# run twice — second run should report cached=True and make zero embedding calls

# manual similarity query (the Day 1 acceptance demo)
python -m scripts.query_vectors <repo_id> "how are options parsed?"

# servers
uvicorn backend.main:app --reload         # API on :8000
streamlit run frontend/app.py             # UI (expects the API already up)
```

Tests run offline (no DB/network). Anything touching Postgres or Voyage needs
real credentials in `.env` and is exercised via the scripts above, not pytest.

## Architecture

### One adaptive pipeline (design target, not yet fully built)

RAG and agentic tool-use are **merged**, not separate modes:
vector search seeds the top-K chunks → Claude gets `read_file`/`grep`/`list_dir`
tools and decides via its own `stop_reason` whether to answer or fetch more →
loop with an iteration cap → answer carries citations verified against what was
actually retrieved/read → the tool-call sequence is surfaced as a reasoning trace.

Build status: **Day 1 done** = indexing pipeline + vector store + FastAPI skeleton.
Day 2 = the tool loop + path-traversal guardrails. Day 3 = citations + trace + chat
UI. Day 4 = eval harness + cost/latency logging. See `README.md` status table.

### Indexing pipeline — `backend/indexing/`

`pipeline.index_repo(repo_url, ref)` orchestrates:
`clone.resolve_sha` (via `git ls-remote` — no download, so the cache can be
checked before paying for a clone) → `repos_store.claim_for_indexing`
(get-or-create) → on cache miss: `clone.shallow_clone` →
`filter.iter_source_files` → `chunk.chunk_file` per file → `embed.embed_documents`
→ `chunks_store.bulk_insert` → `repos_store.mark_ready`; `mark_failed` on any
exception; clone dir always deleted in `finally`.

- **filter.py**: denylist (vendored dirs, lockfiles, binary extensions) + size cap
  + first-8KB binary sniff (NUL byte / invalid UTF-8). Deliberately not an
  extension allowlist — keeps `Makefile`, `Dockerfile`, unusual source extensions.
- **chunk.py**: line-window chunking (default 60 lines, 15 overlap). Line numbers
  are the citation anchors; overlap keeps boundary-straddling code intact in one
  chunk. AST-aware chunking is a deferred stretch goal.
- **embed.py**: Voyage `voyage-code-3`, 1024-dim, `input_type="document"`. The
  query side uses `input_type="query"` (asymmetric embedding) — currently in
  `scripts/query_vectors.py`, moves into the agent on Day 2. Blocking SDK calls
  run via `asyncio.to_thread`; batched ≤128 items with exponential-backoff retry.

### Caching + concurrency — the core trick

The cache key is `(repo_url, commit_sha)`, not the URL — a specific commit is the
stable unit. `repos` has `unique (repo_url, commit_sha)`, which is *both* the cache
lookup and the race guard: `claim_for_indexing` does
`INSERT ... ON CONFLICT DO NOTHING RETURNING id`; concurrent first-time requests
race, one wins the INSERT, the loser gets no row back and re-selects the winner's
row instead of starting a duplicate embedding job. No locks/queue/Redis — the DB
constraint is the coordination primitive. A pre-existing `failed` row is reset to
`indexing` for retry.

### Data layer — raw SQL, no ORM (deliberate)

`asyncpg` only. `backend/db.py` owns a module-global pool; `_init_connection`
registers the `pgvector` codec on every connection so `vector` columns move as
Python lists. `statement_cache_size=0` for pgbouncer (Supabase pooler)
compatibility. All queries live in `backend/store/*_store.py` as functions using
`$1,$2` positional params. `similarity_search` uses `<=>` (cosine distance),
returns `1 - distance` as the score, orders ascending so the HNSW index is used.

### API — `backend/main.py`

FastAPI with a `lifespan` that opens/closes the pool. Endpoints: `GET /health`
(DB ping), `POST /index` (runs the full pipeline **synchronously** — a job queue
is the noted production upgrade), `GET /repos/{repo_id}`. Request/response models
in `backend/models.py`.

### Frontend — `frontend/app.py`

Streamlit, intentionally a thin HTTP client — no business logic, so it can be
replaced without touching backend/agent code.

## Conventions

- `from __future__ import annotations` at the top of every module.
- Frozen `@dataclass(slots=True)` for value types; Pydantic models only at the HTTP boundary.
- Settings come from `backend.config.get_settings()` — never read `os.environ` elsewhere.
- New DB access = a new function in the relevant `backend/store/*_store.py`, raw parameterised SQL.
- Schema changes = a new `migrations/NNNN_*.sql`, idempotent (`IF NOT EXISTS`); `apply_schema` runs them in filename order (no versioning table by design).
- `ruff` config in `ruff.toml` (line length 100, py311 target).
