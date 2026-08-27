# AI Coding Buddy

Point it at a public GitHub repo and ask questions about the codebase. An agent
plans, retrieves, reads across files, and answers **with citations** you can
check.

This is a learning + portfolio project built over ~4 days. The skills it
exercises — multi-document reasoning, tool use, grounded answers, evals — are
the transferable part; the codebase just stands in for a corpus of documents.

## Architecture (target)

One adaptive pipeline, not "RAG mode vs. agent mode":

1. **RAG seeds context** — vector search pulls the top-K relevant chunks for the question.
2. **The agent decides if that's enough** — Claude has tools (`read_file`, `grep`, `list_dir`)
   and uses its own `stop_reason` to answer now or fetch more.
3. **Tool loop** — tool result is appended to the conversation; repeat, capped at a max
   iteration count (tuned from eval data, not guessed).
4. **Citations** — every answer cites file/line spans, verified programmatically against what
   was actually retrieved/read (catches hallucinated citations).
5. **Reasoning trace** — the searched-X → read-Y → answered sequence is surfaced in the UI.

## Status

| Day | Scope | State |
|-----|-------|-------|
| 1 | Scaffold, clone/filter/chunk/embed pipeline, Supabase + pgvector schema, FastAPI skeleton | **in progress** |
| 2 | RAG-seeded agentic tool loop, path-traversal guardrails, iteration cap | — |
| 3 | Citations + grounding check, reasoning trace, Streamlit chat UI | — |
| 4 | Eval harness (golden set + LLM-as-judge), latency/cost logging, README writeup | — |

## Stack

- **Python** throughout. **FastAPI** (async) backend, **Streamlit** frontend (thin HTTP client).
- **Anthropic API** (`claude-sonnet-5`), raw `anthropic` SDK — tool loop built by hand, no agent framework.
- **Voyage AI** `voyage-code-3` embeddings (1024-dim, cosine).
- **Supabase Postgres + pgvector**, accessed with **raw SQL over `asyncpg`** (no ORM — deliberate, to build SQL fluency).

Deferred to "what I'd add for production": Docker, Redis, a job queue, migrations tooling.

## Data model

- **`repos`** — one row per `(repo_url, commit_sha)`, `unique` on that pair. This *is* the cache
  and the duplicate-indexing race guard. `status` ∈ `indexing | ready | failed`.
- **`chunks`** — `repo_id` FK, `file_path`, `start_line`/`end_line` (citation anchors), `content`,
  `embedding vector(1024)`. HNSW cosine index.

Repo embeddings are a shared cache of public code — never deleted on session/tab close. Local
clone dirs are disposable and removed right after embedding.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt        # runtime + dev deps
cp .env.example .env                        # then fill in DATABASE_URL + VOYAGE_API_KEY
```

`DATABASE_URL` is the Supabase connection URI (Project Settings → Database → Connection string,
URI form). Use a plain `postgresql://` scheme — not the SQLAlchemy `postgresql+asyncpg://` prefix.

## Run

```bash
# 1. create the schema (idempotent)
python -m scripts.apply_schema

# 2. start the API
uvicorn backend.main:app --reload

# 3. health check
curl localhost:8000/health          # -> {"status":"ok","db":"ok"}

# 4. index a small repo (CLI path, no server needed)
python -m scripts.index_repo https://github.com/pallets/click

# 5. run it again -> cached=True, zero embedding calls

# 6. query the vector store (Day 1 acceptance demo)
python -m scripts.query_vectors <repo_id> "how are options parsed?"

# optional UI
streamlit run frontend/app.py
```

## Dev

```bash
ruff check .
pytest
```

## Layout

```
backend/
  config.py            typed settings from .env
  db.py                asyncpg pool + pgvector codec registration
  models.py            Pydantic API models
  main.py              FastAPI app (/health, /index, /repos/{id})
  indexing/
    clone.py           resolve SHA via `git ls-remote`, shallow clone, cleanup
    filter.py          denylist + binary sniff + size cap
    chunk.py           line-window chunking with overlap
    embed.py           Voyage batch embedding (input_type="document")
    pipeline.py        cache-check -> clone -> filter -> chunk -> embed -> store
  store/
    repos_store.py     get-or-create (race guard), mark_ready/failed
    chunks_store.py    bulk insert, cosine similarity_search
scripts/
  apply_schema.py      run migrations/*.sql
  index_repo.py        index one repo from the CLI
  query_vectors.py     manual similarity query
frontend/app.py        Streamlit placeholder
migrations/0001_init.sql
tests/                 chunk + filter unit tests
```
