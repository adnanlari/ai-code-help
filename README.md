# AI Coding Buddy

Point it at a public GitHub repo and ask questions about the codebase. An agent
plans, retrieves, reads across files, and answers **with citations** you can
check.

This is a compact learning + portfolio project. The skills it exercises —
multi-document reasoning, tool use, grounded answers, evals — are the
transferable part; the codebase just stands in for a corpus of documents.

## Architecture

One adaptive pipeline, not "RAG mode vs. agent mode":

1. **RAG seeds context** — vector search pulls the top-K relevant chunks for the question.
2. **The agent decides if that's enough** — the LLM has tools (`read_file`, `grep`, `list_dir`)
   and uses its `finish_reason` to answer now or fetch more.
3. **Tool loop** — tool result is appended to the conversation; repeat, capped at a max
   iteration count (tuned from eval data, not guessed).
4. **Citations** — every answer cites file/line spans, verified programmatically against what
   was actually retrieved/read (catches hallucinated citations).
5. **Reasoning trace** — the searched-X → read-Y → answered sequence is surfaced in the UI.

## Status

| Component | State |
|-----------|-------|
| Indexing pipeline (clone → filter → chunk → embed → store) + `(repo_url, commit_sha)` cache | ✅ |
| Vector store (Supabase pgvector) + manual similarity query | ✅ |
| `LLMClient` provider strategy layer (Kimi K2.6, OpenAI-compatible) | ✅ |
| File tools (`read_file` / `grep` / `list_dir`) + path-traversal guardrail + on-demand worktree | ✅ |
| RAG-seeded agentic tool-use loop + iteration cap | ✅ |
| `POST /ask` — retrieval → worktree → agent → answer + trace + token usage | ✅ |
| Programmatic citation grounding (flag-only: `verified` / `unverified_lines` / `unverified_file`) | ✅ |
| Reasoning-trace UI (Streamlit chat) | ⬜ next |
| Eval harness (golden set + retrieval overlap + LLM-as-judge) | ⬜ |
| Structured cost / latency / iteration logging (JSON lines) | ⬜ |

## Stack

- **Python** throughout. **FastAPI** (async) backend, **Streamlit** frontend (thin HTTP client).
- **Kimi K2.6** (Moonshot) as the agent LLM, via its **OpenAI-compatible** API (`openai` SDK
  pointed at Kimi's `base_url`) — tool loop built by hand, no agent framework. Chosen for cost;
  the loop sits behind a thin `LLMClient` interface so the provider (Claude, Gemini, …) is swappable.
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
cp .env.example .env                        # then fill in DATABASE_URL, VOYAGE_API_KEY, KIMI_API_KEY
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
curl -s localhost:8000/health | jq          # -> {"status":"ok","db":"ok"}

# 4. index a repo via the API (clone -> filter -> chunk -> embed -> store)
curl -s localhost:8000/index -H 'content-type: application/json' \
  -d '{"repo_url": "https://github.com/pallets/click"}' | jq
# -> {"repo_id": "...", "commit_sha": "...", "status": "ready",
#     "cached": false, "chunk_count": 312}
#
# same call again -> "cached": true, zero embedding calls
# (CLI equivalent, no server needed: python -m scripts.index_repo <repo_url> [ref])

# 5. check indexing status
curl -s localhost:8000/repos/<repo_id> | jq

# 6. retrieval only — top-K chunks for a query, no agent (CLI)
python -m scripts.query_vectors <repo_id> "how are options parsed?"

# 7. full Q&A through the agent loop (needs KIMI_API_KEY)
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"repo_id": "<repo_id>", "question": "how are options parsed?"}' | jq

# optional UI
streamlit run frontend/app.py
```

## API

Base URL `http://localhost:8000`. All request/response bodies are JSON; models
live in `backend/models.py`. Interactive docs at `/docs` when the server is up.

### `GET /health`
Liveness + DB ping. → `{ "status": "ok", "db": "ok" | "error" }`

### `POST /index`
Index a public repo (or return the cache hit). Runs the whole pipeline
**synchronously** — the request blocks until indexing finishes.

Request:
| field | type | required | notes |
|-------|------|----------|-------|
| `repo_url` | string | yes | HTTPS git URL of a public repo |
| `ref` | string | no | branch / tag / commit SHA (default `HEAD`) |

Response `200`:
```json
{ "repo_id": "uuid", "commit_sha": "sha", "status": "ready" | "indexing" | "failed",
  "cached": false, "chunk_count": 312 }
```
`cached: true` means it was already indexed at that commit — no clone, no embeddings.
`status: "indexing"` means another request is already building it. `400` on a git error.

### `GET /repos/{repo_id}`
Indexing status for one repo.
→ `{ "repo_id", "repo_url", "commit_sha", "status", "indexed_at": "ts|null", "chunk_count" }`
`404` if unknown.

### `POST /ask`
Ask a question about an already-indexed repo. Vector search seeds the top-K
chunks, then the agent loop (tools: `read_file` / `grep` / `list_dir`) runs until
it answers or hits the iteration cap. **Synchronous** — blocks for the whole loop.
Needs `KIMI_API_KEY`.

Request:
| field | type | required | notes |
|-------|------|----------|-------|
| `repo_id` | uuid | yes | from `POST /index` |
| `question` | string | yes | non-empty |
| `top_k` | int | no | retrieval size (1–20; default `TOP_K`, 5) |
| `max_iterations` | int | no | tool-loop cap (1–12; default `AGENT_MAX_ITERATIONS`, 6) |

Response `200`:
```json
{
  "repo_id": "uuid",
  "question": "...",
  "answer": "... with path:line citations ...",
  "stop_reason": "answered" | "max_iterations" | "empty_response",
  "iterations": 3,
  "usage": { "input_tokens": 8123, "output_tokens": 412, "total_tokens": 8535 },
  "retrieved": [ { "file_path": "src/x.py", "start_line": 60, "end_line": 119, "score": 0.71 } ],
  "trace": [ { "index": 1, "tool": "grep", "arguments": { "pattern": "..." },
              "ok": true, "result_preview": "...", "result_chars": 842 } ],
  "grounded": true,
  "citations": [ { "raw": "src/x.py:72", "file_path": "src/x.py", "start_line": 72,
                   "end_line": 72, "status": "verified" } ]
}
```
`retrieved` is the RAG seed; `trace` is every tool call the agent made.

**Citation grounding.** Every `path:line` reference in `answer` is checked against
what the agent was actually shown (retrieved chunks + lines it read via
`read_file`/`grep`). Each is tagged `verified`, `unverified_lines` (right file,
lines never seen), or `unverified_file` (never retrieved or read — likely
fabricated). `grounded` is `true` only when the answer has at least one citation
and all of them are `verified`. This is **flag-only**: a failed citation is
reported, not auto-corrected by re-prompting the model — deterministic, no extra
token cost, and the reader decides. (Rationale in `backend/agent/citations.py`.)

Errors: `404` repo unknown · `409` repo not `ready` · `502` git checkout or LLM provider failure.

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
  main.py              FastAPI app (/health, /index, /repos/{id}, /ask)
  indexing/
    clone.py           resolve SHA via `git ls-remote`, shallow clone, cleanup
    filter.py          denylist + binary sniff + size cap; is_source_file()
    chunk.py           line-window chunking with overlap
    embed.py           Voyage embeddings: embed_documents() + embed_query()
    pipeline.py        cache-check -> clone -> filter -> chunk -> embed -> store
  llm/
    base.py            LLMClient interface + neutral Message/ToolSpec/LLMResponse
    kimi.py            KimiClient (OpenAI-compatible) — Message <-> OpenAI shape
    __init__.py        _PROVIDERS registry + get_llm_client()
  agent/
    guardrail.py       safe_path() — path-traversal defence for tool paths
    tools.py           read_file / grep / list_dir + ToolBox dispatcher
    workspace.py       ensure_worktree() — re-materialise the indexed commit
    loop.py            run_agent() — the hand-rolled tool-use loop
    citations.py       verify_answer() — flag-only citation grounding
    service.py         run_qa() — retrieval + worktree + loop + grounding
  store/
    repos_store.py     claim_for_indexing (race guard + stale reclaim), get_by_id
    chunks_store.py    bulk insert, cosine similarity_search, delete_for_repo
scripts/
  apply_schema.py      run migrations/*.sql in order
  index_repo.py        index one repo from the CLI
  query_vectors.py     manual similarity query (retrieval only)
  reset_repo.py        wipe a repo's index (dev helper)
frontend/app.py        Streamlit placeholder
migrations/            0001_init.sql, 0002_repo_claimed_at.sql
tests/                 offline: chunk, filter, embed batching, llm translation,
                       guardrail, tools, agent loop, agent service, citations
```
