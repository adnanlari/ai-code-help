# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A learning + portfolio project (see `README.md` for the full brief). Point it at a
public GitHub repo; an agent answers questions about that codebase by planning,
retrieving via vector search, reading across files with tools, and citing sources.
The codebase stands in for a document corpus — the transferable skills are
multi-document reasoning, hand-built tool-use loops, grounded answers, and evals.

**This is built to be understood, not just to work.** Prefer hand-rolled
mechanisms over frameworks that hide them (no LangChain/LlamaIndex; the LLM SDK
is called directly and the tool-use loop is written by hand). When adding to the
tool-use loop, RAG, chunking, or the eval harness, keep the reasoning explicit in
comments/docstrings — the owner needs to defend every design choice in an interview.

## LLM provider

The agent LLM is **Kimi K2.6** (Moonshot), called through its **OpenAI-compatible**
API with the `openai` SDK pointed at Kimi's `base_url`. Chosen for cost: ~$0.55/$2.65
per 1M input/output tokens vs. a much higher Claude bill, and Moonshot's prepaid
"recharge" tiers gate rate limits ($10 recharge → Tier1: 100 RPM, unlimited daily).
Key: `KIMI_API_KEY` in `.env`.

The loop is built behind a thin `LLMClient` interface (`backend/llm/base.py`, one
async `complete(messages, *, tools, ...)` method returning a normalised
`LLMResponse`) so the provider is swappable — `get_llm_client()` in
`backend/llm/__init__.py` picks a concrete strategy from `_PROVIDERS` by
`settings.llm_provider`. `KimiClient` (`backend/llm/kimi.py`) translates our
neutral `Message`/`ToolSpec` types to/from the OpenAI shape. Tool-call signalling
is OpenAI-style (`finish_reason == "tool_calls"`, a `tool_calls` array), **not**
Anthropic's `stop_reason` / content blocks. Adding Claude/Gemini = one
`LLMClient` subclass + one line in `_PROVIDERS`; the agent loop never changes.

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

# manual similarity query (retrieval only, no agent)
python -m scripts.query_vectors <repo_id> "how are options parsed?"

# wipe a repo's index so it re-indexes from scratch (dev helper)
python -m scripts.reset_repo <repo_url | repo_id>

# servers
uvicorn backend.main:app --reload         # API on :8000
streamlit run frontend/app.py             # UI (expects the API already up)

# full Q&A once the server is up (needs KIMI_API_KEY too):
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"repo_id": "<uuid>", "question": "how does X work?"}' | jq
```

Tests run offline (no DB / Voyage / git / LLM). Anything touching those needs
real credentials in `.env` and is exercised via the scripts / HTTP, not pytest.

## Architecture

### One adaptive pipeline

RAG and agentic tool-use are **merged**, not separate modes:
vector search seeds the top-K chunks → the LLM gets `read_file`/`grep`/`list_dir`
tools and decides via its `finish_reason` whether to answer or fetch more →
loop with an iteration cap → answer carries citations → the tool-call sequence is
surfaced as a reasoning trace.

Built: indexing pipeline + vector store; the LLM `LLMClient` strategy layer; the
file tools + path-traversal guardrail + on-demand worktree; the tool-use loop;
programmatic citation grounding (flag-only); structured JSON-lines event logging;
`POST /ask` wiring it together. Not built yet: the reasoning-trace UI (Streamlit
is still a placeholder) and the eval harness. See `README.md` for the status table.

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
- **embed.py**: Voyage `voyage-code-3`, 1024-dim. `_embed_sync(inputs, input_type)`
  is shared by `embed_documents` (`"document"`, index side) and `embed_query`
  (`"query"`, search side) — asymmetric embedding. Blocking SDK calls run via
  `asyncio.to_thread`; batched ≤128 items / `EMBED_MAX_BATCH_TOKENS`; optional
  `_RollingLimiter` (`EMBED_RPM`/`EMBED_TPM`, 0 = off) + minute-scale backoff on
  `RateLimitError`.

### Caching + concurrency — the core trick

The cache key is `(repo_url, commit_sha)`, not the URL — a specific commit is the
stable unit. `repos` has `unique (repo_url, commit_sha)`, which is *both* the cache
lookup and the race guard: `claim_for_indexing` does
`INSERT ... ON CONFLICT DO NOTHING RETURNING id`; concurrent first-time requests
race, one wins the INSERT, the loser gets no row back and re-selects the winner's
row instead of starting a duplicate embedding job. No locks/queue/Redis — the DB
constraint is the coordination primitive. `claim_for_indexing` returns a
`ClaimOutcome` (`CREATED` / `RECLAIMED` / `IN_PROGRESS` / `READY`); a `failed`
row, or an `indexing` row whose `claimed_at` is older than `STALE_AFTER`
(crashed run), is `RECLAIMED` and retried — leftover chunks are deleted first.

### Data layer — raw SQL, no ORM (deliberate)

`asyncpg` only. `backend/db.py` owns a module-global pool; `_init_connection`
registers the `pgvector` codec on every connection so `vector` columns move as
Python lists. `statement_cache_size=0` for pgbouncer (Supabase pooler)
compatibility. All queries live in `backend/store/*_store.py` as functions using
`$1,$2` positional params. `similarity_search` uses `<=>` (cosine distance),
returns `1 - distance` as the score, orders ascending so the HNSW index is used.

### Agent / Q&A pipeline — `backend/agent/` + `backend/llm/`

`service.run_qa(repo_id, question, *, top_k, max_iterations)` orchestrates:
`repos_store.get_by_id` (must be `status='ready'`, else `RepoNotFound` /
`RepoNotReady`) → `embed.embed_query` (`input_type="query"`) →
`chunks_store.similarity_search` (the RAG seed, top-K) →
`workspace.ensure_worktree` (re-materialises the **exact** indexed commit via
`git init` + `fetch --depth 1 <sha>` + `checkout FETCH_HEAD`, cached under
`WORKDIR/worktrees/<sha>`; blocking, so run in `asyncio.to_thread`) →
`tools.ToolBox(repo_root)` + `llm.get_llm_client()` → `loop.run_agent` →
`QAOutcome`.

- **guardrail.py**: `safe_path(repo_root, user_path)` — the security boundary.
  The model picks tool paths; this rejects NUL bytes, forces the path relative to
  the root (a leading `/` can't escape), `Path.resolve()`s it (collapses `..`,
  expands symlinks), and requires containment under the root. Runs on every tool
  call.
- **tools.py**: `read_file` / `grep` / `list_dir` as provider-neutral `ToolSpec`s
  + a `ToolBox` dispatcher. Every tool returns a **string** (errors as
  `"error: ..."`, never raised — the model reads and self-corrects). Outputs are
  line/byte/match capped so a tool result can't blow the context window.
  `grep`/`list_dir` reuse `filter.SKIP_DIRS` + `is_source_file` so the agent sees
  the same repo slice that was indexed. `ToolBox.run` is async (fs work via
  `to_thread`), returns `ToolResult(name, ok, content)`.
- **loop.py**: `run_agent(*, question, hits, toolbox, llm, max_iterations)` — the
  hand-rolled tool-use loop. Seed = system prompt + question + formatted
  excerpts. Each pass: `llm.complete(messages, tools=specs)` → append the
  assistant turn → if `resp.wants_tools`, run each via `ToolBox`, append
  `role="tool"` messages, record a `TraceStep`, loop → else the text is the
  answer. On hitting the cap, one final `complete(..., tools=())` forces an
  answer. Returns `AgentResult` (`answer`, `trace`, summed `Usage`, `iterations`,
  `stop_reason`, full `messages`). Provider- and repo-agnostic — depends only on
  `LLMClient` + `ToolBox`; tests drive it with a scripted fake LLM.
- **`agent_max_iterations`** (config, default 6) is the cost brake — each pass is
  a paid LLM call.
- **citations.py**: after the loop, `verify_answer(answer, hits, messages)`
  extracts `path:line` citations and checks each against the **evidence set** —
  retrieved chunk spans + the exact lines shown by `read_file`/`grep` (parsed
  back out of the tool-result messages) + `list_dir` names (existence only). Each
  citation is tagged `verified` / `unverified_lines` / `unverified_file`;
  `grounded` = ≥1 citation and none unverified. **Flag-only by design** — a bad
  citation is reported (`GroundingReport` → `AskResponse.grounded` + `citations`),
  never bounced back to the model for a self-correction retry. Rationale (cost,
  determinism, read-only tool) is in the module docstring; the structured return
  leaves a retry policy addable later without touching this module.

### Observability — `backend/obslog.py`

`log_event(event, **fields)` appends one JSON object per line to
`EVENTS_LOG_PATH` (default `logs/events.jsonl`, gitignored) and echoes it to
stderr (`EVENTS_LOG_STDERR`). Never raises — a write failure is a warning.
`service.run_qa` emits an `ask` event per request (`request_id`, timings
`t_embed_ms`/`t_search_ms`/`t_worktree_ms`/`t_agent_ms`/`t_total_ms`,
`iterations`, `stop_reason`, `tool_calls` breakdown, token counts,
`est_cost_usd` via `llm/pricing.py`, `grounded` / `n_unverified`) and an
`ask_error` on failure; `pipeline.index_repo` emits `index` / `index_error`.
`request_id` is in the `/ask` response and in HTTP error details, so a log line
correlates with a specific call. Aggregate with `jq` / pandas for cost totals,
latency percentiles, and iteration-cap tuning.

### API — `backend/main.py`

FastAPI with a `lifespan` that opens/closes the pool. Endpoints: `GET /health`
(DB ping), `POST /index` (runs the full pipeline **synchronously** — a job queue
is the noted production upgrade), `GET /repos/{repo_id}`, `POST /ask` (calls
`run_qa`; maps `QAOutcome` → `AskResponse` = `request_id` + answer +
`stop_reason` + iterations + token `usage` + `retrieved` chunks + tool `trace` +
`grounded` + per-citation `citations`; 404 / 409 / 502 for not-found / not-ready
/ git+LLM failures, each with the `request_id` in the detail). Also
synchronous — the request blocks for the whole tool loop. Request/response models
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
