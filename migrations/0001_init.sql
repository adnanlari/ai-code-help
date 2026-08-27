-- ============================================================================
-- 0001_init.sql  --  AI Coding Buddy base schema
-- ============================================================================
-- Applied by: python -m scripts.apply_schema
-- Everything here is idempotent (IF NOT EXISTS) so re-running is safe.
-- ============================================================================

-- pgvector: adds the `vector` column type + similarity operators (<->, <=>, <#>).
-- On Supabase the extension ships preinstalled; we just have to enable it.
create extension if not exists vector;

-- ----------------------------------------------------------------------------
-- repos: one row per (repo_url, commit_sha) we have indexed.
--
-- This table IS the cache. Before indexing we look up (repo_url, commit_sha);
-- if a 'ready' row exists we skip cloning/chunking/embedding entirely.
-- ----------------------------------------------------------------------------
create table if not exists repos (
    id          uuid primary key default gen_random_uuid(),
    repo_url    text        not null,
    commit_sha  text        not null,          -- the exact commit we embedded
    status      text        not null default 'indexing'
                check (status in ('indexing', 'ready', 'failed')),
    indexed_at  timestamptz,                   -- set when status -> 'ready'
    created_at  timestamptz not null default now(),

    -- Doubles as (a) the cache key and (b) the race guard: two concurrent
    -- "index this brand-new repo" requests cannot both INSERT. The loser gets a
    -- unique-violation / no row back from ON CONFLICT DO NOTHING and re-selects
    -- the winner's row instead of starting a second embedding job.
    unique (repo_url, commit_sha)
);

-- ----------------------------------------------------------------------------
-- chunks: the embedded pieces of source code for a repo.
--
-- start_line/end_line are 1-based inclusive line numbers in file_path. We keep
-- them so citations can point at "src/foo.py:60-119" and so Day 4 grounding
-- checks can verify a cited span was actually retrieved/read.
-- ----------------------------------------------------------------------------
create table if not exists chunks (
    id          uuid primary key default gen_random_uuid(),
    repo_id     uuid        not null references repos(id) on delete cascade,
    file_path   text        not null,          -- repo-relative, POSIX separators
    start_line  int         not null,
    end_line    int         not null,
    content     text        not null,
    embedding   vector(1024) not null,         -- voyage-code-3 @ output_dimension=1024
    created_at  timestamptz not null default now()
);

-- Every similarity query is scoped to a single repo, so filter by repo_id first.
create index if not exists chunks_repo_id_idx on chunks (repo_id);

-- HNSW vs IVFFlat:
--   * HNSW  - graph index, no training step, no `lists` param to tune, strong
--             recall/latency even on tiny datasets. Slower to build, more RAM.
--   * IVFFlat - needs representative rows present BEFORE you build it, and a
--             tuned `lists` value; poor recall if built on an empty/small table.
-- For a personal-scale project HNSW is the low-friction, high-recall choice.
--
-- vector_cosine_ops pairs with the `<=>` (cosine distance) operator. Voyage
-- returns L2-normalised vectors, so cosine is the natural metric.
create index if not exists chunks_embedding_idx
    on chunks using hnsw (embedding vector_cosine_ops);
