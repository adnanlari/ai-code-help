-- ============================================================================
-- 0002_repo_claimed_at.sql
-- ============================================================================
-- Adds repos.claimed_at: the moment a row was (re)claimed for indexing.
--
-- Without this, a run that dies mid-index (Ctrl-C, crash, killed process)
-- leaves status = 'indexing' with nothing to ever clear it, and every later
-- attempt bails out thinking a job is still running. claimed_at lets us treat
-- an 'indexing' row older than a threshold as stale and safe to retake.
-- ============================================================================

alter table repos add column if not exists claimed_at timestamptz;

-- Backfill existing rows so the staleness check has something to compare.
update repos set claimed_at = coalesce(indexed_at, created_at) where claimed_at is null;
