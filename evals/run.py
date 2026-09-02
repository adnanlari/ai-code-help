"""Run the eval suite from the command line.

    python -m evals.run                      # whole golden set, with the LLM judge
    python -m evals.run --no-judge           # retrieval + system metrics only (no judge calls)
    python -m evals.run --repo click         # only cases for one repo
    python -m evals.run --kind multihop      # only multi-hop cases
    python -m evals.run --id click-help-generation
    python -m evals.run --limit 3
    python -m evals.run --min-pass 0.8       # exit non-zero if pass rate < 0.8 (CI gate)

Writes a JSON report to logs/eval-<timestamp>.json and prints a scorecard.
Needs DATABASE_URL, VOYAGE_API_KEY, KIMI_API_KEY (like /ask).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from backend.obslog import log_event
from evals.dataset import load_golden
from evals.report import format_scorecard, report_dict, summarise
from evals.runner import run_suite


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="evals.run")
    p.add_argument("--repo", help="only cases for this repo slug")
    p.add_argument("--kind", choices=["localized", "multihop"], help="only this kind")
    p.add_argument("--id", dest="ids", action="append", help="run specific case id(s)")
    p.add_argument("--limit", type=int, help="cap the number of cases")
    p.add_argument("--no-judge", action="store_true", help="skip the LLM-as-judge scorer")
    p.add_argument(
        "--min-pass", type=float, default=0.0, help="exit non-zero if pass rate below this"
    )
    p.add_argument("--out", type=Path, help="JSON report path (default logs/eval-<ts>.json)")
    return p.parse_args(argv)


async def _main(argv: list[str]) -> int:
    args = _parse_args(argv)
    cases = load_golden()

    if args.repo:
        cases = [c for c in cases if c.repo.slug == args.repo]
    if args.kind:
        cases = [c for c in cases if c.kind == args.kind]
    if args.ids:
        wanted = set(args.ids)
        cases = [c for c in cases if c.id in wanted]
    if args.limit:
        cases = cases[: args.limit]

    if not cases:
        print("no cases match the given filters", file=sys.stderr)
        return 2

    print(f"running {len(cases)} case(s){' without judge' if args.no_judge else ''} ...\n")
    report = await run_suite(cases, do_judge=not args.no_judge)

    print(format_scorecard(report))

    out = args.out or Path("logs") / f"eval-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report_dict(report), indent=2), encoding="utf-8")
    print(f"\nreport -> {out}")

    s = summarise(report)
    log_event(
        "eval",
        n_cases=s["n_cases"],
        n_errored=s["n_errored"],
        judged=s["judged"],
        pass_rate=s["pass_rate"],
        grounded_rate=s["grounded_rate"],
        mean_recall=s["mean_recall"],
        mean_iterations=s["mean_iterations"],
        total_cost_usd=s["total_cost_usd"],
        p95_latency_ms=s["p95_latency_ms"],
    )

    if args.min_pass and report.judged and s["pass_rate"] < args.min_pass:
        print(f"\nFAIL: pass rate {s['pass_rate']} < --min-pass {args.min_pass}", file=sys.stderr)
        return 1
    if s["n_errored"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main(sys.argv[1:])))
