"""Turn a SuiteReport into a human scorecard + a machine-readable dict."""

from __future__ import annotations

from statistics import mean

from evals.runner import CaseResult, SuiteReport


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, int(round(q * (len(s) - 1))))
    return s[idx]


def _safe_mean(values: list[float]) -> float:
    return round(mean(values), 3) if values else 0.0


def summarise(report: SuiteReport) -> dict:
    res = report.results
    ok = [r for r in res if r.error is None]
    judged = [r for r in ok if r.judge is not None]
    latencies = [r.latency_ms for r in ok]

    by_kind: dict[str, dict] = {}
    for kind in sorted({r.kind for r in res}):
        k = [r for r in ok if r.kind == kind]
        kj = [r for r in k if r.judge is not None]
        by_kind[kind] = {
            "n": len(k),
            "pass_rate": _safe_mean([1.0 if r.passed else 0.0 for r in kj]),
            "mean_recall": _safe_mean([r.retrieval.recall for r in k if r.retrieval]),
            "mean_seed_recall": _safe_mean([r.retrieval.seed_recall for r in k if r.retrieval]),
        }

    return {
        "n_cases": len(res),
        "n_errored": sum(1 for r in res if r.error),
        "judged": report.judged,
        "pass_rate": _safe_mean([1.0 if r.passed else 0.0 for r in judged]),
        "grounded_rate": _safe_mean([1.0 if r.grounded else 0.0 for r in ok]),
        "mean_recall": _safe_mean([r.retrieval.recall for r in ok if r.retrieval]),
        "mean_precision": _safe_mean([r.retrieval.precision for r in ok if r.retrieval]),
        "mean_seed_recall": _safe_mean([r.retrieval.seed_recall for r in ok if r.retrieval]),
        "mean_iterations": _safe_mean([float(r.iterations) for r in ok]),
        "total_cost_usd": round(sum(r.est_cost_usd for r in ok), 4),
        "p50_latency_ms": round(_percentile(latencies, 0.50), 1),
        "p95_latency_ms": round(_percentile(latencies, 0.95), 1),
        "by_kind": by_kind,
    }


def _fail_line(r: CaseResult) -> str:
    if r.error:
        return f"  ERROR  {r.id}: {r.error}"
    why = []
    if r.judge and not r.judge.passed:
        why.append(f"judge: {r.judge.reason or 'facts missing'}")
    if r.retrieval and r.retrieval.recall < 1.0:
        missing = set(r.retrieval.expected) - set(r.retrieval.matched)
        why.append(f"retrieval recall {r.retrieval.recall} (missed {sorted(missing)})")
    if not r.grounded:
        why.append(f"ungrounded ({r.n_unverified} bad citations)")
    return f"  FAIL   {r.id}: " + " | ".join(why or ["(see result)"])


def format_scorecard(report: SuiteReport) -> str:
    s = summarise(report)
    lines = [
        "=" * 64,
        f"EVAL  {s['n_cases']} cases"
        + (f"  ({s['n_errored']} errored)" if s["n_errored"] else "")
        + ("" if report.judged else "   [--no-judge: retrieval/system metrics only]"),
        "=" * 64,
    ]
    if report.judged:
        lines.append(f"  answer pass rate   {_pct(s['pass_rate'])}")
    lines += [
        f"  grounded rate      {_pct(s['grounded_rate'])}",
        f"  retrieval recall   {s['mean_recall']}   (seed-only {s['mean_seed_recall']})",
        f"  retrieval precision{s['mean_precision']:>6}",
        f"  mean iterations    {s['mean_iterations']}",
        f"  total est. cost    ${s['total_cost_usd']}",
        f"  latency p50 / p95  {s['p50_latency_ms']} / {s['p95_latency_ms']} ms",
        "",
        "  by kind:",
    ]
    for kind, k in s["by_kind"].items():
        pr = _pct(k["pass_rate"]) if report.judged else "-"
        lines.append(
            f"    {kind:<10} n={k['n']:<3} pass={pr:<5} "
            f"recall={k['mean_recall']} (seed {k['mean_seed_recall']})"
        )

    problems = [r for r in report.results if r.error or not r.passed]
    if problems:
        lines += ["", "  problems:"]
        lines += [_fail_line(r) for r in problems]
    lines.append("=" * 64)
    return "\n".join(lines)


def report_dict(report: SuiteReport) -> dict:
    return {
        "summary": summarise(report),
        "cases": [
            {
                "id": r.id,
                "kind": r.kind,
                "passed": r.passed,
                "error": r.error,
                "stop_reason": r.stop_reason,
                "iterations": r.iterations,
                "grounded": r.grounded,
                "n_unverified": r.n_unverified,
                "retrieval": None
                if r.retrieval is None
                else {
                    "recall": r.retrieval.recall,
                    "precision": r.retrieval.precision,
                    "seed_recall": r.retrieval.seed_recall,
                    "expected": list(r.retrieval.expected),
                    "used": list(r.retrieval.used),
                    "matched": list(r.retrieval.matched),
                },
                "judge": None
                if r.judge is None
                else {
                    "passed": r.judge.passed,
                    "reason": r.judge.reason,
                    "facts_found": list(r.judge.facts_found),
                },
                "agent_tokens": r.agent_tokens,
                "judge_tokens": r.judge_tokens,
                "est_cost_usd": r.est_cost_usd,
                "latency_ms": r.latency_ms,
                "answer": r.answer,
            }
            for r in report.results
        ],
    }
