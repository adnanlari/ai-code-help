from evals.report import format_scorecard, report_dict, summarise
from evals.runner import CaseResult, SuiteReport
from evals.scorers import JudgeScore, score_retrieval


def _case(cid, kind, *, passed, recall, grounded=True, error=None):
    hits = ["a.py", "b.py"] if recall >= 1 else ["a.py"]
    return CaseResult(
        id=cid,
        kind=kind,
        question="q",
        answer="a",
        stop_reason="answered",
        iterations=3,
        grounded=grounded,
        retrieval=None if error else score_retrieval(["a.py", "b.py"], hits, []),
        judge=None if error else JudgeScore(passed, "reason", (passed,), "raw"),
        est_cost_usd=0.01,
        latency_ms=1000.0 + len(cid),
        error=error,
    )


def _report():
    return SuiteReport(
        judged=True,
        results=[
            _case("loc-pass", "localized", passed=True, recall=1.0),
            _case("loc-fail", "localized", passed=False, recall=0.5, grounded=False),
            _case("multi-pass", "multihop", passed=True, recall=1.0),
            _case("boom", "multihop", passed=False, recall=0.0, error="LLMError: nope"),
        ],
    )


def test_summarise_aggregates():
    s = summarise(_report())
    assert s["n_cases"] == 4
    assert s["n_errored"] == 1
    # judged & non-errored = 3 cases, 2 passed
    assert s["pass_rate"] == round(2 / 3, 3)
    assert s["grounded_rate"] == round(2 / 3, 3)  # 2 of 3 non-errored grounded
    assert s["by_kind"]["localized"]["pass_rate"] == 0.5
    assert s["by_kind"]["multihop"]["pass_rate"] == 1.0  # only the non-errored multihop counts
    assert s["p95_latency_ms"] >= s["p50_latency_ms"]


def test_scorecard_text_has_sections():
    txt = format_scorecard(_report())
    assert "EVAL  4 cases" in txt
    assert "answer pass rate" in txt
    assert "by kind:" in txt
    assert "problems:" in txt
    assert "boom" in txt and "loc-fail" in txt


def test_report_dict_is_json_shaped():
    d = report_dict(_report())
    assert set(d) == {"summary", "cases"}
    assert len(d["cases"]) == 4
    boom = next(c for c in d["cases"] if c["id"] == "boom")
    assert boom["error"].startswith("LLMError")
    assert boom["retrieval"] is None


def test_no_judge_report_still_summarises():
    rep = SuiteReport(
        judged=False,
        results=[
            CaseResult(
                id="x",
                kind="localized",
                question="q",
                retrieval=score_retrieval(["a.py"], ["a.py"], []),
                latency_ms=500.0,
            )
        ],
    )
    s = summarise(rep)
    assert s["judged"] is False
    assert s["pass_rate"] == 0.0  # nothing judged
    assert s["mean_recall"] == 1.0
    assert "retrieval/system metrics only" in format_scorecard(rep)
