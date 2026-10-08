"""COST-001 tests."""
from __future__ import annotations

from evalwarden.checks.cost import CostReportingCheck, summarize_cost
from evalwarden.model import Attempt, Confidence, Severity

from .conftest import make_model


def _priced_attempt(task_id: str, status: str = "pass") -> Attempt:
    return Attempt(
        task_id=task_id,
        status=status,
        score=1.0 if status == "pass" else 0.0,
        tool_calls=4,
        tokens_in=1_000_000,
        tokens_out=1_000_000,
    )


def test_cost_math():
    attempts = [_priced_attempt("t1"), _priced_attempt("t2")]
    summary = summarize_cost(attempts, price_in_per_1m=3.0, price_out_per_1m=15.0)
    assert summary is not None
    assert summary.total_tokens_in == 2_000_000
    assert summary.total_tokens_out == 2_000_000
    assert summary.estimated_usd == 36.0
    assert summary.cost_per_success_usd == 18.0
    assert summary.avg_tool_calls_per_success == 4.0


def test_cost_per_success_with_failures():
    attempts = [_priced_attempt("t1"), _priced_attempt("t2", status="fail")]
    summary = summarize_cost(attempts)
    assert summary is not None
    assert summary.successes == 1
    assert summary.cost_per_success_usd == 36.0


def test_missing_token_data_flagged():
    model = make_model(attempts=[Attempt(task_id="t1", status="pass", score=1.0)])
    findings = CostReportingCheck().run(model)
    assert len(findings) == 1
    assert findings[0].id == "COST-001"
    assert findings[0].severity == Severity.MEDIUM
    assert findings[0].confidence == Confidence.HIGH


def test_present_token_data_no_finding_and_summary_attached():
    model = make_model(attempts=[_priced_attempt("t1")])
    findings = CostReportingCheck().run(model)
    assert findings == []
    assert model.cost_summary is not None
    assert model.cost_summary.cost_per_success_usd == 18.0


def test_no_attempts_no_finding():
    model = make_model(attempts=[])
    assert CostReportingCheck().run(model) == []


# ---------------------------------------------------------------------------
# COST-002 / COST-003 / COST-004: efficiency checks
# ---------------------------------------------------------------------------
from evalwarden.checks.cost import (  # noqa: E402
    RetryMultiplierCheck,
    RunawayAttemptCheck,
    WastedSpendCheck,
)


def _eff_attempt(task_id, status="pass", tokens_in=2000, tokens_out=800, tries=1):
    return Attempt(
        task_id=task_id,
        status=status,
        score=1.0 if status == "pass" else 0.0,
        tool_calls=5,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        tries=tries,
    )


def _run_with_summary(attempts, **kwargs):
    """Attach a cost summary to a model, like COST-001 does."""
    from evalwarden.checks.cost import summarize_cost

    model = make_model(attempts=attempts)
    model.cost_summary = summarize_cost(attempts, **kwargs)
    return model


def test_summary_efficiency_fields():
    attempts = [
        _eff_attempt("t1", "fail", tries=2),
        _eff_attempt("t1", "pass", tries=2),
        _eff_attempt("t2", "fail"),
        _eff_attempt("t3", "fail", tokens_out=22000),  # runaway, and it failed
    ]
    s = summarize_cost(attempts)
    assert s.avg_tries_per_success == 4.0  # only t1 succeeded, after 4 tries
    assert s.max_tries == 2
    assert s.wasted_share > 0.5
    assert s.p50_tokens_out == 800
    assert s.max_tokens_out == 22000
    assert s.max_tokens_out_task == "t3"


def test_summary_budget_capped_success():
    attempts = [
        _eff_attempt("t1", "pass", tokens_in=1000, tokens_out=100),  # $0.0045
        _eff_attempt("t2", "pass", tokens_in=100000, tokens_out=100000),  # $1.80
    ]
    s = summarize_cost(attempts, budget_per_task_usd=0.05)
    assert s.tasks_total == 2
    assert s.tasks_within_budget == 1


def test_cost_002_retry_multiplier_fires():
    attempts = [
        _eff_attempt("t1", "fail", tries=2),
        _eff_attempt("t1", "pass", tries=2),
        _eff_attempt("t2", "pass", tries=3),
    ]
    findings = RetryMultiplierCheck().run(_run_with_summary(attempts))
    assert len(findings) == 1
    assert findings[0].id == "COST-002"
    assert findings[0].severity == Severity.MEDIUM
    assert findings[0].confidence == Confidence.HIGH


def test_cost_002_no_finding_when_efficient():
    attempts = [_eff_attempt("t1", "pass"), _eff_attempt("t2", "pass", tries=2)]
    assert RetryMultiplierCheck().run(_run_with_summary(attempts)) == []


def test_cost_002_no_finding_without_token_data():
    # No priced attempts: COST-001 owns this case; efficiency checks stay quiet.
    model = make_model(attempts=[Attempt(task_id="t1", status="pass", tries=5)])
    assert RetryMultiplierCheck().run(model) == []
    assert WastedSpendCheck().run(model) == []
    assert RunawayAttemptCheck().run(model) == []


def test_cost_003_wasted_spend_fires():
    attempts = [
        _eff_attempt("t1", "fail"),
        _eff_attempt("t2", "fail"),
        _eff_attempt("t3", "pass"),
    ]
    findings = WastedSpendCheck().run(_run_with_summary(attempts))
    assert len(findings) == 1
    assert findings[0].id == "COST-003"
    assert findings[0].severity == Severity.MEDIUM
    assert findings[0].confidence == Confidence.HIGH


def test_cost_003_no_finding_when_mostly_useful():
    attempts = [
        _eff_attempt("t1", "pass"),
        _eff_attempt("t2", "pass"),
        _eff_attempt("t3", "fail", tokens_in=100, tokens_out=100),
    ]
    assert WastedSpendCheck().run(_run_with_summary(attempts)) == []


def test_cost_004_runaway_fires():
    attempts = [_eff_attempt(f"t{i}") for i in range(6)]
    attempts.append(_eff_attempt("t7", "fail", tokens_out=22000))
    findings = RunawayAttemptCheck().run(_run_with_summary(attempts))
    assert len(findings) == 1
    assert findings[0].id == "COST-004"
    assert findings[0].severity == Severity.MEDIUM
    assert findings[0].confidence == Confidence.MEDIUM
    assert "t7" in findings[0].title


def test_cost_004_no_finding_when_uniform():
    attempts = [_eff_attempt(f"t{i}") for i in range(6)]
    assert RunawayAttemptCheck().run(_run_with_summary(attempts)) == []


def test_cost_004_needs_minimum_attempts():
    # Fewer than 5 priced attempts: no baseline to call anything a runaway.
    attempts = [_eff_attempt("t1", tokens_out=50000)]
    assert RunawayAttemptCheck().run(_run_with_summary(attempts)) == []


def test_with_options_configures_cost_check():
    from evalwarden.checks.env_leakage import EnvLeakageCheck

    plain = EnvLeakageCheck()
    assert plain.with_options(price_in_per_1m=1.0) is plain  # base default: self
    configured = CostReportingCheck().with_options(
        price_in_per_1m=1.0, price_out_per_1m=2.0, budget_per_task_usd=0.5
    )
    assert isinstance(configured, CostReportingCheck)
    assert configured.price_in_per_1m == 1.0
    assert configured.price_out_per_1m == 2.0
    assert configured.budget_per_task_usd == 0.5


def test_audit_budget_option_flows_through(tmp_path):
    from typer.testing import CliRunner

    from evalwarden.cli import app

    from .conftest import DEMO_COST_CLEAN

    report = tmp_path / "report.html"
    result = CliRunner().invoke(
        app,
        ["audit", str(DEMO_COST_CLEAN), "--output", str(report), "--budget-per-task", "0.05"],
    )
    assert result.exit_code == 0, result.output
    assert "within $0.05/task budget: 4/4 tasks passed" in result.output
    html = report.read_text()
    assert "Within $0.05/task budget" in html
