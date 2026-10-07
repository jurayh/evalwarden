"""Report card reporter tests."""
from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from evalwarden.checks import BY_ID
from evalwarden.cli import app
from evalwarden.engine import AuditResult, integrity_score
from evalwarden.model import Confidence, Finding, Severity, SourceLocation
from evalwarden.reporters.report_card import (
    category_of,
    category_scores,
    grouped_checks,
    render_index,
    render_report_card,
    slugify,
)

from .conftest import DEMO_HARDENED, DEMO_LEAKY, make_model

runner = CliRunner()


def _finding(check_id, severity=Severity.HIGH):
    return Finding(
        id=check_id,
        title=f"{check_id} title",
        severity=severity,
        confidence=Confidence.HIGH,
        description="why",
        evidence=["direct evidence"],
        locations=[SourceLocation(file="run.json")],
    )


def _result(*findings):
    model = make_model()
    return AuditResult(
        model=model,
        findings=list(findings),
        score=integrity_score(findings),
        verdict="BLOCKED" if findings else "PASS",
        blocked_by=[f for f in findings if f.severity in (Severity.ERROR, Severity.HIGH)],
    )


def test_category_of():
    assert category_of("ENV-001") == "Environment"
    assert category_of("GRAD-002") == "Grader"
    assert category_of("JUDGE-004") == "Judge"
    assert category_of("COST-003") == "Cost and efficiency"
    assert category_of("XXX-001") == "Other"


def test_category_scores_use_same_deductions():
    result = _result(
        _finding("ENV-001", Severity.ERROR),  # -25
        _finding("COST-002", Severity.MEDIUM),  # -5
        _finding("COST-003", Severity.MEDIUM),  # -5
    )
    scores = {c.name: c for c in category_scores(result)}
    assert scores["Environment"].score == 75
    assert scores["Environment"].errors == 1
    assert scores["Cost and efficiency"].score == 90
    assert scores["Cost and efficiency"].mediums == 2
    assert scores["Grader"].score == 100  # no findings: clean, not absent


def test_report_card_renders_verdict_and_categories():
    result = _result(_finding("ENV-001", Severity.ERROR))
    html = render_report_card(result, version="0.0.0-test")
    assert "Integrity report card" in html
    assert "test-eval" in html
    assert "BLOCKED" in html
    assert "Environment" in html
    assert "Cost and efficiency" in html
    assert "ENV-001" in html
    assert "Methodology" in html
    assert "0.0.0-test" in html


def test_report_card_is_self_contained():
    result = _result()
    html = render_report_card(result, version="0.0.0-test").lower()
    assert "<script" not in html
    assert "http://www" not in html and 'src="http' not in html and 'href="http' not in html


def test_report_card_pass_state():
    result = _result()
    html = render_report_card(result, version="0.0.0-test")
    assert "PASS" in html
    assert "clean" in html


def test_report_card_shows_no_secret_values():
    # The card summarizes findings and methodology; it never carries env
    # names, values, or any other raw material from the eval boundary.
    model = make_model(env_vars={"AGENT_TOKEN": "<redacted>"})
    result = AuditResult(
        model=model, findings=[], score=100, verdict="PASS", blocked_by=[]
    )
    html = render_report_card(result, version="0.0.0-test")
    assert "AGENT_TOKEN" not in html
    assert "sk-" not in html


def test_index_lists_cards():
    from evalwarden.reporters.report_card import CardEntry

    cards = [
        CardEntry("eval-a", "BLOCKED", 0, "eval-a.html", "2026-09-28 00:00 UTC"),
        CardEntry("eval-b", "PASS", 100, "eval-b.html", "2026-09-28 00:00 UTC"),
    ]
    html = render_index(cards, version="0.0.0-test")
    assert "eval-a.html" in html and "eval-b.html" in html
    assert "BLOCKED" in html and "PASS" in html
    assert "<script" not in html.lower()


def test_slugify():
    assert slugify("tinycode-leaky-1.0") == "tinycode-leaky-1-0"
    assert slugify("Judge Demo!") == "judge-demo"


def test_grouped_checks_lane_order_and_coverage():
    groups = grouped_checks()
    labels = [label for label, _ in groups]
    assert labels == [
        "Environment",
        "Grader",
        "Cost and efficiency",
        "Judge",
        "Dataset",
        "Trajectory",
        "Noise budget",
    ]
    seen = [cid for _, items in groups for cid, _ in items]
    assert sorted(seen) == sorted(BY_ID)  # every check exactly once
    for _, items in groups:
        ids = [cid for cid, _ in items]
        assert ids == sorted(ids)
        for cid, title in items:
            assert title == BY_ID[cid].meta.title


def test_report_card_checks_run_grouped_title_first():
    html = render_report_card(_result(), version="0.0.0-test")
    # Lane groups appear in audit order.
    labels = [
        "Environment",
        "Grader",
        "Cost and efficiency",
        "Judge",
        "Dataset",
        "Trajectory",
        "Noise budget",
    ]
    positions = [html.index(f'<div class="lanelabel">{label}</div>') for label in labels]
    assert positions == sorted(positions)
    # Each check is its own line: plain title first, code as a muted chip.
    assert (
        '<li><span class="ct">Model judge lacks validation</span> '
        '<span class="code">JUDGE-001</span></li>'
    ) in html
    # The old one-paragraph "ID: Title; ID: Title" format is gone.
    assert "JUDGE-001: Model judge lacks validation" not in html
    assert "; JUDGE-002" not in html
    # A clean card mentions every check id exactly once (in Checks run).
    for cid in BY_ID:
        assert html.count(cid) == 1


def test_report_card_finding_headline_title_first():
    finding = Finding(
        id="JUDGE-001",
        title="Model judge lacks validation",
        severity=Severity.HIGH,
        confidence=Confidence.HIGH,
        description="why",
        evidence=["direct evidence"],
        locations=[SourceLocation(file="run.json")],
    )
    html = render_report_card(_result(finding), version="0.0.0-test")
    assert (
        '<span class="title">Model judge lacks validation</span> '
        '<span class="code">JUDGE-001</span>'
    ) in html
    assert "JUDGE-001: Model judge lacks validation" not in html
    # The masthead blocked-by line leads with the title too, code in parens.
    assert "Blocked by Model judge lacks validation (JUDGE-001)" in html


def test_cli_report_card(tmp_path: Path):
    out = tmp_path / "card.html"
    result = runner.invoke(app, ["report-card", str(DEMO_LEAKY), "--output", str(out)])
    assert result.exit_code == 0, result.output
    html = out.read_text()
    assert "Integrity report card" in html
    assert "BLOCKED" in html


def test_cli_report_cards_with_fixtures(tmp_path: Path):
    out_dir = tmp_path / "cards"
    result = runner.invoke(
        app, ["report-cards", "--fixtures", "leaky,hardened", "--output-dir", str(out_dir)]
    )
    assert result.exit_code == 0, result.output
    names = {p.name for p in out_dir.iterdir()}
    assert "index.html" in names
    cards = [p for p in out_dir.glob("*.html") if p.name != "index.html"]
    assert len(cards) == 2
    index = (out_dir / "index.html").read_text()
    assert all(c.name in index for c in cards)


def test_cli_report_cards_needs_input():
    result = runner.invoke(app, ["report-cards"])
    assert result.exit_code == 2
