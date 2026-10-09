"""CLI tests via typer's CliRunner."""
from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from evalwarden.cli import app

from .conftest import DEMO_HARDENED, DEMO_LEAKY

runner = CliRunner()


def test_audit_leaky_exits_1_and_writes_report(tmp_path: Path):
    report = tmp_path / "report.html"
    result = runner.invoke(app, ["audit", str(DEMO_LEAKY), "--output", str(report)])
    assert result.exit_code == 1, result.output
    assert "BLOCKED" in result.output
    assert "ENV-001" in result.output
    assert "GRAD-001" in result.output
    assert report.is_file()
    html = report.read_text()
    assert "ENV-001" in html
    assert "GRAD-001" in html
    assert "confidence" in html


def test_audit_hardened_exits_0(tmp_path: Path):
    report = tmp_path / "report.html"
    result = runner.invoke(app, ["audit", str(DEMO_HARDENED), "--output", str(report)])
    assert result.exit_code == 0, result.output
    assert "PASS" in result.output
    assert "no blocking findings" in result.output


def test_audit_json_output(tmp_path: Path):
    report = tmp_path / "report.html"
    payload = tmp_path / "findings.json"
    result = runner.invoke(
        app, ["audit", str(DEMO_LEAKY), "--output", str(report), "--json", str(payload)]
    )
    assert result.exit_code == 1
    import json

    data = json.loads(payload.read_text())
    assert data["schema"] == "evalwarden.findings"
    assert data["schema_version"] == "1.0"
    assert data["verdict"] == "BLOCKED"
    assert any(f["check_id"] == "ENV-001" for f in data["findings"])
    assert all("confidence" in f for f in data["findings"])


def test_audit_unknown_path_exits_2():
    result = runner.invoke(app, ["audit", "/does/not/exist"])
    assert result.exit_code == 2


def test_explain_known_check():
    result = runner.invoke(app, ["explain", "ENV-001"])
    assert result.exit_code == 0
    assert "Eval-detection" in result.output


def test_explain_unknown_check():
    result = runner.invoke(app, ["explain", "NOPE-999"])
    assert result.exit_code == 2


def test_fail_on_error_still_blocks_leaky(tmp_path: Path):
    report = tmp_path / "report.html"
    result = runner.invoke(
        app, ["audit", str(DEMO_LEAKY), "--output", str(report), "--fail-on", "error"]
    )
    assert result.exit_code == 1


def test_audit_judge_bad_exits_1_with_all_judge_checks(tmp_path: Path):
    from .conftest import DEMO_JUDGE_BAD

    report = tmp_path / "report.html"
    result = runner.invoke(app, ["audit", str(DEMO_JUDGE_BAD), "--output", str(report)])
    assert result.exit_code == 1, result.output
    assert "BLOCKED" in result.output
    for check_id in ("JUDGE-001", "JUDGE-002", "JUDGE-003", "JUDGE-004", "JUDGE-005", "JUDGE-006"):
        assert check_id in result.output, check_id
    html = report.read_text()
    assert "JUDGE-004" in html


def test_audit_judge_clean_exits_0(tmp_path: Path):
    from .conftest import DEMO_JUDGE_CLEAN

    report = tmp_path / "report.html"
    result = runner.invoke(app, ["audit", str(DEMO_JUDGE_CLEAN), "--output", str(report)])
    assert result.exit_code == 0, result.output
    assert "PASS" in result.output


def test_demo_judge_bad_fixture(tmp_path: Path):
    report = tmp_path / "demo.html"
    result = runner.invoke(app, ["demo", "--fixture", "judge_bad", "--output", str(report)])
    # Demo always exits 0 (findings are the point of the fixture); `audit` is the gate.
    assert result.exit_code == 0, result.output
    assert "JUDGE-004" in result.output
    assert "by design" in result.output
    assert report.is_file()


def test_demo_judge_clean_fixture(tmp_path: Path):
    report = tmp_path / "demo.html"
    result = runner.invoke(app, ["demo", "--fixture", "judge_clean", "--output", str(report)])
    assert result.exit_code == 0, result.output
    assert "PASS" in result.output


def test_demo_unknown_fixture_exits_2():
    result = runner.invoke(app, ["demo", "--fixture", "nope"])
    assert result.exit_code == 2


def test_explain_judge_check():
    result = runner.invoke(app, ["explain", "JUDGE-001"])
    assert result.exit_code == 0
    assert "calibration" in result.output


def test_demo_cost_wasteful_fixture(tmp_path: Path):
    report = tmp_path / "demo.html"
    result = runner.invoke(app, ["demo", "--fixture", "cost_wasteful", "--output", str(report)])
    # Medium findings only: priced waste, but the score itself is not invalidated.
    assert result.exit_code == 0, result.output
    assert "COST-002" in result.output
    assert "COST-003" in result.output
    assert "COST-004" in result.output
    assert "avg tries per success" in result.output
    assert report.is_file()
    assert "COST-004" in report.read_text()


def test_demo_cost_clean_fixture(tmp_path: Path):
    report = tmp_path / "demo.html"
    result = runner.invoke(app, ["demo", "--fixture", "cost_clean", "--output", str(report)])
    assert result.exit_code == 0, result.output
    assert "PASS" in result.output
    assert "COST-002" not in result.output


def test_checks_lists_full_registry_grouped():
    import re

    from evalwarden.checks import REGISTRY
    from evalwarden.reporters.report_card import LANES

    result = runner.invoke(app, ["checks"])
    assert result.exit_code == 0, result.output
    assert result.output.startswith(f"{len(REGISTRY)} checks")
    # Title-first lines for every registered check, no more, no fewer.
    listed = re.findall(r"\(([A-Z]+-\d{3})\)", result.output)
    assert sorted(listed) == sorted(c.meta.id for c in REGISTRY)
    assert len(listed) == len(set(listed))
    for check in REGISTRY:
        assert f"{check.meta.title} ({check.meta.id})" in result.output
    # Lanes appear in the same order the report card uses.
    positions = [result.output.index(f"\n{label}\n") for _, label in LANES]
    assert positions == sorted(positions)
