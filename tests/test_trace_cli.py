"""`evalwarden trace`: the one-command observatory path.

Kill criterion for this slice: from a fresh shell, one command with no
decisions produces the observatory page -- via --demo on a machine with
no sessions, via auto-discovery on a machine with stock session dirs,
or via one explicit path.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from evalwarden.cli import app
from evalwarden.reporters.trace_report import format_usd

from .trace_fixtures import (
    CLAUDE_CLEAN_SESSION,
    CLAUDE_LOOP_SESSION,
    CODEX_CLEAN_FILE,
    CODEX_LOOP_FILE,
    claude_clean_records,
    claude_loop_records,
    codex_clean_records,
    codex_loop_records,
    write_claude_project,
    write_codex_session,
)

runner = CliRunner()


def test_format_usd_keeps_sub_cent_figures_exact():
    assert format_usd(0.01215) == "$0.01215"
    assert format_usd(0.0042) == "$0.0042"
    assert format_usd(0.08625) == "$0.08625"
    assert format_usd(1.5) == "$1.50"
    assert format_usd(1234.5) == "$1,234.50"


def test_trace_demo_first_run(tmp_path: Path):
    report = tmp_path / "trace.html"
    result = runner.invoke(app, ["trace", "--demo", "--output", str(report)])
    assert result.exit_code == 0, result.output
    # The five-line summary: sessions, tokens/spend, waste, phases, report.
    assert "Trace observatory: 5 session(s)" in result.output
    assert "15,550 in / 2,640 out" in result.output
    assert "Wasted on loops: $0.01635 across 7 wasted call(s)" in result.output
    assert "Phases:" in result.output and "explore" in result.output
    assert f"Report: {report}" in result.output
    html = report.read_text(encoding="utf-8")
    assert "Trace observatory" in html
    assert "$0.0042" in html and "$0.01215" in html  # both planted loops, priced exactly
    assert "explore" in html and "review" in html  # phase timeline present
    assert "What this page cannot see" in html  # coverage honesty on the page


def test_trace_explicit_codex_tree(tmp_path: Path):
    root = tmp_path / "sessions" / "2026" / "10" / "02"
    write_codex_session(root, codex_loop_records(), CODEX_LOOP_FILE)
    report = tmp_path / "trace.html"
    result = runner.invoke(app, ["trace", str(tmp_path / "sessions"),
                                 "--output", str(report)])
    assert result.exit_code == 0, result.output
    assert "1 session(s) (codex)" in result.output
    assert "Wasted on loops: $0.01215 across 3 wasted call(s)" in result.output


def test_trace_auto_discovery(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    home = tmp_path / "home"
    project = home / ".claude" / "projects" / "my-proj"
    write_claude_project(project, claude_loop_records(), CLAUDE_LOOP_SESSION)
    write_codex_session(home / ".codex" / "sessions" / "2026" / "10" / "01",
                        codex_clean_records(), CODEX_CLEAN_FILE)
    monkeypatch.setenv("HOME", str(home))
    report = tmp_path / "trace.html"
    result = runner.invoke(app, ["trace", "--output", str(report)])
    assert result.exit_code == 0, result.output
    assert "2 session(s)" in result.output
    assert "claude-code" in result.output and "codex" in result.output


def test_trace_no_sessions_found(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path / "empty-home"))
    result = runner.invoke(app, ["trace", "--output", str(tmp_path / "t.html")])
    assert result.exit_code == 2
    assert "--demo" in result.output  # the error names the way out


def test_trace_since_filters_old_sessions(tmp_path: Path):
    root = tmp_path / "sessions"
    write_codex_session(root / "2026" / "10" / "01", codex_clean_records(), CODEX_CLEAN_FILE)
    write_codex_session(root / "2026" / "10" / "02", codex_loop_records(), CODEX_LOOP_FILE)
    report = tmp_path / "trace.html"
    result = runner.invoke(app, ["trace", str(root), "--since", "2026-10-02",
                                 "--output", str(report)])
    assert result.exit_code == 0, result.output
    assert "1 session(s)" in result.output  # only the Oct 2 loop session
    assert "$0.01215" in result.output
    result = runner.invoke(app, ["trace", str(root), "--since", "2026-10-03",
                                 "--output", str(report)])
    assert result.exit_code == 2  # everything filtered out: a clear error, not an empty page


def test_trace_project_and_agent_filters(tmp_path: Path):
    project = tmp_path / "alpha-proj"
    write_claude_project(project, claude_clean_records(), CLAUDE_CLEAN_SESSION)
    report = tmp_path / "trace.html"
    result = runner.invoke(app, ["trace", str(project), "--agent", "codex",
                                 "--output", str(report)])
    assert result.exit_code == 2  # wrong agent for this source: refused, not empty
    result = runner.invoke(app, ["trace", str(project), "--agent", "claude-code",
                                 "--output", str(report)])
    assert result.exit_code == 0, result.output
    assert "1 session(s) (claude-code)" in result.output


def test_trace_bad_since_exits_2(tmp_path: Path):
    result = runner.invoke(app, ["trace", "--demo", "--since", "yesterday",
                                 "--output", str(tmp_path / "t.html")])
    assert result.exit_code == 2
    assert "--since" in result.output
