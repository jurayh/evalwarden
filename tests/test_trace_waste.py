"""Trajectory waste: loops and unused outputs, priced exactly or not at all."""
from __future__ import annotations

from pathlib import Path

from evalwarden.adapters.claude_code import ClaudeCodeAdapter
from evalwarden.adapters.codex import CodexAdapter
from evalwarden.trace_waste import analyze_waste

from .conftest import make_attempt, make_model, make_step
from .trace_fixtures import (
    CLAUDE_CLEAN_SESSION,
    CLAUDE_LOOP_SESSION,
    CLAUDE_LOOP_WASTE,
    CODEX_LOOP_FILE,
    CODEX_LOOP_WASTE,
    claude_clean_records,
    claude_loop_records,
    codex_loop_records,
    write_claude_project,
    write_codex_session,
)


def _claude_model(tmp_path: Path, records, session: str):
    path = write_claude_project(tmp_path, records, session)
    adapter = ClaudeCodeAdapter()
    return adapter.normalize(adapter.collect(path))


def test_claude_loop_waste_priced_exactly(tmp_path: Path):
    model = _claude_model(tmp_path, claude_loop_records(), CLAUDE_LOOP_SESSION)
    report = analyze_waste(model)
    (loop,) = report.loops
    assert loop.task_id == CLAUDE_LOOP_SESSION
    assert loop.total == 5 and loop.wasted_calls == CLAUDE_LOOP_WASTE["calls"]
    assert (loop.tokens_in, loop.tokens_out) == (
        CLAUDE_LOOP_WASTE["tokens_in"], CLAUDE_LOOP_WASTE["tokens_out"])
    # 800 in @ $3/1M + 120 out @ $15/1M = 0.0024 + 0.0018
    assert loop.usd == 0.0042 and loop.priced
    assert report.total_wasted_usd == 0.0042
    assert report.total_wasted_calls == 4
    assert report.unpriced_wasted_calls == 0
    # No consumption edges in this format: unused-output waste stays
    # silent, exactly as TRAJ-002 does -- it is not reported as zero waste.
    assert report.unused == []


def test_codex_loop_waste_priced_from_turn_deltas(tmp_path: Path):
    path = write_codex_session(tmp_path, codex_loop_records(), CODEX_LOOP_FILE)
    adapter = CodexAdapter()
    model = adapter.normalize(adapter.collect(path))
    report = analyze_waste(model)
    (loop,) = report.loops
    assert loop.total == 4 and loop.wasted_calls == CODEX_LOOP_WASTE["calls"]
    assert (loop.tokens_in, loop.tokens_out) == (
        CODEX_LOOP_WASTE["tokens_in"], CODEX_LOOP_WASTE["tokens_out"])
    # 1800 in @ $3/1M + 450 out @ $15/1M = 0.0054 + 0.00675
    assert loop.usd == 0.01215 and loop.priced
    assert report.unused == []


def test_clean_traces_have_no_waste(tmp_path: Path):
    model = _claude_model(tmp_path, claude_clean_records(), CLAUDE_CLEAN_SESSION)
    report = analyze_waste(model)
    assert report.loops == [] and report.unused == []
    assert report.total_wasted_usd == 0.0 and report.total_wasted_calls == 0


def test_unpriced_loop_is_reported_not_estimated():
    # Spans without recorded per-step tokens: the loop is real, its price
    # is unknown, and the report says so instead of splitting a total.
    spans = [make_step(step_id=f"s{i}", tool="bash", args={"command": "npm test"})
             for i in range(5)]
    report = analyze_waste(make_model(attempts=[make_attempt(task_id="t1", spans=spans)]))
    (loop,) = report.loops
    assert loop.wasted_calls == 4 and not loop.priced
    assert loop.tokens_in is None and loop.usd is None
    assert report.unpriced_wasted_calls == 4
    assert report.total_wasted_usd == 0.0


def test_unused_output_waste_priced_with_recorded_flow():
    # 6 steps; only s0 is consumed (by s1). Unused (final step excluded):
    # s1, s2, s3, s4 -- four outputs consumed by nothing downstream.
    spans = []
    for i in range(6):
        spans.append(make_step(
            step_id=f"s{i}", tool=f"tool_{i}", args={"i": i},
            consumes=["s0"] if i == 1 else [],
        ))
        spans[-1].tokens_in, spans[-1].tokens_out = 10, 5
    report = analyze_waste(make_model(attempts=[make_attempt(task_id="t1", spans=spans)]))
    (unused,) = report.unused
    assert unused.step_ids == ["s1", "s2", "s3", "s4"]
    assert (unused.tokens_in, unused.tokens_out) == (40, 20)
    # 40 in @ $3/1M + 20 out @ $15/1M = 0.00012 + 0.0003
    assert unused.usd == 0.00042 and unused.priced
    assert report.loops == []


def test_unused_stays_silent_without_consumption_data():
    spans = [make_step(step_id=f"s{i}", tool=f"tool_{i}", args={"i": i}) for i in range(8)]
    report = analyze_waste(make_model(attempts=[make_attempt(task_id="t1", spans=spans)]))
    assert report.unused == []
