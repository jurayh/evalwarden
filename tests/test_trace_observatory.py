"""Trace observatory kill criterion, end to end through the audit engine.

- TRAJ-001 fires on the planted real-format loop fixtures (Claude Code
  and Codex) and stays silent on the clean ones.
- TRAJ-002 stays silent on both formats: neither records consumption
  edges, and silent is the honest answer -- not a pass.
- Importer token totals reconcile three ways: the hand-computed
  constants in tests/trace_fixtures.py, an independent re-sum of the
  raw fixture files done here (no adapter code), and the adapter output.
- Segmentation boundaries on the Codex fixture with planted phase
  structure land exactly on the planted change points (asserted in
  test_trace_phases.py and re-checked here through autodetect).
"""
from __future__ import annotations

import json
from pathlib import Path

from evalwarden.engine import audit
from evalwarden.trace_phases import segment_phases

from .trace_fixtures import (
    CLAUDE_CLEAN_SESSION,
    CLAUDE_CLEAN_TOTALS,
    CLAUDE_LOOP_SESSION,
    CLAUDE_LOOP_TOTALS,
    CODEX_CLEAN_FILE,
    CODEX_CLEAN_TOTALS,
    CODEX_LOOP_FILE,
    CODEX_LOOP_PHASES,
    CODEX_LOOP_TOTALS,
    claude_clean_records,
    claude_loop_records,
    codex_clean_records,
    codex_loop_records,
    write_claude_project,
    write_codex_session,
)


def _finding_ids(result) -> set[str]:
    return {f.id for f in result.findings}


def _raw_claude_totals(path: Path) -> tuple[int, int]:
    """Independent re-sum of a Claude session file (no adapter code)."""
    tokens_in = tokens_out = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("type") == "assistant":
            usage = record["message"]["usage"]
            tokens_in += usage["input_tokens"]
            tokens_out += usage["output_tokens"]
    return tokens_in, tokens_out


def _raw_codex_totals(path: Path) -> tuple[int, int]:
    """Independent read of a rollout: the final cumulative token_count."""
    final = (0, 0)
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("type") == "event_msg" and record["payload"].get("type") == "token_count":
            total = record["payload"]["info"]["total_token_usage"]
            final = (total["input_tokens"], total["output_tokens"])
    return final


def test_kill_criterion_claude_loop_fires_clean_silent(tmp_path: Path):
    loop_dir = tmp_path / "loop"
    write_claude_project(loop_dir, claude_loop_records(), CLAUDE_LOOP_SESSION)
    result = audit(loop_dir, adapter_name="claude-code")
    assert result.model.adapter_name == "claude-code"
    assert "TRAJ-001" in _finding_ids(result)
    assert "TRAJ-002" not in _finding_ids(result)  # no edges recorded: silent

    clean_dir = tmp_path / "clean"
    write_claude_project(clean_dir, claude_clean_records(), CLAUDE_CLEAN_SESSION)
    result = audit(clean_dir, adapter_name="claude-code")
    assert "TRAJ-001" not in _finding_ids(result)
    assert "TRAJ-002" not in _finding_ids(result)


def test_kill_criterion_codex_loop_fires_clean_silent(tmp_path: Path):
    loop_path = write_codex_session(tmp_path, codex_loop_records(), CODEX_LOOP_FILE)
    result = audit(loop_path, adapter_name="codex")
    assert result.model.adapter_name == "codex"
    assert "TRAJ-001" in _finding_ids(result)
    assert "TRAJ-002" not in _finding_ids(result)

    clean_path = write_codex_session(tmp_path / "clean", codex_clean_records(), CODEX_CLEAN_FILE)
    result = audit(clean_path, adapter_name="codex")
    assert "TRAJ-001" not in _finding_ids(result)
    assert "TRAJ-002" not in _finding_ids(result)


def test_autodetect_picks_the_trace_adapters(tmp_path: Path):
    claude_file = write_claude_project(
        tmp_path / "p", claude_clean_records(), CLAUDE_CLEAN_SESSION)
    assert audit(claude_file).model.adapter_name == "claude-code"

    codex_file = write_codex_session(
        tmp_path / "c", codex_clean_records(), CODEX_CLEAN_FILE)
    assert audit(codex_file).model.adapter_name == "codex"


def test_reconciliation_three_ways(tmp_path: Path):
    claude_file = write_claude_project(
        tmp_path / "p", claude_loop_records(), CLAUDE_LOOP_SESSION)
    raw = _raw_claude_totals(claude_file)
    assert raw == (CLAUDE_LOOP_TOTALS["tokens_in"], CLAUDE_LOOP_TOTALS["tokens_out"])
    (attempt,) = audit(claude_file, adapter_name="claude-code").model.attempts
    assert (attempt.tokens_in, attempt.tokens_out) == raw

    claude_clean = write_claude_project(
        tmp_path / "p2", claude_clean_records(), CLAUDE_CLEAN_SESSION)
    raw = _raw_claude_totals(claude_clean)
    assert raw == (CLAUDE_CLEAN_TOTALS["tokens_in"], CLAUDE_CLEAN_TOTALS["tokens_out"])

    codex_file = write_codex_session(tmp_path / "c", codex_clean_records(), CODEX_CLEAN_FILE)
    raw = _raw_codex_totals(codex_file)
    assert raw == (CODEX_CLEAN_TOTALS["tokens_in"], CODEX_CLEAN_TOTALS["tokens_out"])
    (attempt,) = audit(codex_file, adapter_name="codex").model.attempts
    assert (attempt.tokens_in, attempt.tokens_out) == raw

    codex_loop = write_codex_session(tmp_path / "c2", codex_loop_records(), CODEX_LOOP_FILE)
    raw = _raw_codex_totals(codex_loop)
    assert raw == (CODEX_LOOP_TOTALS["tokens_in"], CODEX_LOOP_TOTALS["tokens_out"])


def test_trace_audits_no_longer_fire_cost_003(tmp_path: Path):
    # The wart, fixed: trace imports record no outcomes, so COST-003
    # (wasted spend over recorded outcomes) stays silent; TRAJ-001,
    # which needs no outcomes, still fires on the planted loop.
    loop_path = write_codex_session(tmp_path, codex_loop_records(), CODEX_LOOP_FILE)
    result = audit(loop_path, adapter_name="codex")
    assert "TRAJ-001" in _finding_ids(result)
    assert "COST-003" not in _finding_ids(result)

    clean_dir = tmp_path / "clean"
    write_claude_project(clean_dir, claude_clean_records(), CLAUDE_CLEAN_SESSION)
    result = audit(clean_dir, adapter_name="claude-code")
    assert "COST-003" not in _finding_ids(result)


def test_adapters_stamp_started_at_and_source(tmp_path: Path):
    path = write_claude_project(
        tmp_path / "p", claude_clean_records(), CLAUDE_CLEAN_SESSION)
    model = audit(path, adapter_name="claude-code").model
    task = next(t for t in model.tasks if t.id == CLAUDE_CLEAN_SESSION)
    assert task.metadata["started_at"].startswith("2026-10-01T09:00:00")
    assert task.metadata["source_file"] == f"{CLAUDE_CLEAN_SESSION}.jsonl"


def test_canonical_round_trip_preserves_span_tokens(tmp_path: Path):
    from evalwarden.model import IntegrityModel

    path = write_claude_project(
        tmp_path / "p", claude_clean_records(), CLAUDE_CLEAN_SESSION)
    model = audit(path, adapter_name="claude-code").model
    restored = IntegrityModel.from_canonical_json(model.to_canonical_json())
    (attempt,) = [a for a in restored.attempts if a.task_id == CLAUDE_CLEAN_SESSION]
    assert [(s.tokens_in, s.tokens_out) for s in attempt.spans] == [
        (100, 20), (200, 30), (300, 40), (150, 25)]


def test_segmentation_through_the_engine(tmp_path: Path):
    codex_file = write_codex_session(tmp_path, codex_loop_records(), CODEX_LOOP_FILE)
    result = audit(codex_file)  # autodetected
    (attempt,) = result.model.attempts
    segments = segment_phases(attempt.spans)
    assert [(s.label, s.start, s.end) for s in segments] == CODEX_LOOP_PHASES
