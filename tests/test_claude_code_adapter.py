"""Claude Code adapter: session JSONL -> spans, usage, sub-agent sidechains.

Fixtures are faithful reconstructions of the documented format (see
tests/trace_fixtures.py); expected totals there are hand-computed from
the usage constants the fixtures are built with.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from evalwarden.adapters import AuditError
from evalwarden.adapters.claude_code import ClaudeCodeAdapter
from evalwarden.model import Confidence

from .trace_fixtures import (
    CLAUDE_CLEAN_SESSION,
    CLAUDE_CLEAN_TOTALS,
    CLAUDE_LOOP_SESSION,
    CLAUDE_LOOP_TOTALS,
    CLAUDE_SUBAGENT_TOTALS,
    claude_clean_records,
    claude_loop_records,
    claude_subagent_records,
    write_claude_project,
)

adapter = ClaudeCodeAdapter()


def _normalize(path: Path):
    return adapter.normalize(adapter.collect(path))


def _clean_project(tmp_path: Path) -> Path:
    write_claude_project(tmp_path, claude_clean_records(), CLAUDE_CLEAN_SESSION,
                         subagent_records=claude_subagent_records())
    return tmp_path


def test_clean_project_sessions_and_subagent(tmp_path: Path):
    model = _normalize(_clean_project(tmp_path))
    assert model.adapter_name == "claude-code"
    assert len(model.attempts) == 2  # main session + sub-agent sidechain
    main, sub = model.attempts
    assert main.task_id == CLAUDE_CLEAN_SESSION
    assert sub.task_id == f"{CLAUDE_CLEAN_SESSION}:subagent:a1b2"
    # Prompts come from the first user records, per file.
    prompts = {t.id: t.prompt for t in model.tasks}
    assert "fix the typo in README" in prompts[CLAUDE_CLEAN_SESSION]
    assert prompts[sub.task_id] == "Scan the docs for TODO markers."


def test_clean_token_reconciliation(tmp_path: Path):
    main, sub = _normalize(_clean_project(tmp_path)).attempts
    assert (main.tokens_in, main.tokens_out) == (
        CLAUDE_CLEAN_TOTALS["tokens_in"], CLAUDE_CLEAN_TOTALS["tokens_out"])
    assert (sub.tokens_in, sub.tokens_out) == (
        CLAUDE_SUBAGENT_TOTALS["tokens_in"], CLAUDE_SUBAGENT_TOTALS["tokens_out"])


def test_clean_spans_tools_args_outputs_and_step_tokens(tmp_path: Path):
    main, _ = _normalize(_clean_project(tmp_path)).attempts
    assert [s.tool for s in main.spans] == ["Read", "Grep", "Edit", "Bash"]
    assert [s.step_id for s in main.spans] == ["toolu_01", "toolu_02", "toolu_03", "toolu_04"]
    assert main.spans[0].args == {"file_path": "README.md"}
    assert main.spans[2].args["new_string"] == "the"
    assert "teh typo" in (main.spans[0].output or "")
    assert main.tool_calls == 4
    assert main.model_id == "claude-sonnet-4-5"
    assert main.latency_s == 120.0  # 09:00:00 -> 09:02:00
    # One tool call per assistant record: record usage lands on the span.
    assert [(s.tokens_in, s.tokens_out) for s in main.spans] == [
        (100, 20), (200, 30), (300, 40), (150, 25)]
    # No consumption edges exist in this format, and none are invented.
    assert all(s.consumes == [] for s in main.spans)
    assert main.status == "incomplete"  # no outcome recorded, none fabricated


def test_cache_totals_reported_not_folded_into_tokens(tmp_path: Path):
    model = _normalize(_clean_project(tmp_path))
    note = next(u for u in model.unsupported if "cache tokens" in u)
    assert "cache_creation=30" in note and "cache_read=580" in note


def test_loop_session_spans_and_totals(tmp_path: Path):
    path = write_claude_project(tmp_path, claude_loop_records(), CLAUDE_LOOP_SESSION)
    model = _normalize(path)  # a single session file as input
    (attempt,) = model.attempts
    assert (attempt.tokens_in, attempt.tokens_out) == (
        CLAUDE_LOOP_TOTALS["tokens_in"], CLAUDE_LOOP_TOTALS["tokens_out"])
    assert len(attempt.spans) == CLAUDE_LOOP_TOTALS["spans"]
    bash = [s for s in attempt.spans if s.tool == "Bash"]
    assert len(bash) == 5
    assert all(s.args == {"command": "npm test", "description": "Run tests"} for s in bash)
    assert all((s.tokens_in, s.tokens_out) == (200, 30) for s in bash)


def test_bookkeeping_record_types_are_counted_and_skipped(tmp_path: Path):
    # Real session files (trace-commons corpus, Oct 2026) interleave
    # bookkeeping records that carry no messages, tool calls, or usage.
    records = claude_clean_records()
    for rtype in ("mode", "file-history-snapshot", "ai-title",
                  "permission-mode", "pr-link"):
        records.insert(0, {"type": rtype, "sessionId": CLAUDE_CLEAN_SESSION,
                           "uuid": f"bk-{rtype}",
                           "timestamp": "2026-10-01T09:00:00.000Z"})
    path = write_claude_project(tmp_path, records, CLAUDE_CLEAN_SESSION)
    model = _normalize(path)
    (attempt,) = [a for a in model.attempts if a.task_id == CLAUDE_CLEAN_SESSION]
    assert (attempt.tokens_in, attempt.tokens_out) == (
        CLAUDE_CLEAN_TOTALS["tokens_in"], CLAUDE_CLEAN_TOTALS["tokens_out"])
    assert len(attempt.spans) == CLAUDE_CLEAN_TOTALS["spans"]
    assert any("bookkeeping record(s) skipped" in note
               and "file-history-snapshot x1" in note
               for note in model.unsupported)


def test_unknown_record_type_fails_loudly(tmp_path: Path):
    records = claude_clean_records()
    records.append({"type": "future-record", "sessionId": CLAUDE_CLEAN_SESSION,
                    "uuid": "x1", "timestamp": "2026-10-01T09:03:00.000Z"})
    path = write_claude_project(tmp_path, records, CLAUDE_CLEAN_SESSION)
    with pytest.raises(AuditError, match="unknown Claude Code record type 'future-record'"):
        _normalize(path)


def test_malformed_jsonl_fails_with_line(tmp_path: Path):
    path = write_claude_project(tmp_path, claude_clean_records(), CLAUDE_CLEAN_SESSION)
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"type": "user",\n')
    with pytest.raises(AuditError, match="invalid JSON"):
        adapter.collect(path)


def test_multi_tool_record_shares_usage_and_is_not_split(tmp_path: Path):
    records = [
        {"parentUuid": None, "isSidechain": False, "sessionId": "sess-x",
         "type": "user", "uuid": "u1", "timestamp": "2026-10-01T09:00:00.000Z",
         "message": {"role": "user", "content": "do two things"}},
        {"parentUuid": "u1", "isSidechain": False, "sessionId": "sess-x",
         "type": "assistant", "uuid": "a1", "timestamp": "2026-10-01T09:00:10.000Z",
         "message": {"model": "claude-sonnet-4-5", "role": "assistant",
                     "content": [
                         {"type": "tool_use", "id": "t1", "name": "Read",
                          "input": {"file_path": "a"}},
                         {"type": "tool_use", "id": "t2", "name": "Read",
                          "input": {"file_path": "b"}},
                     ],
                     "usage": {"input_tokens": 500, "output_tokens": 50,
                               "cache_creation_input_tokens": 0,
                               "cache_read_input_tokens": 0}}},
    ]
    path = write_claude_project(tmp_path, records, "sess-x")
    model = _normalize(path)
    (attempt,) = model.attempts
    assert (attempt.tokens_in, attempt.tokens_out) == (500, 50)  # totals still reconcile
    assert [s.tokens_in for s in attempt.spans] == [None, None]  # never split
    assert any("several tool calls under one usage total" in u for u in model.unsupported)


def test_sidechain_records_in_main_file_are_counted_not_merged(tmp_path: Path):
    records = claude_clean_records() + [
        {"parentUuid": None, "isSidechain": True, "sessionId": CLAUDE_CLEAN_SESSION,
         "type": "user", "uuid": "su", "timestamp": "2026-10-01T09:05:00.000Z",
         "message": {"role": "user", "content": "sidechain prompt"}},
    ]
    path = write_claude_project(tmp_path, records, CLAUDE_CLEAN_SESSION)
    model = _normalize(path)
    (attempt,) = model.attempts
    assert len(attempt.spans) == CLAUDE_CLEAN_TOTALS["spans"]
    assert any("sidechain record(s) skipped" in u for u in model.unsupported)


def test_detect(tmp_path: Path):
    project = _clean_project(tmp_path)
    session_file = project / f"{CLAUDE_CLEAN_SESSION}.jsonl"
    assert adapter.detect(session_file) == Confidence.HIGH
    assert adapter.detect(project) == Confidence.HIGH
    other = tmp_path / "plain.jsonl"
    other.write_text(json.dumps({"hello": "world"}) + "\n", encoding="utf-8")
    assert adapter.detect(other) == Confidence.LOW
