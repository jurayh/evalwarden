"""Codex adapter: rollout JSONL -> spans, cumulative-token deltas, filename ids.

Fixtures are faithful reconstructions of the documented format (see
tests/trace_fixtures.py); expected totals there are hand-computed from
the cumulative snapshots the fixtures are built with.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from evalwarden.adapters import AuditError
from evalwarden.adapters.codex import CodexAdapter
from evalwarden.model import Confidence

from .trace_fixtures import (
    CODEX_CLEAN_FILE,
    CODEX_CLEAN_SESSION,
    CODEX_CLEAN_TOTALS,
    CODEX_LINEAGE_ID,
    CODEX_LOOP_FILE,
    CODEX_LOOP_SESSION,
    CODEX_LOOP_TOTALS,
    codex_clean_records,
    codex_loop_records,
    write_codex_session,
)

adapter = CodexAdapter()


def _normalize(path: Path):
    return adapter.normalize(adapter.collect(path))


def test_clean_rollout_reconciliation_and_filename_session_id(tmp_path: Path):
    path = write_codex_session(tmp_path, codex_clean_records(), CODEX_CLEAN_FILE)
    model = _normalize(path)
    (attempt,) = model.attempts
    # The filename UUID is the session id; session_meta's lineage id is not.
    assert attempt.task_id == CODEX_CLEAN_SESSION
    assert attempt.task_id != CODEX_LINEAGE_ID
    assert (attempt.tokens_in, attempt.tokens_out) == (
        CODEX_CLEAN_TOTALS["tokens_in"], CODEX_CLEAN_TOTALS["tokens_out"])
    assert attempt.model_id == "gpt-5-codex"
    assert {t.id: t.prompt for t in model.tasks}[CODEX_CLEAN_SESSION] == \
        "Add a --verbose flag to the CLI and test it."


def test_clean_spans_come_from_response_item_only(tmp_path: Path):
    path = write_codex_session(tmp_path, codex_clean_records(), CODEX_CLEAN_FILE)
    model = _normalize(path)
    (attempt,) = model.attempts
    # 4 calls in response_item; the event_msg duplicates add none.
    assert [s.tool for s in attempt.spans] == ["shell", "shell", "apply_patch", "shell"]
    assert [s.step_id for s in attempt.spans] == ["call_1", "call_2", "call_3", "call_4"]
    assert attempt.spans[0].args == {"command": ["ls", "src"]}  # arguments JSON string parsed
    assert attempt.spans[2].args["input"].startswith("*** Begin Patch")
    assert "3 passed" in (attempt.spans[3].output or "")
    assert all(s.consumes == [] for s in attempt.spans)  # no edges recorded, none invented
    assert attempt.status == "incomplete"


def test_clean_span_token_attribution_from_cumulative_deltas(tmp_path: Path):
    path = write_codex_session(tmp_path, codex_clean_records(), CODEX_CLEAN_FILE)
    model = _normalize(path)
    (attempt,) = model.attempts
    by_id = {s.step_id: s for s in attempt.spans}
    # First snapshot covers call_1 + call_2 together: a shared delta is
    # never split, so both stay unpriced.
    assert by_id["call_1"].tokens_in is None and by_id["call_2"].tokens_in is None
    # One call per later snapshot window: exact deltas (1500, 160), (1700, 240).
    assert (by_id["call_3"].tokens_in, by_id["call_3"].tokens_out) == (1500, 160)
    assert (by_id["call_4"].tokens_in, by_id["call_4"].tokens_out) == (1700, 240)
    notes = "\n".join(model.unsupported)
    assert "lineage root" in notes
    assert "cumulative" in notes and "counted once" in notes  # verbatim repeat snapshot
    assert "cached_input=1500" in notes and "reasoning_output=150" in notes
    assert "event_msg record(s) ignored as duplicates" in notes


def test_loop_rollout_in_nested_sessions_tree(tmp_path: Path):
    root = tmp_path / "sessions" / "2026" / "10" / "02"
    write_codex_session(root, codex_loop_records(), CODEX_LOOP_FILE)
    model = _normalize(tmp_path / "sessions")
    (attempt,) = model.attempts
    assert attempt.task_id == CODEX_LOOP_SESSION
    assert (attempt.tokens_in, attempt.tokens_out) == (
        CODEX_LOOP_TOTALS["tokens_in"], CODEX_LOOP_TOTALS["tokens_out"])
    assert len(attempt.spans) == CODEX_LOOP_TOTALS["spans"]
    pytest_spans = [s for s in attempt.spans
                    if s.args.get("command", [None])[0] == "pytest"]
    assert len(pytest_spans) == 4
    # Each pytest ran as its own turn: every one carries its (600, 150) delta.
    assert all((s.tokens_in, s.tokens_out) == (600, 150) for s in pytest_spans)


def test_newer_payload_types_definitions_skipped_calls_become_spans(tmp_path: Path):
    # Shapes from real rollouts (AletheiaResearch corpus pass, Oct 2026).
    records = codex_clean_records()
    records[3:3] = [
        {"timestamp": "2026-10-01T09:00:03.000Z", "type": "response_item",
         "payload": {"type": "tool_schema", "name": "apply_patch",
                     "schema": {"type": "object"}, "source": "teich"}},
        {"timestamp": "2026-10-01T09:00:04.000Z", "type": "response_item",
         "payload": {"type": "tool_search_call", "call_id": "call_ts",
                     "status": "completed", "execution": "client",
                     "arguments": {"query": "api key helper", "limit": 5}}},
        {"timestamp": "2026-10-01T09:00:05.000Z", "type": "response_item",
         "payload": {"type": "tool_search_output", "call_id": "call_ts",
                     "status": "completed",
                     "tools": [{"type": "namespace", "name": "mcp__github"}]}},
        {"timestamp": "2026-10-01T09:00:06.000Z", "type": "response_item",
         "payload": {"type": "web_search_call", "status": "completed",
                     "action": {"type": "search", "query": "embeddings api",
                                "queries": ["embeddings api"]}}},
        {"timestamp": "2026-10-01T09:00:07.000Z", "type": "response_item",
         "payload": {"type": "image_generation_call", "id": "ig_1",
                     "status": "generating", "revised_prompt": "a hero image"}},
    ]
    path = write_codex_session(tmp_path, records, CODEX_CLEAN_FILE)
    model = _normalize(path)
    (attempt,) = model.attempts
    by_tool = {s.tool: s for s in attempt.spans}
    # Token totals are untouched by the new record shapes.
    assert (attempt.tokens_in, attempt.tokens_out) == (
        CODEX_CLEAN_TOTALS["tokens_in"], CODEX_CLEAN_TOTALS["tokens_out"])
    # The definition adds no span: only the fixture's real patch call
    # is named apply_patch, and exactly 3 spans were added (the 3 calls).
    assert len([s for s in attempt.spans if s.tool == "apply_patch"]) == 1
    assert len(attempt.spans) == CODEX_CLEAN_TOTALS["spans"] + 3
    assert by_tool["tool_search"].args == {"query": "api key helper", "limit": 5}
    assert "mcp__github" in (by_tool["tool_search"].output or "")
    assert by_tool["web_search"].args["query"] == "embeddings api"
    assert by_tool["web_search"].output is None  # none recorded in this stream
    assert by_tool["image_generation"].args == {"revised_prompt": "a hero image"}
    assert by_tool["image_generation"].step_id == "ig_1"
    assert any("tool_schema record(s) skipped" in u for u in model.unsupported)
    assert any("carry no output" in u for u in model.unsupported)


def test_unknown_record_type_fails_loudly(tmp_path: Path):
    records = codex_clean_records()
    records.append({"timestamp": "2026-10-01T09:02:00.000Z", "type": "future_stream",
                    "payload": {}})
    path = write_codex_session(tmp_path, records, CODEX_CLEAN_FILE)
    with pytest.raises(AuditError, match="unknown Codex record type 'future_stream'"):
        _normalize(path)


def test_unknown_response_item_payload_fails_loudly(tmp_path: Path):
    records = codex_clean_records()
    records.append({"timestamp": "2026-10-01T09:02:00.000Z", "type": "response_item",
                    "payload": {"type": "hologram_call", "call_id": "call_9"}})
    path = write_codex_session(tmp_path, records, CODEX_CLEAN_FILE)
    with pytest.raises(AuditError, match="unknown response_item payload type 'hologram_call'"):
        _normalize(path)


def test_unknown_event_msg_type_is_noted_not_fatal(tmp_path: Path):
    records = codex_clean_records()
    records.append({"timestamp": "2026-10-01T09:02:00.000Z", "type": "event_msg",
                    "payload": {"type": "confetti", "message": "yay"}})
    path = write_codex_session(tmp_path, records, CODEX_CLEAN_FILE)
    model = _normalize(path)
    assert any("unrecognized type 'confetti'" in u for u in model.unsupported)


def test_decreasing_cumulative_totals_fail_loudly(tmp_path: Path):
    records = codex_clean_records()
    records.append({"timestamp": "2026-10-01T09:03:00.000Z", "type": "event_msg",
                    "payload": {"type": "token_count", "info": {"total_token_usage": {
                        "input_tokens": 10, "cached_input_tokens": 0,
                        "output_tokens": 5, "reasoning_output_tokens": 0,
                        "total_tokens": 15}}}})
    path = write_codex_session(tmp_path, records, CODEX_CLEAN_FILE)
    with pytest.raises(AuditError, match="cumulative token totals decreased"):
        _normalize(path)


def test_filename_without_uuid_fails_loudly(tmp_path: Path):
    path = write_codex_session(tmp_path, codex_clean_records(), "rollout-plain.jsonl")
    with pytest.raises(AuditError, match="cannot determine the Codex session id"):
        adapter.collect(path)


def test_detect(tmp_path: Path):
    path = write_codex_session(tmp_path, codex_clean_records(), CODEX_CLEAN_FILE)
    assert adapter.detect(path) == Confidence.HIGH
    assert adapter.detect(tmp_path) == Confidence.HIGH
    other = tmp_path / "plain.jsonl"
    other.write_text(json.dumps({"hello": "world"}) + "\n", encoding="utf-8")
    assert adapter.detect(other) == Confidence.LOW
