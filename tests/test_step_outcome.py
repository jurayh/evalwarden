"""Step-outcome field: TrajectoryStep.outcome population and contract.

The field records only whether the harness reported a single tool call
as completed ("ok") or errored ("error"); None when the source records
no per-call outcome. It never says whether the work succeeded, and no
adapter derives it from output text.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from evalwarden.adapters import AuditError
from evalwarden.adapters.inspect_ai import InspectAdapter, _normalize_spans
from evalwarden.adapters.promptfoo import PromptfooAdapter
from evalwarden.canonical import CanonicalValidationError, model_from_jsonl
from evalwarden.engine import audit
from evalwarden.model import IntegrityModel, TrajectoryStep
from evalwarden.trace_phases import segment_phases

from .conftest import DEMO_AGENT_LOOP, DEMO_AGENT_ORDINARY, DEMO_PROMPTFOO_CLEAN
from .trace_fixtures import (
    CLAUDE_CLEAN_SESSION,
    CLAUDE_LOOP_SESSION,
    CODEX_CLEAN_FILE,
    claude_clean_records,
    claude_loop_records,
    codex_clean_records,
    write_claude_project,
    write_codex_session,
)


def _document_with_spans(spans: list[dict]) -> dict:
    return {
        "schema": "evalwarden.model",
        "schema_version": "1.0",
        "eval_id": "outcome-test",
        "tasks": [{"id": "t1", "prompt": "prompt"}],
        "attempts": [{"task_id": "t1", "status": "pass", "spans": spans}],
    }


# ------------------------------------------------------------- canonical


def test_canonical_round_trip_preserves_span_outcomes():
    model = IntegrityModel.from_canonical_document(
        _document_with_spans(
            [
                {"step_id": "s1", "tool": "read", "outcome": "ok"},
                {"step_id": "s2", "tool": "bash", "outcome": "error"},
                {"step_id": "s3", "tool": "read"},
            ]
        )
    )
    assert [s.outcome for s in model.attempts[0].spans] == ["ok", "error", None]

    document = model.to_canonical_document()
    span_dicts = document["attempts"][0]["spans"]
    assert span_dicts[0]["outcome"] == "ok"
    assert span_dicts[1]["outcome"] == "error"
    # Conditional emission: outcome-less spans carry no outcome key.
    assert "outcome" not in span_dicts[2]

    restored = IntegrityModel.from_canonical_document(document)
    assert [s.outcome for s in restored.attempts[0].spans] == ["ok", "error", None]


def test_canonical_export_without_outcomes_has_no_outcome_keys():
    model = IntegrityModel.from_canonical_document(
        _document_with_spans([{"step_id": "s1", "tool": "read"}])
    )
    document = model.to_canonical_document()
    (span_dict,) = document["attempts"][0]["spans"]
    assert list(span_dict) == ["step_id", "tool", "args", "output", "consumes"]
    assert '"outcome"' not in model.to_canonical_json()


def test_canonical_rejects_unknown_span_outcome_with_field_path():
    with pytest.raises(
        CanonicalValidationError, match=r"\$\.attempts\[0\]\.spans\[0\]\.outcome"
    ):
        IntegrityModel.from_canonical_document(
            _document_with_spans(
                [{"step_id": "s1", "tool": "bash", "outcome": "test_failed"}]
            )
        )


def test_jsonl_span_records_carry_outcome():
    imported = model_from_jsonl(
        "\n".join(
            [
                '{"type":"eval","schema":"evalwarden.model","schema_version":"1.0","eval_id":"j"}',
                '{"type":"task","id":"t1","prompt":"prompt"}',
                '{"type":"span","task_id":"t1","step_id":"s1","tool":"bash","outcome":"error"}',
                '{"type":"span","task_id":"t1","step_id":"s2","tool":"read"}',
                '{"type":"attempt","task_id":"t1","status":"pass"}',
            ]
        )
    )
    assert [s.outcome for s in imported.attempts[0].spans] == ["error", None]


# ------------------------------------------------------- Inspect JSON


def test_inspect_json_spans_carry_outcome_and_validate_it():
    spans = _normalize_spans(
        [
            {"task_id": "t1", "step_id": "s1", "tool": "bash", "outcome": "error"},
            {"task_id": "t1", "step_id": "s2", "tool": "read"},
        ],
        [],
    )
    assert [s.outcome for s in spans["t1"]] == ["error", None]
    with pytest.raises(AuditError, match="'outcome' must be 'ok' or 'error'"):
        _normalize_spans(
            [{"task_id": "t1", "tool": "bash", "outcome": "passed"}], []
        )


def test_inspect_demo_trajectories_populate_outcomes():
    model = InspectAdapter().normalize(InspectAdapter().collect(DEMO_AGENT_LOOP))
    spans = {s.step_id: s for a in model.attempts if a.task_id == "t1"
             for s in a.spans}
    # The repeated pytest runs completed; their failures live in output
    # text, so the outcomes are "ok" -- the semantics rule, exercised end
    # to end through the demo artifact.
    assert spans["s3"].outcome == "ok"
    assert spans["s3"].output == "1 failed, 2 passed"
    assert spans["s0"].outcome is None  # key absent in the artifact

    ordinary = InspectAdapter().normalize(
        InspectAdapter().collect(DEMO_AGENT_ORDINARY)
    )
    outcomes = [s.outcome for a in ordinary.attempts for s in a.spans]
    assert outcomes.count("ok") == 3
    assert all(o in ("ok", None) for o in outcomes)


# ----------------------------------------------------------- Claude Code


def test_claude_outcomes_come_only_from_the_is_error_flag(tmp_path: Path):
    path = write_claude_project(
        tmp_path / "p", claude_clean_records(), CLAUDE_CLEAN_SESSION)
    model = audit(path, adapter_name="claude-code").model
    (attempt,) = [a for a in model.attempts if a.task_id == CLAUDE_CLEAN_SESSION]
    # Fixture: toolu_01 and toolu_03 carry is_error False; toolu_02 and
    # toolu_04 carry no flag at all.
    assert [s.outcome for s in attempt.spans] == ["ok", None, "ok", None]


def test_claude_error_flag_populates_error(tmp_path: Path):
    records = claude_clean_records()
    for record in records:
        content = (record.get("message") or {}).get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("tool_use_id") == "toolu_02":
                    block["is_error"] = True
    path = write_claude_project(tmp_path / "p", records, CLAUDE_CLEAN_SESSION)
    model = audit(path, adapter_name="claude-code").model
    (attempt,) = [a for a in model.attempts if a.task_id == CLAUDE_CLEAN_SESSION]
    assert attempt.spans[1].outcome == "error"


def test_claude_completed_calls_with_failing_text_are_ok(tmp_path: Path):
    # The loop fixture's five Bash results return "1 failing test" text
    # with is_error False: the calls completed, so outcome is "ok".
    path = write_claude_project(
        tmp_path / "p", claude_loop_records(), CLAUDE_LOOP_SESSION)
    model = audit(path, adapter_name="claude-code").model
    (attempt,) = [a for a in model.attempts if a.task_id == CLAUDE_LOOP_SESSION]
    bash_spans = [s for s in attempt.spans if s.tool == "Bash"]
    assert len(bash_spans) == 5
    assert all(s.outcome == "ok" for s in bash_spans)
    assert all(s.output == "1 failing test: test_checkout" for s in bash_spans)


# ------------------------------------------------------ Codex / Promptfoo


def test_codex_spans_stay_silent_on_outcome(tmp_path: Path):
    path = write_codex_session(tmp_path, codex_clean_records(), CODEX_CLEAN_FILE)
    model = audit(path).model
    (attempt,) = model.attempts
    assert attempt.spans  # spans exist; only the outcome is absent
    assert all(s.outcome is None for s in attempt.spans)


def test_promptfoo_builds_no_spans():
    model = PromptfooAdapter().normalize(
        PromptfooAdapter().collect(DEMO_PROMPTFOO_CLEAN)
    )
    assert model.attempts
    assert all(a.spans == [] for a in model.attempts)


# ------------------------------------------------- TRAJ-003 expressibility


def test_stalled_oscillation_fixture_is_expressible():
    """The fixture whose absence killed the TRAJ-003 spike, now writable.

    Repeated edit->test cycles with changing arguments throughout, no
    intervening explore/plan/review phase, ending without a successful
    test. Phase segmentation yields the alternation and the test-phase
    spans carry outcome "error" from a structured source (here, the
    fixture's own construction). Building the check is a separate spike.
    """
    spans: list[TrajectoryStep] = []
    for cycle in range(3):
        for part in range(2):
            spans.append(TrajectoryStep(
                step_id=f"e{cycle}{part}", tool="Edit",
                args={"file_path": f"src/mod{cycle}_{part}.py",
                      "old_string": f"v{cycle}", "new_string": f"v{cycle + 1}"},
                outcome="ok",
            ))
        for part in range(2):
            spans.append(TrajectoryStep(
                step_id=f"t{cycle}{part}", tool="Bash",
                args={"command": f"pytest -q tests/test_mod{cycle}_{part}.py"},
                output=f"runner crashed on cycle {cycle}",
                outcome="error",
            ))
    segments = segment_phases(spans)
    # Phases are multi-call runs (single-step runs are absorbed by the
    # segmenter's smoothing, as in real trajectories).
    assert [seg.label for seg in segments] == ["edit", "test"] * 3
    test_spans = [s for s in spans if s.step_id.startswith("t")]
    assert all(s.outcome == "error" for s in test_spans)
    # Arguments change every cycle: nothing here is an exact loop.
    assert len({json.dumps(s.args, sort_keys=True) for s in spans}) == len(spans)
