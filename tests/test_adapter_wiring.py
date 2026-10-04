"""Adapter wiring tests: the Inspect adapter feeds the TRAJ, NOISE, and
panel-coverage lanes from recorded artifact data -- and only from recorded
data. Fields the artifact does not carry stay empty, so the checks that
need them stay silent instead of fed on invented values.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from evalwarden.adapters import AuditError
from evalwarden.adapters.inspect_ai import MAX_SPAN_OUTPUT_CHARS, InspectAdapter
from evalwarden.engine import audit

from .conftest import DEMO_AGENT_LOOP, DEMO_AGENT_ORDINARY, DEMO_JUDGE_PANEL

adapter = InspectAdapter()


def _write(path: Path, name: str, text: str) -> None:
    (path / name).write_text(text, encoding="utf-8")


def _artifact(tmp_path: Path, *, dataset: dict, run: dict | None = None,
              judge_run: dict | None = None, trajectories: str | None = None) -> Path:
    _write(tmp_path, "dataset.json", json.dumps(dataset))
    _write(tmp_path, "grader.json", json.dumps({"kind": "script"}))
    if run is not None:
        _write(tmp_path, "run.json", json.dumps(run))
    if judge_run is not None:
        _write(tmp_path, "judge_run.json", json.dumps(judge_run))
    if trajectories is not None:
        _write(tmp_path, "trajectories.jsonl", trajectories)
    return tmp_path


def _normalize(path: Path):
    return adapter.normalize(adapter.collect(path))


# ------------------------------------------------------------- spans (TRAJ)


def test_spans_attach_with_step_ids_and_consumes(tmp_path: Path):
    path = _artifact(
        tmp_path,
        dataset={"tasks": [{"id": "t1", "prompt": "do it"}]},
        run={"attempts": [{"task_id": "t1", "status": "pass", "tool_calls": 2}]},
        trajectories=(
            '{"task_id": "t1", "step_id": "a", "tool": "read_file", "args": {"path": "x"}, "consumes": []}\n'
            '{"task_id": "t1", "tool": "edit_file", "args": {"path": "x"}, "consumes": ["a"]}\n'
        ),
    )
    model = _normalize(path)
    spans = model.attempts[0].spans
    assert [s.step_id for s in spans] == ["a", "s1"]  # missing id synthesized in order
    assert spans[1].consumes == ["a"]  # recorded data flow, carried verbatim
    assert model.digests["trajectories.jsonl"]


def test_span_output_truncated_at_boundary(tmp_path: Path):
    long_output = "x" * (MAX_SPAN_OUTPUT_CHARS + 100)
    path = _artifact(
        tmp_path,
        dataset={"tasks": [{"id": "t1", "prompt": "do it"}]},
        run={"attempts": [{"task_id": "t1", "status": "pass"}]},
        trajectories=json.dumps(
            {"task_id": "t1", "tool": "bash", "args": {}, "output": long_output}
        ),
    )
    model = _normalize(path)
    assert len(model.attempts[0].spans[0].output) == MAX_SPAN_OUTPUT_CHARS
    # The truncation is reported, not silent.
    assert any("truncated" in u for u in model.unsupported)


def test_spans_without_recorded_attempt_reported(tmp_path: Path):
    path = _artifact(
        tmp_path,
        dataset={"tasks": [{"id": "t1", "prompt": "do it"}]},
        run={"attempts": [{"task_id": "t1", "status": "pass"}]},
        trajectories='{"task_id": "ghost", "tool": "bash", "args": {}}\n',
    )
    model = _normalize(path)
    assert model.attempts[0].spans == []
    assert any("ghost" in u and "no recorded attempt" in u for u in model.unsupported)


def test_trajectories_fail_clearly(tmp_path: Path):
    # malformed JSON line
    d = tmp_path / "bad-json"
    d.mkdir()
    path = _artifact(d, dataset={"tasks": []}, trajectories='{"task_id": "t1",\n')
    with pytest.raises(AuditError, match="line 1"):
        adapter.collect(path)
    # record missing the required tool
    d2 = tmp_path / "no-tool"
    d2.mkdir()
    path = _artifact(d2, dataset={"tasks": []}, trajectories='{"task_id": "t1"}\n')
    with pytest.raises(AuditError, match="'task_id' and 'tool'"):
        _normalize(path)


def test_spans_absent_means_no_spans(tmp_path: Path):
    path = _artifact(
        tmp_path,
        dataset={"tasks": [{"id": "t1", "prompt": "do it"}]},
        run={"attempts": [{"task_id": "t1", "status": "pass"}]},
    )
    model = _normalize(path)
    assert model.attempts[0].spans == []


# ------------------------------------------------- judge identity (JUDGE-007/008/009)


def test_judge_id_and_confidence_populated(tmp_path: Path):
    judge_run = {
        "judge": {"model": "panel"},
        "judgments": [
            {"task_id": "t1", "candidates": ["sol-a", "sol-b"], "winner": "sol-a",
             "judge_id": "judge-a", "confidence": 0.8},
            {"task_id": "t1", "candidates": ["sol-a", "sol-b"], "winner": "sol-b",
             "judge_id": "judge-b", "confidence": 1.4},  # out of range: absent, not clamped
            {"task_id": "t2", "candidates": ["sol-a", "sol-b"], "winner": "sol-a"},
        ],
    }
    path = _artifact(
        tmp_path,
        dataset={"tasks": [{"id": "t1", "prompt": "q"}, {"id": "t2", "prompt": "q2"}]},
        judge_run=judge_run,
    )
    model = _normalize(path)
    assert [j.judge_id for j in model.judgments] == ["judge-a", "judge-b", None]
    assert [j.confidence for j in model.judgments] == [0.8, None, None]


# -------------------------------------------------- failure modes (DATA-003 / JUDGE-009)


def test_failure_mode_from_field_metadata_and_taxonomy(tmp_path: Path):
    dataset = {
        "failure_mode_taxonomy": ["verbosity-gaming", "sycophancy"],
        "tasks": [
            {"id": "t1", "prompt": "a", "failure_mode": "verbosity-gaming"},
            {"id": "t2", "prompt": "b", "metadata": {"failure_mode": "sycophancy"}},
            {"id": "t3", "prompt": "c", "metadata": {"kind": "misc"}},
        ],
    }
    path = _artifact(tmp_path, dataset=dataset)
    model = _normalize(path)
    assert [t.failure_mode for t in model.tasks] == [
        "verbosity-gaming", "sycophancy", None,  # untagged stays untagged
    ]
    assert model.failure_mode_taxonomy == ["verbosity-gaming", "sycophancy"]
    # The taxonomy key is understood, not reported as an unsupported field.
    assert not any("failure_mode_taxonomy" in u for u in model.unsupported)


# ------------------------------------------------------- run scores (NOISE)


def test_run_scores_and_claimed_delta_populated(tmp_path: Path):
    run = {
        "attempts": [],
        "run_scores": [
            {"task_id": "t1", "score": 0.9, "generation_index": 0},
            {"task_id": "t1", "score": 0.8, "generation_index": 1, "grade_index": 2},
        ],
        "claimed_delta": 0.03,
    }
    path = _artifact(tmp_path, dataset={"tasks": []}, run=run)
    model = _normalize(path)
    assert [(r.score, r.generation_index, r.grade_index) for r in model.run_scores] == [
        (0.9, 0, 0), (0.8, 1, 2),
    ]
    assert model.claimed_delta == 0.03
    # NOISE evidence cites the file the scores actually live in.
    assert model.artifact_file("run_scores.json") == "run.json"


def test_run_scores_bad_types_fail_clearly(tmp_path: Path):
    path = _artifact(
        tmp_path,
        dataset={"tasks": []},
        run={"run_scores": [{"task_id": "t1", "score": "high"}]},
    )
    with pytest.raises(AuditError, match="must be a number"):
        _normalize(path)


# ------------------------------------------------------- fixture end-to-end


def _finding_ids(path: Path) -> list[str]:
    return [f.id for f in audit(path).findings]


def test_agent_loop_fixture_fires_traj(tmp_path: Path):
    ids = _finding_ids(DEMO_AGENT_LOOP)
    assert "TRAJ-001" in ids  # the planted 5x identical test command
    assert "TRAJ-002" in ids  # recorded flow: most t1 outputs consumed by nothing
    # Nothing recorded for the other lanes: they stay silent.
    assert not {"NOISE-001", "NOISE-002", "JUDGE-009", "DATA-003"} & set(ids)


def test_agent_ordinary_fixture_is_finding_free():
    result = audit(DEMO_AGENT_ORDINARY)
    assert result.findings == []
    # The wiring is live (spans and run scores are read), just quiet:
    # silence here is a verdict on the data, not missing plumbing.
    assert any(a.spans for a in result.model.attempts)
    assert len(result.model.run_scores) == 36


def test_judge_panel_fixture_fires_only_judge_009():
    result = audit(DEMO_JUDGE_PANEL)
    assert [f.id for f in result.findings] == ["JUDGE-009"]
    # Both judges are attributed, failure modes tagged, taxonomy declared.
    assert {j.judge_id for j in result.model.judgments} == {"judge-a", "judge-b"}
    assert all(j.confidence is not None for j in result.model.judgments)
    assert result.model.failure_mode_taxonomy == ["factual-recall", "sycophancy"]


def _hashes(path: Path) -> dict[str, str]:
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(path.iterdir())
        if p.is_file()
    }


@pytest.mark.parametrize("fixture", [DEMO_AGENT_LOOP, DEMO_AGENT_ORDINARY, DEMO_JUDGE_PANEL])
def test_new_fixtures_are_read_only(fixture: Path):
    before = _hashes(fixture)
    audit(fixture)
    assert _hashes(fixture) == before
