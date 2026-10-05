"""Native Inspect AI `.eval` ingest tests.

The two `.eval` fixtures were generated with Inspect AI 0.3.276 (real
`eval()` runs against mockllm; see each fixture's README), and validated
with Inspect's own `read_eval_log` before being committed. They are
native schema version 2 archives: ZIP members `header.json`,
`samples/<id>_epoch_<n>.json`, `summaries.json`.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from evalwarden.adapters import AuditError, autodetect
from evalwarden.adapters.inspect_ai import InspectAdapter
from evalwarden.cli import app
from evalwarden.engine import audit
from evalwarden.model import Confidence

from .conftest import NATIVE_LOOP_LOG, NATIVE_ORDINARY_LOG

adapter = InspectAdapter()
runner = CliRunner()


def _normalize(path: Path):
    return adapter.normalize(adapter.collect(path))


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- detection


def test_detect_eval_file_high():
    assert adapter.detect(NATIVE_LOOP_LOG) == Confidence.HIGH


def test_detect_directory_with_one_eval_high(tmp_path: Path):
    shutil.copy(NATIVE_LOOP_LOG, tmp_path / "log.eval")
    assert adapter.detect(tmp_path) == Confidence.HIGH


def test_detect_other_file_low(tmp_path: Path):
    other = tmp_path / "notes.txt"
    other.write_text("not a log")
    assert adapter.detect(other) == Confidence.LOW


def test_autodetect_eval_file_picks_inspect():
    assert autodetect(NATIVE_LOOP_LOG).name == "inspect"


def test_directory_with_several_eval_logs_fails_clearly(tmp_path: Path):
    shutil.copy(NATIVE_LOOP_LOG, tmp_path / "a.eval")
    shutil.copy(NATIVE_ORDINARY_LOG, tmp_path / "b.eval")
    with pytest.raises(AuditError, match="audit one .eval file at a time"):
        audit(tmp_path)


def test_audit_directory_holding_one_eval(tmp_path: Path):
    shutil.copy(NATIVE_LOOP_LOG, tmp_path / "log.eval")
    result = audit(tmp_path)
    assert result.model.eval_id == "evalwarden-native-loop"
    assert "TRAJ-001" in [f.id for f in result.findings]


# ------------------------------------------------------------- loop fixture


def test_native_loop_model_shape():
    model = _normalize(NATIVE_LOOP_LOG)
    assert model.eval_id == "evalwarden-native-loop"  # the Inspect task name
    assert model.adapter_name == "inspect"
    assert [t.id for t in model.tasks] == ["loop-1", "plain-1", "plain-2"]
    loop_task = model.tasks[0]
    assert loop_task.target == "done"
    # failure_mode comes from the sample's explicit metadata, nowhere else.
    assert loop_task.failure_mode == "tool-loop"
    assert loop_task.metadata["source"] == "inspect-native-fixture"
    assert model.failure_mode_taxonomy == []  # native logs declare none
    assert model.claimed_delta is None
    assert len(model.attempts) == 3
    assert all(a.model_id == "mockllm/model" for a in model.attempts)
    assert all(a.score == 1.0 and a.status == "pass" for a in model.attempts)
    assert model.run_scores == []  # one epoch: nothing repeated to measure
    assert model.judgments == []  # scalar scores are not judgments
    assert model.digests[NATIVE_LOOP_LOG.name] == _hash(NATIVE_LOOP_LOG)
    # Evidence cites the file the data actually lives in.
    assert model.artifact_file("trajectories.jsonl") == NATIVE_LOOP_LOG.name
    assert model.artifact_file("run_scores.json") == NATIVE_LOOP_LOG.name


def test_native_loop_spans_use_native_call_ids():
    model = _normalize(NATIVE_LOOP_LOG)
    loop_attempt = next(a for a in model.attempts if a.task_id == "loop-1")
    lookups = [s for s in loop_attempt.spans if s.tool == "lookup"]
    assert [s.step_id for s in lookups] == [f"loop-call-{i}" for i in range(1, 6)]
    assert all(s.args == {"query": "status"} for s in lookups)
    assert all(s.output == "fact[status]=verified" for s in lookups)
    # No consumption edges exist in a native log, and none are invented.
    assert all(s.consumes == [] for s in loop_attempt.spans)
    assert loop_attempt.tool_calls == len(loop_attempt.spans)


def test_native_span_output_truncated_at_boundary():
    model = _normalize(NATIVE_LOOP_LOG)
    plain2 = next(a for a in model.attempts if a.task_id == "plain-2")
    long_span = next(s for s in plain2.spans if s.tool == "lookup")
    assert len(long_span.output) == 512
    assert any("truncated" in u for u in model.unsupported)


def test_native_loop_audit_fires_traj_001_only():
    result = audit(NATIVE_LOOP_LOG)
    ids = [f.id for f in result.findings]
    assert "TRAJ-001" in ids  # the planted 5x identical lookup
    # TRAJ-002 cannot fire: no data-flow edges are recorded natively.
    assert "TRAJ-002" not in ids
    assert not {"NOISE-001", "NOISE-002"} & set(ids)


# ---------------------------------------------------------- ordinary fixture


def test_native_ordinary_model_is_populated():
    model = _normalize(NATIVE_ORDINARY_LOG)
    assert model.eval_id == "evalwarden-native-ordinary"
    assert len(model.tasks) == 12
    first = model.tasks[0]
    assert first.target == "answer-01"
    assert first.choices == ["answer-01", "wrong-choice"]
    assert len(model.attempts) == 36  # 12 samples x 3 epochs
    assert all(a.spans for a in model.attempts)  # spans on every epoch
    assert all(a.tokens_in is not None and a.tokens_out is not None for a in model.attempts)
    # Repeated epochs of the one scorer become run scores, epoch as the
    # generation index; judge/environment axes are never fabricated.
    assert len(model.run_scores) == 36
    assert {r.generation_index for r in model.run_scores} == {0, 1, 2}
    assert all(r.score == 1.0 for r in model.run_scores)
    assert all(r.grade_index == 0 and r.environment_index == 0 for r in model.run_scores)


def test_native_ordinary_audit_has_no_traj_or_noise_findings():
    result = audit(NATIVE_ORDINARY_LOG)
    ids = {f.id for f in result.findings}
    assert not {"TRAJ-001", "TRAJ-002", "NOISE-001", "NOISE-002"} & ids
    # Silence is a verdict, not missing plumbing (see the model test).
    assert len(result.model.run_scores) == 36
    assert any(a.spans for a in result.model.attempts)


# ---------------------------------------------------------------- read-only


@pytest.mark.parametrize("fixture", [NATIVE_LOOP_LOG, NATIVE_ORDINARY_LOG])
def test_native_audit_is_read_only(fixture: Path):
    before = _hash(fixture)
    audit(fixture)
    assert _hash(fixture) == before


# ------------------------------------------------------- malformed archives


def _write_zip(path: Path, members: dict[str, str]) -> Path:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, text in members.items():
            archive.writestr(name, text)
    return path


def _header(**overrides) -> str:
    header = {
        "version": 2,
        "status": "success",
        "eval": {"task": "t", "model": "m", "dataset": {}, "scorers": []},
        "results": {"total_samples": 0, "scores": []},
    }
    header.update(overrides)
    return json.dumps(header)


def test_not_a_zip_fails_clearly(tmp_path: Path):
    bogus = tmp_path / "bogus.eval"
    bogus.write_bytes(b"this is not a zip archive")
    with pytest.raises(AuditError, match="not a readable ZIP"):
        _normalize(bogus)


def test_missing_header_fails_clearly(tmp_path: Path):
    path = _write_zip(tmp_path / "noheader.eval", {"summaries.json": "[]"})
    with pytest.raises(AuditError, match="missing header.json"):
        _normalize(path)


def test_unsupported_schema_version_fails_clearly(tmp_path: Path):
    path = _write_zip(
        tmp_path / "future.eval", {"header.json": _header(version=99)}
    )
    with pytest.raises(AuditError, match="unsupported native Inspect log schema version 99"):
        _normalize(path)


def test_unfinished_log_fails_clearly(tmp_path: Path):
    path = _write_zip(
        tmp_path / "started.eval", {"header.json": _header(status="started")}
    )
    with pytest.raises(AuditError, match="only a .*completed .* log can be audited"):
        _normalize(path)


def test_log_without_samples_fails_clearly(tmp_path: Path):
    path = _write_zip(tmp_path / "nosamples.eval", {"header.json": _header()})
    with pytest.raises(AuditError, match="records no full samples"):
        _normalize(path)


def test_corrupt_sample_json_fails_clearly(tmp_path: Path):
    path = _write_zip(
        tmp_path / "corrupt.eval",
        {"header.json": _header(), "samples/x_epoch_1.json": "{not json"},
    )
    with pytest.raises(AuditError, match="invalid JSON in archive member"):
        _normalize(path)


# ------------------------------------------------------- CLI on a native log


def test_cli_audit_native_log(tmp_path: Path):
    report = tmp_path / "report.html"
    result = runner.invoke(
        app, ["audit", str(NATIVE_LOOP_LOG), "--output", str(report)]
    )
    assert result.exit_code == 1, result.output  # TRAJ-001 is HIGH
    assert "TRAJ-001" in result.output
    assert "evalwarden-native-loop" in result.output
    assert report.is_file()


def test_cli_audit_native_log_explicit_adapter(tmp_path: Path):
    report = tmp_path / "report.html"
    result = runner.invoke(
        app,
        ["audit", str(NATIVE_ORDINARY_LOG), "--output", str(report),
         "--adapter", "inspect"],
    )
    assert result.exit_code in (0, 1), result.output
    assert "TRAJ-001" not in result.output
    assert "NOISE-001" not in result.output


# ------------------------------------------- native vs JSON adapter parity


def test_native_and_json_paths_agree_on_shared_fields(tmp_path: Path):
    """The same eval through both Inspect paths must agree where they overlap.

    Intentional differences (asserted, not hidden): the native path knows
    the model id (the JSON run record does not carry one here), and the
    native eval_id is the Inspect task name, which the JSON dataset below
    reuses as its eval_id so the comparison is apples to apples.
    """
    native = _normalize(NATIVE_LOOP_LOG)

    dataset = {
        "schema_version": "evalwarden-artifact-v1",
        "eval_id": native.eval_id,
        "tasks": [
            {
                "id": t.id,
                "prompt": t.prompt,
                "target": t.target,
                "choices": t.choices,
                "failure_mode": t.failure_mode,
                "metadata": t.metadata,
            }
            for t in native.tasks
        ],
    }
    run = {
        "attempts": [
            {
                "task_id": a.task_id,
                "status": a.status,
                "score": a.score,
                "tool_calls": a.tool_calls,
                "tokens_in": a.tokens_in,
                "tokens_out": a.tokens_out,
                "latency_s": a.latency_s,
                "tries": a.tries,
            }
            for a in native.attempts
        ]
    }
    trajectories = "".join(
        json.dumps(
            {
                "task_id": a.task_id,
                "step_id": s.step_id,
                "tool": s.tool,
                "args": s.args,
                "output": s.output,
                "consumes": s.consumes,
            }
        )
        + "\n"
        for a in native.attempts
        for s in a.spans
    )
    (tmp_path / "dataset.json").write_text(json.dumps(dataset))
    (tmp_path / "grader.json").write_text(json.dumps({"kind": "script"}))
    (tmp_path / "run.json").write_text(json.dumps(run))
    (tmp_path / "trajectories.jsonl").write_text(trajectories)
    via_json = _normalize(tmp_path)

    assert via_json.eval_id == native.eval_id
    for json_task, native_task in zip(via_json.tasks, native.tasks):
        assert (json_task.id, json_task.prompt, json_task.target,
                json_task.choices, json_task.failure_mode,
                json_task.metadata) == (
                    native_task.id, native_task.prompt, native_task.target,
                    native_task.choices, native_task.failure_mode,
                    native_task.metadata)
    for json_attempt, native_attempt in zip(via_json.attempts, native.attempts):
        assert (json_attempt.task_id, json_attempt.status, json_attempt.score,
                json_attempt.tool_calls, json_attempt.tokens_in,
                json_attempt.tokens_out, json_attempt.latency_s,
                json_attempt.tries) == (
                    native_attempt.task_id, native_attempt.status,
                    native_attempt.score, native_attempt.tool_calls,
                    native_attempt.tokens_in, native_attempt.tokens_out,
                    native_attempt.latency_s, native_attempt.tries)
        assert [(s.step_id, s.tool, s.args, s.output, s.consumes)
                for s in json_attempt.spans] == [
                    (s.step_id, s.tool, s.args, s.output, s.consumes)
                    for s in native_attempt.spans]
    # The one deliberate gap: native attempts know their model.
    assert all(a.model_id == "mockllm/model" for a in native.attempts)
    assert all(a.model_id is None for a in via_json.attempts)
