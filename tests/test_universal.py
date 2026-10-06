"""Universal importer: canonical JSON, JSONL streams, and score-table CSV."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from evalwarden.adapters import AuditError, autodetect
from evalwarden.canonical import (
    CanonicalValidationError,
    model_from_csv,
    model_from_jsonl,
)
from evalwarden.engine import audit
from evalwarden.model import IntegrityModel

from .conftest import DEMO_JUDGE_BAD


def _finding_projection(result):
    return [
        (finding.id, finding.severity.value, finding.title, finding.evidence)
        for finding in result.findings
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_canonical_round_trip_preserves_model_and_findings(tmp_path: Path):
    original = audit(DEMO_JUDGE_BAD)
    document = original.model.to_canonical_document()
    assert document["schema"] == "evalwarden.model"
    assert document["schema_version"] == "1.0"

    reimported = IntegrityModel.from_canonical_document(document)
    assert reimported.tasks == original.model.tasks
    assert reimported.environment == original.model.environment
    assert reimported.grader == original.model.grader
    assert reimported.attempts == original.model.attempts
    assert reimported.judgments == original.model.judgments
    assert reimported.run_scores == original.model.run_scores
    assert reimported.claimed_delta == original.model.claimed_delta
    assert reimported.unsupported == original.model.unsupported

    path = tmp_path / "judge-bad.evalwarden.json"
    path.write_text(original.model.to_canonical_json(), encoding="utf-8")
    assert autodetect(path).name == "universal"
    round_tripped = audit(path)
    assert round_tripped.model.adapter_name == "universal"
    assert round_tripped.model.digests[path.name]
    assert round_tripped.model.artifact_file("tasks.json") == path.name
    assert _finding_projection(round_tripped) == _finding_projection(original)


def test_canonical_environment_carries_names_never_values():
    model = IntegrityModel.from_canonical_document(
        {
            "schema": "evalwarden.model",
            "schema_version": "1.0",
            "eval_id": "env-test",
            "environment": {"env_vars": ["API_TOKEN"], "mounts": []},
        }
    )
    assert model.environment.env_vars == {"API_TOKEN": "<redacted>"}

    with pytest.raises(CanonicalValidationError, match=r"\$\.environment\.env_vars"):
        IntegrityModel.from_canonical_document(
            {
                "schema": "evalwarden.model",
                "schema_version": "1.0",
                "eval_id": "env-test",
                "environment": {"env_vars": {"API_TOKEN": "secret"}},
            }
        )


def test_canonical_unknown_fields_are_reported_not_imported():
    model = IntegrityModel.from_canonical_document(
        {
            "schema": "evalwarden.model",
            "schema_version": "1.0",
            "eval_id": "unknown-fields",
            "future_root": 1,
            "tasks": [
                {"id": "t1", "prompt": "prompt", "future_task_field": True}
            ],
        }
    )
    assert any("$.future_root" in note for note in model.unsupported)
    assert any("$.tasks[0].future_task_field" in note for note in model.unsupported)
    assert model.tasks[0].metadata == {}


def test_canonical_errors_name_the_field_path():
    with pytest.raises(CanonicalValidationError, match=r"\$\.tasks\[0\]\.prompt"):
        IntegrityModel.from_canonical_document(
            {
                "schema": "evalwarden.model",
                "schema_version": "1.0",
                "eval_id": "bad",
                "tasks": [{"id": "t1", "prompt": 42}],
            }
        )

    with pytest.raises(CanonicalValidationError, match=r"\$\.attempts\[0\]\.task_id"):
        IntegrityModel.from_canonical_document(
            {
                "schema": "evalwarden.model",
                "schema_version": "1.0",
                "eval_id": "bad-reference",
                "tasks": [{"id": "t1", "prompt": "prompt"}],
                "attempts": [{"task_id": "missing", "status": "pass"}],
            }
        )

    with pytest.raises(CanonicalValidationError, match="invalid JSON at line 1 column"):
        IntegrityModel.from_canonical_json('{"schema":')


def _foreign_jsonl() -> str:
    lines = [
        '{"type":"eval","schema":"evalwarden.model","schema_version":"1.0",'
        '"eval_id":"foreign-harness","claimed_delta":0.01}',
        '{"type":"environment","env_vars":[],"mounts":[]}',
        '{"type":"grader","kind":"script"}',
        '{"type":"task","id":"loop-task","prompt":"Check service status."}',
    ]
    for index in range(2, 13):
        lines.append(
            f'{{"type":"task","id":"t{index:02d}","prompt":"Answer item {index}."}}'
        )
    lines.append(
        '{"type":"attempt","task_id":"loop-task","status":"pass","score":1.0}'
    )
    for index in range(2, 13):
        lines.append(
            f'{{"type":"attempt","task_id":"t{index:02d}","status":"pass","score":1.0}}'
        )
    for step in range(1, 6):
        lines.append(
            '{"type":"span","task_id":"loop-task","attempt_index":0,'
            f'"step_id":"call-{step}","tool":"status",'
            '"args":{"resource":"billing"}}'
        )
    # A harness-authored repeated-measures stream: judge re-grades move the
    # score by 0.20, fresh generations by only 0.01.  No Evalwarden
    # serializer is involved in producing these records.
    for task_index in range(1, 13):
        task_id = "loop-task" if task_index == 1 else f"t{task_index:02d}"
        base = 0.72 + task_index * 0.005
        for generation, generation_effect in enumerate((0.0, 0.01, 0.02)):
            for grade, grade_effect in enumerate((-0.20, 0.0, 0.20)):
                score = base + generation_effect + grade_effect
                lines.append(
                    json.dumps(
                        {
                            "type": "run_score",
                            "task_id": task_id,
                            "score": round(score, 3),
                            "generation_index": generation,
                            "grade_index": grade,
                            "environment_index": 0,
                        }
                    )
                )
    return "\n".join(lines) + "\n"


def test_jsonl_foreign_harness_fires_traj_and_noise(tmp_path: Path):
    path = tmp_path / "foreign.jsonl"
    path.write_text(_foreign_jsonl(), encoding="utf-8")
    before = _sha256(path)

    assert autodetect(path).name == "universal"
    result = audit(path)

    assert _sha256(path) == before
    assert result.model.adapter_name == "universal"
    assert len(result.model.tasks) == 12
    assert len(result.model.run_scores) == 108
    loop_attempt = next(
        attempt for attempt in result.model.attempts if attempt.task_id == "loop-task"
    )
    assert len(loop_attempt.spans) == 5
    finding_ids = {finding.id for finding in result.findings}
    assert {"TRAJ-001", "NOISE-001", "NOISE-002"} <= finding_ids
    assert "TRAJ-002" not in finding_ids


def test_jsonl_span_may_precede_its_attempt():
    imported = model_from_jsonl(
        "\n".join(
            [
                '{"type":"eval","schema":"evalwarden.model","schema_version":"1.0","eval_id":"order"}',
                '{"type":"task","id":"t1","prompt":"prompt"}',
                '{"type":"span","task_id":"t1","step_id":"s1","tool":"read"}',
                '{"type":"attempt","task_id":"t1","status":"pass"}',
            ]
        )
    )
    assert imported.attempts[0].spans[0].step_id == "s1"


def test_jsonl_errors_name_the_line_and_field():
    with pytest.raises(CanonicalValidationError, match="line 3: invalid JSON"):
        model_from_jsonl(
            "\n".join(
                [
                    '{"type":"eval","schema":"evalwarden.model","schema_version":"1.0","eval_id":"x"}',
                    '{"type":"task","id":"t1","prompt":"prompt"}',
                    '{"type":"task",',
                ]
            )
        )

    with pytest.raises(
        CanonicalValidationError, match=r"line 3: \$\.run_scores\[0\]\.score"
    ):
        model_from_jsonl(
            "\n".join(
                [
                    '{"type":"eval","schema":"evalwarden.model","schema_version":"1.0","eval_id":"x"}',
                    '{"type":"task","id":"t1","prompt":"prompt"}',
                    '{"type":"run_score","task_id":"t1","score":"high"}',
                ]
            )
        )


def test_jsonl_adapter_surfaces_errors_as_audit_errors(tmp_path: Path):
    path = tmp_path / "broken.jsonl"
    path.write_text('{"type":"task","id":"t1","prompt":"prompt"}\n', encoding="utf-8")
    with pytest.raises(AuditError, match="line 1.*first record"):
        audit(path)


def _saturated_csv() -> str:
    lines = ["task_id,score,status,model_id"]
    for index in range(1, 31):
        for model_id in ("model-a", "model-b"):
            lines.append(f"task-{index:02d},1.0,pass,{model_id}")
    return "\n".join(lines) + "\n"


def test_csv_score_table_supports_only_what_it_states(tmp_path: Path):
    path = tmp_path / "scores.csv"
    path.write_text(_saturated_csv(), encoding="utf-8")
    before = _sha256(path)

    assert autodetect(path).name == "universal"
    result = audit(path)

    assert _sha256(path) == before
    assert result.model.adapter_name == "universal"
    assert len(result.model.tasks) == 30
    assert len(result.model.attempts) == 60
    assert result.model.run_scores == []
    finding_ids = {finding.id for finding in result.findings}
    assert {"DATA-001", "COST-001"} <= finding_ids
    assert not finding_ids.intersection(
        {"JUDGE-001", "TRAJ-001", "NOISE-001", "DATA-002", "DATA-003"}
    )
    assert any(
        "judgments, and trajectories are not carried" in note
        for note in result.model.unsupported
    )
    assert any(
        "no run_id or condition index column" in note
        for note in result.model.unsupported
    )


def test_csv_run_mapping_and_multiple_model_guard():
    model = model_from_csv(
        "\n".join(
            [
                "task_id,score,run_id,grade_index",
                "t1,0.5,run-a,0",
                "t1,0.7,run-a,1",
                "t1,0.6,run-b,0",
            ]
        ),
        eval_id="csv-runs",
        source_name="runs.csv",
    )
    assert model.attempts == []
    assert [
        (row.generation_index, row.grade_index) for row in model.run_scores
    ] == [(0, 0), (0, 1), (1, 0)]
    assert any("no status column" in note for note in model.unsupported)

    mixed = model_from_csv(
        "\n".join(
            [
                "task_id,score,model_id,run_id",
                "t1,0.5,model-a,run-a",
                "t1,0.7,model-b,run-a",
            ]
        ),
        source_name="mixed.csv",
    )
    assert mixed.run_scores == []
    assert any("multiple model_id values" in note for note in mixed.unsupported)


def test_csv_errors_name_the_row_and_field():
    with pytest.raises(CanonicalValidationError, match=r"line 1: \$\.score column"):
        model_from_csv("task_id,status\nt1,pass\n")

    with pytest.raises(CanonicalValidationError, match=r"row 3: \$\.score"):
        model_from_csv("task_id,score\nt1,0.5\nt2,not-a-number\n")

    with pytest.raises(CanonicalValidationError, match="row 3: duplicate score"):
        model_from_csv(
            "task_id,score,run_id\nt1,0.5,run-a\nt1,0.6,run-a\n"
        )
