"""Canonical Evalwarden input formats.

This module is the universal importer: it translates the versioned
``evalwarden.model`` document, an equivalent JSONL record stream, and a
minimal score-table CSV into :class:`IntegrityModel`.  It is deliberately a
translation layer, not a second model.  Values are carried only when the
input states them; in particular, the importer never infers trajectory
consumption edges, failure modes, judge identities, reference labels, or
noise axes.

The public Python entry points are ``model_to_document``,
``model_from_document``, ``model_from_json``, ``model_from_jsonl``, and
``model_from_csv``.  ``IntegrityModel`` also exposes the canonical JSON
entry points as methods for convenience.
"""
from __future__ import annotations

import csv
import io
import json
import math
import re
from typing import Any, Mapping

from .model import (
    Attempt,
    Environment,
    Grader,
    IntegrityModel,
    Judgment,
    Mount,
    RunItemScore,
    TaskSample,
    TrajectoryStep,
)

SCHEMA_ID = "evalwarden.model"
SCHEMA_VERSION = "1.0"
CANONICAL_ADAPTER_NAME = "universal"
CANONICAL_ADAPTER_VERSION = "1.0.0"

_ATTEMPT_STATUSES = {"pass", "fail", "error", "incomplete"}
_MOUNT_MODES = {"ro", "rw"}
_MOUNT_ACCESS = {"none", "read", "write"}
_JSONL_TYPES = {
    "eval",
    "environment",
    "grader",
    "task",
    "attempt",
    "judgment",
    "run_score",
    "span",
}
_CSV_COLUMNS = {
    "task_id",
    "score",
    "status",
    "model_id",
    "run_id",
    "run_index",
    "generation_index",
    "grade_index",
    "environment_index",
}


class CanonicalValidationError(ValueError):
    """A canonical input is malformed; the message names the exact place."""


def _fail(path: str, message: str, locations: Mapping[str, int] | None = None) -> None:
    raise CanonicalValidationError(f"{_where(path, locations)}: {message}")


def _where(path: str, locations: Mapping[str, int] | None = None) -> str:
    """Format a field path, adding a JSONL line number when one is known."""
    if not locations:
        return path
    candidate = path
    while candidate:
        if candidate in locations:
            return f"line {locations[candidate]}: {path}"
        if candidate == "$":
            break
        shortened = re.sub(r"(\.[^.\[\]]+|\[\d+\])$", "", candidate)
        if shortened == candidate:
            break
        candidate = shortened
    return path


def _is_bool(value: Any) -> bool:
    return isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _object(value: Any, path: str, locations: Mapping[str, int] | None = None) -> dict:
    if not isinstance(value, dict):
        _fail(path, "must be a JSON object", locations)
    return value


def _list(value: Any, path: str, locations: Mapping[str, int] | None = None) -> list:
    if not isinstance(value, list):
        _fail(path, "must be an array", locations)
    return value


def _string(
    value: Any,
    path: str,
    locations: Mapping[str, int] | None = None,
    *,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        qualifier = "" if allow_empty else "non-empty "
        _fail(path, f"must be a {qualifier}string", locations)
    return value


def _optional_string(
    value: Any,
    path: str,
    locations: Mapping[str, int] | None = None,
) -> str | None:
    if value is None:
        return None
    return _string(value, path, locations, allow_empty=True)


def _boolean(value: Any, path: str, locations: Mapping[str, int] | None = None) -> bool:
    if not isinstance(value, bool):
        _fail(path, "must be a boolean", locations)
    return value


def _optional_boolean(
    value: Any,
    path: str,
    locations: Mapping[str, int] | None = None,
) -> bool | None:
    if value is None:
        return None
    return _boolean(value, path, locations)


def _integer(
    value: Any,
    path: str,
    locations: Mapping[str, int] | None = None,
    *,
    minimum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(path, "must be an integer", locations)
    if minimum is not None and value < minimum:
        _fail(path, f"must be at least {minimum}", locations)
    return value


def _number(
    value: Any,
    path: str,
    locations: Mapping[str, int] | None = None,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if not _is_number(value) or not math.isfinite(float(value)):
        _fail(path, "must be a finite number", locations)
    number = float(value)
    if minimum is not None and number < minimum:
        _fail(path, f"must be at least {minimum:g}", locations)
    if maximum is not None and number > maximum:
        _fail(path, f"must be at most {maximum:g}", locations)
    return number


def _optional_number(
    value: Any,
    path: str,
    locations: Mapping[str, int] | None = None,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    if value is None:
        return None
    return _number(value, path, locations, minimum=minimum, maximum=maximum)


def _optional_integer(
    value: Any,
    path: str,
    locations: Mapping[str, int] | None = None,
    *,
    minimum: int | None = None,
) -> int | None:
    if value is None:
        return None
    return _integer(value, path, locations, minimum=minimum)


def _string_list(
    value: Any,
    path: str,
    locations: Mapping[str, int] | None = None,
    *,
    allow_empty_strings: bool = False,
) -> list[str]:
    values = _list(value, path, locations)
    return [
        _string(item, f"{path}[{index}]", locations, allow_empty=allow_empty_strings)
        for index, item in enumerate(values)
    ]


def _json_value(value: Any, path: str, locations: Mapping[str, int] | None = None) -> Any:
    """Validate and copy a JSON value (used for metadata and tool args)."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if _is_number(value):
        if not math.isfinite(float(value)):
            _fail(path, "must contain only finite numbers", locations)
        return value
    if isinstance(value, list):
        return [
            _json_value(item, f"{path}[{index}]", locations)
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        copied: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                _fail(path, "object keys must be strings", locations)
            copied[key] = _json_value(item, f"{path}.{key}", locations)
        return copied
    _fail(path, "must be a JSON value", locations)


def _unknown_fields(
    value: dict,
    known: set[str],
    path: str,
    unsupported: list[str],
    locations: Mapping[str, int] | None = None,
) -> None:
    for key in value:
        if key not in known:
            unsupported.append(
                f"{_where(f'{path}.{key}', locations)} "
                f"(not part of {SCHEMA_ID} {SCHEMA_VERSION}; ignored)"
            )


def _parse_environment(
    raw: Any,
    path: str,
    unsupported: list[str],
    locations: Mapping[str, int] | None = None,
) -> Environment:
    data = _object(raw, path, locations)
    _unknown_fields(data, {"env_vars", "mounts"}, path, unsupported, locations)
    names = _string_list(data.get("env_vars", []), f"{path}.env_vars", locations)
    if len(names) != len(set(names)):
        _fail(f"{path}.env_vars", "must not contain duplicate names", locations)
    mounts: list[Mount] = []
    for index, raw_mount in enumerate(_list(data.get("mounts", []), f"{path}.mounts", locations)):
        mount_path = f"{path}.mounts[{index}]"
        mount = _object(raw_mount, mount_path, locations)
        _unknown_fields(
            mount, {"path", "mode", "agent_access"}, mount_path, unsupported, locations
        )
        mode = _string(mount.get("mode", "ro"), f"{mount_path}.mode", locations)
        if mode not in _MOUNT_MODES:
            _fail(f"{mount_path}.mode", "must be 'ro' or 'rw'", locations)
        access = _string(
            mount.get("agent_access", "read"), f"{mount_path}.agent_access", locations
        )
        if access not in _MOUNT_ACCESS:
            _fail(
                f"{mount_path}.agent_access",
                "must be 'none', 'read', or 'write'",
                locations,
            )
        mounts.append(
            Mount(
                path=_string(mount.get("path"), f"{mount_path}.path", locations),
                mode=mode,
                agent_access=access,
            )
        )
    # The wire format carries names only.  This is intentional: environment
    # values can be secrets, and no integrity check needs them.
    return Environment(env_vars={name: "<redacted>" for name in names}, mounts=mounts)


def _parse_grader(
    raw: Any,
    path: str,
    unsupported: list[str],
    locations: Mapping[str, int] | None = None,
) -> Grader:
    data = _object(raw, path, locations)
    known = {
        "kind",
        "verifier_path",
        "verifier_writable_by_agent",
        "verifier_rule",
        "accepts_empty_output",
        "tests",
        "judge_model",
        "judge_family",
        "protocol",
        "counterbalanced",
        "temperature",
        "repeats",
        "rubric_criteria",
        "scale_anchors",
        "reference_labels",
    }
    _unknown_fields(data, known, path, unsupported, locations)
    scale_anchors_raw = _object(
        data.get("scale_anchors", {}), f"{path}.scale_anchors", locations
    )
    scale_anchors = {
        str(key): _string(value, f"{path}.scale_anchors.{key}", locations, allow_empty=True)
        for key, value in scale_anchors_raw.items()
    }
    labels_raw = _object(
        data.get("reference_labels", {}), f"{path}.reference_labels", locations
    )
    reference_labels = {
        str(key): _string(value, f"{path}.reference_labels.{key}", locations, allow_empty=True)
        for key, value in labels_raw.items()
    }
    return Grader(
        kind=_string(data.get("kind", "script"), f"{path}.kind", locations),
        verifier_path=_optional_string(data.get("verifier_path"), f"{path}.verifier_path", locations),
        verifier_writable_by_agent=_boolean(
            data.get("verifier_writable_by_agent", False),
            f"{path}.verifier_writable_by_agent",
            locations,
        ),
        verifier_rule=_optional_string(data.get("verifier_rule"), f"{path}.verifier_rule", locations),
        accepts_empty_output=_boolean(
            data.get("accepts_empty_output", False),
            f"{path}.accepts_empty_output",
            locations,
        ),
        tests=_string_list(data.get("tests", []), f"{path}.tests", locations, allow_empty_strings=True),
        judge_model=_optional_string(data.get("judge_model"), f"{path}.judge_model", locations),
        judge_family=_optional_string(data.get("judge_family"), f"{path}.judge_family", locations),
        protocol=_optional_string(data.get("protocol"), f"{path}.protocol", locations),
        counterbalanced=_optional_boolean(
            data.get("counterbalanced"), f"{path}.counterbalanced", locations
        ),
        temperature=_optional_number(data.get("temperature"), f"{path}.temperature", locations),
        repeats=_integer(data.get("repeats", 1), f"{path}.repeats", locations, minimum=0),
        rubric_criteria=_string_list(
            data.get("rubric_criteria", []), f"{path}.rubric_criteria", locations, allow_empty_strings=True
        ),
        scale_anchors=scale_anchors,
        reference_labels=reference_labels,
    )


def _parse_task(
    raw: Any,
    path: str,
    unsupported: list[str],
    locations: Mapping[str, int] | None = None,
) -> TaskSample:
    data = _object(raw, path, locations)
    _unknown_fields(
        data,
        {"id", "prompt", "target", "choices", "failure_mode", "metadata"},
        path,
        unsupported,
        locations,
    )
    target_raw = data.get("target")
    if target_raw is None or isinstance(target_raw, str):
        target: str | list[str] | None = target_raw
    elif isinstance(target_raw, list):
        target = _string_list(target_raw, f"{path}.target", locations, allow_empty_strings=True)
    else:
        _fail(f"{path}.target", "must be a string, an array of strings, or null", locations)
    choices_raw = data.get("choices")
    choices = (
        None
        if choices_raw is None
        else _string_list(choices_raw, f"{path}.choices", locations, allow_empty_strings=True)
    )
    metadata_raw = _object(data.get("metadata", {}), f"{path}.metadata", locations)
    return TaskSample(
        id=_string(data.get("id"), f"{path}.id", locations),
        prompt=_string(data.get("prompt"), f"{path}.prompt", locations, allow_empty=True),
        target=target,
        choices=choices,
        failure_mode=_optional_string(data.get("failure_mode"), f"{path}.failure_mode", locations),
        metadata=_json_value(metadata_raw, f"{path}.metadata", locations),
    )


def _parse_span(
    raw: Any,
    path: str,
    unsupported: list[str],
    locations: Mapping[str, int] | None = None,
) -> TrajectoryStep:
    data = _object(raw, path, locations)
    _unknown_fields(
        data, {"step_id", "tool", "args", "output", "consumes"}, path, unsupported, locations
    )
    args_raw = _object(data.get("args", {}), f"{path}.args", locations)
    return TrajectoryStep(
        step_id=_string(data.get("step_id"), f"{path}.step_id", locations),
        tool=_string(data.get("tool"), f"{path}.tool", locations),
        args=_json_value(args_raw, f"{path}.args", locations),
        output=_optional_string(data.get("output"), f"{path}.output", locations),
        consumes=_string_list(data.get("consumes", []), f"{path}.consumes", locations),
    )


def _parse_attempt(
    raw: Any,
    path: str,
    unsupported: list[str],
    locations: Mapping[str, int] | None = None,
) -> Attempt:
    data = _object(raw, path, locations)
    _unknown_fields(
        data,
        {
            "task_id",
            "status",
            "score",
            "tool_calls",
            "actions",
            "tokens_in",
            "tokens_out",
            "latency_s",
            "tries",
            "empty_submission",
            "output",
            "model_id",
            "spans",
        },
        path,
        unsupported,
        locations,
    )
    status = _string(data.get("status"), f"{path}.status", locations)
    if status not in _ATTEMPT_STATUSES:
        _fail(
            f"{path}.status",
            "must be one of 'pass', 'fail', 'error', or 'incomplete'",
            locations,
        )
    spans = [
        _parse_span(item, f"{path}.spans[{index}]", unsupported, locations)
        for index, item in enumerate(_list(data.get("spans", []), f"{path}.spans", locations))
    ]
    return Attempt(
        task_id=_string(data.get("task_id"), f"{path}.task_id", locations),
        status=status,
        score=_optional_number(data.get("score"), f"{path}.score", locations),
        tool_calls=_integer(data.get("tool_calls", 0), f"{path}.tool_calls", locations, minimum=0),
        actions=_string_list(data.get("actions", []), f"{path}.actions", locations, allow_empty_strings=True),
        tokens_in=_optional_integer(data.get("tokens_in"), f"{path}.tokens_in", locations, minimum=0),
        tokens_out=_optional_integer(data.get("tokens_out"), f"{path}.tokens_out", locations, minimum=0),
        latency_s=_optional_number(data.get("latency_s"), f"{path}.latency_s", locations, minimum=0.0),
        tries=_integer(data.get("tries", 1), f"{path}.tries", locations, minimum=1),
        empty_submission=_boolean(
            data.get("empty_submission", False), f"{path}.empty_submission", locations
        ),
        output=_optional_string(data.get("output"), f"{path}.output", locations),
        model_id=_optional_string(data.get("model_id"), f"{path}.model_id", locations),
        spans=spans,
    )


def _parse_judgment(
    raw: Any,
    path: str,
    unsupported: list[str],
    locations: Mapping[str, int] | None = None,
) -> Judgment:
    data = _object(raw, path, locations)
    _unknown_fields(
        data,
        {
            "task_id",
            "candidates",
            "presentation_order",
            "winner",
            "scores",
            "lengths",
            "repeat_index",
            "confidence",
            "judge_id",
        },
        path,
        unsupported,
        locations,
    )
    scores_raw = _object(data.get("scores", {}), f"{path}.scores", locations)
    scores = {
        str(key): _number(value, f"{path}.scores.{key}", locations)
        for key, value in scores_raw.items()
    }
    lengths_raw = _object(data.get("lengths", {}), f"{path}.lengths", locations)
    lengths = {
        str(key): _integer(value, f"{path}.lengths.{key}", locations, minimum=0)
        for key, value in lengths_raw.items()
    }
    return Judgment(
        task_id=_string(data.get("task_id"), f"{path}.task_id", locations),
        candidates=_string_list(data.get("candidates", []), f"{path}.candidates", locations),
        presentation_order=_string_list(
            data.get("presentation_order", []), f"{path}.presentation_order", locations
        ),
        winner=_optional_string(data.get("winner"), f"{path}.winner", locations),
        scores=scores,
        lengths=lengths,
        repeat_index=_integer(data.get("repeat_index", 0), f"{path}.repeat_index", locations, minimum=0),
        confidence=_optional_number(
            data.get("confidence"), f"{path}.confidence", locations, minimum=0.0, maximum=1.0
        ),
        judge_id=_optional_string(data.get("judge_id"), f"{path}.judge_id", locations),
    )


def _parse_run_score(
    raw: Any,
    path: str,
    unsupported: list[str],
    locations: Mapping[str, int] | None = None,
) -> RunItemScore:
    data = _object(raw, path, locations)
    _unknown_fields(
        data,
        {"task_id", "score", "generation_index", "grade_index", "environment_index"},
        path,
        unsupported,
        locations,
    )
    return RunItemScore(
        task_id=_string(data.get("task_id"), f"{path}.task_id", locations),
        score=_number(data.get("score"), f"{path}.score", locations),
        generation_index=_integer(
            data.get("generation_index", 0), f"{path}.generation_index", locations, minimum=0
        ),
        grade_index=_integer(data.get("grade_index", 0), f"{path}.grade_index", locations, minimum=0),
        environment_index=_integer(
            data.get("environment_index", 0), f"{path}.environment_index", locations, minimum=0
        ),
    )


def _parse_document(
    document: Any,
    *,
    adapter_name: str = CANONICAL_ADAPTER_NAME,
    adapter_version: str = CANONICAL_ADAPTER_VERSION,
    locations: Mapping[str, int] | None = None,
) -> IntegrityModel:
    root_path = "$"
    data = _object(document, root_path, locations)
    unsupported: list[str] = []
    _unknown_fields(
        data,
        {
            "schema",
            "schema_version",
            "eval_id",
            "failure_mode_taxonomy",
            "environment",
            "grader",
            "tasks",
            "attempts",
            "judgments",
            "run_scores",
            "claimed_delta",
            "unsupported",
        },
        root_path,
        unsupported,
        locations,
    )
    if data.get("schema") != SCHEMA_ID:
        _fail("$.schema", f"must be {SCHEMA_ID!r}", locations)
    if data.get("schema_version") != SCHEMA_VERSION:
        _fail(
            "$.schema_version",
            f"must be {SCHEMA_VERSION!r} (this importer understands only that version)",
            locations,
        )
    declared_unsupported = _string_list(
        data.get("unsupported", []), "$.unsupported", locations, allow_empty_strings=True
    )
    unsupported = [*declared_unsupported, *unsupported]

    tasks = [
        _parse_task(item, f"$.tasks[{index}]", unsupported, locations)
        for index, item in enumerate(_list(data.get("tasks", []), "$.tasks", locations))
    ]
    seen_tasks: dict[str, int] = {}
    for index, task in enumerate(tasks):
        if task.id in seen_tasks:
            _fail(f"$.tasks[{index}].id", f"duplicates task id {task.id!r}", locations)
        seen_tasks[task.id] = index

    environment = _parse_environment(
        data.get("environment", {}), "$.environment", unsupported, locations
    )
    grader = _parse_grader(data.get("grader", {}), "$.grader", unsupported, locations)
    attempts = [
        _parse_attempt(item, f"$.attempts[{index}]", unsupported, locations)
        for index, item in enumerate(_list(data.get("attempts", []), "$.attempts", locations))
    ]
    judgments = [
        _parse_judgment(item, f"$.judgments[{index}]", unsupported, locations)
        for index, item in enumerate(
            _list(data.get("judgments", []), "$.judgments", locations)
        )
    ]
    run_scores = [
        _parse_run_score(item, f"$.run_scores[{index}]", unsupported, locations)
        for index, item in enumerate(
            _list(data.get("run_scores", []), "$.run_scores", locations)
        )
    ]

    task_ids = set(seen_tasks)
    for index, attempt in enumerate(attempts):
        if attempt.task_id not in task_ids:
            _fail(
                f"$.attempts[{index}].task_id",
                f"references unknown task {attempt.task_id!r}",
                locations,
            )
        step_ids: set[str] = set()
        for span_index, span in enumerate(attempt.spans):
            if span.step_id in step_ids:
                _fail(
                    f"$.attempts[{index}].spans[{span_index}].step_id",
                    f"duplicates step id {span.step_id!r} in this attempt",
                    locations,
                )
            step_ids.add(span.step_id)
        for span_index, span in enumerate(attempt.spans):
            for consume_index, consumed in enumerate(span.consumes):
                if consumed not in step_ids:
                    _fail(
                        f"$.attempts[{index}].spans[{span_index}].consumes[{consume_index}]",
                        f"references unknown step {consumed!r} in this attempt",
                        locations,
                    )
    for index, judgment in enumerate(judgments):
        if judgment.task_id not in task_ids:
            _fail(
                f"$.judgments[{index}].task_id",
                f"references unknown task {judgment.task_id!r}",
                locations,
            )
    for index, run_score in enumerate(run_scores):
        if run_score.task_id not in task_ids:
            _fail(
                f"$.run_scores[{index}].task_id",
                f"references unknown task {run_score.task_id!r}",
                locations,
            )

    return IntegrityModel(
        eval_id=_string(data.get("eval_id"), "$.eval_id", locations),
        adapter_name=adapter_name,
        adapter_version=adapter_version,
        tasks=tasks,
        failure_mode_taxonomy=_string_list(
            data.get("failure_mode_taxonomy", []), "$.failure_mode_taxonomy", locations
        ),
        environment=environment,
        grader=grader,
        attempts=attempts,
        judgments=judgments,
        run_scores=run_scores,
        claimed_delta=_optional_number(data.get("claimed_delta"), "$.claimed_delta", locations),
        unsupported=unsupported,
    )


def model_from_document(
    document: Mapping[str, Any],
    *,
    adapter_name: str = CANONICAL_ADAPTER_NAME,
    adapter_version: str = CANONICAL_ADAPTER_VERSION,
) -> IntegrityModel:
    """Build an integrity model from a parsed canonical JSON document."""
    return _parse_document(
        document, adapter_name=adapter_name, adapter_version=adapter_version
    )


def _reject_json_constant(value: str) -> None:
    raise CanonicalValidationError(f"$: invalid JSON constant {value!r}")


def _loads_json(text: str, *, path: str = "$") -> Any:
    try:
        return json.loads(text, parse_constant=_reject_json_constant)
    except json.JSONDecodeError as exc:
        raise CanonicalValidationError(
            f"{path}: invalid JSON at line {exc.lineno} column {exc.colno}: {exc.msg}"
        ) from exc


def model_from_json(text: str | bytes) -> IntegrityModel:
    """Build an integrity model from canonical JSON text or bytes."""
    if isinstance(text, bytes):
        try:
            text = text.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CanonicalValidationError("$: canonical JSON must be UTF-8") from exc
    return model_from_document(_loads_json(text))


def model_to_document(model: IntegrityModel) -> dict[str, Any]:
    """Serialize the semantic content of a model as canonical JSON data.

    Runtime provenance (source adapter, digests, file aliases) and derived
    cost summaries are not eval content and are deliberately excluded.
    Environment values are also excluded: only variable names cross this
    boundary, matching every adapter's redaction rule.
    """
    return {
        "schema": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "eval_id": model.eval_id,
        "failure_mode_taxonomy": list(model.failure_mode_taxonomy),
        "environment": {
            "env_vars": list(model.environment.env_vars),
            "mounts": [
                {
                    "path": mount.path,
                    "mode": mount.mode,
                    "agent_access": mount.agent_access,
                }
                for mount in model.environment.mounts
            ],
        },
        "grader": {
            "kind": model.grader.kind,
            "verifier_path": model.grader.verifier_path,
            "verifier_writable_by_agent": model.grader.verifier_writable_by_agent,
            "verifier_rule": model.grader.verifier_rule,
            "accepts_empty_output": model.grader.accepts_empty_output,
            "tests": list(model.grader.tests),
            "judge_model": model.grader.judge_model,
            "judge_family": model.grader.judge_family,
            "protocol": model.grader.protocol,
            "counterbalanced": model.grader.counterbalanced,
            "temperature": model.grader.temperature,
            "repeats": model.grader.repeats,
            "rubric_criteria": list(model.grader.rubric_criteria),
            "scale_anchors": dict(model.grader.scale_anchors),
            "reference_labels": dict(model.grader.reference_labels),
        },
        "tasks": [
            {
                "id": task.id,
                "prompt": task.prompt,
                "target": _json_value(task.target, f"$.tasks[{index}].target"),
                "choices": list(task.choices) if task.choices is not None else None,
                "failure_mode": task.failure_mode,
                "metadata": _json_value(task.metadata, f"$.tasks[{index}].metadata"),
            }
            for index, task in enumerate(model.tasks)
        ],
        "attempts": [
            {
                "task_id": attempt.task_id,
                "status": attempt.status,
                "score": attempt.score,
                "tool_calls": attempt.tool_calls,
                "actions": list(attempt.actions),
                "tokens_in": attempt.tokens_in,
                "tokens_out": attempt.tokens_out,
                "latency_s": attempt.latency_s,
                "tries": attempt.tries,
                "empty_submission": attempt.empty_submission,
                "output": attempt.output,
                "model_id": attempt.model_id,
                "spans": [
                    {
                        "step_id": span.step_id,
                        "tool": span.tool,
                        "args": _json_value(span.args, "$.attempts[].spans[].args"),
                        "output": span.output,
                        "consumes": list(span.consumes),
                    }
                    for span in attempt.spans
                ],
            }
            for attempt in model.attempts
        ],
        "judgments": [
            {
                "task_id": judgment.task_id,
                "candidates": list(judgment.candidates),
                "presentation_order": list(judgment.presentation_order),
                "winner": judgment.winner,
                "scores": dict(judgment.scores),
                "lengths": dict(judgment.lengths),
                "repeat_index": judgment.repeat_index,
                "confidence": judgment.confidence,
                "judge_id": judgment.judge_id,
            }
            for judgment in model.judgments
        ],
        "run_scores": [
            {
                "task_id": run_score.task_id,
                "score": run_score.score,
                "generation_index": run_score.generation_index,
                "grade_index": run_score.grade_index,
                "environment_index": run_score.environment_index,
            }
            for run_score in model.run_scores
        ],
        "claimed_delta": model.claimed_delta,
        "unsupported": list(model.unsupported),
    }


def dumps_canonical(model: IntegrityModel, *, indent: int | None = 2) -> str:
    """Serialize a model to canonical JSON text."""
    return json.dumps(model_to_document(model), indent=indent, ensure_ascii=False) + "\n"


def model_from_jsonl(text: str | bytes) -> IntegrityModel:
    """Build a model from a canonical JSONL record stream.

    The first non-blank record must be ``{"type": "eval", ...}``.
    ``environment`` and ``grader`` records may appear at most once.  Span
    records use ``task_id`` plus ``attempt_index`` (the zero-based attempt
    occurrence for that task, default 0) and may appear before or after the
    attempt they belong to; attempts may also carry nested ``spans``.
    """
    if isinstance(text, bytes):
        try:
            text = text.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CanonicalValidationError("line 1: JSONL input must be UTF-8") from exc

    document: dict[str, Any] | None = None
    locations: dict[str, int] = {}
    pending_spans: list[dict[str, Any]] = []
    eval_unknown_fields: list[tuple[int, str]] = []
    seen_environment = False
    seen_grader = False

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line, parse_constant=_reject_json_constant)
        except CanonicalValidationError as exc:
            raise CanonicalValidationError(f"line {line_number}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise CanonicalValidationError(
                f"line {line_number}: invalid JSON at column {exc.colno}: {exc.msg}"
            ) from exc
        if not isinstance(record, dict):
            raise CanonicalValidationError(f"line {line_number}: record must be a JSON object")
        record_type = record.get("type")
        if not isinstance(record_type, str) or not record_type:
            raise CanonicalValidationError(f"line {line_number}: $.type is required")
        if record_type not in _JSONL_TYPES:
            raise CanonicalValidationError(
                f"line {line_number}: $.type {record_type!r} is not a canonical record type"
            )
        if document is None and record_type != "eval":
            raise CanonicalValidationError(
                f"line {line_number}: the first record must have $.type 'eval'"
            )

        body = {key: value for key, value in record.items() if key != "type"}
        if record_type == "eval":
            if document is not None:
                raise CanonicalValidationError(
                    f"line {line_number}: duplicate 'eval' record"
                )
            document = {
                "schema": body.get("schema"),
                "schema_version": body.get("schema_version"),
                "eval_id": body.get("eval_id"),
                "failure_mode_taxonomy": body.get("failure_mode_taxonomy", []),
                "environment": {},
                "grader": {},
                "tasks": [],
                "attempts": [],
                "judgments": [],
                "run_scores": [],
                "claimed_delta": body.get("claimed_delta"),
                "unsupported": body.get("unsupported", []),
            }
            for key in body:
                if key not in {
                    "schema",
                    "schema_version",
                    "eval_id",
                    "failure_mode_taxonomy",
                    "claimed_delta",
                    "unsupported",
                }:
                    eval_unknown_fields.append((line_number, key))
            locations["$"] = line_number
            continue

        assert document is not None  # guarded by the first-record check
        if record_type == "environment":
            if seen_environment:
                raise CanonicalValidationError(
                    f"line {line_number}: duplicate 'environment' record"
                )
            seen_environment = True
            document["environment"] = body
            locations["$.environment"] = line_number
        elif record_type == "grader":
            if seen_grader:
                raise CanonicalValidationError(f"line {line_number}: duplicate 'grader' record")
            seen_grader = True
            document["grader"] = body
            locations["$.grader"] = line_number
        elif record_type == "task":
            index = len(document["tasks"])
            document["tasks"].append(body)
            locations[f"$.tasks[{index}]"] = line_number
        elif record_type == "attempt":
            nested_spans = body.pop("spans", [])
            if not isinstance(nested_spans, list):
                raise CanonicalValidationError(
                    f"line {line_number}: $.spans must be an array"
                )
            index = len(document["attempts"])
            body["spans"] = nested_spans
            document["attempts"].append(body)
            locations[f"$.attempts[{index}]"] = line_number
        elif record_type == "judgment":
            index = len(document["judgments"])
            document["judgments"].append(body)
            locations[f"$.judgments[{index}]"] = line_number
        elif record_type == "run_score":
            index = len(document["run_scores"])
            document["run_scores"].append(body)
            locations[f"$.run_scores[{index}]"] = line_number
        elif record_type == "span":
            task_id = body.pop("task_id", None)
            attempt_index = body.pop("attempt_index", 0)
            if not isinstance(task_id, str) or not task_id:
                raise CanonicalValidationError(
                    f"line {line_number}: $.task_id must be a non-empty string"
                )
            if isinstance(attempt_index, bool) or not isinstance(attempt_index, int) or attempt_index < 0:
                raise CanonicalValidationError(
                    f"line {line_number}: $.attempt_index must be a non-negative integer"
                )
            pending_spans.append(
                {
                    "line": line_number,
                    "task_id": task_id,
                    "attempt_index": attempt_index,
                    "record": body,
                }
            )

    if document is None:
        raise CanonicalValidationError("line 1: JSONL stream has no 'eval' record")

    model = _parse_document(document, locations=locations)
    for line_number, key in eval_unknown_fields:
        model.unsupported.append(
            f"line {line_number}: $.{key} "
            f"(not part of {SCHEMA_ID} {SCHEMA_VERSION}; ignored)"
        )

    attempts_by_task: dict[str, list[Attempt]] = {}
    for attempt in model.attempts:
        attempts_by_task.setdefault(attempt.task_id, []).append(attempt)
    parsed_pending: list[tuple[int, Attempt, TrajectoryStep]] = []
    for pending in pending_spans:
        line_number = int(pending["line"])
        candidates = attempts_by_task.get(str(pending["task_id"]), [])
        attempt_index = int(pending["attempt_index"])
        if attempt_index >= len(candidates):
            raise CanonicalValidationError(
                f"line {line_number}: $.task_id {pending['task_id']!r} with "
                f"$.attempt_index {attempt_index} has no matching attempt record"
            )
        span = _parse_span(
            pending["record"],
            "$.span",
            model.unsupported,
            locations={"$.span": line_number},
        )
        parsed_pending.append((line_number, candidates[attempt_index], span))
    for line_number, attempt, span in parsed_pending:
        if any(existing.step_id == span.step_id for existing in attempt.spans):
            raise CanonicalValidationError(
                f"line {line_number}: $.step_id {span.step_id!r} duplicates a step "
                "in the target attempt"
            )
        attempt.spans.append(span)
    for line_number, attempt, span in parsed_pending:
        step_ids = {existing.step_id for existing in attempt.spans}
        for consumed in span.consumes:
            if consumed not in step_ids:
                raise CanonicalValidationError(
                    f"line {line_number}: $.consumes references unknown step "
                    f"{consumed!r} in the target attempt"
                )
    return model


def model_from_csv(
    text: str | bytes,
    *,
    eval_id: str = "csv-eval",
    source_name: str = "scores.csv",
) -> IntegrityModel:
    """Build a model from the minimal Evalwarden score-table CSV contract.

    Required columns are ``task_id`` and ``score``.  An attempt is created
    only for a row whose ``status`` cell is present, because a score alone
    does not state pass/fail.  Run scores are created only when a run or
    condition column states repeated-run structure, and only for a single
    model series: ``RunItemScore`` has no model dimension, so mixing model
    series into one NOISE series would fabricate a measurement.
    """
    if isinstance(text, bytes):
        try:
            text = text.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise CanonicalValidationError("line 1: CSV input must be UTF-8") from exc
    else:
        text = text.lstrip("\ufeff")

    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise CanonicalValidationError("line 1: CSV header row is required")
    headers = [name.strip() if isinstance(name, str) else "" for name in reader.fieldnames]
    if len(headers) != len(set(headers)):
        raise CanonicalValidationError("line 1: CSV header contains duplicate columns")
    if any(not header for header in headers):
        raise CanonicalValidationError("line 1: CSV header contains an empty column name")
    reader.fieldnames = headers
    for required in ("task_id", "score"):
        if required not in headers:
            raise CanonicalValidationError(
                f"line 1: $.{required} column is required in the CSV header"
            )

    unsupported: list[str] = []
    for header in headers:
        if header not in _CSV_COLUMNS:
            unsupported.append(
                f"{source_name}: column {header!r} (not part of the CSV contract; ignored)"
            )
    unsupported.append(
        f"{source_name}: task prompts, grader, environment, judgments, and "
        "trajectories are not carried by the CSV contract; checks needing "
        "them cannot be assessed"
    )

    rows: list[dict[str, Any]] = []
    task_order: list[str] = []
    seen_tasks: set[str] = set()
    run_id_generations: dict[str, int] = {}
    next_generation = 0
    has_repeat_structure = False

    for raw_row in reader:
        row_number = reader.line_num
        if None in raw_row:
            raise CanonicalValidationError(
                f"row {row_number}: too many fields for the CSV header"
            )
        row = {
            key: (value.strip() if isinstance(value, str) else "")
            for key, value in raw_row.items()
        }
        if not any(row.values()):
            continue
        task_id = row.get("task_id", "")
        if not task_id:
            raise CanonicalValidationError(f"row {row_number}: $.task_id must be a non-empty string")
        score_text = row.get("score", "")
        if not score_text:
            raise CanonicalValidationError(f"row {row_number}: $.score is required")
        try:
            score = float(score_text)
        except ValueError as exc:
            raise CanonicalValidationError(
                f"row {row_number}: $.score must be a finite number"
            ) from exc
        if not math.isfinite(score):
            raise CanonicalValidationError(f"row {row_number}: $.score must be a finite number")

        status = row.get("status", "") or None
        if status is not None and status not in _ATTEMPT_STATUSES:
            raise CanonicalValidationError(
                f"row {row_number}: $.status must be one of 'pass', 'fail', "
                "'error', or 'incomplete'"
            )
        model_id = row.get("model_id", "") or None

        def optional_index(column: str) -> int | None:
            value = row.get(column, "")
            if not value:
                return None
            if not re.fullmatch(r"\d+", value):
                raise CanonicalValidationError(
                    f"row {row_number}: $.{column} must be a non-negative integer"
                )
            return int(value)

        generation = optional_index("generation_index")
        run_index = optional_index("run_index")
        if generation is not None and run_index is not None and generation != run_index:
            raise CanonicalValidationError(
                f"row {row_number}: $.generation_index and $.run_index disagree"
            )
        if generation is None:
            generation = run_index
        grade = optional_index("grade_index")
        environment = optional_index("environment_index")
        run_id = row.get("run_id", "") or None
        if run_id is not None or generation is not None or grade is not None or environment is not None:
            has_repeat_structure = True
        if run_id is not None:
            if run_id in run_id_generations:
                if generation is not None and generation != run_id_generations[run_id]:
                    raise CanonicalValidationError(
                        f"row {row_number}: $.run_id {run_id!r} was already mapped "
                        f"to generation {run_id_generations[run_id]}"
                    )
                generation = run_id_generations[run_id]
            else:
                if generation is None:
                    generation = next_generation
                run_id_generations[run_id] = generation
                next_generation = max(next_generation, generation + 1)
        parsed = {
            "row_number": row_number,
            "task_id": task_id,
            "score": score,
            "status": status,
            "model_id": model_id,
            "generation_index": generation if generation is not None else 0,
            "grade_index": grade if grade is not None else 0,
            "environment_index": environment if environment is not None else 0,
        }
        rows.append(parsed)
        if task_id not in seen_tasks:
            seen_tasks.add(task_id)
            task_order.append(task_id)

    if not rows:
        raise CanonicalValidationError("line 1: CSV contains no data rows")

    attempts = [
        Attempt(
            task_id=str(row["task_id"]),
            status=str(row["status"]),
            score=float(row["score"]),
            model_id=row["model_id"],
        )
        for row in rows
        if row["status"] is not None
    ]
    if "status" not in headers:
        unsupported.append(
            f"{source_name}: no status column; scores do not state pass/fail, "
            "so no attempts were created"
        )

    run_scores: list[RunItemScore] = []
    model_ids = {row["model_id"] for row in rows if row["model_id"] is not None}
    if not has_repeat_structure:
        unsupported.append(
            f"{source_name}: no run_id or condition index column states "
            "repeated-run structure, so no run scores were created"
        )
    elif len(model_ids) > 1:
        unsupported.append(
            f"{source_name}: multiple model_id values are present; run scores "
            "were not created because one NOISE series cannot represent "
            "several models"
        )
    else:
        seen_conditions: dict[tuple[str, int, int, int], int] = {}
        for row in rows:
            key = (
                str(row["task_id"]),
                int(row["generation_index"]),
                int(row["grade_index"]),
                int(row["environment_index"]),
            )
            if key in seen_conditions:
                raise CanonicalValidationError(
                    f"row {row['row_number']}: duplicate score for task "
                    f"{key[0]!r} under the same run conditions as row "
                    f"{seen_conditions[key]}; the CSV contract cannot "
                    "distinguish the two observations"
                )
            seen_conditions[key] = int(row["row_number"])
            run_scores.append(
                RunItemScore(
                    task_id=key[0],
                    score=float(row["score"]),
                    generation_index=key[1],
                    grade_index=key[2],
                    environment_index=key[3],
                )
            )

    return IntegrityModel(
        eval_id=eval_id,
        adapter_name=CANONICAL_ADAPTER_NAME,
        adapter_version=CANONICAL_ADAPTER_VERSION,
        tasks=[TaskSample(id=task_id, prompt="") for task_id in task_order],
        attempts=attempts,
        run_scores=run_scores,
        unsupported=unsupported,
    )


__all__ = [
    "CANONICAL_ADAPTER_NAME",
    "CANONICAL_ADAPTER_VERSION",
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "CanonicalValidationError",
    "dumps_canonical",
    "model_from_csv",
    "model_from_document",
    "model_from_json",
    "model_from_jsonl",
    "model_to_document",
]
