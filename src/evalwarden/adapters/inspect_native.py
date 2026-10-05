"""Native Inspect AI ``.eval`` log translation.

This module is the native half of the Inspect adapter. It translates the
JSON documents inside a real Inspect ``.eval`` archive (schema versions 1
and 2) into the framework-neutral integrity model.

Only facts the log itself records are mapped:

- ``header.json`` supplies the eval/task/model identity and scorer list;
- ``samples/*.json`` supplies samples, epochs, outputs, usage, scores, and
  events;
- tool events become trajectory spans, with the native call id as the
  step id and the tool result truncated at the same 512-character
  boundary as the JSON span wiring;
- repeated epochs of one selected score become ``run_scores`` with the
  epoch as the generation index. No judge or environment axis is
  fabricated, and no downstream consumption edge is inferred, so
  TRAJ-002 stays silent on native logs by design.

The archive has already been read into memory by
:mod:`evalwarden.adapters._inspect_archive`; nothing here touches disk.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from ..model import (
    Attempt,
    Environment,
    Grader,
    IntegrityModel,
    Judgment,
    RunItemScore,
    TaskSample,
    TrajectoryStep,
)
from . import AuditError

NATIVE_SCHEMA_VERSIONS = {1, 2}
MAX_SPAN_OUTPUT_CHARS = 512  # same truncation boundary as the JSON span wiring
_ATTACHMENT_PREFIX = "attachment://"

# Inspect categorical score values (inspect_ai.scorer): correct, partial,
# incorrect, and no-answer. Numeric strings are parsed as numbers first.
_CATEGORY_SCORES = {
    "C": 1.0,
    "CORRECT": 1.0,
    "PASS": 1.0,
    "TRUE": 1.0,
    "P": 0.5,
    "PARTIAL": 0.5,
    "I": 0.0,
    "INCORRECT": 0.0,
    "FAIL": 0.0,
    "FALSE": 0.0,
    "N": 0.0,
    "NOANSWER": 0.0,
    "NO_ANSWER": 0.0,
}


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _json_member(members: dict[str, bytes], name: str, path: Path) -> Any:
    raw = members.get(name)
    if raw is None:
        raise AuditError(f"{path}: native Inspect log is missing {name}")
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"{path}: invalid JSON in archive member {name}: {exc}") from exc


def _require_dict(value: Any, *, path: Path, member: str, field: str) -> dict:
    if not isinstance(value, dict):
        raise AuditError(f"{path}: {member} field {field!r} must be an object")
    return value


def _resolve_attachment(text: str, attachments: dict) -> str:
    if text.startswith(_ATTACHMENT_PREFIX):
        resolved = attachments.get(text[len(_ATTACHMENT_PREFIX):])
        if isinstance(resolved, str):
            return resolved
    return text


def _content_text(content: Any, attachments: dict, flags: set[str]) -> str:
    """Text of an Inspect content value. Non-text parts are omitted and
    flagged once by the caller through ``flags``."""
    if content is None:
        return ""
    if isinstance(content, str):
        return _resolve_attachment(content, attachments)
    if isinstance(content, list):
        return "".join(_content_text(item, attachments, flags) for item in content)
    if isinstance(content, dict):
        content_type = content.get("type")
        if content_type in (None, "text") and isinstance(content.get("text"), str):
            return _resolve_attachment(content["text"], attachments)
        if isinstance(content.get("content"), (str, list)):
            return _content_text(content["content"], attachments, flags)
        flags.add("non_text_content")
        return ""
    flags.add("non_text_content")
    return ""


def _prompt_text(sample: dict, flags: set[str]) -> str:
    attachments = sample.get("attachments") or {}
    raw = sample.get("input")
    if isinstance(raw, str):
        return _resolve_attachment(raw, attachments)
    if isinstance(raw, list):
        parts: list[str] = []
        for message in raw:
            if isinstance(message, dict):
                text = _content_text(message.get("content"), attachments, flags)
            else:
                text = _content_text(message, attachments, flags)
            if text:
                parts.append(text)
        return "\n".join(parts)
    if raw is None:
        return ""
    flags.add("non_text_content")
    return ""


def _target_value(sample: dict, *, path: Path, member: str) -> str | list[str] | None:
    if "target" not in sample:
        raise AuditError(f"{path}: {member} is missing the required 'target' field")
    target = sample["target"]
    if target is None or isinstance(target, str):
        return target
    if isinstance(target, list):
        return [str(item) for item in target]
    return str(target)


def _score_number(value: Any) -> float | None:
    """Scalar value of an Inspect score, or None when it has none.

    Booleans and numbers map directly. Strings map through Inspect's
    categorical score vocabulary (C/P/I/N) or a numeric parse; arbitrary
    category labels and structured values have no honest scalar.
    """
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if _is_number(value):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        try:
            number = float(text)
        except ValueError:
            return _CATEGORY_SCORES.get(text.upper())
        return number if math.isfinite(number) else None
    return None


def _score_status(number: float | None, sample: dict) -> str:
    if sample.get("error"):
        return "error"
    if sample.get("limit"):
        return "incomplete"
    if number is None:
        return "incomplete"
    return "pass" if number >= 0.5 else "fail"


def _tool_result_text(event: dict, attachments: dict, flags: set[str]) -> str | None:
    result = event.get("result")
    if result is not None:
        return _content_text(result, attachments, flags)
    error = event.get("error")
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return error["message"]
    if isinstance(error, str):
        return error
    return None


def _spans_for_sample(
    sample: dict, *, path: Path, member: str, flags: set[str]
) -> list[TrajectoryStep]:
    events = sample.get("events")
    if not isinstance(events, list):
        raise AuditError(f"{path}: {member} field 'events' must be a list")
    attachments = sample.get("attachments") or {}
    spans: list[TrajectoryStep] = []
    seen_ids: set[str] = set()
    truncated = 0
    for event in events:
        if not isinstance(event, dict) or event.get("event") != "tool":
            continue
        function = event.get("function")
        if not isinstance(function, str) or not function:
            raise AuditError(f"{path}: {member} has a tool event with no function name")
        arguments = event.get("arguments")
        if not isinstance(arguments, dict):
            raise AuditError(
                f"{path}: {member} tool event {function!r} arguments must be an object"
            )
        step_id = str(event.get("id") or event.get("uuid") or f"s{len(spans)}")
        if step_id in seen_ids:
            raise AuditError(
                f"{path}: {member} repeats tool call id {step_id!r}; "
                "trajectory step ids must be unique"
            )
        seen_ids.add(step_id)
        output = _tool_result_text(event, attachments, flags)
        if output is not None and len(output) > MAX_SPAN_OUTPUT_CHARS:
            output = output[:MAX_SPAN_OUTPUT_CHARS]
            truncated += 1
        spans.append(
            TrajectoryStep(
                step_id=step_id,
                tool=function,
                args=dict(arguments),
                output=output,
                consumes=[],  # native logs record no data-flow edges; never inferred
            )
        )
    if truncated:
        flags.add("truncated_outputs")
    return spans


def _score_entries(header: dict) -> list[dict]:
    results = header.get("results") or {}
    entries = results.get("scores") or []
    return [entry for entry in entries if isinstance(entry, dict)]


def _primary_score_name(header: dict, spec: dict) -> str | None:
    """The one score series attempts and run_scores may use.

    The log's declared headline score wins. Otherwise a single score
    series is unambiguous. With several series and no headline there is
    no honest primary, and none is selected.
    """
    results = header.get("results") or {}
    headline = results.get("headline") or spec.get("headline_metric") or {}
    if isinstance(headline, dict):
        named = headline.get("score")
        if isinstance(named, str) and named:
            return named
    entries = _score_entries(header)
    if len(entries) == 1 and isinstance(entries[0].get("name"), str):
        return entries[0]["name"]
    return None


def _scorer_name_for(header: dict, score_name: str | None) -> str | None:
    if score_name is None:
        return None
    for entry in _score_entries(header):
        if entry.get("name") == score_name and isinstance(entry.get("scorer"), str):
            return entry["scorer"]
    return score_name


def _all_scorer_names(header: dict, spec: dict) -> list[str]:
    names: list[str] = []
    for scorer in spec.get("scorers") or []:
        if isinstance(scorer, dict) and isinstance(scorer.get("name"), str):
            if scorer["name"] not in names:
                names.append(scorer["name"])
    for entry in _score_entries(header):
        scorer = entry.get("scorer")
        if isinstance(scorer, str) and scorer not in names:
            names.append(scorer)
    return names


def _is_model_scorer(name: str) -> bool:
    lowered = name.lower()
    return "model_graded" in lowered or lowered in {"model_graded_qa", "model_graded_fact"}


def _grader_model(spec: dict) -> str | None:
    roles = spec.get("model_roles") or {}
    grader_role = roles.get("grader") if isinstance(roles, dict) else None
    role_entries = grader_role if isinstance(grader_role, list) else [grader_role]
    for entry in role_entries:
        if isinstance(entry, dict) and isinstance(entry.get("model"), str):
            return entry["model"]
    for scorer in spec.get("scorers") or []:
        if isinstance(scorer, dict) and _is_model_scorer(str(scorer.get("name", ""))):
            options = scorer.get("options") or {}
            if isinstance(options.get("model"), str):
                return options["model"]
    return None


def _grader(spec: dict, header: dict) -> Grader:
    scorer_names = _all_scorer_names(header, spec)
    model_scorers = [name for name in scorer_names if _is_model_scorer(name)]
    kind = "judge" if model_scorers else "script"

    temperature: float | None = None
    rubric: list[str] = []
    scale_anchors: dict[str, str] = {}
    for scorer in spec.get("scorers") or []:
        if not isinstance(scorer, dict) or scorer.get("name") not in model_scorers:
            continue
        options = scorer.get("options") or {}
        if _is_number(options.get("temperature")):
            temperature = float(options["temperature"])
        for key in ("criteria", "criterion", "rubric"):
            value = options.get(key)
            if isinstance(value, str) and value:
                rubric.append(value)
            elif isinstance(value, list):
                rubric.extend(str(item) for item in value if isinstance(item, str))
        anchors = options.get("scale_anchors") or options.get("scale")
        if isinstance(anchors, dict):
            scale_anchors.update({str(k): str(v) for k, v in anchors.items()})

    return Grader(
        kind=kind,
        tests=[str(entry.get("name")) for entry in _score_entries(header) if entry.get("name")],
        judge_model=_grader_model(spec) if kind == "judge" else None,
        temperature=temperature,
        rubric_criteria=rubric,
        scale_anchors=scale_anchors,
    )


def _explicit_reference_labels(
    header: dict, samples: list[dict]
) -> dict[str, str]:
    """Reference labels only from fields that declare themselves labels.

    A sample target is the answer key the scorer consumed; it is not, by
    itself, a human calibration label set, so targets are never copied
    here.
    """
    labels: dict[str, str] = {}
    for container in (header.get("eval", {}).get("metadata") or {}, header.get("metadata") or {}):
        if isinstance(container, dict):
            declared = container.get("reference_labels")
            if isinstance(declared, dict):
                labels.update({str(k): str(v) for k, v in declared.items()})
    for sample in samples:
        task_id = str(sample.get("id"))
        metadata = sample.get("metadata") or {}
        if isinstance(metadata, dict) and "reference_label" in metadata:
            value = metadata["reference_label"]
            if isinstance(value, (str, int, float, bool)):
                labels[task_id] = str(value)
        for score in (sample.get("scores") or {}).values():
            if not isinstance(score, dict):
                continue
            score_meta = score.get("metadata") or {}
            if isinstance(score_meta, dict) and "reference_label" in score_meta:
                value = score_meta["reference_label"]
                if isinstance(value, (str, int, float, bool)):
                    labels.setdefault(task_id, str(value))
    return labels


def _judgments(
    samples: list[dict], header: dict, spec: dict
) -> list[Judgment]:
    """Judgments only where a score record is itself a structured verdict.

    A dict-valued score is a candidate->score map: that is a judgment in
    the model's sense. Scalar scores are attempt scores, not judgments,
    and are never re-dressed as one. ``judge_id`` is the producing
    scorer's name, set only when the log used more than one scorer.
    """
    scorer_names = _all_scorer_names(header, spec)
    multiple = len(scorer_names) > 1
    judgments: list[Judgment] = []
    for sample in samples:
        task_id = str(sample.get("id"))
        for score_name, score in (sample.get("scores") or {}).items():
            if not isinstance(score, dict):
                continue
            value = score.get("value")
            metadata = score.get("metadata") or {}
            scorer = _scorer_name_for(header, str(score_name))
            judge_id = scorer if multiple and scorer else None
            confidence = metadata.get("confidence") if isinstance(metadata, dict) else None
            confidence = (
                float(confidence)
                if _is_number(confidence) and 0.0 <= float(confidence) <= 1.0
                else None
            )
            if isinstance(value, dict) and value and all(
                _is_number(item) for item in value.values()
            ):
                judgments.append(
                    Judgment(
                        task_id=task_id,
                        candidates=sorted(str(key) for key in value),
                        scores={str(key): float(item) for key, item in value.items()},
                        confidence=confidence,
                        judge_id=judge_id,
                    )
                )
            elif (
                isinstance(metadata, dict)
                and isinstance(metadata.get("winner"), str)
                and isinstance(metadata.get("candidates"), list)
            ):
                candidates = [str(item) for item in metadata["candidates"]]
                order = metadata.get("presentation_order")
                judgments.append(
                    Judgment(
                        task_id=task_id,
                        candidates=sorted(candidates),
                        presentation_order=(
                            [str(item) for item in order]
                            if isinstance(order, list)
                            else candidates
                        ),
                        winner=metadata["winner"],
                        confidence=confidence,
                        judge_id=judge_id,
                    )
                )
    return judgments


def normalize_native_log(
    bundle: dict,
    *,
    adapter_name: str,
    adapter_version: str,
) -> IntegrityModel:
    """Translate one parsed native ``.eval`` archive into the model."""
    path: Path = bundle["native_eval"]
    members: dict[str, bytes] = bundle["native_members"]
    filename = path.name

    header = _json_member(members, "header.json", path)
    if not isinstance(header, dict):
        raise AuditError(f"{path}: header.json must contain a JSON object")
    version = header.get("version")
    if version not in NATIVE_SCHEMA_VERSIONS:
        raise AuditError(
            f"{path}: unsupported native Inspect log schema version {version!r} "
            f"(supported: {sorted(NATIVE_SCHEMA_VERSIONS)})"
        )
    status = header.get("status")
    if status != "success":
        raise AuditError(
            f"{path}: native Inspect log status is {status!r}; only a "
            "completed (success) log can be audited, not a partial one"
        )
    spec = _require_dict(header.get("eval"), path=path, member="header.json", field="eval")
    task_name = spec.get("task")
    model_name = spec.get("model")
    if not isinstance(task_name, str) or not task_name:
        raise AuditError(f"{path}: header.json eval.task is missing")
    if not isinstance(model_name, str) or not model_name:
        raise AuditError(f"{path}: header.json eval.model is missing")
    results = header.get("results")
    if not isinstance(results, dict):
        raise AuditError(f"{path}: header.json has no results object (log not finished)")

    start_raw = members.get("_journal/start.json")
    if start_raw is not None:
        start = _json_member(members, "_journal/start.json", path)
        if isinstance(start, dict):
            if start.get("version") not in (None, version):
                raise AuditError(f"{path}: journal version disagrees with header.json")
            start_spec = start.get("eval") or {}
            if isinstance(start_spec, dict) and start_spec.get("eval_id") not in (
                None,
                spec.get("eval_id"),
            ):
                raise AuditError(f"{path}: journal eval_id disagrees with header.json")

    sample_members = sorted(
        name
        for name in members
        if name.startswith("samples/") and name.endswith(".json")
    )
    if not sample_members:
        raise AuditError(
            f"{path}: native Inspect log records no full samples "
            "(it may have been written with sample logging disabled)"
        )
    samples: list[tuple[str, dict]] = []
    for member in sample_members:
        sample = _json_member(members, member, path)
        if not isinstance(sample, dict):
            raise AuditError(f"{path}: {member} must contain a JSON object")
        if sample.get("id") is None:
            raise AuditError(f"{path}: {member} is missing the required 'id' field")
        epoch = sample.get("epoch")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
            raise AuditError(f"{path}: {member} field 'epoch' must be a positive integer")
        if not isinstance(sample.get("metadata") or {}, dict):
            raise AuditError(f"{path}: {member} field 'metadata' must be an object")
        samples.append((member, sample))

    summaries_raw = members.get("summaries.json")
    if summaries_raw is not None:
        summaries = _json_member(members, "summaries.json", path)
        if isinstance(summaries, list) and len(summaries) != len(samples):
            raise AuditError(
                f"{path}: summaries.json lists {len(summaries)} samples but "
                f"{len(samples)} sample files are present (log is inconsistent)"
            )
    total_samples = results.get("total_samples")
    if isinstance(total_samples, int) and total_samples != len(samples):
        raise AuditError(
            f"{path}: header results claim {total_samples} samples but "
            f"{len(samples)} sample files are present (log is incomplete)"
        )
    dataset = spec.get("dataset") or {}
    declared_ids = dataset.get("sample_ids") if isinstance(dataset, dict) else None
    if isinstance(declared_ids, list):
        present_ids = {str(sample.get("id")) for _, sample in samples}
        if {str(item) for item in declared_ids} != present_ids:
            raise AuditError(
                f"{path}: sample files do not match the dataset sample_ids "
                "declared in header.json (log is incomplete)"
            )

    samples.sort(key=lambda item: (str(item[1].get("id")), int(item[1].get("epoch", 1))))
    seen_epochs: set[tuple[str, int]] = set()
    for member, sample in samples:
        key = (str(sample["id"]), int(sample["epoch"]))
        if key in seen_epochs:
            raise AuditError(f"{path}: duplicate sample {key[0]!r} epoch {key[1]} ({member})")
        seen_epochs.add(key)

    flags: set[str] = set()
    tasks: list[TaskSample] = []
    task_signatures: dict[str, tuple[str, str, str]] = {}
    attempts: list[Attempt] = []
    primary_name = _primary_score_name(header, spec)
    epochs_seen: set[int] = set()
    any_total_cost = False
    any_explanations = False
    structured_primary = False

    for member, sample in samples:
        task_id = str(sample["id"])
        epoch = int(sample["epoch"])
        epochs_seen.add(epoch)
        prompt = _prompt_text(sample, flags)
        target = _target_value(sample, path=path, member=member)
        choices_raw = sample.get("choices")
        if choices_raw is not None and not isinstance(choices_raw, list):
            raise AuditError(f"{path}: {member} field 'choices' must be a list")
        choices = [str(item) for item in choices_raw] if choices_raw else None
        signature = (
            prompt,
            json.dumps(target, sort_keys=True),
            json.dumps(choices, sort_keys=True),
        )
        metadata = dict(sample.get("metadata") or {})
        if task_id not in task_signatures:
            task_signatures[task_id] = signature
            failure_mode = metadata.get("failure_mode")
            tasks.append(
                TaskSample(
                    id=task_id,
                    prompt=prompt,
                    target=target,
                    choices=choices,
                    failure_mode=(
                        failure_mode
                        if isinstance(failure_mode, str) and failure_mode
                        else None
                    ),
                    metadata=metadata,
                )
            )
        elif task_signatures[task_id] != signature:
            raise AuditError(
                f"{path}: sample {task_id!r} changes input/target/choices "
                "between epochs; the model cannot represent that"
            )

        scores = sample.get("scores") or {}
        if not isinstance(scores, dict):
            raise AuditError(f"{path}: {member} field 'scores' must be an object")
        for score in scores.values():
            if isinstance(score, dict) and score.get("explanation"):
                any_explanations = True
        primary_score = scores.get(primary_name) if primary_name else None
        number: float | None = None
        if isinstance(primary_score, dict):
            number = _score_number(primary_score.get("value"))
            if number is None and isinstance(
                primary_score.get("value"), (dict, list)
            ):
                structured_primary = True
        status = _score_status(number, sample)

        usage = sample.get("model_usage") or {}
        tokens_in: int | None = None
        tokens_out: int | None = None
        if isinstance(usage, dict) and usage:
            tokens_in = 0
            tokens_out = 0
            for entry in usage.values():
                if not isinstance(entry, dict):
                    continue
                tokens_in += int(entry.get("input_tokens") or 0)
                tokens_out += int(entry.get("output_tokens") or 0)
                if entry.get("total_cost") is not None:
                    any_total_cost = True
        else:
            output_usage = (sample.get("output") or {}).get("usage")
            if isinstance(output_usage, dict):
                tokens_in = int(output_usage.get("input_tokens") or 0)
                tokens_out = int(output_usage.get("output_tokens") or 0)
                if output_usage.get("total_cost") is not None:
                    any_total_cost = True
        latency = sample.get("total_time")
        if not _is_number(latency):
            latency = sample.get("working_time")
        retries = sample.get("error_retries") or []
        spans = _spans_for_sample(sample, path=path, member=member, flags=flags)
        output = sample.get("output") or {}
        completion = output.get("completion") if isinstance(output, dict) else None
        attempts.append(
            Attempt(
                task_id=task_id,
                status=status,
                score=number,
                tool_calls=len(spans),
                actions=[span.tool for span in spans],
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                latency_s=float(latency) if _is_number(latency) else None,
                tries=1 + (len(retries) if isinstance(retries, list) else 0),
                empty_submission=bool(
                    status == "pass"
                    and isinstance(completion, str)
                    and not completion.strip()
                ),
                model_id=model_name,
                spans=spans,
            )
        )

    run_scores: list[RunItemScore] = []
    if primary_name is not None and len(epochs_seen) >= 2:
        sample_by_key = {
            (str(sample["id"]), int(sample["epoch"])): sample for _, sample in samples
        }
        per_task_epochs: dict[str, list[int]] = {}
        for _, sample in samples:
            per_task_epochs.setdefault(str(sample["id"]), []).append(int(sample["epoch"]))
        attempt_index: dict[str, int] = {}
        for attempt in attempts:
            task_epochs = sorted(per_task_epochs[attempt.task_id])
            position = attempt_index.get(attempt.task_id, 0)
            attempt_index[attempt.task_id] = position + 1
            epoch = task_epochs[position]
            sample = sample_by_key[(attempt.task_id, epoch)]
            score = (sample.get("scores") or {}).get(primary_name)
            if isinstance(score, dict):
                value = _score_number(score.get("value"))
                if value is not None:
                    run_scores.append(
                        RunItemScore(
                            task_id=attempt.task_id,
                            score=value,
                            generation_index=epoch - 1,
                        )
                    )

    grader = _grader(spec, header)
    grader.reference_labels = _explicit_reference_labels(
        header, [sample for _, sample in samples]
    )
    judgments = _judgments([sample for _, sample in samples], header, spec)

    unsupported: list[str] = []
    if any(attempt.spans for attempt in attempts):
        unsupported.append(
            f"{filename}: ToolEvent records carry no downstream consumption "
            "edges; TRAJ-002 cannot be assessed on native logs"
        )
    if "non_text_content" in flags:
        unsupported.append(
            f"{filename}: non-text sample/tool content was omitted at the "
            "adapter boundary (text only is kept)"
        )
    if "truncated_outputs" in flags:
        unsupported.append(
            f"{filename}: span output(s) truncated to {MAX_SPAN_OUTPUT_CHARS} "
            "chars at the adapter boundary"
        )
    if any_explanations:
        unsupported.append(
            f"{filename}: scorer explanations redacted (raw scorer text is "
            "not stored; structured verdicts only)"
        )
    if any_total_cost:
        unsupported.append(
            f"{filename}: the log records per-model total_cost, which the "
            "model has no per-attempt field for; cost checks use token "
            "estimates instead"
        )
    if structured_primary:
        unsupported.append(
            f"{filename}: the headline scorer returns structured scores, "
            "so no scalar attempt score was fabricated from them"
        )
    if primary_name is None and len(_all_scorer_names(header, spec)) > 1:
        unsupported.append(
            f"{filename}: several scorers and no headline score declared; "
            "no primary score series was selected"
        )
    if isinstance(spec.get("sandbox"), dict):
        unsupported.append(
            f"{filename}: eval.sandbox is a sandbox spec, not an "
            "agent-visible environment record; ENV-001 sees no native "
            "environment"
        )

    aliases = {
        canonical: filename
        for canonical in (
            "tasks.json",
            "attempts.json",
            "run.json",
            "environment.json",
            "grader.json",
            "judge_run.json",
            "run_scores.json",
            "trajectories.jsonl",
        )
    }
    return IntegrityModel(
        eval_id=task_name,
        adapter_name=adapter_name,
        adapter_version=adapter_version,
        tasks=tasks,
        environment=Environment(),
        grader=grader,
        attempts=attempts,
        judgments=judgments,
        run_scores=run_scores,
        unsupported=unsupported,
        digests=dict(bundle.get("digests", {})),
        file_aliases=aliases,
    )
