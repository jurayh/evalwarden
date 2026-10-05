"""Inspect AI adapter (v0.3).

Reads Inspect eval artifacts in two forms:

1. A real Inspect AI `.eval` log (a ZIP archive of JSON documents:
   `header.json`, `samples/<id>_epoch_<n>.json`, `summaries.json`),
   pointed to directly or as the single `.eval` file in a directory.
   Native logs are parsed in memory by
   :mod:`evalwarden.adapters.inspect_native`; nothing is extracted to
   disk. See that module for the field mapping and its limits (notably:
   ToolEvent records carry no consumption edges, so TRAJ-002 stays
   silent on native logs).
2. An Inspect-style eval artifact directory -- the normalized input
   format, modeled on Inspect's Task / dataset / scorer / .eval-log
   concepts:

    eval-artifact/
      dataset.json       samples: [{id, prompt, metadata}]
      environment.json   env vars visible to the solver, mounts
      grader.json        verifier config, pass conditions, tests
      run.json           per-task attempts with status, usage, actions
      judge_run.json     (optional) model-judge config, judgments, reference labels
      trajectories.jsonl (optional) per-span tool-call records (see below)

Field wiring: the adapter populates the integrity model's extended fields
only from data the artifact actually records. Nothing is invented.

- Trajectory spans come from `trajectories.jsonl`, one JSON object per
  line: {"task_id", "tool", "args", "output", "consumes", "step_id"}.
  `output` is truncated at the boundary (MAX_SPAN_OUTPUT_CHARS). `consumes`
  is the recorded data-flow: step_ids whose outputs this step used. It is
  carried verbatim, never inferred -- when an artifact records no
  consumption edges, the trajectory checks that need them stay silent.
- Judgment.judge_id comes from a per-judgment `judge_id` (the scorer
  identity, for multi-judge panels); `confidence` from a per-judgment
  numeric `confidence` in [0, 1]. Neither is synthesized when absent.
- TaskSample.failure_mode comes from a task's own `failure_mode` field or
  its `metadata.failure_mode` -- an explicit tag in the data, never a
  guess. A dataset-level `failure_mode_taxonomy` list declares the modes
  the eval claims to cover.
- RunItemScore rows come from `run.json:run_scores`, where each row tags
  which run conditions were re-rolled (generation / grade / environment
  indices). `run.json:claimed_delta` is the eval's own claimed score
  change, carried only when the artifact states one.

This is a read-only translation layer. The adapter pins the schema
versions it understands and fails clearly on anything else.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..model import (
    Attempt,
    Confidence,
    Environment,
    Grader,
    IntegrityModel,
    Judgment,
    Mount,
    RunItemScore,
    TaskSample,
    TrajectoryStep,
)
from . import AuditError, register
from ._inspect_archive import read_archive_members
from .inspect_native import normalize_native_log

ADAPTER_NAME = "inspect"
ADAPTER_VERSION = "0.3.0"
SCHEMA_VERSION = "evalwarden-artifact-v1"
MAX_SPAN_OUTPUT_CHARS = 512  # tool results are truncated at the boundary

_SPAN_KEYS = {"task_id", "step_id", "tool", "args", "output", "consumes"}


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AuditError(f"missing required file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise AuditError(f"invalid JSON in {path}: {exc}") from exc


def _read_jsonl(path: Path) -> list[dict]:
    """Parse a JSONL span file. Malformed lines fail clearly, with the line."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise AuditError(f"missing required file: {path}") from exc
    records: list[dict] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AuditError(f"invalid JSON in {path} line {lineno}: {exc}") from exc
        if not isinstance(record, dict):
            raise AuditError(f"{path} line {lineno}: span record must be a JSON object")
        records.append(record)
    return records


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _failure_mode(task: dict) -> str | None:
    """Explicit failure-mode tag: the task's own field, else its metadata.

    Adapters never invent tags; an eval that does not tag its items leaves
    the coverage checks nothing to slice by, and they stay silent.
    """
    direct = task.get("failure_mode")
    if isinstance(direct, str) and direct:
        return direct
    metadata = task.get("metadata")
    if isinstance(metadata, dict):
        tagged = metadata.get("failure_mode")
        if isinstance(tagged, str) and tagged:
            return tagged
    return None


def _confidence(value: Any) -> float | None:
    """A recorded judge confidence: numeric, in [0, 1]. Anything else is absent."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if 0.0 <= value <= 1.0 else None


def _normalize_spans(
    records: list[dict], unsupported: list[str]
) -> dict[str, list[TrajectoryStep]]:
    """Group span records by task_id, in recorded order.

    `consumes` edges are carried verbatim from the record: the harness
    recorded the data flow, the adapter only translates it. Outputs are
    truncated at the boundary; raw tool text beyond the cap never enters
    the model.
    """
    spans_by_task: dict[str, list[TrajectoryStep]] = {}
    truncated = 0
    unknown_keys: set[str] = set()
    for i, record in enumerate(records):
        unknown_keys.update(set(record) - _SPAN_KEYS)
        task_id = record.get("task_id")
        tool = record.get("tool")
        if not task_id or not tool:
            raise AuditError(
                f"trajectories.jsonl record {i + 1}: 'task_id' and 'tool' are required"
            )
        args = record.get("args") or {}
        if not isinstance(args, dict):
            args = {"value": args}
        output = record.get("output")
        if output is not None and not isinstance(output, str):
            output = str(output)
        if output is not None and len(output) > MAX_SPAN_OUTPUT_CHARS:
            output = output[:MAX_SPAN_OUTPUT_CHARS]
            truncated += 1
        steps = spans_by_task.setdefault(str(task_id), [])
        steps.append(
            TrajectoryStep(
                # Step ids are adapter-local identifiers: taken from the
                # record when present, else assigned in recorded order.
                step_id=str(record.get("step_id") or f"s{len(steps)}"),
                tool=str(tool),
                args=dict(args),
                output=output,
                consumes=[str(c) for c in (record.get("consumes") or [])],
            )
        )
    for key in sorted(unknown_keys):
        unsupported.append(f"trajectories.jsonl:{key} (not part of the span schema; ignored)")
    if truncated:
        unsupported.append(
            f"trajectories.jsonl: {truncated} span output(s) truncated to "
            f"{MAX_SPAN_OUTPUT_CHARS} chars at the adapter boundary"
        )
    return spans_by_task


def _normalize_run_scores(run: dict) -> list[RunItemScore]:
    """Repeated-run per-item scores, with the re-rolled conditions tagged.

    Rows are carried as recorded; indices default to 0 (a single condition)
    when the artifact does not tag them, which leaves the noise estimators
    nothing to decompose -- they report total variance or stay silent.
    """
    scores: list[RunItemScore] = []
    for i, row in enumerate(run.get("run_scores") or []):
        if not isinstance(row, dict):
            raise AuditError(f"run.json: run_scores[{i}] must be an object")
        score = row.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise AuditError(f"run.json: run_scores[{i}].score must be a number")
        indices: dict[str, int] = {}
        for field in ("generation_index", "grade_index", "environment_index"):
            value = row.get(field, 0)
            if isinstance(value, bool) or not isinstance(value, int):
                raise AuditError(f"run.json: run_scores[{i}].{field} must be an integer")
            indices[field] = value
        scores.append(
            RunItemScore(
                task_id=str(row.get("task_id", "")),
                score=float(score),
                generation_index=indices["generation_index"],
                grade_index=indices["grade_index"],
                environment_index=indices["environment_index"],
            )
        )
    return scores


def _native_eval_path(path: Path) -> Path | None:
    """The native ``.eval`` log for `path`, if there is exactly one.

    A file with the ``.eval`` suffix is a native log. A directory is a
    native log container only when it holds exactly one ``.eval`` file;
    several logs in one directory are separate evals, and merging them
    into one model would invent a single score out of many -- collect
    fails clearly instead.
    """
    if path.is_file():
        return path if path.suffix == ".eval" else None
    if path.is_dir():
        logs = sorted(p for p in path.glob("*.eval") if p.is_file())
        if len(logs) > 1:
            raise AuditError(
                f"{path}: directory contains {len(logs)} native Inspect "
                ".eval logs; audit one .eval file at a time "
                "(report-cards can batch several paths)"
            )
        if logs:
            return logs[0]
    return None


@register
class InspectAdapter:
    name = ADAPTER_NAME
    version = ADAPTER_VERSION

    def detect(self, path: Path) -> Confidence:
        if path.is_file():
            return Confidence.HIGH if path.suffix == ".eval" else Confidence.LOW
        if not path.is_dir():
            return Confidence.LOW
        if any(p.is_file() for p in path.glob("*.eval")):
            return Confidence.HIGH
        has_dataset = (path / "dataset.json").is_file()
        has_grader = (path / "grader.json").is_file()
        if has_dataset and has_grader:
            return Confidence.HIGH
        if has_dataset:
            return Confidence.MEDIUM
        return Confidence.LOW

    def collect(self, path: Path) -> dict:
        """Read-only: files are opened for reading and never modified."""
        native_path = _native_eval_path(path)
        if native_path is not None:
            # Native .eval log: parse the archive in memory. Members are
            # never extracted to disk, and the file itself is only read.
            return {
                "root": path,
                "native_eval": native_path,
                "native_members": read_archive_members(native_path),
                "digests": {native_path.name: _digest(native_path)},
            }
        bundle: dict[str, Any] = {"root": path}
        digests: dict[str, str] = {}
        for fname in (
            "dataset.json",
            "environment.json",
            "grader.json",
            "run.json",
            "judge_run.json",
        ):
            fpath = path / fname
            if fpath.is_file():
                bundle[fname] = _read_json(fpath)
                digests[fname] = _digest(fpath)
        traj_path = path / "trajectories.jsonl"
        if traj_path.is_file():
            bundle["trajectories.jsonl"] = _read_jsonl(traj_path)
            digests["trajectories.jsonl"] = _digest(traj_path)
        for fname in ("gold_map.json",):
            fpath = path / fname
            if fpath.is_file():
                # Recorded for the digest manifest only; never parsed for content
                # beyond its existence (it is the leak, not the evidence).
                digests[fname] = _digest(fpath)
                bundle.setdefault("extra_files", []).append(fname)
        bundle["digests"] = digests
        return bundle

    def normalize(self, bundle: dict) -> IntegrityModel:
        if "native_members" in bundle:
            return normalize_native_log(
                bundle, adapter_name=self.name, adapter_version=self.version
            )
        root: Path = bundle["root"]
        dataset = bundle.get("dataset.json", {})
        environment = bundle.get("environment.json", {})
        grader_cfg = bundle.get("grader.json", {})
        run = bundle.get("run.json", {})

        if dataset.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION and "tasks" not in dataset:
            raise AuditError(
                f"unsupported dataset schema (expected {SCHEMA_VERSION!r} with a 'tasks' list)"
            )

        tasks = [
            TaskSample(
                id=str(t.get("id", f"task-{i}")),
                prompt=str(t.get("prompt", "")),
                target=t.get("target"),
                choices=(
                    [str(c) for c in t["choices"]]
                    if isinstance(t.get("choices"), list)
                    else None
                ),
                failure_mode=_failure_mode(t),
                metadata=dict(t.get("metadata", {})),
            )
            for i, t in enumerate(dataset.get("tasks", []))
        ]
        taxonomy = [
            str(mode)
            for mode in (dataset.get("failure_mode_taxonomy") or [])
            if isinstance(mode, str) and mode
        ]

        env_vars = {
            str(name): "<redacted>"  # values are never stored; names are the signal
            for name in (environment.get("env") or {})
        }
        mounts = [
            Mount(
                path=str(m.get("path", "")),
                mode=str(m.get("mode", "ro")),
                agent_access=str(m.get("agent_access", "read")),
            )
            for m in (environment.get("mounts") or [])
        ]

        verifier = grader_cfg.get("verifier") or {}
        judge = grader_cfg.get("judge") or {}
        grader = Grader(
            kind=str(grader_cfg.get("kind", "script")),
            verifier_path=verifier.get("path"),
            verifier_writable_by_agent=bool(verifier.get("writable_by_agent", False)),
            accepts_empty_output=bool(grader_cfg.get("accepts_empty_output", False)),
            tests=[str(t) for t in (grader_cfg.get("tests") or [])],
            judge_model=judge.get("model"),
            judge_family=judge.get("family"),
            protocol=judge.get("protocol"),
            counterbalanced=judge.get("counterbalanced"),
            temperature=judge.get("temperature"),
            repeats=int(judge.get("repeats", 1)),
            rubric_criteria=[str(c) for c in (judge.get("rubric_criteria") or [])],
            scale_anchors={str(k): str(v) for k, v in (judge.get("scale_anchors") or {}).items()},
        )

        attempts = [
            Attempt(
                task_id=str(a.get("task_id", "")),
                status=str(a.get("status", "error")),
                score=a.get("score"),
                tool_calls=int(a.get("tool_calls", 0)),
                actions=[str(x) for x in (a.get("actions") or [])],
                tokens_in=a.get("tokens_in"),
                tokens_out=a.get("tokens_out"),
                latency_s=a.get("latency_s"),
                tries=int(a.get("tries", 1)),
                empty_submission=bool(a.get("empty_submission", False)),
            )
            for a in (run.get("attempts") or [])
        ]

        unsupported: list[str] = []

        # Trajectory spans attach to the attempt that produced them. When a
        # task somehow has several attempts, spans go to the first and the
        # attribution is reported rather than silently duplicated.
        spans_by_task = _normalize_spans(bundle.get("trajectories.jsonl", []), unsupported)
        attached: set[str] = set()
        for attempt in attempts:
            group = spans_by_task.get(attempt.task_id)
            if not group:
                continue
            if attempt.task_id in attached:
                unsupported.append(
                    f"trajectories.jsonl: spans for task {attempt.task_id} "
                    "attached to the first attempt only"
                )
                continue
            attempt.spans = group
            attached.add(attempt.task_id)
        for task_id in spans_by_task:
            if task_id not in attached:
                unsupported.append(
                    f"trajectories.jsonl: spans for task {task_id!r} have no recorded attempt"
                )

        run_scores = _normalize_run_scores(run)
        claimed = run.get("claimed_delta")
        if claimed is not None and (
            isinstance(claimed, bool) or not isinstance(claimed, (int, float))
        ):
            raise AuditError("run.json: claimed_delta must be a number")
        claimed_delta = float(claimed) if claimed is not None else None

        judge_run = bundle.get("judge_run.json", {})
        judgments: list[Judgment] = []
        for j in judge_run.get("judgments", []):
            candidates = sorted(str(c) for c in (j.get("candidates") or []))
            judgments.append(
                Judgment(
                    task_id=str(j.get("task_id", "")),
                    candidates=candidates,
                    presentation_order=[str(c) for c in (j.get("presentation_order") or candidates)],
                    winner=j.get("winner"),
                    scores={str(k): float(v) for k, v in (j.get("scores") or {}).items()},
                    lengths={str(k): int(v) for k, v in (j.get("lengths") or {}).items()},
                    repeat_index=int(j.get("repeat_index", 0)),
                    confidence=_confidence(j.get("confidence")),
                    judge_id=str(j["judge_id"]) if j.get("judge_id") else None,
                )
            )
        if any("rationale" in j for j in judge_run.get("judgments", [])):
            # Raw judge text is redacted at the boundary by design, not parsed.
            # Reported here so the redaction is explicit, not silent.
            unsupported.append("judge_run.json:judgments[].rationale (redacted: raw judge text not stored)")
        grader.reference_labels = {
            str(k): str(v) for k, v in (judge_run.get("reference_labels") or {}).items()
        }
        for fname, content in bundle.items():
            if fname in ("root", "digests", "extra_files"):
                continue
            if isinstance(content, dict):
                known = {
                    "dataset.json": {"schema_version", "eval_id", "tasks", "failure_mode_taxonomy"},
                    "environment.json": {"env", "mounts", "notes"},
                    "grader.json": {"kind", "verifier", "accepts_empty_output", "tests", "judge"},
                    "run.json": {"solver", "attempts", "notes", "run_scores", "claimed_delta"},
                    "judge_run.json": {"judge", "judgments", "reference_labels", "notes"},
                }.get(fname, set())
                for key in content:
                    if key not in known:
                        unsupported.append(f"{fname}:{key}")

        return IntegrityModel(
            eval_id=str(dataset.get("eval_id", root.name)),
            adapter_name=self.name,
            adapter_version=self.version,
            tasks=tasks,
            failure_mode_taxonomy=taxonomy,
            environment=Environment(env_vars=env_vars, mounts=mounts),
            grader=grader,
            attempts=attempts,
            judgments=judgments,
            run_scores=run_scores,
            claimed_delta=claimed_delta,
            unsupported=unsupported,
            digests=dict(bundle.get("digests", {})),
            # Checks cite canonical artifact names; point them at the real files.
            file_aliases={
                "tasks.json": "dataset.json",
                "attempts.json": "run.json",
                "run_scores.json": "run.json",
            },
        )
