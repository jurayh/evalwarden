"""Codex rollout adapter.

Reads OpenAI Codex CLI rollout files (JSONL, one ``{timestamp, type,
payload}`` record per line) from ``$CODEX_HOME/sessions/``:

    rollout-<ISO timestamp>-<session uuid>.jsonl

The input may be a single rollout file or a directory, which is searched
recursively for ``rollout-*.jsonl`` files. Each rollout becomes one
attempt whose task id is the session id.

Format traps, handled explicitly (all documented by independent parser
authors; see the 2026-10-05 trace-tooling scan):

- ``response_item`` lines are the authoritative stream. ``event_msg``
  lines are a parallel UI duplicate (``user_message``, ``agent_message``,
  ``patch_apply_end`` mirror content already in ``response_item``), so
  spans and prompts are built from ``response_item`` only -- counting
  both streams would double every tool call. ``event_msg`` is read for
  exactly one thing: ``token_count``.
- ``token_count`` carries *cumulative* session totals, not per-turn
  usage. Per-turn usage is the delta between successive snapshots; a
  verbatim repeat of a snapshot is a zero delta, not new usage. A
  snapshot whose totals decrease breaks the cumulative invariant and
  fails loudly. Attempt ``tokens_in`` / ``tokens_out`` are the final
  cumulative input / output totals (cached input is a subset of input,
  reasoning output a subset of output; both subsets are reported in
  ``unsupported``, never added on top).
- The session id is the UUID in the rollout *filename*. The
  ``session_meta`` payload id is a lineage root that can differ, and is
  never used as the session id (a mismatch is reported, not hidden).
- A turn's token delta is attributed to a span only when exactly one
  tool call was made since the previous snapshot. Deltas covering
  several calls stay unattributed rather than being split, which would
  fabricate per-step precision.
- The format records no consumption edges, so every span's ``consumes``
  stays empty and TRAJ-002 stays silent. Rollouts record no pass/fail
  outcome, so attempts are marked ``incomplete`` and COST-lane
  wasted-spend does not apply (see ``evalwarden.trace_waste``).

The parser is pinned to the record and payload types below and fails
loudly on unknown top-level or ``response_item`` payload types.
Unknown ``event_msg`` payload types are counted and reported instead:
that stream is ignored by design, so an unfamiliar UI event loses no
trajectory data.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from ..model import (
    Attempt,
    Confidence,
    Environment,
    Grader,
    IntegrityModel,
    TaskSample,
    TrajectoryStep,
)
from . import AuditError, register
from ._io import digest_file, parse_ts, read_jsonl

ADAPTER_NAME = "codex"
ADAPTER_VERSION = "0.1.0"
MAX_SPAN_OUTPUT_CHARS = 512  # same truncation boundary as the other adapters

KNOWN_RECORD_TYPES = {"session_meta", "turn_context", "response_item", "event_msg"}
KNOWN_RESPONSE_ITEM_TYPES = {
    "message",
    "reasoning",
    "function_call",
    "function_call_output",
    "custom_tool_call",
    "custom_tool_call_output",
    # Observed in real rollouts (AletheiaResearch corpus pass, Oct 2026):
    # tool_schema records are tool *definitions* interleaved in the
    # stream -- counted and skipped, never spans. The search and image
    # call types are genuine calls and become spans; tool_search pairs
    # with its output by call_id, while web_search and image_generation
    # record no output in this stream.
    "tool_schema",
    "tool_search_call",
    "tool_search_output",
    "web_search_call",
    "image_generation_call",
}
KNOWN_EVENT_MSG_TYPES = {
    "user_message",
    "agent_message",
    "agent_reasoning",
    "token_count",
    "task_started",
    "task_complete",
    "turn_aborted",
    "patch_apply_begin",
    "patch_apply_end",
    "exec_command_begin",
    "exec_command_end",
    "error",
    "warning",
    "context_compacted",
    "thread_rolled_back",
}

_SESSION_ID_RE = re.compile(
    r"([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})\.jsonl$"
)


def _session_id_from_filename(path: Path) -> str:
    match = _SESSION_ID_RE.search(path.name)
    if not match:
        raise AuditError(
            f"{path}: cannot determine the Codex session id: rollout "
            "filenames end with the session UUID "
            "(rollout-<timestamp>-<uuid>.jsonl), and the filename -- not "
            "session_meta -- is the authoritative source"
        )
    return match.group(1)


def _looks_like_codex(path: Path) -> bool:
    """Cheap peek for detect(): a session_meta / rollout record shape."""
    if not _SESSION_ID_RE.search(path.name) and not path.name.startswith("rollout-"):
        return False
    try:
        with path.open(encoding="utf-8") as fh:
            for _ in range(50):
                line = fh.readline()
                if not line:
                    break
                try:
                    record = json.loads(line)
                except ValueError:
                    return False
                if isinstance(record, dict) and record.get("type") in KNOWN_RECORD_TYPES \
                        and isinstance(record.get("payload"), dict):
                    return True
    except OSError:
        return False
    return False


def _rollout_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("rollout-*.jsonl") if p.is_file())


def _message_text(payload: dict) -> str:
    content = payload.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts)
    return ""


def _output_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, default=str)


class _RolloutBuilder:
    """Accumulates one rollout file into an attempt + coverage notes."""

    def __init__(self, path: Path, session_id: str):
        self.path = path
        self.session_id = session_id
        self.prompt = ""
        self.model_id: str | None = None
        self.cli_version: str | None = None
        self.meta_id: str | None = None
        self.spans: list[TrajectoryStep] = []
        self.spans_by_id: dict[str, TrajectoryStep] = {}
        self.pending_spans: list[TrajectoryStep] = []  # spans since the last snapshot
        self.prev_total: tuple[int, int] | None = None
        self.final_total: tuple[int, int] | None = None
        self.final_cached: int | None = None
        self.final_reasoning: int | None = None
        self.snapshots = 0
        self.duplicate_snapshots = 0
        self.multi_call_deltas = 0
        self.notes: list[str] = []
        self.type_counts: dict[str, int] = {}
        self.unknown_event_types: dict[str, int] = {}
        self.duplicate_event_records = 0
        self.truncated = 0
        self.orphan_outputs = 0
        self.tool_schema_count = 0
        self.no_output_calls = 0  # web_search / image_generation spans
        self.first_ts: datetime | None = None
        self.last_ts: datetime | None = None

    def feed(self, record: dict, lineno: int) -> None:
        rtype = record.get("type")
        if rtype not in KNOWN_RECORD_TYPES:
            raise AuditError(
                f"{self.path} line {lineno}: unknown Codex record type "
                f"{rtype!r} (this adapter is pinned to "
                f"{sorted(KNOWN_RECORD_TYPES)}; unrecognized records are "
                "never silently dropped)"
            )
        self.type_counts[rtype] = self.type_counts.get(rtype, 0) + 1
        payload = record.get("payload")
        if not isinstance(payload, dict):
            raise AuditError(f"{self.path} line {lineno}: record payload must be an object")
        ts = parse_ts(record.get("timestamp"))
        if ts is not None:
            self.first_ts = ts if self.first_ts is None else min(self.first_ts, ts)
            self.last_ts = ts if self.last_ts is None else max(self.last_ts, ts)
        if rtype == "session_meta":
            if isinstance(payload.get("id"), str):
                self.meta_id = payload["id"]
            if isinstance(payload.get("cli_version"), str):
                self.cli_version = payload["cli_version"]
        elif rtype == "turn_context":
            if self.model_id is None and isinstance(payload.get("model"), str):
                self.model_id = payload["model"]
        elif rtype == "response_item":
            self._feed_response_item(payload, lineno)
        elif rtype == "event_msg":
            self._feed_event_msg(payload, lineno)

    def _feed_response_item(self, payload: dict, lineno: int) -> None:
        ptype = payload.get("type")
        if ptype not in KNOWN_RESPONSE_ITEM_TYPES:
            raise AuditError(
                f"{self.path} line {lineno}: unknown response_item payload "
                f"type {ptype!r} (pinned to {sorted(KNOWN_RESPONSE_ITEM_TYPES)}; "
                "response_item is the authoritative stream, so an "
                "unrecognized item is never silently dropped)"
            )
        if ptype == "message":
            if payload.get("role") == "user" and not self.prompt:
                self.prompt = _message_text(payload).strip()
            return
        if ptype == "reasoning":
            return  # reasoning content is not a tool call; counted by type only
        if ptype == "tool_schema":
            # A tool definition (name + JSON schema), not a call.
            self.tool_schema_count += 1
            return
        if ptype in ("function_call", "custom_tool_call", "tool_search_call",
                     "web_search_call", "image_generation_call"):
            call_id = str(payload.get("call_id") or payload.get("id")
                          or f"s{len(self.spans)}")
            if call_id in self.spans_by_id:
                raise AuditError(
                    f"{self.path} line {lineno}: duplicate call_id {call_id!r}; "
                    "trajectory step ids must be unique"
                )
            if ptype == "function_call":
                name = str(payload.get("name") or "")
                raw_args = payload.get("arguments")
                if isinstance(raw_args, dict):
                    args = dict(raw_args)
                elif isinstance(raw_args, str):
                    try:
                        parsed = json.loads(raw_args)
                    except ValueError:
                        parsed = None
                    args = dict(parsed) if isinstance(parsed, dict) \
                        else {"raw_arguments": raw_args}
                else:
                    args = {"value": raw_args}
            elif ptype == "custom_tool_call":
                # custom_tool_call: the payload's input is the argument
                name = str(payload.get("name") or "")
                raw_input = payload.get("input")
                args = {"input": raw_input if isinstance(raw_input, str)
                        else _output_text(raw_input)}
            elif ptype == "tool_search_call":
                name = "tool_search"
                raw_args = payload.get("arguments")
                args = dict(raw_args) if isinstance(raw_args, dict) \
                    else {"value": raw_args}
            elif ptype == "web_search_call":
                name = "web_search"
                action = payload.get("action")
                args = dict(action) if isinstance(action, dict) \
                    else {"value": action}
                self.no_output_calls += 1  # no output record in this stream
            else:  # image_generation_call
                name = "image_generation"
                prompt = payload.get("revised_prompt")
                args = {"revised_prompt": prompt} if isinstance(prompt, str) else {}
                self.no_output_calls += 1  # result lands outside this stream
            span = TrajectoryStep(
                step_id=call_id,
                tool=name,
                args=args,
                output=None,
                consumes=[],  # the format records no consumption edges; never inferred
            )
            self.spans.append(span)
            self.spans_by_id[call_id] = span
            self.pending_spans.append(span)
            return
        # function_call_output / custom_tool_call_output / tool_search_output
        call_id = str(payload.get("call_id") or "")
        span = self.spans_by_id.get(call_id)
        if span is None:
            self.orphan_outputs += 1
            return
        text = _output_text(payload.get("tools") if ptype == "tool_search_output"
                            else payload.get("output"))
        if len(text) > MAX_SPAN_OUTPUT_CHARS:
            text = text[:MAX_SPAN_OUTPUT_CHARS]
            self.truncated += 1
        span.output = text

    def _feed_event_msg(self, payload: dict, lineno: int) -> None:
        ptype = payload.get("type")
        if ptype == "token_count":
            self._feed_token_count(payload, lineno)
            return
        if ptype not in KNOWN_EVENT_MSG_TYPES:
            self.unknown_event_types[str(ptype)] = \
                self.unknown_event_types.get(str(ptype), 0) + 1
            return
        # Every other known event_msg type duplicates response_item
        # content (or UI state) and is counted, never turned into spans.
        self.duplicate_event_records += 1

    def _feed_token_count(self, payload: dict, lineno: int) -> None:
        info = payload.get("info")
        total = info.get("total_token_usage") if isinstance(info, dict) else None
        if not isinstance(total, dict):
            self.notes.append(
                f"{self.path.name}: a token_count event carried no "
                "total_token_usage and was skipped"
            )
            return
        current = (int(total.get("input_tokens") or 0), int(total.get("output_tokens") or 0))
        self.snapshots += 1
        # The first snapshot's delta is its own total: cumulative counters
        # start at zero when the session starts.
        base = self.prev_total if self.prev_total is not None else (0, 0)
        delta = (current[0] - base[0], current[1] - base[1])
        if delta[0] < 0 or delta[1] < 0:
            raise AuditError(
                f"{self.path} line {lineno}: cumulative token totals "
                f"decreased ({self.prev_total} -> {current}); token_count "
                "totals are cumulative, so this file breaks the format "
                "invariant"
            )
        if delta == (0, 0) and self.prev_total is not None:
            self.duplicate_snapshots += 1
        if len(self.pending_spans) == 1:
            self.pending_spans[0].tokens_in = delta[0]
            self.pending_spans[0].tokens_out = delta[1]
        elif len(self.pending_spans) > 1:
            self.multi_call_deltas += 1
        # Zero pending spans: the delta belongs to message/reasoning turns
        # with no tool call; it still counts in attempt totals.
        self.prev_total = current
        self.final_total = current
        cached = total.get("cached_input_tokens")
        self.final_cached = int(cached) if isinstance(cached, (int, float)) else None
        reasoning = total.get("reasoning_output_tokens")
        self.final_reasoning = int(reasoning) if isinstance(reasoning, (int, float)) else None
        self.pending_spans = []

    def finish(self) -> tuple[TaskSample, Attempt]:
        name = self.path.name
        if self.meta_id and self.meta_id != self.session_id:
            self.notes.append(
                f"{name}: session_meta id {self.meta_id} is the lineage root, "
                f"not the session id; the filename UUID {self.session_id} is used"
            )
        if self.final_total is not None:
            self.notes.append(
                f"{name}: token_count totals are cumulative; per-turn usage "
                f"was computed as deltas over {self.snapshots} snapshot(s)"
                + (f" ({self.duplicate_snapshots} verbatim repeat(s) counted once)"
                   if self.duplicate_snapshots else "")
            )
            if self.final_cached is not None or self.final_reasoning is not None:
                self.notes.append(
                    f"{name}: cached_input={self.final_cached} is a subset of "
                    f"tokens_in and reasoning_output={self.final_reasoning} a "
                    "subset of tokens_out; neither is added on top"
                )
        else:
            self.notes.append(f"{name}: no token_count events; token totals unknown")
        if self.multi_call_deltas:
            self.notes.append(
                f"{name}: {self.multi_call_deltas} token delta(s) covered "
                "several tool calls; those spans carry no per-step tokens "
                "(a shared delta is never split)"
            )
        if self.duplicate_event_records:
            self.notes.append(
                f"{name}: {self.duplicate_event_records} event_msg record(s) "
                "ignored as duplicates of the response_item stream "
                "(token_count excepted)"
            )
        for ptype, count in sorted(self.unknown_event_types.items()):
            self.notes.append(
                f"{name}: {count} event_msg record(s) of unrecognized type "
                f"{ptype!r} ignored (the event_msg stream is not used for "
                "spans, so no trajectory data is lost)"
            )
        if self.spans:
            self.notes.append(
                f"{name}: Codex rollouts record no consumption edges; "
                "TRAJ-002 cannot be assessed on this import"
            )
        if self.truncated:
            self.notes.append(
                f"{name}: {self.truncated} tool output(s) truncated to "
                f"{MAX_SPAN_OUTPUT_CHARS} chars at the adapter boundary"
            )
        if self.orphan_outputs:
            self.notes.append(
                f"{name}: {self.orphan_outputs} tool output record(s) named "
                "a call_id with no matching call in this file"
            )
        if self.tool_schema_count:
            self.notes.append(
                f"{name}: {self.tool_schema_count} tool_schema record(s) "
                "skipped; they carry tool definitions, not calls"
            )
        if self.no_output_calls:
            self.notes.append(
                f"{name}: {self.no_output_calls} web_search / "
                "image_generation call(s) carry no output; the rollout "
                "records their results outside the response_item stream"
            )
        self.notes.append(
            f"{name}: rollouts record no pass/fail outcome; the attempt is "
            "marked incomplete and COST-lane wasted-spend does not apply"
        )
        latency = None
        if self.first_ts is not None and self.last_ts is not None:
            latency = (self.last_ts - self.first_ts).total_seconds()
        metadata: dict = {"source_file": name, "source_path": str(self.path)}
        if self.first_ts is not None:
            metadata["started_at"] = self.first_ts.isoformat()
        task = TaskSample(id=self.session_id, prompt=self.prompt, metadata=metadata)
        attempt = Attempt(
            task_id=self.session_id,
            status="incomplete",  # no outcome is recorded; never fabricated
            tool_calls=len(self.spans),
            actions=[s.tool for s in self.spans],
            tokens_in=self.final_total[0] if self.final_total else None,
            tokens_out=self.final_total[1] if self.final_total else None,
            latency_s=latency,
            model_id=self.model_id,
            spans=self.spans,
        )
        return task, attempt


@register
class CodexAdapter:
    name = ADAPTER_NAME
    version = ADAPTER_VERSION

    def detect(self, path: Path) -> Confidence:
        if path.is_file():
            return Confidence.HIGH if _looks_like_codex(path) else Confidence.LOW
        if path.is_dir():
            for candidate in _rollout_files(path)[:5]:
                if _looks_like_codex(candidate):
                    return Confidence.HIGH
            return Confidence.LOW
        return Confidence.LOW

    def collect(self, path: Path) -> dict:
        """Read-only: rollout files are opened for reading and hashed."""
        files = _rollout_files(path)
        if not files:
            raise AuditError(f"codex adapter found no rollout-*.jsonl files under {path}")
        sources = []
        digests: dict[str, str] = {}
        for file_path in files:
            sources.append({
                "path": file_path,
                "records": read_jsonl(file_path, kind="rollout file"),
                "session_id": _session_id_from_filename(file_path),
            })
            digests[file_path.name] = digest_file(file_path)
        return {"root": path, "sources": sources, "digests": digests}

    def normalize(self, bundle: dict) -> IntegrityModel:
        root: Path = bundle["root"]
        tasks: list[TaskSample] = []
        attempts: list[Attempt] = []
        unsupported: list[str] = []
        for source in bundle["sources"]:
            builder = _RolloutBuilder(source["path"], source["session_id"])
            for lineno, record in enumerate(source["records"], start=1):
                builder.feed(record, lineno)
            task, attempt = builder.finish()
            tasks.append(task)
            attempts.append(attempt)
            unsupported.extend(builder.notes)
        eval_id = root.name if root.is_dir() else attempts[0].task_id
        first_file = bundle["sources"][0]["path"].name
        return IntegrityModel(
            eval_id=eval_id,
            adapter_name=self.name,
            adapter_version=self.version,
            tasks=tasks,
            environment=Environment(),
            grader=Grader(),
            attempts=attempts,
            unsupported=unsupported,
            digests=dict(bundle.get("digests", {})),
            file_aliases={
                "trajectories.jsonl": first_file,
                "attempts.json": first_file,
                "run.json": first_file,
                "tasks.json": first_file,
            },
        )
