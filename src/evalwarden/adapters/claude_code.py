"""Claude Code session adapter.

Reads Claude Code session transcripts (JSONL, one record per line) from
``~/.claude/projects/<project-slug>/``:

    <session-id>.jsonl                     one file per session
    <session-id>/subagents/agent-<id>.jsonl   sub-agent sidechains
    <session-id>/subagents/agent-<id>.meta.json   sidechain metadata

The input may be a single ``.jsonl`` session file, a directory holding
one session file (plus its ``subagents/`` sidechains), or a project
directory holding several session files. Each session file becomes one
attempt; each sub-agent sidechain becomes its own attempt, because a
sidechain is a separate agent run whose spans and usage must not be
merged into the parent session's trajectory.

Format notes (reverse-engineered; Anthropic documents the location and
states the entry format is internal and changes between versions, so
this parser is pinned to the record types below and fails loudly on any
other type instead of silently dropping data):

- Assistant records carry ``message.usage`` per record (input / output /
  cache-creation / cache-read tokens). Attempt ``tokens_in`` /
  ``tokens_out`` are the input / output sums; cache totals are reported
  in ``unsupported`` because the model has no cache fields and folding
  them into input would double-count (cache reads are a subset story
  the format keeps separate).
- Tool calls are ``tool_use`` content blocks on assistant records; the
  block id becomes the step id. Tool results return as ``tool_result``
  blocks on user records and are matched back by that id.
- A record's usage is attributed to a span only when the record made
  exactly one tool call. A record with several tool calls shares one
  usage total, and splitting it would fabricate per-step precision, so
  those spans carry no per-step tokens (reported in ``unsupported``).
- Neither the format nor this adapter records consumption edges, so
  every span's ``consumes`` stays empty and TRAJ-002 stays silent.
- Session files record no pass/fail outcome, so attempts are marked
  ``incomplete``: claiming ``pass`` would fabricate the outcome join.
  COST-lane wasted-spend (which keys on pass) is therefore not
  meaningful on trace imports; use ``evalwarden.trace_waste`` instead.
"""
from __future__ import annotations

import json
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

ADAPTER_NAME = "claude-code"
ADAPTER_VERSION = "0.1.0"
MAX_SPAN_OUTPUT_CHARS = 512  # same truncation boundary as the other adapters

# Record types this parser is pinned to (observed in Claude Code
# transcripts as of the 2026-10-05 format scan). Anything else fails
# loudly: the format shifts between versions, and a silent drop would
# lose trajectory data without a trace.
KNOWN_RECORD_TYPES = {
    "user",
    "assistant",
    "system",
    "summary",
    "queue-operation",
    "attachment",
    "last-prompt",
}

# Bookkeeping record types observed in real session files (trace-commons
# corpus pass, Oct 2026): Claude Code writes them alongside the
# conversation, and a full-corpus scan verified they carry no messages,
# no tool calls, and no usage -- they record file-history snapshots,
# session titles, and mode state. They are counted and skipped, with
# the counts surfaced in the coverage notes; anything outside both sets
# still fails loudly above.
SKIPPED_RECORD_TYPES = {
    "file-history-snapshot",
    "ai-title",
    "mode",
    "permission-mode",
    "pr-link",
}


def _looks_like_claude(path: Path) -> bool:
    """Cheap peek for detect(): session markers in the first records."""
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
                if not isinstance(record, dict):
                    continue
                if "sessionId" in record and (
                    "parentUuid" in record
                    or record.get("type") in KNOWN_RECORD_TYPES | SKIPPED_RECORD_TYPES
                ):
                    return True
    except OSError:
        return False
    return False


def _session_files(path: Path) -> list[tuple[Path, Path | None, str | None]]:
    """(file, parent session file, subagent id) triples for the input.

    A single file is one session. A directory contributes every top-level
    ``*.jsonl`` as a session, plus each session's ``subagents/*.jsonl``
    sidechains (attributed to that session, parsed as separate attempts).
    """
    if path.is_file():
        return [(path, None, None)]
    triples: list[tuple[Path, Path | None, str | None]] = []
    for session_file in sorted(p for p in path.glob("*.jsonl") if p.is_file()):
        triples.append((session_file, None, None))
        sub_dir = path / session_file.stem / "subagents"
        if sub_dir.is_dir():
            for sub_file in sorted(p for p in sub_dir.glob("agent-*.jsonl") if p.is_file()):
                agent_id = sub_file.stem.removeprefix("agent-")
                triples.append((sub_file, session_file, agent_id))
    return triples


def _block_text(content: Any) -> str:
    """Plain text of a message content value (str or content-block list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return ""


def _tool_result_text(block: dict) -> str:
    content = block.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return _block_text(content)
    return ""


def _usage_int(usage: dict, key: str) -> int:
    value = usage.get(key)
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


class _SessionBuilder:
    """Accumulates one session file into an attempt + coverage notes."""

    def __init__(self, path: Path, *, session_hint: str, subagent_id: str | None,
                 meta_description: str | None):
        self.path = path
        self.session_hint = session_hint
        self.subagent_id = subagent_id
        self.meta_description = meta_description
        self.session_id: str | None = None
        self.prompt: str = ""
        self.model_id: str | None = None
        self.spans: list[TrajectoryStep] = []
        self.spans_by_id: dict[str, TrajectoryStep] = {}
        self.tokens_in = 0
        self.tokens_out = 0
        self.cache_creation = 0
        self.cache_read = 0
        self.has_usage = False
        self.first_ts: datetime | None = None
        self.last_ts: datetime | None = None
        self.notes: list[str] = []
        self.type_counts: dict[str, int] = {}
        self.skipped_counts: dict[str, int] = {}
        self.sidechain_records_in_main = 0
        self.multi_tool_records = 0
        self.truncated = 0
        self.orphan_results = 0

    @property
    def task_id(self) -> str:
        base = self.session_id or self.session_hint
        if self.subagent_id:
            return f"{base}:subagent:{self.subagent_id}"
        return base

    def feed(self, record: dict, lineno: int) -> None:
        rtype = record.get("type")
        if rtype not in KNOWN_RECORD_TYPES and rtype not in SKIPPED_RECORD_TYPES:
            raise AuditError(
                f"{self.path} line {lineno}: unknown Claude Code record type "
                f"{rtype!r} (this adapter is pinned to "
                f"{sorted(KNOWN_RECORD_TYPES | SKIPPED_RECORD_TYPES)}; the "
                "internal format shifts between versions and unrecognized "
                "records are never silently dropped)"
            )
        if self.session_id is None and isinstance(record.get("sessionId"), str):
            self.session_id = record["sessionId"]
        ts = parse_ts(record.get("timestamp"))
        if ts is not None:
            self.first_ts = ts if self.first_ts is None else min(self.first_ts, ts)
            self.last_ts = ts if self.last_ts is None else max(self.last_ts, ts)
        if rtype in SKIPPED_RECORD_TYPES:
            self.skipped_counts[rtype] = self.skipped_counts.get(rtype, 0) + 1
            return
        self.type_counts[rtype] = self.type_counts.get(rtype, 0) + 1
        # Sidechain records inside a *main* session file belong to a
        # sub-agent run; their data lives in the sidechain file, which is
        # parsed as its own attempt. Counting them here keeps the skip
        # explicit rather than silent.
        if self.subagent_id is None and record.get("isSidechain") is True:
            self.sidechain_records_in_main += 1
            return
        if rtype == "user":
            self._feed_user(record)
        elif rtype == "assistant":
            self._feed_assistant(record, lineno)

    def _feed_user(self, record: dict) -> None:
        message = record.get("message") or {}
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    span = self.spans_by_id.get(str(block.get("tool_use_id")))
                    if span is None:
                        self.orphan_results += 1
                        continue
                    text = _tool_result_text(block)
                    if len(text) > MAX_SPAN_OUTPUT_CHARS:
                        text = text[:MAX_SPAN_OUTPUT_CHARS]
                        self.truncated += 1
                    span.output = text
        if not self.prompt:
            text = _block_text(content).strip()
            if text:
                self.prompt = text

    def _feed_assistant(self, record: dict, lineno: int) -> None:
        message = record.get("message") or {}
        if not isinstance(message, dict):
            return
        if self.model_id is None and isinstance(message.get("model"), str):
            self.model_id = message["model"]
        usage = message.get("usage")
        rec_in = rec_out = 0
        if isinstance(usage, dict):
            self.has_usage = True
            rec_in = _usage_int(usage, "input_tokens")
            rec_out = _usage_int(usage, "output_tokens")
            self.tokens_in += rec_in
            self.tokens_out += rec_out
            self.cache_creation += _usage_int(usage, "cache_creation_input_tokens")
            self.cache_read += _usage_int(usage, "cache_read_input_tokens")
        content = message.get("content")
        blocks = content if isinstance(content, list) else []
        tool_blocks = [b for b in blocks if isinstance(b, dict) and b.get("type") == "tool_use"]
        new_spans: list[TrajectoryStep] = []
        for block in tool_blocks:
            step_id = str(block.get("id") or f"s{len(self.spans) + len(new_spans)}")
            if step_id in self.spans_by_id:
                raise AuditError(
                    f"{self.path} line {lineno}: duplicate tool_use id {step_id!r}; "
                    "trajectory step ids must be unique"
                )
            inp = block.get("input")
            span = TrajectoryStep(
                step_id=step_id,
                tool=str(block.get("name") or ""),
                args=dict(inp) if isinstance(inp, dict) else {"value": inp},
                output=None,
                consumes=[],  # the format records no consumption edges; never inferred
            )
            self.spans.append(span)
            self.spans_by_id[step_id] = span
            new_spans.append(span)
        if len(new_spans) == 1 and isinstance(usage, dict):
            new_spans[0].tokens_in = rec_in
            new_spans[0].tokens_out = rec_out
        elif len(new_spans) > 1:
            self.multi_tool_records += 1

    def finish(self) -> tuple[TaskSample, Attempt]:
        name = self.path.name
        if self.session_id and self.subagent_id is None and self.session_id != self.session_hint:
            self.notes.append(
                f"{name}: sessionId {self.session_id} differs from the filename "
                f"stem {self.session_hint}; the recorded sessionId wins"
            )
        if self.cache_creation or self.cache_read:
            self.notes.append(
                f"{name}: cache tokens are not part of tokens_in/out "
                f"(cache_creation={self.cache_creation}, cache_read={self.cache_read})"
            )
        if self.spans:
            self.notes.append(
                f"{name}: Claude Code records no consumption edges; "
                "TRAJ-002 cannot be assessed on this import"
            )
        if self.sidechain_records_in_main:
            self.notes.append(
                f"{name}: {self.sidechain_records_in_main} sidechain record(s) "
                "skipped in the main file; sub-agent files are parsed as "
                "separate attempts"
            )
        if self.multi_tool_records:
            self.notes.append(
                f"{name}: {self.multi_tool_records} assistant record(s) made "
                "several tool calls under one usage total; those spans carry "
                "no per-step tokens (a shared total is never split)"
            )
        if self.truncated:
            self.notes.append(
                f"{name}: {self.truncated} tool result(s) truncated to "
                f"{MAX_SPAN_OUTPUT_CHARS} chars at the adapter boundary"
            )
        if self.orphan_results:
            self.notes.append(
                f"{name}: {self.orphan_results} tool_result block(s) named a "
                "tool_use id with no matching call in this file"
            )
        if self.skipped_counts:
            total = sum(self.skipped_counts.values())
            detail = ", ".join(
                f"{t} x{n}" for t, n in sorted(self.skipped_counts.items()))
            self.notes.append(
                f"{name}: {total} bookkeeping record(s) skipped ({detail}); "
                "these types carry no tool calls or usage"
            )
        self.notes.append(
            f"{name}: session files record no pass/fail outcome; the attempt "
            "is marked incomplete and COST-lane wasted-spend does not apply"
        )
        prompt = self.prompt or self.meta_description or ""
        latency = None
        if self.first_ts is not None and self.last_ts is not None:
            latency = (self.last_ts - self.first_ts).total_seconds()
        metadata: dict = {"source_file": name, "source_path": str(self.path)}
        if self.first_ts is not None:
            metadata["started_at"] = self.first_ts.isoformat()
        task = TaskSample(id=self.task_id, prompt=prompt, metadata=metadata)
        attempt = Attempt(
            task_id=self.task_id,
            status="incomplete",  # no outcome is recorded; never fabricated
            tool_calls=len(self.spans),
            actions=[s.tool for s in self.spans],
            tokens_in=self.tokens_in if self.has_usage else None,
            tokens_out=self.tokens_out if self.has_usage else None,
            latency_s=latency,
            model_id=self.model_id,
            spans=self.spans,
        )
        return task, attempt


@register
class ClaudeCodeAdapter:
    name = ADAPTER_NAME
    version = ADAPTER_VERSION

    def detect(self, path: Path) -> Confidence:
        if path.is_file():
            return Confidence.HIGH if path.suffix == ".jsonl" and _looks_like_claude(path) \
                else Confidence.LOW
        if path.is_dir():
            for candidate in sorted(p for p in path.glob("*.jsonl") if p.is_file()):
                if _looks_like_claude(candidate):
                    return Confidence.HIGH
            return Confidence.LOW
        return Confidence.LOW

    def collect(self, path: Path) -> dict:
        """Read-only: session files are opened for reading and hashed."""
        triples = _session_files(path)
        if not triples or (path.is_dir() and not any(t[1] is None and t[0].is_file() for t in triples)):
            raise AuditError(f"claude-code adapter found no session .jsonl files under {path}")
        sources = []
        digests: dict[str, str] = {}
        for file_path, parent, agent_id in triples:
            if not file_path.is_file():
                continue
            meta_description = None
            if agent_id is not None:
                # agent-<id>.jsonl -> agent-<id>.meta.json
                meta_path = file_path.parent / (file_path.stem + ".meta.json")
                if meta_path.is_file():
                    try:
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        meta = {}
                    if isinstance(meta, dict) and isinstance(meta.get("description"), str):
                        meta_description = meta["description"]
                    digests[meta_path.name] = digest_file(meta_path)
            sources.append({
                "path": file_path,
                "records": read_jsonl(file_path),
                "parent_stem": parent.stem if parent else None,
                "agent_id": agent_id,
                "meta_description": meta_description,
            })
            digests[file_path.name] = digest_file(file_path)
        if not sources:
            raise AuditError(f"claude-code adapter found no readable session files under {path}")
        return {"root": path, "sources": sources, "digests": digests}

    def normalize(self, bundle: dict) -> IntegrityModel:
        root: Path = bundle["root"]
        tasks: list[TaskSample] = []
        attempts: list[Attempt] = []
        unsupported: list[str] = []
        for source in bundle["sources"]:
            file_path: Path = source["path"]
            hint = source["parent_stem"] or file_path.stem
            if source["agent_id"] is not None and source["parent_stem"]:
                hint = source["parent_stem"]
            builder = _SessionBuilder(
                file_path,
                session_hint=hint,
                subagent_id=source["agent_id"],
                meta_description=source["meta_description"],
            )
            # Sub-agent files carry the *parent* sessionId in their records;
            # the attempt identity comes from the parent filename + agent id.
            if source["agent_id"] is not None:
                builder.session_id = source["parent_stem"]
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
