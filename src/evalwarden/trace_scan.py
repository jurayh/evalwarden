"""Trace observatory scanning: find sessions, load them, filter them.

This is the ease-of-use layer over the trace adapters. It knows where
Claude Code and Codex keep sessions on a stock install
(``~/.claude/projects/``, ``~/.codex/sessions/``), loads each source
root through its adapter, and applies the report-level filters
(``--since``, ``--project``, ``--agent``). Everything is read-only and
offline, like the adapters themselves.

Filters work on what the adapters recorded: each imported task carries
``started_at`` / ``source_path`` in its metadata (stamped from the
session file's own records), so ``--since`` compares real session
start times, never file mtimes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .adapters import AuditError, autodetect
from .adapters.claude_code import ClaudeCodeAdapter
from .adapters.codex import CodexAdapter
from .model import IntegrityModel

TRACE_ADAPTERS = ("claude-code", "codex")


@dataclass
class TraceSource:
    """One root to import: a project dir, a sessions tree, or one file."""

    label: str  # display name (usually the root's directory name)
    agent: str  # adapter name: "claude-code" | "codex"
    path: Path


def discover(home: Path | None = None) -> list[TraceSource]:
    """Stock install locations under `home` (default: the real home)."""
    home = home if home is not None else Path.home()
    sources: list[TraceSource] = []
    claude_root = home / ".claude" / "projects"
    if claude_root.is_dir():
        # Each subdirectory is one project holding session files; a
        # session file directly in the root counts as its own project.
        if any(p.is_file() for p in claude_root.glob("*.jsonl")):
            sources.append(TraceSource(claude_root.name, "claude-code", claude_root))
        for project in sorted(p for p in claude_root.iterdir() if p.is_dir()):
            if any(f.is_file() for f in project.glob("*.jsonl")):
                sources.append(TraceSource(project.name, "claude-code", project))
    codex_root = home / ".codex" / "sessions"
    if codex_root.is_dir() and any(p.is_file() for p in codex_root.rglob("rollout-*.jsonl")):
        sources.append(TraceSource("codex-sessions", "codex", codex_root))
    return sources


def demo_sources() -> list[TraceSource]:
    """The packaged demo fixtures (planted loops, planted phases)."""
    import evalwarden

    demo_dir = Path(evalwarden.__file__).resolve().parent / "demo" / "trace_observatory"
    if not demo_dir.is_dir():
        raise AuditError("trace demo fixtures not found (expected demo/trace_observatory/)")
    return [
        TraceSource("demo-claude", "claude-code", demo_dir / "claude"),
        TraceSource("demo-codex", "codex", demo_dir / "codex"),
    ]


def load_source(source: TraceSource) -> IntegrityModel:
    adapter = ClaudeCodeAdapter() if source.agent == "claude-code" else CodexAdapter()
    return adapter.normalize(adapter.collect(source.path))


def load_path(path: Path) -> IntegrityModel:
    """One explicit path, adapter chosen by autodetect."""
    adapter = autodetect(path)
    if adapter.name not in TRACE_ADAPTERS:
        raise AuditError(
            f"{path}: detected adapter {adapter.name!r} is not a trace adapter; "
            "the trace observatory reads Claude Code or Codex sessions"
        )
    return adapter.normalize(adapter.collect(path))


def parse_since(value: str, *, now: datetime | None = None) -> datetime:
    """Parse a --since value: ``7d``, ``24h``, ``2w``, or ``YYYY-MM-DD``."""
    now = now if now is not None else datetime.now(timezone.utc)
    match = re.fullmatch(r"(\d+)([dwh])", value.strip())
    if match:
        amount, unit = int(match.group(1)), match.group(2)
        delta = {"d": timedelta(days=amount), "w": timedelta(weeks=amount),
                 "h": timedelta(hours=amount)}[unit]
        return now - delta
    try:
        day = datetime.strptime(value.strip(), "%Y-%m-%d")
    except ValueError:
        raise AuditError(
            f"--since {value!r}: expected a duration like 7d / 24h / 2w or a date YYYY-MM-DD"
        ) from None
    return day.replace(tzinfo=timezone.utc)


def _started_at(model: IntegrityModel, task_id: str) -> datetime | None:
    for task in model.tasks:
        if task.id == task_id:
            raw = task.metadata.get("started_at")
            if isinstance(raw, str):
                try:
                    return datetime.fromisoformat(raw)
                except ValueError:
                    return None
    return None


def filter_since(models: list[IntegrityModel], cutoff: datetime) -> list[IntegrityModel]:
    """Drop attempts that started before `cutoff` (unknown starts are kept)."""
    kept_models: list[IntegrityModel] = []
    for model in models:
        kept = []
        for attempt in model.attempts:
            started = _started_at(model, attempt.task_id)
            if started is None or started >= cutoff:
                kept.append(attempt)
        if not kept:
            continue
        kept_ids = {a.task_id for a in kept}
        model.attempts = kept
        model.tasks = [t for t in model.tasks if t.id in kept_ids]
        kept_models.append(model)
    return kept_models
