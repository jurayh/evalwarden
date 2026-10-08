"""Shared file-reading helpers for adapters.

Every adapter reads the same kinds of files (JSON documents, JSONL record
streams, source files to digest for the reproducibility manifest) and must
fail the same way when they are missing or malformed: an
:class:`AuditError` naming the file and, for JSONL, the offending line.

Error wording is part of each adapter's surface, so the helpers take the
adapter's own nouns instead of imposing one vocabulary: ``kind`` names the
file ("session file", "rollout file", "required file") and ``record``
names a JSONL line's payload ("record", "span record").
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from . import AuditError


def digest_file(path: Path) -> str:
    """SHA-256 hex digest of a source file, for reproducibility manifests."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    """Parse a JSON document; missing or malformed files fail clearly."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AuditError(f"missing required file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise AuditError(f"invalid JSON in {path}: {exc}") from exc


def read_jsonl(path: Path, *, kind: str = "session file", record: str = "record") -> list[dict]:
    """Parse a JSONL file of JSON objects. Malformed lines fail clearly, with the line."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise AuditError(f"missing {kind}: {path}") from exc
    except UnicodeDecodeError as exc:
        raise AuditError(f"{path}: {kind} must be UTF-8 text") from exc
    records: list[dict] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AuditError(f"invalid JSON in {path} line {lineno}: {exc}") from exc
        if not isinstance(parsed, dict):
            raise AuditError(f"{path} line {lineno}: {record} must be a JSON object")
        records.append(parsed)
    return records


def parse_ts(value: Any) -> datetime | None:
    """Parse an ISO-8601 timestamp (``Z`` suffix allowed); None if unusable."""
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
