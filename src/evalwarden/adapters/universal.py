"""Universal importer adapter.

Reads a single Evalwarden canonical input file:

- ``*.json`` containing an ``evalwarden.model`` 1.0 document (or a JSONL
  stream stored under a JSON name),
- ``*.jsonl`` containing typed ``evalwarden.model`` records, or
- ``*.csv`` containing the minimal score-table contract.

The translation itself lives in :mod:`evalwarden.canonical`; this adapter
only detects, reads, preserves the digest, and points check evidence at
the actual source file.  Like every adapter it is read-only and stdlib
only.  The CSV contract is intentionally narrow: it never fabricates
judgments, spans, grader data, or data-flow edges.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..canonical import (
    CANONICAL_ADAPTER_NAME,
    CANONICAL_ADAPTER_VERSION,
    SCHEMA_ID,
    CanonicalValidationError,
    model_from_csv,
    model_from_json,
    model_from_jsonl,
)
from ..model import Confidence, IntegrityModel
from . import AuditError, register
from ._io import digest_file

ADAPTER_NAME = CANONICAL_ADAPTER_NAME
ADAPTER_VERSION = CANONICAL_ADAPTER_VERSION

_CANONICAL_FILES = (
    "tasks.json",
    "environment.json",
    "grader.json",
    "run.json",
    "judge_run.json",
    "attempts.json",
    "run_scores.json",
)


def _looks_like_jsonl(text: str) -> bool:
    """Whether the first complete JSON line is a typed canonical record."""
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            return False
        return isinstance(record, dict) and record.get("type") in {
            "eval",
            "environment",
            "grader",
            "task",
            "attempt",
            "judgment",
            "run_score",
            "span",
        }
    return False


def _format_for(path: Path, text: str) -> str:
    if path.suffix.lower() == ".jsonl":
        return "jsonl"
    if path.suffix.lower() == ".csv":
        return "csv"
    if _looks_like_jsonl(text):
        return "jsonl"
    return "json"


@register
class UniversalAdapter:
    name = ADAPTER_NAME
    version = ADAPTER_VERSION

    def detect(self, path: Path) -> Confidence:
        if not path.is_file():
            return Confidence.LOW
        suffix = path.suffix.lower()
        if suffix in {".jsonl", ".csv"}:
            return Confidence.HIGH
        if suffix != ".json":
            return Confidence.LOW
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return Confidence.LOW
        try:
            document = json.loads(text)
        except ValueError:
            # Claim files that visibly name the schema even when malformed,
            # so collect() can report the precise JSON error instead of a
            # generic "no adapter recognizes this path".
            if SCHEMA_ID in text[:8192]:
                return Confidence.MEDIUM
            return Confidence.HIGH if _looks_like_jsonl(text) else Confidence.LOW
        if isinstance(document, dict) and document.get("schema") == SCHEMA_ID:
            return Confidence.HIGH
        return Confidence.LOW

    def collect(self, path: Path) -> dict[str, Any]:
        """Read-only: the input file is opened for reading and hashed."""
        if not path.is_file():
            raise AuditError(
                f"universal adapter expects a single .json, .jsonl, or .csv file: {path}"
            )
        if path.suffix.lower() not in {".json", ".jsonl", ".csv"}:
            raise AuditError(
                f"universal adapter does not recognize the file extension of {path}"
            )
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise AuditError(f"{path}: input must be UTF-8 text") from exc
        except OSError as exc:
            raise AuditError(f"could not read {path}: {exc}") from exc
        return {
            "root": path,
            "format": _format_for(path, text),
            "text": text,
            "digests": {path.name: digest_file(path)},
        }

    def normalize(self, bundle: dict) -> IntegrityModel:
        path: Path = bundle["root"]
        text: str = bundle["text"]
        try:
            if bundle["format"] == "jsonl":
                model = model_from_jsonl(text)
            elif bundle["format"] == "csv":
                model = model_from_csv(
                    text, eval_id=path.stem, source_name=path.name
                )
            else:
                model = model_from_json(text)
        except CanonicalValidationError as exc:
            raise AuditError(str(exc)) from exc
        model.digests = dict(bundle.get("digests", {}))
        model.file_aliases = {name: path.name for name in _CANONICAL_FILES}
        return model
