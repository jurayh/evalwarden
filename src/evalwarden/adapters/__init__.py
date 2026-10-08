"""Adapter protocol and registry.

Adapters are the ONLY harness-specific code in the project. They discover and
translate; the core never becomes a harness. Each adapter:

- is read-only (never mutates the input files),
- works offline (no network in default mode),
- preserves source locations,
- pins its own version and the schema versions it understands,
- reports unsupported fields explicitly instead of silently dropping them.

Five adapters ship: Inspect AI eval artifacts (artifact directories and
native ``.eval`` logs), Promptfoo results, Claude Code session files, Codex
rollout files, and a universal importer for the canonical
``evalwarden.model`` JSON, JSONL, and CSV formats. New adapters plug into
REGISTRY with the same contract.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..model import Confidence, IntegrityModel


class AuditError(Exception):
    """The audit could not complete (exit code 2)."""


class Adapter(Protocol):
    name: str
    version: str

    def detect(self, path: Path) -> Confidence:
        """How confident are we that this adapter understands `path`?"""
        ...

    def collect(self, path: Path) -> dict:
        """Read-only collection of raw evidence from the artifact."""
        ...

    def normalize(self, bundle: dict) -> IntegrityModel:
        """Translate raw evidence into the framework-neutral integrity model."""
        ...


REGISTRY: list[Adapter] = []


def register(adapter: Adapter) -> Adapter:
    # Accept either an instance or a class (instantiated here) so adapters can
    # use either `@register` on the class or `register(MyAdapter())`.
    REGISTRY.append(adapter() if isinstance(adapter, type) else adapter)
    return adapter


def autodetect(path: Path) -> Adapter:
    """Pick the most confident adapter for `path`, or raise AuditError."""
    if not REGISTRY:
        raise AuditError("no adapters registered")
    ranked = sorted(
        ((adapter.detect(path), adapter) for adapter in REGISTRY),
        key=lambda item: (item[0] == Confidence.HIGH, item[0] == Confidence.MEDIUM),
        reverse=True,
    )
    confidence, adapter = ranked[0]
    if confidence == Confidence.LOW:
        raise AuditError(
            f"no adapter recognizes {path} "
            f"(best guess: {adapter.name}, confidence=low). "
            "Expected an eval artifact directory (see demo/leaky for the layout)."
        )
    return adapter
