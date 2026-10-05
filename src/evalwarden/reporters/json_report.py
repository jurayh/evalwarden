"""JSON reporter: the machine-readable findings document.

The document is a public contract, identified by the top-level `schema` /
`schema_version` fields:

    {
      "schema": "evalwarden.findings",
      "schema_version": "1.0",
      "tool": {"name": "evalwarden", "version": "..."},
      "eval_id": ..., "adapter": {"name": ..., "version": ...},
      "verdict": "BLOCKED" | "PASS",
      "score": 0-100 (diagnostic only),
      "findings": [ <finding record>, ... ],   # see reporters.records
      "unsupported": [ coverage gaps the adapter saw but could not translate ]
    }

Finding records are built by `reporters.records` and shared with the SARIF
renderer so the two machine formats can never drift apart.
"""
from __future__ import annotations

import json

from .. import __version__
from ..engine import AuditResult
from .records import finding_record

SCHEMA_NAME = "evalwarden.findings"
SCHEMA_VERSION = "1.0"


def findings_document(result: AuditResult) -> dict:
    """Build the versioned JSON findings document for an audit result."""
    model = result.model
    return {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "tool": {"name": "evalwarden", "version": __version__},
        "eval_id": model.eval_id,
        "adapter": {"name": model.adapter_name, "version": model.adapter_version},
        "verdict": result.verdict,
        "score": result.score,
        "findings": [finding_record(f) for f in result.findings],
        "unsupported": list(model.unsupported),
    }


def render_json(result: AuditResult) -> str:
    """Render an audit result as the JSON findings document."""
    return json.dumps(findings_document(result), indent=2)
