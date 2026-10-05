"""Shared finding serialization for the machine-readable reporters.

JSON and SARIF are two renderings of the same records. This module owns the
record shapes so the formats cannot drift apart: the JSON findings document
embeds `finding_record` dicts verbatim, and the SARIF renderer maps the same
records onto SARIF 2.1.0 results.
"""
from __future__ import annotations

from ..model import Finding, Severity, SourceLocation

#: SARIF 2.1.0 result level per severity. SARIF cannot distinguish ERROR
#: from HIGH; both mean the result should block, so both map to "error".
SARIF_LEVELS: dict[Severity, str] = {
    Severity.ERROR: "error",
    Severity.HIGH: "error",
    Severity.MEDIUM: "warning",
    Severity.LOW: "note",
}


def location_record(location: SourceLocation) -> dict:
    """One artifact location as a plain dict.

    Only facts the check actually recorded are emitted: `line` and
    `excerpt` keys appear only when the location carries them.
    """
    record: dict = {"file": location.file}
    if location.line is not None:
        record["line"] = location.line
    if location.excerpt:
        record["excerpt"] = location.excerpt
    return record


def finding_record(finding: Finding) -> dict:
    """The stable public record for one finding, shared by JSON and SARIF."""
    return {
        "check_id": finding.id,
        "severity": finding.severity.value,
        "confidence": finding.confidence.value,
        "title": finding.title,
        "message": finding.description,
        "evidence": list(finding.evidence),
        "locations": [location_record(loc) for loc in finding.locations],
        "remediation": finding.remediation,
        "fingerprint": finding.fingerprint,
    }


def sarif_level(severity: Severity) -> str:
    """The SARIF 2.1.0 result level for a severity (see SARIF_LEVELS)."""
    return SARIF_LEVELS[severity]
