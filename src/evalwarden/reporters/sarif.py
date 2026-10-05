"""SARIF 2.1.0 reporter: findings for code-scanning consumers.

Rules come from the check registry -- the same `CheckMeta` text that
`evalwarden explain` prints -- and results come from the shared finding
records (`reporters.records`), so SARIF and JSON always describe the same
findings.

Severity maps to SARIF levels as ERROR/HIGH -> error, MEDIUM -> warning,
LOW -> note. A result's location points at the audited artifact file when
the finding names one (prefixed with `base_uri`, the artifact path, when
given); a finding with no file gets a logical location naming the eval. A
file or line is never invented.
"""
from __future__ import annotations

import json

from .. import __version__
from ..checks import REGISTRY
from ..engine import AuditResult
from ..model import Finding
from .records import sarif_level

SARIF_VERSION = "2.1.0"
SARIF_SCHEMA = "https://json.schemastore.org/sarif-2.1.0.json"
INFORMATION_URI = "https://github.com/jurayh/evalwarden"


def _rule_name(check_id: str) -> str:
    """SARIF rule name: the check id as a plain identifier (ENV-001 -> ENV001)."""
    return check_id.replace("-", "")


def _rules() -> list[dict]:
    """All registered checks as SARIF reporting descriptors."""
    rules = []
    for check in REGISTRY:
        meta = check.meta
        rules.append(
            {
                "id": meta.id,
                "name": _rule_name(meta.id),
                "shortDescription": {"text": meta.title},
                "fullDescription": {"text": meta.threat},
                "help": {"text": f"{meta.threat}\n\nFix: {meta.remediation}"},
            }
        )
    return rules


def _locations(finding: Finding, eval_id: str, base_uri: str | None) -> list[dict]:
    """SARIF locations for a finding; logical fallback, never invented files."""
    if not finding.locations:
        return [{"logicalLocations": [{"name": eval_id, "kind": "module"}]}]
    locations = []
    for loc in finding.locations:
        uri = loc.file
        if base_uri and not uri.startswith("/"):
            uri = f"{base_uri.rstrip('/')}/{uri}"
        physical: dict = {"artifactLocation": {"uri": uri}}
        region: dict = {}
        if loc.line is not None:
            region["startLine"] = loc.line
        if loc.excerpt:
            region["snippet"] = {"text": loc.excerpt}
        if region:
            physical["region"] = region
        locations.append({"physicalLocation": physical})
    return locations


def _result(finding: Finding, eval_id: str, base_uri: str | None) -> dict:
    return {
        "ruleId": finding.id,
        "level": sarif_level(finding.severity),
        "message": {"text": finding.title},
        "locations": _locations(finding, eval_id, base_uri),
        "partialFingerprints": {"evalwarden/v1": finding.fingerprint},
        "properties": {
            "severity": finding.severity.value,
            "confidence": finding.confidence.value,
            "evidence": list(finding.evidence),
            "remediation": finding.remediation,
        },
    }


def sarif_document(result: AuditResult, base_uri: str | None = None) -> dict:
    """Build the SARIF 2.1.0 log for an audit result.

    `base_uri` (the audited artifact's path, as given on the command line)
    prefixes relative artifact locations so consumers can resolve them;
    locations are used as recorded when it is None.
    """
    model = result.model
    rules = _rules()
    known = {rule["id"] for rule in rules}
    for finding in result.findings:
        # Defensive: a result must never reference a rule that is absent.
        if finding.id not in known:
            rules.append(
                {
                    "id": finding.id,
                    "name": _rule_name(finding.id),
                    "shortDescription": {"text": finding.title},
                    "fullDescription": {"text": finding.description},
                    "help": {"text": finding.remediation},
                }
            )
            known.add(finding.id)
    run = {
        "tool": {
            "driver": {
                "name": "evalwarden",
                "version": __version__,
                "informationUri": INFORMATION_URI,
                "rules": rules,
            }
        },
        "results": [_result(f, model.eval_id, base_uri) for f in result.findings],
        "properties": {
            "eval_id": model.eval_id,
            "verdict": result.verdict,
            "integrity_score": result.score,
        },
    }
    return {"$schema": SARIF_SCHEMA, "version": SARIF_VERSION, "runs": [run]}


def render_sarif(result: AuditResult, base_uri: str | None = None) -> str:
    """Render an audit result as a SARIF 2.1.0 log."""
    return json.dumps(sarif_document(result, base_uri), indent=2)
