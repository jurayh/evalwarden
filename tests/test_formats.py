"""Machine-format tests: JSON/SARIF reporters, --format parity, action.yml.

The kill criterion lives here: the leaky fixture must yield the same set of
findings (check id + SARIF level + artifact file) in text, JSON, and SARIF;
the hardened fixture must be empty everywhere with exit 0; audit exit codes
must be identical across formats.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

import evalwarden
from evalwarden.checks import REGISTRY
from evalwarden.cli import app
from evalwarden.engine import AuditResult, audit
from evalwarden.model import Confidence, Finding, Severity, SourceLocation
from evalwarden.reporters.json_report import findings_document, render_json
from evalwarden.reporters.records import location_record, sarif_level
from evalwarden.reporters.sarif import render_sarif, sarif_document

from .conftest import DEMO_HARDENED, DEMO_LEAKY, REPO_ROOT, make_model

runner = CliRunner()

_HEADER_RE = re.compile(r"^[EHML] ([A-Z]+-\d{3}) \[(\w+)\|confidence:\w+\] ")
_AT_RE = re.compile(r"^\s+at: (\S+)")


def _extract_json(text: str) -> dict:
    """Parse the first JSON document in CLI output (status notes may follow)."""
    doc, _ = json.JSONDecoder().raw_decode(text[text.index("{") :])
    return doc


def _file_of(rendered: str) -> str:
    """Artifact filename from a rendered location (`file` or `file:line`)."""
    head, _, tail = rendered.rpartition(":")
    return head if tail.isdigit() else rendered


def _text_findings(output: str) -> set[tuple[str, str, str | None]]:
    """(check id, SARIF level, artifact file) triples from terminal output."""
    by_key: dict[tuple[str, str], set[str]] = {}
    current: tuple[str, str] | None = None
    for line in output.splitlines():
        header = _HEADER_RE.match(line)
        if header:
            current = (header.group(1), sarif_level(Severity(header.group(2))))
            by_key.setdefault(current, set())
            continue
        at = _AT_RE.match(line)
        if at and current is not None:
            by_key[current].add(_file_of(at.group(1)))
    return {
        (check_id, level, filename)
        for (check_id, level), files in by_key.items()
        for filename in (files or {None})
    }


def _json_findings(doc: dict) -> set[tuple[str, str, str | None]]:
    triples: set[tuple[str, str, str | None]] = set()
    for finding in doc["findings"]:
        level = sarif_level(Severity(finding["severity"]))
        files = {loc["file"] for loc in finding["locations"]} or {None}
        triples |= {(finding["check_id"], level, filename) for filename in files}
    return triples


def _sarif_findings(doc: dict, base: str) -> set[tuple[str, str, str | None]]:
    triples: set[tuple[str, str, str | None]] = set()
    for sarif_result in doc["runs"][0]["results"]:
        files = set()
        for location in sarif_result["locations"]:
            physical = location.get("physicalLocation")
            if physical:
                uri = physical["artifactLocation"]["uri"]
                if uri.startswith(base + "/"):
                    uri = uri[len(base) + 1 :]
                files.add(uri)
        triples |= {
            (sarif_result["ruleId"], sarif_result["level"], filename)
            for filename in (files or {None})
        }
    return triples


# ---------------------------------------------------------------- kill criterion


def test_format_parity_on_leaky(tmp_path: Path):
    outputs, codes = {}, {}
    for fmt in ("text", "json", "sarif"):
        invoked = runner.invoke(
            app,
            [
                "audit",
                str(DEMO_LEAKY),
                "--output",
                str(tmp_path / f"report-{fmt}.html"),
                "--format",
                fmt,
            ],
        )
        outputs[fmt] = invoked.output
        codes[fmt] = invoked.exit_code
    assert codes == {"text": 1, "json": 1, "sarif": 1}

    text_set = _text_findings(outputs["text"])
    json_set = _json_findings(_extract_json(outputs["json"]))
    sarif_set = _sarif_findings(_extract_json(outputs["sarif"]), str(DEMO_LEAKY))

    assert text_set, "leaky fixture must produce findings"
    assert {check_id for check_id, _, _ in text_set} >= {"ENV-001", "GRAD-001"}
    assert text_set == json_set == sarif_set


@pytest.mark.parametrize("fmt", ["text", "json", "sarif"])
def test_hardened_is_empty_and_passes_in_every_format(tmp_path: Path, fmt: str):
    invoked = runner.invoke(
        app,
        [
            "audit",
            str(DEMO_HARDENED),
            "--output",
            str(tmp_path / "report.html"),
            "--format",
            fmt,
        ],
    )
    assert invoked.exit_code == 0, invoked.output
    if fmt == "json":
        assert _extract_json(invoked.output)["findings"] == []
    if fmt == "sarif":
        assert _extract_json(invoked.output)["runs"][0]["results"] == []


def test_audit_unknown_format_exits_2(tmp_path: Path):
    invoked = runner.invoke(
        app,
        [
            "audit",
            str(DEMO_LEAKY),
            "--output",
            str(tmp_path / "report.html"),
            "--format",
            "yaml",
        ],
    )
    assert invoked.exit_code == 2


# ---------------------------------------------------------------- JSON schema


def test_json_document_schema(leaky_dir: Path):
    doc = findings_document(audit(leaky_dir))
    assert doc["schema"] == "evalwarden.findings"
    assert doc["schema_version"] == "1.0"
    assert doc["tool"] == {"name": "evalwarden", "version": evalwarden.__version__}
    assert doc["eval_id"]
    assert doc["adapter"]["name"]
    assert doc["verdict"] == "BLOCKED"
    assert isinstance(doc["score"], int)
    assert doc["findings"]
    for record in doc["findings"]:
        assert set(record) == {
            "check_id",
            "severity",
            "confidence",
            "title",
            "message",
            "evidence",
            "locations",
            "remediation",
            "fingerprint",
        }
        assert record["remediation"], "every finding record carries the fix"
        assert record["fingerprint"]


def test_render_json_round_trips(leaky_dir: Path):
    assert json.loads(render_json(audit(leaky_dir)))["schema"] == "evalwarden.findings"


def test_location_record_omits_absent_fields():
    assert location_record(SourceLocation(file="a.json")) == {"file": "a.json"}
    assert location_record(SourceLocation(file="a.json", line=3, excerpt="e")) == {
        "file": "a.json",
        "line": 3,
        "excerpt": "e",
    }


# ---------------------------------------------------------------- SARIF validity


def test_sarif_structure_and_registry_rules(leaky_dir: Path):
    result = audit(leaky_dir)
    doc = sarif_document(result, base_uri="/artifacts/leaky")
    assert doc["version"] == "2.1.0"
    assert doc["$schema"].endswith("sarif-2.1.0.json")
    assert len(doc["runs"]) == 1

    run = doc["runs"][0]
    driver = run["tool"]["driver"]
    assert driver["name"] == "evalwarden"
    assert driver["version"] == evalwarden.__version__

    rules = driver["rules"]
    assert {rule["id"] for rule in rules} == {check.meta.id for check in REGISTRY}
    for rule in rules:
        assert rule["name"] == rule["id"].replace("-", "")
        assert rule["shortDescription"]["text"]
        assert rule["fullDescription"]["text"]
        assert "Fix:" in rule["help"]["text"]

    assert len(run["results"]) == len(result.findings)
    for sarif_result, finding in zip(run["results"], result.findings):
        assert sarif_result["ruleId"] == finding.id
        assert sarif_result["level"] == sarif_level(finding.severity)
        assert sarif_result["message"]["text"] == finding.title
        assert sarif_result["partialFingerprints"]["evalwarden/v1"] == finding.fingerprint
        properties = sarif_result["properties"]
        assert properties["severity"] == finding.severity.value
        assert properties["confidence"] == finding.confidence.value
        for location in sarif_result["locations"]:
            physical = location["physicalLocation"]
            assert physical["artifactLocation"]["uri"].startswith("/artifacts/leaky/")
            # This fixture records no line numbers: none may be invented.
            assert "startLine" not in physical.get("region", {})


@pytest.mark.parametrize(
    ("severity", "level"),
    [
        (Severity.ERROR, "error"),
        (Severity.HIGH, "error"),
        (Severity.MEDIUM, "warning"),
        (Severity.LOW, "note"),
    ],
)
def test_sarif_level_mapping(severity: Severity, level: str):
    assert sarif_level(severity) == level


def _bare_result(findings: list[Finding]) -> AuditResult:
    return AuditResult(
        model=make_model(), findings=findings, score=100, verdict="PASS", blocked_by=[]
    )


def test_sarif_logical_location_when_finding_has_no_file():
    finding = Finding(
        id="TRAJ-001",
        title="looping tool call",
        severity=Severity.HIGH,
        confidence=Confidence.HIGH,
        description="the trajectory repeats itself",
    )
    doc = sarif_document(_bare_result([finding]))
    (location,) = doc["runs"][0]["results"][0]["locations"]
    assert location == {"logicalLocations": [{"name": "test-eval", "kind": "module"}]}


def test_sarif_region_only_from_recorded_line_and_excerpt():
    finding = Finding(
        id="ENV-001",
        title="leak",
        severity=Severity.ERROR,
        confidence=Confidence.HIGH,
        description="d",
        locations=[SourceLocation(file="run.json", line=7, excerpt="TASK_ID=...")],
    )
    doc = sarif_document(_bare_result([finding]), base_uri="evals/demo")
    (location,) = doc["runs"][0]["results"][0]["locations"]
    physical = location["physicalLocation"]
    assert physical["artifactLocation"]["uri"] == "evals/demo/run.json"
    assert physical["region"] == {"startLine": 7, "snippet": {"text": "TASK_ID=..."}}

    no_base = sarif_document(_bare_result([finding]))
    uri = no_base["runs"][0]["results"][0]["locations"][0]["physicalLocation"][
        "artifactLocation"
    ]["uri"]
    assert uri == "run.json"


def test_render_sarif_round_trips(leaky_dir: Path):
    doc = json.loads(render_sarif(audit(leaky_dir)))
    assert doc["version"] == "2.1.0"
    assert doc["runs"][0]["results"]


# ---------------------------------------------------------------- GitHub Action


def test_action_yml_shape():
    action = yaml.safe_load((REPO_ROOT / "action.yml").read_text(encoding="utf-8"))
    assert action["runs"]["using"] == "composite"
    assert {"path", "format", "version", "fail-on"} <= set(action["inputs"])
    assert action["outputs"]["exit-code"]["value"] == "${{ steps.audit.outputs.exit-code }}"

    steps = action["runs"]["steps"]
    script = "\n".join(step.get("run", "") for step in steps)
    assert "evalwarden==${{ inputs.version }}" in script
    assert "--format sarif" in script
    assert ".github/workflows" not in script

    upload = next(
        step
        for step in steps
        if step.get("uses", "").startswith("github/codeql-action/upload-sarif@")
    )
    assert upload["with"]["sarif_file"] == "evalwarden.sarif"
    assert "sarif" in upload["if"]

    enforce = steps[-1]
    assert "steps.audit.outputs.exit-code" in enforce["run"]
    assert 'exit "$code"' in enforce["run"]
