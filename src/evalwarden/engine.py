"""Audit engine: adapter -> checks -> score -> verdict.

The pipeline is deliberately boring: detect the harness, translate to the
normalized model, run every registered check, score the findings. No network,
no mutation, no surprises.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import adapters  # noqa: F401  (the adapter modules imported below register themselves)
from .adapters import AuditError, autodetect
from .adapters import inspect_ai  # noqa: F401  (registers itself on import)
from .adapters import promptfoo  # noqa: F401  (registers itself on import)
from .adapters import claude_code  # noqa: F401  (registers itself on import)
from .adapters import codex  # noqa: F401  (registers itself on import)
from .adapters import universal  # noqa: F401  (registers itself on import)
from .checks import REGISTRY
from .checks.diff import PairedRegressionCheck, SeriesComparison
from .model import SEVERITY_ORDER, Finding, IntegrityModel, Severity


@dataclass
class AuditResult:
    model: IntegrityModel
    findings: list[Finding]
    score: int  # 0-100, diagnostic only: an explainable prioritization aid.
    verdict: str  # "BLOCKED" | "PASS"
    blocked_by: list[Finding]


def integrity_score(findings: list[Finding]) -> int:
    """Explainable, fixed deduction per severity. Diagnostic, not a certification."""
    deductions = {Severity.ERROR: 25, Severity.HIGH: 10, Severity.MEDIUM: 5, Severity.LOW: 1}
    score = 100 - sum(deductions.get(f.severity, 0) for f in findings)
    return max(0, score)


def sort_findings(findings: list[Finding]) -> list[Finding]:
    order = {sev: i for i, sev in enumerate(SEVERITY_ORDER)}
    return sorted(findings, key=lambda f: (order[f.severity], f.id, f.title))


_FAIL_ON_LEVELS = {
    "error": {Severity.ERROR},
    "high": {Severity.ERROR, Severity.HIGH},
    "medium": {Severity.ERROR, Severity.HIGH, Severity.MEDIUM},
    "low": {Severity.ERROR, Severity.HIGH, Severity.MEDIUM, Severity.LOW},
}


def audit(
    path: str | Path,
    adapter_name: str = "auto",
    price_in_per_1m: float = 3.0,
    price_out_per_1m: float = 15.0,
    budget_per_task_usd: float | None = None,
    training_file: str | Path | None = None,
) -> AuditResult:
    target = Path(path)
    if not target.exists():
        raise AuditError(f"no such eval artifact: {target}")

    # The training file is a side input for DATA-004 (train/eval
    # leakage): an eval artifact cannot contain the fine-tune's corpus,
    # so it travels beside the model and reaches the check through
    # with_options, the same channel the differ uses for its baseline.
    training_records = None
    if training_file is not None:
        from .leakage import TrainingFileError, load_training_file

        try:
            training_records = load_training_file(training_file)
        except TrainingFileError as exc:
            raise AuditError(str(exc)) from exc

    if adapter_name == "auto":
        adapter = autodetect(target)
    else:
        matches = [a for a in adapters.REGISTRY if a.name == adapter_name]
        if not matches:
            raise AuditError(f"unknown adapter {adapter_name!r}")
        adapter = matches[0]

    bundle = adapter.collect(target)
    model = adapter.normalize(bundle)

    # Every check gets the runtime options; checks that need them (token
    # prices, per-task budget) override Check.with_options. No special-casing.
    options = {
        "price_in_per_1m": price_in_per_1m,
        "price_out_per_1m": price_out_per_1m,
        "budget_per_task_usd": budget_per_task_usd,
        "training_records": training_records,
    }
    checks = [c.with_options(**options) for c in REGISTRY]
    findings: list[Finding] = []
    for check in checks:
        findings.extend(check.run(model))
    findings = sort_findings(findings)

    score = integrity_score(findings)
    blocked_by = [f for f in findings if f.severity in _FAIL_ON_LEVELS["high"]]
    verdict = "BLOCKED" if blocked_by else "PASS"
    return AuditResult(
        model=model, findings=findings, score=score, verdict=verdict, blocked_by=blocked_by
    )


def audit_with_policy(
    path: str | Path,
    adapter_name: str = "auto",
    fail_on: str = "high",
    price_in_per_1m: float = 3.0,
    price_out_per_1m: float = 15.0,
    budget_per_task_usd: float | None = None,
    training_file: str | Path | None = None,
) -> tuple[AuditResult, bool]:
    """Run the audit; return (result, policy_failed)."""
    result = audit(
        path,
        adapter_name,
        price_in_per_1m,
        price_out_per_1m,
        budget_per_task_usd,
        training_file=training_file,
    )
    levels = _FAIL_ON_LEVELS[fail_on]
    policy_failed = any(f.severity in levels for f in result.findings)
    return result, policy_failed


def load_model(path: str | Path, adapter_name: str = "auto") -> IntegrityModel:
    """Load one eval artifact into the normalized model.

    The front half of the audit pipeline (detect, collect, normalize),
    exposed on its own for comparative work: the differ consumes two
    models loaded exactly the way `audit` loads one.
    """
    target = Path(path)
    if not target.exists():
        raise AuditError(f"no such eval artifact: {target}")
    if adapter_name == "auto":
        adapter = autodetect(target)
    else:
        matches = [a for a in adapters.REGISTRY if a.name == adapter_name]
        if not matches:
            raise AuditError(f"unknown adapter {adapter_name!r}")
        adapter = matches[0]
    bundle = adapter.collect(target)
    return adapter.normalize(bundle)


@dataclass
class DiffResult:
    baseline: IntegrityModel
    candidate: IntegrityModel
    comparisons: list[SeriesComparison]  # per paired model series
    notes: list[str]  # series that existed on one side only
    findings: list[Finding]
    score: int  # 0-100, the audit deduction table over the diff findings
    verdict: str  # "BLOCKED" | "PASS"
    blocked_by: list[Finding]


def diff_models(
    baseline: IntegrityModel,
    candidate: IntegrityModel,
) -> DiffResult:
    """Compare two normalized result sets with the DIFF lane (DIFF-001).

    Verdict semantics mirror the audit's: BLOCKED when any finding is
    HIGH or above. The score reuses the audit's deduction table over the
    diff findings; it prioritizes, it does not certify.
    """
    check = PairedRegressionCheck().with_options(baseline_model=baseline)
    comparisons, notes = check.comparisons(candidate)
    findings = sort_findings(check.run(candidate))
    score = integrity_score(findings)
    blocked_by = [f for f in findings if f.severity in _FAIL_ON_LEVELS["high"]]
    return DiffResult(
        baseline=baseline,
        candidate=candidate,
        comparisons=comparisons,
        notes=notes,
        findings=findings,
        score=score,
        verdict="BLOCKED" if blocked_by else "PASS",
        blocked_by=blocked_by,
    )


def diff_pair(
    baseline_path: str | Path,
    candidate_path: str | Path,
    adapter_name: str = "auto",
    fail_on: str = "high",
) -> tuple[DiffResult, bool]:
    """Load two artifacts and diff them; return (result, policy_failed)."""
    baseline = load_model(baseline_path, adapter_name)
    candidate = load_model(candidate_path, adapter_name)
    result = diff_models(baseline, candidate)
    levels = _FAIL_ON_LEVELS[fail_on]
    policy_failed = any(f.severity in levels for f in result.findings)
    return result, policy_failed
