"""Eval noise budget checks: is a score change signal or weather?

NOISE-001 asks where the noise lives: repeated runs re-roll fresh
generations (sampling), re-grades of frozen outputs (judge noise), and
repeated environments (flakiness), and when one die casts most of the
variance, averaging more of the others buys nothing. NOISE-002 asks
whether a claimed improvement clears the noise those dice produce, or
sits inside it.

Both read per-item scores from repeated runs (IntegrityModel.run_scores)
and decompose the variance with the pooled within-cell estimators in
evalwarden.noise. Precision-first throughout: a source the run conditions
never varied is reported as unmeasured, never as zero; a claim below the
partial floor is within noise for certain (the floor only grows), while
a claim above it with unmeasured sources is honestly undecidable. A
single run produces no findings -- there is nothing to decompose.
"""
from __future__ import annotations

from ..model import Confidence, Finding, IntegrityModel, Severity, SourceLocation
from ..noise import (
    ENVIRONMENT,
    INTERACTION,
    JUDGE,
    SAMPLING,
    SOURCES,
    NoiseBudget,
    claim_verdict,
    noise_budget,
)
from .base import Check, CheckMeta

MIN_ITEMS = 10  # items with repeated scores before a budget means anything
DOMINANCE_SHARE = 0.5  # one source at or above this share owns the budget
MAX_UNMEASURED_SHOWN = 3

_SOURCE_LABEL = {
    SAMPLING: "sampling (fresh generations)",
    JUDGE: "judge (re-grades of frozen outputs)",
    ENVIRONMENT: "environment (tool/env flakiness)",
    INTERACTION: "interactions between run conditions",
}

_SOURCE_FIX = {
    SAMPLING: (
        "Average more fresh generations per item, lower sampling "
        "temperature, or accept a wider noise floor before comparing runs."
    ),
    JUDGE: (
        "Pin judge temperature, average repeated grades per item, or "
        "tighten the rubric: re-grading frozen outputs should not move "
        "scores this much."
    ),
    ENVIRONMENT: (
        "Stabilize the environment (pin tool versions, seed or mock "
        "flaky calls, retry infrastructure errors): repeated executions "
        "of the same condition should not move scores this much."
    ),
    INTERACTION: (
        "Run conditions interact: changing one condition changes what the "
        "others do. Freeze two of the three axes while studying the third."
    ),
}


def _noise_loc(model: IntegrityModel, budget: NoiseBudget) -> SourceLocation:
    return SourceLocation(
        file=model.artifact_file("run_scores.json"),
        excerpt=(
            f"run_scores[] over {budget.n_items} items "
            f"({budget.n_scores} scores)"
        ),
    )


def _budget_evidence(budget: NoiseBudget) -> list[str]:
    evidence: list[str] = []
    shares = budget.shares()
    for source in SOURCES:
        value = budget.components.get(source)
        if value is None:
            evidence.append(
                f"{_SOURCE_LABEL[source]}: unmeasured "
                f"({budget.levels[source]} level(s) observed) -- the data "
                "says nothing about this source, so it contributes no "
                "estimate, not a zero."
            )
        else:
            evidence.append(
                f"{_SOURCE_LABEL[source]}: variance {value:.4f} "
                f"({shares.get(source, 0.0):.0%} of measured noise)."
            )
    if budget.interaction is not None:
        evidence.append(
            f"interactions: variance {budget.interaction:.4f} "
            f"({shares.get(INTERACTION, 0.0):.0%} of measured noise)."
        )
    evidence.append(
        f"noise floor +/-{budget.noise_floor:.3f}: two protocol-identical "
        "scores must differ by at least this to mean anything at 95% "
        "confidence."
    )
    return evidence


def _tuples(model: IntegrityModel) -> list[tuple[str, float, int, int, int]]:
    return [
        (
            rs.task_id,
            rs.score,
            rs.generation_index,
            rs.grade_index,
            rs.environment_index,
        )
        for rs in model.run_scores
    ]


class DominantNoiseCheck(Check):
    meta = CheckMeta(
        id="NOISE-001",
        title="One noise source dominates the eval budget",
        threat=(
            "Repeated runs re-roll fresh generations, re-grades, and "
            "environments. When one source casts most of the variance, "
            "score comparisons mostly measure that source: averaging more "
            "of the quiet axes buys nothing, and run-to-run deltas are one "
            "die's mood. Repeated-run scores with run-condition tags are "
            "required; without them there is no budget to dominate."
        ),
        remediation=(
            "Attack the dominant source first: it, not the others, sets "
            "the noise floor."
        ),
    )

    def run(self, model: IntegrityModel) -> list[Finding]:
        if not model.run_scores:
            return []
        budget = noise_budget(_tuples(model))
        if budget is None or budget.n_items < MIN_ITEMS:
            return []
        if len(budget.measured_sources) < 2:
            # One measured axis is total variance, not a decomposition:
            # it cannot attribute dominance to anything.
            return []
        dominant = budget.dominant()
        if dominant is None or dominant[1] < DOMINANCE_SHARE:
            return []
        source, share = dominant
        return [
            Finding(
                id=self.meta.id,
                title=(
                    f"{_SOURCE_LABEL[source]} casts {share:.0%} of the "
                    "eval's noise budget"
                ),
                severity=Severity.MEDIUM,
                confidence=Confidence.MEDIUM,
                description=self.meta.threat,
                evidence=_budget_evidence(budget),
                locations=[_noise_loc(model, budget)],
                remediation=_SOURCE_FIX[source],
            )
        ]


class ClaimedDeltaCheck(Check):
    meta = CheckMeta(
        id="NOISE-002",
        title="Claimed score change versus the noise floor",
        threat=(
            "An eval that reports an improvement smaller than its own "
            "run-to-run noise is reporting weather as progress. Repeated "
            "runs set a floor: deltas inside it carry no signal, and a "
            "delta claimed without repeated-run data cannot be judged "
            "at all."
        ),
        remediation=(
            "Report score changes with the noise floor attached: re-run "
            "both configurations under the same protocol until the delta "
            "clears the floor, or withdraw the improvement claim."
        ),
    )

    def run(self, model: IntegrityModel) -> list[Finding]:
        claimed = model.claimed_delta
        if claimed is None or not model.run_scores:
            return []
        budget = noise_budget(_tuples(model))
        if budget is None or budget.n_items < MIN_ITEMS:
            return [
                Finding(
                    id=self.meta.id,
                    title=(
                        f"Claimed improvement of {claimed:+.3f} cannot be "
                        "judged: too few repeated-run scores to measure "
                        "the noise floor"
                    ),
                    severity=Severity.MEDIUM,
                    confidence=Confidence.LOW,
                    description=self.meta.threat,
                    evidence=[
                        f"claimed delta {claimed:+.3f}, but "
                        f"{len(model.run_scores)} repeated-run score(s) "
                        f"over {len({rs.task_id for rs in model.run_scores})} "
                        f"item(s): below the {MIN_ITEMS}-item minimum for "
                        "a variance estimate, so no verdict is honest.",
                        "Statistical signal: collect repeated runs under "
                        "recorded conditions before claiming a delta.",
                    ],
                    locations=[
                        SourceLocation(
                            file=model.artifact_file("run_scores.json"),
                            excerpt="run_scores[] (claimed delta unjudged)",
                        )
                    ],
                    remediation=self.meta.remediation,
                )
            ]
        verdict = claim_verdict(claimed, budget)
        if verdict == "real":
            return []
        if verdict == "cant_tell":
            unmeasured = ", ".join(
                _SOURCE_LABEL[s] for s in budget.unmeasured_sources
            )
            return [
                Finding(
                    id=self.meta.id,
                    title=(
                        f"Claimed improvement of {claimed:+.3f} cannot be "
                        "judged: it clears the partial noise floor "
                        f"(+/-{budget.noise_floor:.3f}) but {unmeasured} "
                        "noise is unmeasured"
                    ),
                    severity=Severity.MEDIUM,
                    confidence=Confidence.LOW,
                    description=self.meta.threat,
                    evidence=[
                        *_budget_evidence(budget),
                        "The floor above counts measured sources only; the "
                        "unmeasured sources can only raise it, so the claim "
                        "is undecidable at this protocol.",
                    ],
                    locations=[_noise_loc(model, budget)],
                    remediation=self.meta.remediation,
                )
            ]
        return [
            Finding(
                id=self.meta.id,
                title=(
                    f"Claimed improvement of {claimed:+.3f} is inside the "
                    f"noise floor (+/-{budget.noise_floor:.3f})"
                ),
                severity=Severity.MEDIUM,
                confidence=Confidence.MEDIUM,
                description=self.meta.threat,
                evidence=[
                    *_budget_evidence(budget),
                    f"claimed delta {claimed:+.3f} vs noise floor "
                    f"{budget.noise_floor:.3f}: re-running the same "
                    "protocol produces differences this large from noise "
                    "alone.",
                    "Statistical signal: confirm on independent runs "
                    "before reporting an improvement.",
                ],
                locations=[_noise_loc(model, budget)],
                remediation=self.meta.remediation,
            )
        ]


CHECKS = [
    DominantNoiseCheck(),  # NOISE-001
    ClaimedDeltaCheck(),  # NOISE-002
]
