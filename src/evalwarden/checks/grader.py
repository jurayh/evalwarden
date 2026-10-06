"""Grader checks.

GRAD-001: the verifier/oracle is writable by the agent. If the agent can edit
the thing that decides pass/fail, the score is whatever the agent wants.

GRAD-002: the grader grants credit without completion. Two deterministic
signals: the grader is explicitly configured to accept empty output, or a run
record shows an explicitly empty submission scored as a pass.

GRAD-003: the verifier's verdicts depend on surface form. Cached outputs are
re-scored against logically isomorphic clones of each task (renamed
entities, reordered elements, reparameterized values) under the scoring
rule the artifact declares. A rule that checks real task success scores
clones like originals; a rule matching the answer's surface string loses
passes it should keep.
"""
from __future__ import annotations

from ..clones import SUPPORTED_RULES, clone_gap, parse_cloneable
from ..model import Confidence, Finding, IntegrityModel, Severity, SourceLocation
from .base import Check, CheckMeta

GAP_AT = 0.25  # clone gap at or above this is verifier surface-sensitivity
MIN_CLONE_ITEMS = 8  # cloneable items needed before a gap means anything
MAX_FLIPS_SHOWN = 5  # flipped task ids shown per finding


class VerifierWritableCheck(Check):
    meta = CheckMeta(
        id="GRAD-001",
        title="Verifier writable by the agent",
        threat=(
            "If the agent can write to the verifier, oracle, or test files, it can "
            "change the measurement itself: weaken assertions, plant expected "
            "outputs, or mark itself passed. The score then reflects the agent's "
            "access, not its capability."
        ),
        remediation=(
            "Mount the verifier and all test/oracle files read-only for the agent "
            "(writable only by the harness), and verify ownership and permissions "
            "in CI."
        ),
    )

    def run(self, model: IntegrityModel) -> list[Finding]:
        grader = model.grader
        if not grader.verifier_writable_by_agent:
            return []
        path = grader.verifier_path or "<unknown verifier path>"
        return [
            Finding(
                id=self.meta.id,
                title=f"Verifier writable by the agent: {path}",
                severity=Severity.ERROR,
                confidence=Confidence.HIGH,
                description=self.meta.threat,
                evidence=[
                    f"grader declares verifier {path!r} writable by the agent.",
                    "An agent with write access to the oracle can alter pass/fail "
                    "outcomes directly.",
                ],
                locations=[SourceLocation(file=model.artifact_file("grader.json"), excerpt="verifier.writable_by_agent")],
                remediation=self.meta.remediation,
            )
        ]


class EmptyPathCheck(Check):
    meta = CheckMeta(
        id="GRAD-002",
        title="Grader grants credit without completion",
        threat=(
            "A grader that passes empty, errored, or trivial submissions inflates "
            "scores: agents get credit for doing nothing. Common causes are "
            "misconfigured scorers, tests that assert nothing, or pipelines that "
            "treat missing output as success."
        ),
        remediation=(
            "Require non-empty, task-relevant output before scoring; add a control "
            "case that submits nothing and assert it fails; fix the scorer "
            "configuration."
        ),
    )

    def run(self, model: IntegrityModel) -> list[Finding]:
        findings: list[Finding] = []
        if model.grader.accepts_empty_output:
            findings.append(
                Finding(
                    id=self.meta.id,
                    title="Grader is configured to accept empty output",
                    severity=Severity.ERROR,
                    confidence=Confidence.HIGH,
                    description=self.meta.threat,
                    evidence=[
                        "grader.json sets accepts_empty_output=true.",
                        "Any submission -- including no submission at all -- can score as a pass.",
                    ],
                    locations=[SourceLocation(file=model.artifact_file("grader.json"), excerpt="accepts_empty_output")],
                    remediation=self.meta.remediation,
                )
            )
        for attempt in model.attempts:
            # Only flag when the run record explicitly marks the submission as
            # empty AND the grader passed it. A zero-tool-call pass on a real
            # answer is legitimate and must not be flagged.
            if attempt.empty_submission and attempt.status == "pass":
                findings.append(
                    Finding(
                        id=self.meta.id,
                        title=f"Empty submission scored as pass: {attempt.task_id}",
                        severity=Severity.ERROR,
                        confidence=Confidence.HIGH,
                        description=self.meta.threat,
                        evidence=[
                            f"task {attempt.task_id!r}: submission marked empty, "
                            f"status={attempt.status!r}, score={attempt.score}.",
                            "Credit was granted without any completed work.",
                        ],
                        locations=[SourceLocation(file=model.artifact_file("run.json"), excerpt=f"attempts[] task_id={attempt.task_id}")],
                        remediation=self.meta.remediation,
                    )
                )
        return findings


class CloneGapCheck(Check):
    meta = CheckMeta(
        id="GRAD-003",
        title="Verifier verdicts depend on surface form (clone gap)",
        threat=(
            "A verifier that rewards the answer's surface form instead of "
            "task success can be gamed by memorizing instance-specific "
            "strings: the score then measures pattern matching against the "
            "stored answer, not whether the task was solved. The gap is "
            "invisible on the original items, because memorized outputs "
            "match them exactly."
        ),
        remediation=(
            "Score the underlying task success, not the rendered answer: "
            "parse the output and compare it semantically (set, numeric, "
            "or structural equality) against the reference, so verdicts "
            "are invariant under renamed entities, reordered elements, "
            "and reparameterized values."
        ),
    )

    def run(self, model: IntegrityModel) -> list[Finding]:
        rule = model.grader.verifier_rule
        if rule is None or rule not in SUPPORTED_RULES:
            # No declared rule, or a rule this audit does not implement:
            # there is nothing honest to re-score against. Stay silent.
            return []
        outputs: dict[str, str] = {}
        for attempt in model.attempts:
            if attempt.output is not None and attempt.task_id not in outputs:
                outputs[attempt.task_id] = attempt.output
        tasks = [
            parsed
            for task in model.tasks
            if (
                parsed := parse_cloneable(
                    task.id, task.metadata, task.target, outputs.get(task.id)
                )
            )
            is not None
        ]
        stats = clone_gap(tasks, rule)
        if stats is None or stats.n_items < MIN_CLONE_ITEMS:
            return []
        if stats.gap < GAP_AT:
            return []
        flipped = ", ".join(stats.flipped_ids[:MAX_FLIPS_SHOWN])
        if stats.flips > MAX_FLIPS_SHOWN:
            flipped += f", ... (+{stats.flips - MAX_FLIPS_SHOWN} more)"
        return [
            Finding(
                id=self.meta.id,
                title=(
                    f"Clone gap {stats.gap:.2f} under declared rule "
                    f"{rule!r}: {stats.clone_passes}/{stats.n_items} clone "
                    f"passes vs {stats.original_passes}/{stats.n_items} "
                    "on originals"
                ),
                severity=Severity.HIGH,
                confidence=Confidence.MEDIUM,
                description=self.meta.threat,
                evidence=[
                    f"declared verifier rule {rule!r} re-scored on "
                    f"{stats.n_items} cloneable selection tasks: pass rate "
                    f"{stats.original_rate:.0%} on originals, "
                    f"{stats.clone_rate:.0%} on isomorphic clones "
                    f"(renamed entities, reordered rosters, rescaled values).",
                    f"clone gap {stats.gap:.2f} at or above {GAP_AT:.2f}: "
                    "a quarter or more of the passes depend on the "
                    "instance's surface form, not on task success.",
                    f"{stats.flips} item(s) flipped pass -> fail under "
                    f"cloning: {flipped}.",
                    "Scope: the declared rule was re-scored, not the "
                    "verifier's code (which this audit never executes), "
                    "and clones cover the one task family the dataset "
                    "marks cloneable. A faithful declaration plus a gap "
                    "this size is the signal; either alone is not.",
                ],
                locations=[
                    SourceLocation(
                        file=model.artifact_file("grader.json"),
                        excerpt="verifier.rule",
                    ),
                    SourceLocation(
                        file=model.artifact_file("tasks.json"),
                        excerpt=(
                            f"tasks[].metadata.clone_family over "
                            f"{stats.n_items} cloneable items"
                        ),
                    ),
                ],
                remediation=self.meta.remediation,
            )
        ]
