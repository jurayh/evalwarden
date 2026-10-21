"""Dataset audit checks: is the eval itself worth running?

DATA-001 measures saturation: when every model gets the same items right, the
dataset no longer discriminates and reported scores are theater.
DATA-002 measures redundancy: near-duplicate items waste eval budget and
inflate confidence in coverage.
DATA-003 measures elicitation coverage: whether each declared failure mode
has enough tagged items to actually surface the behavior -- the count-side
mirror of JUDGE-009's panel blind spots.
DATA-004 measures train/eval leakage: whether eval items appear, exactly
or nearly, in the training data the model was fine-tuned on. It needs a
second input no eval artifact carries -- the training file -- so it runs
only when the auditor supplies one (`audit --training-file`); without
one it stays silent, because an artifact alone cannot prove a negative
about data it does not contain.

Precision-first throughout: the checks stay silent when the data cannot
support the claim. Saturation needs per-item results from at least two
models (Attempt.model_id); redundancy needs enough task prompts to make a
fraction meaningful; coverage needs failure-mode tags on enough items to
make a per-mode count meaningful. A clean, discriminating dataset produces
no findings.
"""
from __future__ import annotations

from collections import defaultdict

from ..coverage import mode_item_counts, thin_modes
from ..dataset import redundancy_stats, saturation_stats
from ..leakage import (
    NEAR_CONTAINMENT,
    NEAR_JACCARD,
    LeakageMatch,
    LeakageStats,
    TrainingRecord,
    item_text,
    leakage_stats,
)
from ..model import Attempt, Confidence, Finding, IntegrityModel, Severity, SourceLocation
from .base import Check, CheckMeta

MIN_MODELS = 2  # saturation is about models no longer differing
MIN_COMMON_ITEMS = 30  # common items needed before calling a dataset saturated
SATURATED_AT = 0.5  # dead fraction at or above this is saturation
MIN_TASKS = 20  # prompts needed before calling a dataset redundant
DUPLICATE_AT = 0.10  # item duplicate fraction at or above this is redundancy
SIMILARITY = 0.8  # shingle-Jaccard at or above this is a near-duplicate
ELICIT_MIN = 30  # items carrying a failure mode below this cannot elicit it reliably
MIN_TAGGED_ITEMS = 30  # tagged items needed before per-mode counts mean anything
MAX_COVERAGE_MODES = 5  # thin/absent modes shown per finding
CONTAMINATED_ERROR_AT = 0.10  # contaminated wins at/above this share of passes
CONTAMINATED_HIGH_AT = 0.03  # contaminated wins at/above this share of passes
MAX_LEAK_MATCHES = 5  # matched pairs listed per leakage finding


def _data_loc(model: IntegrityModel, excerpt: str) -> SourceLocation:
    return SourceLocation(file=model.artifact_file("attempts.json"), excerpt=excerpt)


def _correct(attempt: Attempt) -> bool:
    if attempt.score is not None:
        return attempt.score >= 0.5
    return attempt.status == "pass"


class SaturationCheck(Check):
    meta = CheckMeta(
        id="DATA-001",
        title="Eval dataset looks saturated",
        threat=(
            "Most items are answered correctly by every model, so the dataset "
            "no longer discriminates between systems. Scores on it measure "
            "ceiling effects, not capability differences."
        ),
        remediation=(
            "Retire or refresh the saturated items: replace them with harder "
            "variants, upsample the disputed items that still separate "
            "models, or report scores with the saturation caveat attached."
        ),
    )

    def run(self, model: IntegrityModel) -> list[Finding]:
        by_model: dict[str, dict[str, bool]] = defaultdict(dict)
        for attempt in model.attempts:
            if attempt.model_id is None:
                continue
            by_model[attempt.model_id][attempt.task_id] = _correct(attempt)
        stats = saturation_stats(by_model)
        if stats is None or stats.n_items < MIN_COMMON_ITEMS:
            return []
        if stats.dead_fraction < SATURATED_AT:
            return []
        acc = ", ".join(f"{m}={a:.0%}" for m, a in stats.per_model_accuracy.items())
        return [
            Finding(
                id=self.meta.id,
                title=(
                    f"Dataset saturated: {stats.dead_fraction:.0%} of "
                    f"{stats.n_items} items answered correctly by all "
                    f"{stats.n_models} models"
                ),
                severity=Severity.MEDIUM,
                confidence=Confidence.MEDIUM,
                description=self.meta.threat,
                evidence=[
                    f"dead fraction {stats.dead_fraction:.0%} "
                    f"({stats.dead_fraction * stats.n_items:.0f}/{stats.n_items}) "
                    f"at or above {SATURATED_AT:.0%}: these items discriminate nothing.",
                    f"disputed fraction {stats.disputed_fraction:.0%}: only these "
                    f"items still separate the models.",
                    f"headroom {stats.headroom:.0%}; per-model accuracy: {acc}.",
                    "Statistical signal: confirm on a held-out item set before "
                    "retiring anything.",
                ],
                locations=[
                    _data_loc(
                        model,
                        f"attempts[].model_id/score over {stats.n_items} common items",
                    )
                ],
                remediation=self.meta.remediation,
            )
        ]


class RedundancyCheck(Check):
    meta = CheckMeta(
        id="DATA-002",
        title="Eval dataset contains near-duplicate items",
        threat=(
            "Near-duplicate prompts inflate the item count without adding "
            "coverage: the eval costs more to run than its effective size "
            "justifies, and scores overweight the duplicated behavior."
        ),
        remediation=(
            "Deduplicate the set: keep one representative per near-duplicate "
            "cluster, or rewrite the variants to test genuinely different "
            "behavior."
        ),
    )

    def run(self, model: IntegrityModel) -> list[Finding]:
        prompts = {t.id: t.prompt for t in model.tasks if t.prompt.strip()}
        if len(prompts) < MIN_TASKS:
            return []
        stats = redundancy_stats(prompts, threshold=SIMILARITY)
        if stats.duplicate_item_fraction < DUPLICATE_AT:
            return []
        examples = [
            f"{a} ~ {b} (similarity {s:.2f})" for a, b, s in stats.examples
        ]
        return [
            Finding(
                id=self.meta.id,
                title=(
                    f"Dataset redundant: {stats.duplicate_item_fraction:.0%} of "
                    f"{stats.n_items} items have a near-duplicate "
                    f"({stats.near_duplicate_pairs} pairs)"
                ),
                severity=Severity.LOW,
                confidence=Confidence.MEDIUM,
                description=self.meta.threat,
                evidence=[
                    f"duplicate item fraction {stats.duplicate_item_fraction:.0%} "
                    f"at or above {DUPLICATE_AT:.0%} "
                    f"(shingle-Jaccard >= {SIMILARITY}).",
                    *[f"near-duplicate: {e}" for e in examples],
                ],
                locations=[
                    SourceLocation(
                        file=model.artifact_file("tasks.json"),
                        excerpt=f"tasks[].prompt over {stats.n_items} items",
                    )
                ],
                remediation=self.meta.remediation,
            )
        ]


class ElicitationCoverageCheck(Check):
    meta = CheckMeta(
        id="DATA-003",
        title="Failure modes under-elicited",
        threat=(
            "A failure mode with only a handful of eliciting items cannot "
            "surface the behavior it names: the eval claims coverage it "
            "never exercises, and a declared mode with no items at all is a "
            "promise the dataset does not even attempt to keep."
        ),
        remediation=(
            "Author more items for the thin modes until each declared "
            "failure mode has enough eliciting samples to measure, or drop "
            "the mode from the eval's declared coverage."
        ),
    )

    def run(self, model: IntegrityModel) -> list[Finding]:
        mode_of = {
            t.id: t.failure_mode for t in model.tasks if t.failure_mode is not None
        }
        if len(mode_of) < MIN_TAGGED_ITEMS:
            # Too few tagged items to distinguish "modes are thin" from
            # "the adapter recorded no tags": stay silent.
            return []
        counts = mode_item_counts(mode_of)
        thin, absent = thin_modes(counts, model.failure_mode_taxonomy, ELICIT_MIN)
        if not thin and not absent:
            return []
        evidence: list[str] = []
        shown = 0
        for mode in absent:
            if shown >= MAX_COVERAGE_MODES:
                break
            evidence.append(
                f"declared mode '{mode}' has no items: the eval claims "
                "coverage it never elicits."
            )
            shown += 1
        for mode, count in thin:
            if shown >= MAX_COVERAGE_MODES:
                break
            evidence.append(
                f"mode '{mode}' has {count} item(s), below {ELICIT_MIN}: "
                "too few samples to elicit the failure mode reliably."
            )
            shown += 1
        total = len(absent) + len(thin)
        if total > shown:
            evidence.append(
                f"... and {total - shown} more under-elicited mode(s) "
                f"({len(counts)} modes over {len(mode_of)} tagged items)."
            )
        parts: list[str] = []
        if absent:
            parts.append(f"absent: {', '.join(absent)}")
        if thin:
            parts.append(
                "thin: "
                + ", ".join(f"{mode} ({count})" for mode, count in thin)
            )
        return [
            Finding(
                id=self.meta.id,
                title=(
                    f"Failure modes under-elicited ({'; '.join(parts)})"
                ),
                severity=Severity.MEDIUM,
                confidence=Confidence.MEDIUM,
                description=self.meta.threat,
                evidence=evidence,
                locations=[
                    SourceLocation(
                        file=model.artifact_file("tasks.json"),
                        excerpt=(
                            f"tasks[].failure_mode over {len(mode_of)} "
                            "tagged items"
                        ),
                    )
                ],
                remediation=self.meta.remediation,
            )
        ]


def _leak_severity(contaminated_share: float, wins: int) -> Severity:
    """Severity scales with contaminated wins, not raw overlap.

    Leaked items the model failed prove a boundary breach but did not
    inflate the score; leaked items it passed are wins it may not have
    earned, and their share of all passes is the inflation estimate.
    """
    if contaminated_share >= CONTAMINATED_ERROR_AT:
        return Severity.ERROR
    if contaminated_share >= CONTAMINATED_HIGH_AT:
        return Severity.HIGH
    if wins >= 1:
        return Severity.MEDIUM
    return Severity.LOW


class LeakageCheck(Check):
    meta = CheckMeta(
        id="DATA-004",
        title="Training data leaks into the eval set",
        threat=(
            "Eval items that appear in the fine-tune's training data -- "
            "verbatim or as lightly edited copies -- measure memory, not "
            "capability. The damning subset is the leaked items the model "
            "passed: those wins may be memorized, and their share of all "
            "passes is an upper-bound estimate of how much of the reported "
            "score the leak could account for."
        ),
        remediation=(
            "Decontaminate and re-measure: drop or rewrite the matched "
            "training rows (and their near-duplicates), retrain or select "
            "a checkpoint trained without them, and re-run the eval. "
            "Report scores with the contaminated-win count attached "
            "until a clean run exists."
        ),
    )

    def __init__(self, training_records: list[TrainingRecord] | None = None):
        self.training_records = training_records

    def with_options(self, **kwargs) -> "LeakageCheck":
        return LeakageCheck(
            training_records=kwargs.get("training_records", self.training_records)
        )

    def run(self, model: IntegrityModel) -> list[Finding]:
        if not self.training_records:
            # No training file supplied: an eval artifact alone cannot
            # speak about training data it does not contain. Silence.
            return []
        eval_texts = {
            task.id: item_text(task.prompt, task.target)
            for task in model.tasks
            if task.prompt.strip()
        }
        if not eval_texts:
            return []
        passed_ids = {
            attempt.task_id for attempt in model.attempts if _correct(attempt)
        }
        stats = leakage_stats(eval_texts, self.training_records, passed_ids)
        findings: list[Finding] = []
        for tier, matches, confidence, threshold_note in (
            (
                "exact",
                stats.exact,
                Confidence.HIGH,
                "exact tier: normalized prompt+target text identical "
                "to a training row",
            ),
            (
                "near-duplicate",
                stats.near,
                Confidence.MEDIUM,
                f"near-duplicate tier: shingle-Jaccard >= {NEAR_JACCARD} "
                f"or containment >= {NEAR_CONTAINMENT} (3-token shingles "
                "over normalized text, shingles shared by over half "
                "the eval set excluded as scaffolding)",
            ),
        ):
            finding = self._tier_finding(model, stats, tier, matches, confidence, threshold_note)
            if finding is not None:
                findings.append(finding)
        return findings

    def _tier_finding(
        self,
        model: IntegrityModel,
        stats: LeakageStats,
        tier: str,
        matches: list[LeakageMatch],
        confidence: Confidence,
        threshold_note: str,
    ) -> Finding | None:
        wins = sum(1 for m in matches if m.passed)
        share = wins / stats.total_passes if stats.total_passes else 0.0
        # A lone leaked item the model failed is a hygiene note for the
        # combined report, not a finding: it proves neither score
        # inflation nor a pattern. Anything more fires.
        if len(matches) < 2 and wins == 0:
            return None
        if not matches:
            return None
        shown = matches[:MAX_LEAK_MATCHES]
        evidence = [
            f"overlap: {len(matches)} of {stats.n_eval_items} eval items "
            f"match the {tier} tier against {stats.n_training_rows} "
            f"training rows (raw overlap rate "
            f"{len(matches) / stats.n_eval_items:.0%} for this tier; "
            f"{stats.overlap_rate:.0%} across tiers).",
            f"contaminated wins: {wins} of the matched items were passed; "
            f"that is {share:.0%} of all {stats.total_passes} passed "
            "items -- the headline number, since leaked-and-failed "
            "items did not inflate the score.",
            threshold_note + ".",
            *(
                f"match: training row {m.train_id} <-> eval item "
                f"{m.eval_id} (jaccard {m.jaccard:.2f}, containment "
                f"{m.containment:.2f}, outcome "
                f"{'passed' if m.passed else 'failed'})"
                for m in shown
            ),
        ]
        if len(matches) > len(shown):
            evidence.append(
                f"... and {len(matches) - len(shown)} more {tier} "
                "match(es), each verifiable the same way."
            )
        if tier != "exact":
            evidence.append(
                "Lexical detection only: heavily paraphrased copies can "
                "fall below the threshold, so a silent near-duplicate "
                "tier is not proof of a clean boundary."
            )
        if wins:
            title = (
                f"Train/eval leakage ({tier}): {wins} contaminated "
                f"win(s), {share:.0%} of all passes match training data"
            )
        else:
            title = (
                f"Train/eval leakage ({tier}): {len(matches)} eval "
                "item(s) match training data, none passed"
            )
        return Finding(
            id=self.meta.id,
            title=title,
            severity=_leak_severity(share, wins),
            confidence=confidence,
            description=self.meta.threat,
            evidence=evidence,
            locations=[
                SourceLocation(
                    file=model.artifact_file("tasks.json"),
                    excerpt=(
                        f"tasks[].prompt/target vs the supplied training "
                        f"file ({len(matches)} {tier} match(es))"
                    ),
                )
            ],
            remediation=self.meta.remediation,
        )


CHECKS = [
    SaturationCheck(),  # DATA-001
    RedundancyCheck(),  # DATA-002
    ElicitationCoverageCheck(),  # DATA-003
    LeakageCheck(),  # DATA-004
]
