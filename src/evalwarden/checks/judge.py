"""Judge reliability and calibration checks.

JUDGE-001 / JUDGE-002 come straight from the spec: a model judge with no
labeled calibration set, and a pairwise protocol without order
counterbalancing. JUDGE-003..006 are the v1 deterministic calibration slice:
self-consistency across repeats, agreement with reference labels, and
deterministic heuristics for position and verbosity bias. JUDGE-007 is the
probabilistic calibration slice: it measures whether the judge's stated
confidence tracks its empirical accuracy on labeled items. JUDGE-008 is the
ensemble slice: in a multi-judge panel, it measures whether each judge
contributes independent signal or just burns budget agreeing with the rest.
JUDGE-009 is the coverage slice: the judge x sample mirror of JUDGE-008 --
whether the panel, between its members, catches each failure mode the eval
claims to test, or whether whole modes pass unseen by every judge.

Precision-first throughout: bias heuristics carry minimum-sample guards and
Medium severity/confidence labels, and a clean, well-run judge produces no
findings. Raw judgment text never reaches these checks -- the adapter redacts
it at the boundary; only structured verdicts are analyzed.
"""
from __future__ import annotations

from collections import defaultdict

from ..calibration import bin_pairs, expected_calibration_error, mean_signed_gap
from ..coverage import blind_spots, mode_catch_table
from ..ensemble import (
    ablation_value,
    max_agreement_partner,
    unique_contribution,
)
from ..model import Confidence, Finding, IntegrityModel, Judgment, Severity, SourceLocation
from .base import Check, CheckMeta

MIN_REPEATS = 3          # repeats needed before calling a judge inconsistent
MIN_LABELS = 10          # labeled items needed before judging calibration
AGREEMENT_FLOOR = 0.75   # reference agreement below this is a finding
MIN_PAIRS = 20           # pairs needed before naming a bias
BIAS_RATE = 0.65         # win rate at or beyond this (either side) is a finding
SCORE_RANGE_FRAC = 0.2   # score spread beyond this fraction of scale is inconsistent
MIN_CALIBRATION_PAIRS = 30  # labeled confidence pairs needed before judging calibration
ECE_THRESHOLD = 0.15     # ECE at or above this is clear miscalibration
GAP_MARGIN = 0.10        # |mean signed gap| at or above this names a direction
MIN_PANEL_JUDGES = 2     # fewer judges than this: no panel to judge redundant
MIN_COMMON_ITEMS = 30    # common items below this: stay silent
MIN_FLAG_ITEMS = 50      # common items below this: measure, but never flag
REDUNDANCY_RATE = 0.05   # unique contribution below this is a finding
CATCH_BAR = 0.70         # judge x mode agreement at or above this is a catch
MIN_MODE_ITEMS = 20      # labeled items of a mode needed before naming a blind spot
MIN_MODE_JUDGE_ITEMS = 10  # labeled verdicts per judge on a mode for a measurable rate
MIN_MEASURING_JUDGES = 2  # judges with measurable rates needed to call a panel blind
MAX_BLIND_SPOTS = 5      # blind-spot modes shown per finding


def _is_judge(model: IntegrityModel) -> bool:
    return model.grader.kind == "judge"


def _judge_loc(model: IntegrityModel, excerpt: str) -> SourceLocation:
    return SourceLocation(file=model.artifact_file("judge_run.json"), excerpt=excerpt)


class UnvalidatedJudgeCheck(Check):
    meta = CheckMeta(
        id="JUDGE-001",
        title="Model judge lacks validation",
        threat=(
            "A model judge with no labeled calibration set is an unvalidated "
            "measurement instrument: its scores have no demonstrated relationship "
            "to human judgment. Unanchored rubrics and unpinned sampling "
            "temperature make the verdicts drift further without anyone noticing."
        ),
        remediation=(
            "Collect a labeled calibration set and report judge-vs-human "
            "agreement before trusting scores; anchor the rubric scale with "
            "explicit examples per level; pin temperature to 0 or take multiple "
            "repeats per item."
        ),
    )

    def run(self, model: IntegrityModel) -> list[Finding]:
        if not _is_judge(model):
            return []
        findings: list[Finding] = []
        grader = model.grader
        if not grader.reference_labels:
            findings.append(
                Finding(
                    id=self.meta.id,
                    title="Model judge has no labeled calibration set",
                    severity=Severity.HIGH,
                    confidence=Confidence.HIGH,
                    description=self.meta.threat,
                    evidence=[
                        f"grader uses model judge {grader.judge_model or '<unknown model>'} "
                        "with no reference labels attached.",
                        "Judge scores are unvalidated against human judgment.",
                    ],
                    locations=[_judge_loc(model, "reference_labels (absent)")],
                    remediation=self.meta.remediation,
                )
            )
        if not grader.scale_anchors:
            findings.append(
                Finding(
                    id=self.meta.id,
                    title="Judge rubric has no explicit scale anchors",
                    severity=Severity.MEDIUM,
                    confidence=Confidence.HIGH,
                    description=self.meta.threat,
                    evidence=[
                        "judge rubric defines criteria but no anchored scale levels.",
                        "Without anchors, score meanings drift between runs and reviewers.",
                    ],
                    locations=[_judge_loc(model, "judge.scale_anchors (absent)")],
                    remediation=self.meta.remediation,
                )
            )
        temp = grader.temperature
        if temp is not None and temp > 0 and grader.repeats < 3:
            findings.append(
                Finding(
                    id=self.meta.id,
                    title=f"Judge samples at temperature {temp} with only {grader.repeats} repeat(s)",
                    severity=Severity.MEDIUM,
                    confidence=Confidence.MEDIUM,
                    description=self.meta.threat,
                    evidence=[
                        f"temperature={temp}, repeats={grader.repeats}: sampling noise "
                        "can flip verdicts with no way to detect it.",
                    ],
                    locations=[_judge_loc(model, "judge.temperature / judge.repeats")],
                    remediation=self.meta.remediation,
                )
            )
        return findings


class PairOrderCheck(Check):
    meta = CheckMeta(
        id="JUDGE-002",
        title="Pairwise order not counterbalanced",
        threat=(
            "LLM judges exhibit position bias: they systematically prefer the "
            "first (or second) presented answer. A pairwise protocol that always "
            "shows candidates in the same order bakes that bias into every "
            "verdict instead of averaging it out."
        ),
        remediation=(
            "Counterbalance presentation order (AB and BA) for every pair, "
            "randomize order per judgment, and record the order with each verdict."
        ),
    )

    def run(self, model: IntegrityModel) -> list[Finding]:
        if not _is_judge(model):
            return []
        if model.grader.protocol != "pairwise":
            # Only a declared pairwise protocol can be AB-only. A pointwise or
            # undeclared protocol has no order to counterbalance.
            return []
        if model.grader.counterbalanced:
            return []
        return [
            Finding(
                id=self.meta.id,
                title="Pairwise protocol without order counterbalancing",
                severity=Severity.HIGH,
                confidence=Confidence.HIGH,
                description=self.meta.threat,
                evidence=[
                    "judge protocol is pairwise and counterbalanced is not enabled.",
                    "Presentation order is a known confound for LLM judges; "
                    "verdicts may reflect position rather than quality.",
                ],
                locations=[_judge_loc(model, "judge.protocol=pairwise, judge.counterbalanced (absent/false)")],
                remediation=self.meta.remediation,
            )
        ]


def _repeat_groups(judgments: list[Judgment]) -> dict[tuple[str, tuple[str, ...]], list[Judgment]]:
    groups: dict[tuple[str, tuple[str, ...]], list[Judgment]] = defaultdict(list)
    for j in judgments:
        groups[(j.task_id, tuple(j.candidates))].append(j)
    return groups


class SelfConsistencyCheck(Check):
    meta = CheckMeta(
        id="JUDGE-003",
        title="Judge contradicts itself on repeated judgments",
        threat=(
            "A judge that flips its verdict on the identical input is measuring "
            "sampling noise, not quality. Reported scores then depend on which "
            "draw the harness happened to keep."
        ),
        remediation=(
            "Pin temperature to 0 for judging; if sampling is required, take "
            "several repeats per item and aggregate by majority or mean instead "
            "of keeping a single draw."
        ),
    )

    def run(self, model: IntegrityModel) -> list[Finding]:
        if not _is_judge(model):
            return []
        findings: list[Finding] = []
        scale_max = max(
            (s for j in model.judgments for s in j.scores.values()), default=0.0
        )
        for (task_id, candidates), group in sorted(_repeat_groups(model.judgments).items()):
            if len(group) < MIN_REPEATS:
                continue
            winners = {j.winner for j in group if j.winner}
            if len(winners) > 1:
                findings.append(
                    Finding(
                        id=self.meta.id,
                        title=f"Judge flips verdict on {task_id} ({len(group)} repeats)",
                        severity=Severity.HIGH,
                        confidence=Confidence.HIGH,
                        description=self.meta.threat,
                        evidence=[
                            f"task {task_id!r}: {len(group)} repeated judgments of "
                            f"{list(candidates)} produced winners {sorted(winners)}.",
                            "The same input yields different verdicts across repeats.",
                        ],
                        locations=[_judge_loc(model, f"judgments[] task_id={task_id} (repeat_index 0..{len(group)-1})")],
                        remediation=self.meta.remediation,
                    )
                )
                continue
            scored = [j for j in group if j.scores]
            if scored and scale_max > 0:
                all_scores = [s for j in scored for s in j.scores.values()]
                spread = max(all_scores) - min(all_scores)
                if spread > SCORE_RANGE_FRAC * scale_max:
                    findings.append(
                        Finding(
                            id=self.meta.id,
                            title=f"Judge scores unstable on {task_id} ({len(group)} repeats)",
                            severity=Severity.HIGH,
                            confidence=Confidence.HIGH,
                            description=self.meta.threat,
                            evidence=[
                                f"task {task_id!r}: scores spread {spread:.2f} across "
                                f"{len(group)} repeats (scale max {scale_max:.2f}).",
                            ],
                            locations=[_judge_loc(model, f"judgments[] task_id={task_id} (repeat_index 0..{len(group)-1})")],
                            remediation=self.meta.remediation,
                        )
                    )
        return findings


class ReferenceAgreementCheck(Check):
    meta = CheckMeta(
        id="JUDGE-004",
        title="Judge disagrees with reference labels",
        threat=(
            "Reference labels are the ground truth the judge is supposed to "
            "approximate. Systematic disagreement means the reported scores do "
            "not track human judgment on the labeled set."
        ),
        remediation=(
            "Investigate the mismatches: fix the rubric, change the judge model, "
            "or restrict claims to the subset where the judge is validated. "
            "Do not report scores the judge cannot reproduce on labeled data."
        ),
    )

    def run(self, model: IntegrityModel) -> list[Finding]:
        if not _is_judge(model):
            return []
        labels = model.grader.reference_labels
        if not labels:
            return []
        compared = 0
        matched = 0
        mismatches: list[str] = []
        for j in model.judgments:
            label = labels.get(j.task_id)
            if label is None:
                continue
            if j.winner is not None:
                compared += 1
                if j.winner == label:
                    matched += 1
                elif len(mismatches) < 5:
                    mismatches.append(f"{j.task_id}: judge={j.winner}, label={label}")
            elif j.scores:
                try:
                    expected = float(label)
                except ValueError:
                    continue
                compared += 1
                mean_score = sum(j.scores.values()) / len(j.scores)
                scale = max(abs(expected), max((abs(s) for s in j.scores.values()), default=0.0), 1e-9)
                if abs(mean_score - expected) <= 0.1 * scale:
                    matched += 1
                elif len(mismatches) < 5:
                    mismatches.append(
                        f"{j.task_id}: judge mean={mean_score:.2f}, label={label}"
                    )
        if compared < MIN_LABELS:
            return []
        agreement = matched / compared
        if agreement >= AGREEMENT_FLOOR:
            return []
        return [
            Finding(
                id=self.meta.id,
                title=f"Judge agrees with reference labels on only {matched}/{compared} items",
                severity=Severity.HIGH,
                confidence=Confidence.HIGH,
                description=self.meta.threat,
                evidence=[
                    f"agreement {agreement:.0%} ({matched}/{compared}) below "
                    f"{AGREEMENT_FLOOR:.0%} over labeled items.",
                    *[f"mismatch: {m}" for m in mismatches],
                ],
                locations=[_judge_loc(model, "reference_labels vs judgments[].winner/scores")],
                remediation=self.meta.remediation,
            )
        ]


def _decided_pairs(judgments: list[Judgment]) -> list[Judgment]:
    return [
        j
        for j in judgments
        if j.winner is not None
        and len(j.presentation_order) >= 2
        and j.winner in j.presentation_order
    ]


class PositionBiasCheck(Check):
    meta = CheckMeta(
        id="JUDGE-005",
        title="Position bias: presentation order predicts the winner",
        threat=(
            "When the first-presented candidate wins far more often than chance, "
            "verdicts track position rather than quality. Rankings built on "
            "these judgments inherit the bias."
        ),
        remediation=(
            "Counterbalance order (JUDGE-002); re-run affected pairs in both "
            "orders and keep only order-robust verdicts."
        ),
    )

    def run(self, model: IntegrityModel) -> list[Finding]:
        if not _is_judge(model):
            return []
        pairs = _decided_pairs(model.judgments)
        if len(pairs) < MIN_PAIRS:
            return []
        first_wins = sum(1 for j in pairs if j.winner == j.presentation_order[0])
        rate = first_wins / len(pairs)
        if rate < BIAS_RATE and rate > 1 - BIAS_RATE:
            return []
        side = "first-presented" if rate >= BIAS_RATE else "second-presented"
        return [
            Finding(
                id=self.meta.id,
                title=f"Position bias: {side} candidate wins {rate:.0%} of {len(pairs)} pairs",
                severity=Severity.MEDIUM,
                confidence=Confidence.MEDIUM,
                description=self.meta.threat,
                evidence=[
                    f"{first_wins}/{len(pairs)} decided pairs won by the "
                    f"first-presented candidate ({rate:.0%}).",
                    "Heuristic signal: corroborate with counterbalanced re-runs "
                    "before treating it as proof.",
                ],
                locations=[_judge_loc(model, "judgments[].presentation_order vs judgments[].winner")],
                remediation=self.meta.remediation,
            )
        ]


class VerbosityBiasCheck(Check):
    meta = CheckMeta(
        id="JUDGE-006",
        title="Verbosity bias: longer answers win disproportionately",
        threat=(
            "LLM judges tend to reward longer responses independent of quality. "
            "When longer candidates win far more often than chance, scores "
            "reward verbosity rather than the rubric."
        ),
        remediation=(
            "Add a rubric criterion that penalizes padding; control for length "
            "in analysis; consider length-normalized comparisons."
        ),
    )

    def run(self, model: IntegrityModel) -> list[Finding]:
        if not _is_judge(model):
            return []
        pairs = [
            j
            for j in _decided_pairs(model.judgments)
            if len(j.candidates) == 2
            and all(c in j.lengths for c in j.candidates)
            and len({j.lengths[c] for c in j.candidates}) == 2
        ]
        if len(pairs) < MIN_PAIRS:
            return []
        longer_wins = sum(
            1
            for j in pairs
            if j.lengths[j.winner] == max(j.lengths[c] for c in j.candidates)
        )
        rate = longer_wins / len(pairs)
        if rate < BIAS_RATE:
            return []
        return [
            Finding(
                id=self.meta.id,
                title=f"Verbosity bias: longer answer wins {rate:.0%} of {len(pairs)} pairs",
                severity=Severity.MEDIUM,
                confidence=Confidence.MEDIUM,
                description=self.meta.threat,
                evidence=[
                    f"{longer_wins}/{len(pairs)} decided pairs won by the longer "
                    f"candidate ({rate:.0%}).",
                    "Heuristic signal: corroborate against reference labels "
                    "before treating it as proof.",
                ],
                locations=[_judge_loc(model, "judgments[].lengths vs judgments[].winner")],
                remediation=self.meta.remediation,
            )
        ]


class JudgeCalibrationCheck(Check):
    meta = CheckMeta(
        id="JUDGE-007",
        title="Judge confidence miscalibrated",
        threat=(
            "A judge whose stated confidence does not track its empirical "
            "accuracy is an uncalibrated instrument: confidence-gated decisions "
            "(abstention, routing, weighting) rest on numbers that mean nothing. "
            "Systematic overconfidence is the common failure mode."
        ),
        remediation=(
            "Recalibrate the judge (temperature or Platt scaling on a held-out "
            "labeled set), collect more labeled judgments, or stop making "
            "decisions on raw judge confidence until it is validated."
        ),
    )

    def run(self, model: IntegrityModel) -> list[Finding]:
        if not _is_judge(model):
            return []
        labels = model.grader.reference_labels
        if not labels:
            return []
        # Narrow: pairwise judgments where the winner is compared to the
        # reference label, and only where the harness recorded confidence.
        pairs: list[tuple[float, bool]] = []
        for j in model.judgments:
            label = labels.get(j.task_id)
            if label is None or j.winner is None or j.confidence is None:
                continue
            conf = j.confidence
            if (
                isinstance(conf, bool)
                or not isinstance(conf, (int, float))
                or not 0.0 <= conf <= 1.0
            ):
                continue
            pairs.append((float(conf), j.winner == label))
        if len(pairs) < MIN_CALIBRATION_PAIRS:
            return []
        ece = expected_calibration_error(pairs)
        if ece < ECE_THRESHOLD:
            return []
        gap = mean_signed_gap(pairs)
        if gap >= GAP_MARGIN:
            verdict = "overconfident"
        elif gap <= -GAP_MARGIN:
            verdict = "underconfident"
        else:
            verdict = "miscalibrated"
        worst = max(
            bin_pairs(pairs), key=lambda b: abs(b.accuracy - b.mean_confidence)
        )
        return [
            Finding(
                id=self.meta.id,
                title=f"Judge {verdict}: stated confidence does not track accuracy",
                severity=Severity.MEDIUM,
                confidence=Confidence.MEDIUM,
                description=self.meta.threat,
                evidence=[
                    f"expected calibration error {ece:.2f} across {len(pairs)} "
                    f"labeled pairwise judgments (threshold {ECE_THRESHOLD:.2f}).",
                    f"mean signed gap {gap:+.2f}: stated confidence runs "
                    f"{abs(gap):.0%} "
                    f"{'above' if gap >= 0 else 'below'} empirical accuracy.",
                    f"worst bin [{worst.lo:.1f}, {worst.hi:.1f}]: stated "
                    f"confidence {worst.mean_confidence:.2f} but accuracy "
                    f"{worst.accuracy:.2f} over {worst.count} judgments.",
                    "Statistical signal: corroborate on a held-out labeled set "
                    "before treating it as proof.",
                ],
                locations=[_judge_loc(model, "judgments[].confidence vs reference_labels")],
                remediation=self.meta.remediation,
            )
        ]


def _panel_verdict_of(j: Judgment) -> str | None:
    """One comparable verdict per judgment: pairwise winner, else top score."""
    if j.winner is not None:
        return j.winner
    if j.scores:
        top = max(j.scores.values())
        return min(c for c, s in j.scores.items() if s == top)
    return None


class EnsembleRedundancyCheck(Check):
    meta = CheckMeta(
        id="JUDGE-008",
        title="Redundant judges in panel",
        threat=(
            "A judge that never differs from the rest of the panel adds no "
            "independent signal: the panel pays for N votes and gets fewer. "
            "Copied or near-identical judges silently collapse the ensemble "
            "into a single effective vote while the budget says otherwise."
        ),
        remediation=(
            "Drop or diversify the redundant judges: keep one member of each "
            "mutually agreeing bloc, or replace copies with judges that "
            "disagree productively. When two judges agree perfectly, verdict "
            "data alone cannot tell who copies whom -- investigate the "
            "pipeline, not just the votes."
        ),
    )

    def run(self, model: IntegrityModel) -> list[Finding]:
        if not _is_judge(model):
            return []
        # First verdict per (judge, task); unattributed judgments cannot be
        # placed in a panel.
        by_judge: dict[str, dict[str, str]] = {}
        for j in model.judgments:
            if j.judge_id is None:
                continue
            verdict = _panel_verdict_of(j)
            if verdict is None:
                continue
            by_judge.setdefault(j.judge_id, {}).setdefault(j.task_id, verdict)
        judge_ids = sorted(by_judge)
        if len(judge_ids) < MIN_PANEL_JUDGES:
            return []
        common = set(by_judge[judge_ids[0]])
        for j in judge_ids[1:]:
            common &= set(by_judge[j])
        if len(common) < MIN_COMMON_ITEMS:
            return []
        votes = {j: {t: by_judge[j][t] for t in common} for j in judge_ids}
        stats = {j: unique_contribution(votes, j) for j in judge_ids}
        if len(common) < MIN_FLAG_ITEMS:
            return []
        flagged = [j for j in judge_ids if stats[j][0] < REDUNDANCY_RATE]
        if not flagged:
            return []
        labels = model.grader.reference_labels
        truth = {t: labels[t] for t in common if t in labels}
        evidence = []
        for j in flagged:
            rate, differs, decisive = stats[j]
            partner, agree = max_agreement_partner(votes, j)
            line = (
                f"judge '{j}': unique contribution {rate:.1%} "
                f"({differs}/{decisive} decisive items) -- differs from the "
                f"majority of the other judges on fewer than "
                f"{REDUNDANCY_RATE:.0%} of items."
            )
            if partner is not None:
                line += f" Mirrors judge '{partner}' on {agree:.0%} of items."
            if truth:
                abl = ablation_value(votes, truth, j)
                line += (
                    f" Ablation value {abl:+.2f}: removing '{j}' changes "
                    f"panel truth-agreement by {abl:+.0%}."
                )
            evidence.append(line)
        evidence.append(
            "When two judges agree perfectly, both are flagged: the data "
            "cannot tell who copies whom."
        )
        evidence.append(
            "Statistical signal: corroborate on a held-out item set before "
            "treating it as proof."
        )
        return [
            Finding(
                id=self.meta.id,
                title=(
                    f"Redundant judges in panel: {', '.join(flagged)} "
                    f"({len(common)} common items)"
                ),
                severity=Severity.MEDIUM,
                confidence=Confidence.MEDIUM,
                description=self.meta.threat,
                evidence=evidence,
                locations=[_judge_loc(model, "judgments[].judge_id vs judgments[].winner")],
                remediation=self.meta.remediation,
            )
        ]


class PanelBlindSpotCheck(Check):
    meta = CheckMeta(
        id="JUDGE-009",
        title="Panel blind spots: failure modes no judge catches",
        threat=(
            "A judge panel can look healthy in aggregate while every member "
            "fails on the same failure mode. Reported scores then carry no "
            "signal about that mode at all: items designed to expose it pass "
            "through the panel unseen, and the eval certifies robustness it "
            "never actually measured."
        ),
        remediation=(
            "Add or retrain a judge that catches the blind modes (validate "
            "per-mode catch rates against the reference labels, not just "
            "aggregate agreement), or narrow the eval's claims to the "
            "failure modes the panel demonstrably catches."
        ),
    )

    def run(self, model: IntegrityModel) -> list[Finding]:
        if not _is_judge(model):
            return []
        labels = model.grader.reference_labels
        if not labels:
            # No ground truth: catch rates are unmeasurable, stay silent.
            return []
        mode_of = {
            t.id: t.failure_mode for t in model.tasks if t.failure_mode is not None
        }
        if not mode_of:
            # No failure-mode tags: nothing to slice coverage by.
            return []
        # First pairwise verdict per (judge, task). Narrow to pairwise like
        # JUDGE-007: a pointwise top-score is not comparable to a reference
        # winner, and treating it as one would manufacture blind spots.
        verdicts: dict[str, dict[str, str]] = {}
        for j in model.judgments:
            if j.judge_id is None or j.winner is None:
                continue
            verdicts.setdefault(j.judge_id, {}).setdefault(j.task_id, j.winner)
        if len(verdicts) < MIN_MEASURING_JUDGES:
            # A single attributable judge is not a panel; per-mode weakness
            # of one judge is JUDGE-004 territory, not a blind spot claim.
            return []
        table = mode_catch_table(verdicts, labels, mode_of)
        spots = blind_spots(
            table,
            catch_bar=CATCH_BAR,
            min_mode_items=MIN_MODE_ITEMS,
            min_judge_items=MIN_MODE_JUDGE_ITEMS,
            min_judges=MIN_MEASURING_JUDGES,
        )
        if not spots:
            return []
        evidence: list[str] = []
        for mc in spots[:MAX_BLIND_SPOTS]:
            measurable = mc.measurable(MIN_MODE_JUDGE_ITEMS)
            per_judge = ", ".join(
                f"{jc.judge_id} {jc.rate:.0%} ({jc.n_caught}/{jc.n_labeled})"
                for jc in measurable
            )
            evidence.append(
                f"mode '{mc.mode}': no judge catches it -- best rate "
                f"{mc.best_rate:.0%} below the {CATCH_BAR:.0%} bar over "
                f"{mc.n_labeled_items} labeled items; per-judge: {per_judge}."
            )
        if len(spots) > MAX_BLIND_SPOTS:
            evidence.append(
                f"... and {len(spots) - MAX_BLIND_SPOTS} more blind-spot mode(s)."
            )
        evidence.append(
            "Statistical signal: corroborate on a held-out labeled set "
            "before treating it as proof."
        )
        modes = ", ".join(f"'{mc.mode}'" for mc in spots)
        return [
            Finding(
                id=self.meta.id,
                title=f"Panel blind spots: no judge catches {modes}",
                severity=Severity.MEDIUM,
                confidence=Confidence.MEDIUM,
                description=self.meta.threat,
                evidence=evidence,
                locations=[
                    _judge_loc(
                        model,
                        "judgments[].judge_id vs reference_labels "
                        "by tasks[].failure_mode",
                    )
                ],
                remediation=self.meta.remediation,
            )
        ]


CHECKS = [
    UnvalidatedJudgeCheck(),  # JUDGE-001
    PairOrderCheck(),  # JUDGE-002
    SelfConsistencyCheck(),  # JUDGE-003
    ReferenceAgreementCheck(),  # JUDGE-004
    PositionBiasCheck(),  # JUDGE-005
    VerbosityBiasCheck(),  # JUDGE-006
    JudgeCalibrationCheck(),  # JUDGE-007
    EnsembleRedundancyCheck(),  # JUDGE-008
    PanelBlindSpotCheck(),  # JUDGE-009
]
