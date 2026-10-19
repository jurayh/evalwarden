"""Paired run comparison checks: what got worse, where, and is it real?

DIFF-001 is the suite's first comparative check. Every other lane audits
one artifact at a time, but the question after a fine-tune, a prompt
change, or a model swap is comparative, and aggregate deltas cannot
answer it: two runs can tie on score while items flip in both directions
underneath, with the regressions concentrated in one slice. DIFF-001
aligns a baseline and a candidate result set item by item and reports
the movement -- flips, their significance, and the slices they cluster
in -- instead of the net.

The check needs two models, so it is driven differently from the rest
of the registry: `evalwarden diff` loads both artifacts through the
adapters exactly as `audit` loads one, and hands the baseline to this
check via with_options. Under a plain `audit` there is no baseline, and
the check stays silent -- a lone artifact has nothing to be compared
against, and silence beats a fabricated comparison.

Pairing, decided here because the model leaves it open (attempts are
recorded per task, optionally per model):
- Attempts first reduce to one outcome per item within a *model
  series*: the attempts sharing a model_id (None is the unnamed single
  model most artifacts carry). Repeated attempts on an item -- Inspect
  epochs land here -- average, matching how Inspect itself aggregates
  epochs; an item passes at the DATA lane's 0.5 threshold.
- Series are the pairing unit. When each side has exactly one series
  they pair regardless of model name: a fine-tune renames the model,
  and the comparison is the point. With several series per side, series
  pair on equal model_id only (the same model under two prompts, say),
  and unpaired series are reported in the coverage evidence, never
  silently merged into another model's diff.
- Alignment is by task_id. Coverage (paired over the union of items)
  and the paired count gate every verdict: below MIN_PAIRED_ITEMS paired
  items or MIN_COVERAGE coverage, the two artifacts did not run the
  same eval closely enough for a diff to mean anything, so the check
  says so in an advisory finding and computes no regression verdict
  rather than diffing unpaired aggregates.

Gate rules, both prongs always required (significance without
materiality blocks on trivia at large n; materiality without
significance blocks on weather at small n):
- Global: McNemar exact p < ALPHA and net regression at or above
  NET_REGRESSION_MIN of the paired items.
- Slice: a slice of at least MIN_SLICE_ITEMS paired items whose own
  split is significant (p < SLICE_ALPHA) with at least
  MIN_SLICE_REGRESSIONS regressions and a net regression rate at least
  SLICE_EXCESS above the global rate. The excess-over-global condition
  is the multiplicity guard: with a handful of metadata keys the per-
  slice alpha alone would eventually cry wolf, so a slice must be not
  just asymmetric but materially worse than the run it sits in.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from ..diffing import Alignment, DiffStats, align, diff_stats
from ..model import Attempt, Confidence, Finding, IntegrityModel, Severity, SourceLocation
from .base import Check, CheckMeta

MIN_PAIRED_ITEMS = 30  # paired items before a verdict means anything (DATA lane floor)
MIN_COVERAGE = 0.8  # paired share of the item union below this: not the same eval
ALPHA = 0.01  # McNemar significance for the global gate
NET_REGRESSION_MIN = 0.02  # net regression share of paired items the gate requires
MIN_SLICE_ITEMS = 20  # paired items a slice needs before it can be flagged
MIN_SLICE_REGRESSIONS = 5  # regressions a slice needs before it can be flagged
SLICE_ALPHA = 0.01  # McNemar significance within the slice
SLICE_EXCESS = 0.10  # slice net rate must beat the global rate by this much
MAX_WORST_SHOWN = 5  # worst regressions listed per finding
MAX_SLICE_FINDINGS = 5  # slice findings per series; the rest fold into evidence


def _attempt_value(attempt: Attempt) -> float:
    """One attempt as an item outcome in [0, 1] (DATA lane correctness)."""
    if attempt.score is not None:
        return attempt.score
    return 1.0 if attempt.status == "pass" else 0.0


def series_outcomes(model: IntegrityModel) -> dict[str | None, dict[str, float]]:
    """Reduce a model's attempts to {model_id: {task_id: item score}}.

    Repeated attempts on one item average. Attempts referencing tasks
    the artifact does not define still count: the run attempted them,
    and alignment, not the dataset listing, decides what pairs.
    """
    grouped: dict[tuple[str | None, str], list[float]] = defaultdict(list)
    for attempt in model.attempts:
        grouped[(attempt.model_id, attempt.task_id)].append(_attempt_value(attempt))
    series: dict[str | None, dict[str, float]] = defaultdict(dict)
    for (model_id, task_id), values in grouped.items():
        series[model_id][task_id] = sum(values) / len(values)
    return dict(series)


def pair_series(
    baseline: dict[str | None, dict[str, float]],
    candidate: dict[str | None, dict[str, float]],
) -> tuple[list[tuple[str | None, str | None]], list[str | None], list[str | None]]:
    """Pair model series across the two artifacts.

    Returns (pairs, unpaired baseline keys, unpaired candidate keys),
    where a pair is (baseline model_id, candidate model_id). The
    single-series rule pairs the only series on each side even when the
    names differ (a fine-tune renames its model); otherwise series pair
    on equal names only.
    """
    if len(baseline) == 1 and len(candidate) == 1:
        return [(next(iter(baseline)), next(iter(candidate)))], [], []
    pairs = [(key, key) for key in sorted(baseline.keys() & candidate.keys(), key=str)]
    paired_base = {b for b, _ in pairs}
    paired_cand = {c for _, c in pairs}
    return (
        pairs,
        [key for key in baseline if key not in paired_base],
        [key for key in candidate if key not in paired_cand],
    )


def _series_label(model_id: str | None) -> str:
    return model_id if model_id is not None else "the unnamed model"


def _pair_label(base_id: str | None, cand_id: str | None) -> str:
    if base_id == cand_id:
        return _series_label(base_id)
    return f"{_series_label(base_id)} -> {_series_label(cand_id)}"


def _task_tags(
    baseline: IntegrityModel,
    candidate: IntegrityModel,
    task_ids: list[str],
) -> dict[str, dict[str, str]]:
    """Slice tags for paired items: task failure_mode plus scalar metadata.

    The candidate's task records win (it is the run under judgment);
    baseline records fill ids the candidate does not define. Only
    scalar metadata values slice -- a list or dict tag cannot name a
    group of items.
    """
    by_id = {task.id: task for task in baseline.tasks}
    by_id.update({task.id: task for task in candidate.tasks})
    tags: dict[str, dict[str, str]] = {}
    for task_id in task_ids:
        task = by_id.get(task_id)
        if task is None:
            continue
        entry: dict[str, str] = {}
        if task.failure_mode:
            entry["failure_mode"] = task.failure_mode
        for key, value in task.metadata.items():
            if isinstance(value, (str, int, float, bool)):
                entry[str(key)] = str(value)
        if entry:
            tags[task_id] = entry
    return tags


def _diff_loc(model: IntegrityModel, excerpt: str) -> SourceLocation:
    return SourceLocation(file=model.artifact_file("attempts.json"), excerpt=excerpt)


def _coverage_evidence(alignment: Alignment) -> str:
    return (
        f"alignment: {alignment.n_paired} paired items, "
        f"{len(alignment.baseline_only)} baseline-only, "
        f"{len(alignment.candidate_only)} candidate-only "
        f"(coverage {alignment.coverage:.0%} of the item union)"
    )


@dataclass
class SeriesComparison:
    """One paired series, compared: alignment, flip stats, judgeability."""

    label: str  # e.g. "base-model -> ft-model"
    alignment: Alignment
    stats: DiffStats
    tags: dict[str, dict[str, str]]
    judgeable: bool  # paired n and coverage both clear their floors
    unjudged_reason: str = ""  # why not, when judgeable is False


class PairedRegressionCheck(Check):
    meta = CheckMeta(
        id="DIFF-001",
        title="Paired regression versus a baseline run",
        threat=(
            "Aggregate scores net movement against itself: a candidate "
            "can hold its headline score while items flip in both "
            "directions underneath, with the regressions concentrated "
            "in one slice of the eval. A single-artifact audit sees none "
            "of that movement; only a paired, per-item comparison "
            "against the previous run shows what got worse, where, and "
            "whether the movement clears noise."
        ),
        remediation=(
            "Inspect the regressed items and any flagged slice before "
            "shipping the candidate: fix the regression, or re-run both "
            "configurations to confirm it is real, and report flip "
            "counts alongside the aggregate delta."
        ),
    )

    def __init__(self, baseline: IntegrityModel | None = None):
        self.baseline = baseline

    def with_options(self, **kwargs) -> "PairedRegressionCheck":
        return PairedRegressionCheck(
            baseline=kwargs.get("baseline_model", self.baseline)
        )

    def comparisons(
        self, candidate: IntegrityModel
    ) -> tuple[list[SeriesComparison], list[str]]:
        """Compare the baseline against ``candidate``, series by series.

        Returns (comparisons, unpaired notes): one SeriesComparison per
        paired model series, plus a note per series that exists on only
        one side and so was not compared. Empty comparisons means
        nothing paired at all. Presentation layers (the diff command's
        summary) read the same numbers the findings were computed from.
        """
        if self.baseline is None:
            return [], []
        baseline = self.baseline
        base_series = series_outcomes(baseline)
        cand_series = series_outcomes(candidate)
        pairs, unpaired_base, unpaired_cand = pair_series(base_series, cand_series)
        comparisons: list[SeriesComparison] = []
        for base_id, cand_id in pairs:
            label = _pair_label(base_id, cand_id)
            alignment = align(base_series[base_id], cand_series[cand_id])
            tags = _task_tags(baseline, candidate, alignment.paired_ids)
            stats = diff_stats(
                [
                    (tid, base_series[base_id][tid], cand_series[cand_id][tid])
                    for tid in alignment.paired_ids
                ],
                tags,
            )
            reasons = []
            if stats.n_paired < MIN_PAIRED_ITEMS:
                reasons.append(
                    f"only {stats.n_paired} paired items "
                    f"(minimum {MIN_PAIRED_ITEMS})"
                )
            if alignment.coverage < MIN_COVERAGE:
                reasons.append(
                    f"alignment coverage {alignment.coverage:.0%} "
                    f"(minimum {MIN_COVERAGE:.0%}): the two artifacts did "
                    "not run the same item set"
                )
            comparisons.append(
                SeriesComparison(
                    label=label,
                    alignment=alignment,
                    stats=stats,
                    tags=tags,
                    judgeable=not reasons,
                    unjudged_reason="; ".join(reasons),
                )
            )
        notes = [
            f"baseline series {_series_label(key)} has no counterpart "
            "in the candidate artifact and was not compared"
            for key in unpaired_base
        ] + [
            f"candidate series {_series_label(key)} has no counterpart "
            "in the baseline artifact and was not compared"
            for key in unpaired_cand
        ]
        return comparisons, notes

    def run(self, model: IntegrityModel) -> list[Finding]:
        if self.baseline is None:
            # Plain audit: nothing to compare against. Silence, by design.
            return []
        comparisons, notes = self.comparisons(model)
        findings: list[Finding] = []
        if not comparisons:
            base_series = series_outcomes(self.baseline)
            cand_series = series_outcomes(model)
            findings.append(
                self._unjudged(
                    model,
                    "no model series pairs across the two artifacts "
                    f"(baseline series: {[_series_label(k) for k in base_series]}; "
                    f"candidate series: {[_series_label(k) for k in cand_series]})",
                )
            )
            return findings
        multi = len(comparisons) > 1
        for comparison in comparisons:
            findings.extend(self._series_findings(model, comparison, multi))
        for note in notes:
            findings.append(self._unjudged(model, note))
        return findings

    # -- findings per paired series -------------------------------------

    def _series_findings(
        self,
        model: IntegrityModel,
        comparison: SeriesComparison,
        multi: bool,
    ) -> list[Finding]:
        label = comparison.label
        alignment = comparison.alignment
        stats = comparison.stats
        tags = comparison.tags
        if not comparison.judgeable:
            return [
                self._unjudged(
                    model,
                    f"series {label}: {comparison.unjudged_reason}",
                    alignment,
                )
            ]
        findings: list[Finding] = []
        prefix = f"{label}: " if (multi or label != "the unnamed model") else ""
        if (
            stats.p_value < ALPHA
            and stats.net_regression_rate >= NET_REGRESSION_MIN
        ):
            findings.append(
                Finding(
                    id=self.meta.id,
                    title=(
                        f"{prefix}candidate regressed: {stats.regressions} "
                        f"pass-to-fail vs {stats.gains} fail-to-pass flips "
                        f"(net regression {stats.net_regression_rate:.0%} of "
                        f"{stats.n_paired} paired items, McNemar "
                        f"p={stats.p_value:.3g})"
                    ),
                    severity=Severity.HIGH,
                    confidence=Confidence.HIGH,
                    description=self.meta.threat,
                    evidence=self._global_evidence(alignment, stats),
                    locations=[
                        _diff_loc(
                            model,
                            f"attempts[] paired against the baseline over "
                            f"{stats.n_paired} items (series {label})",
                        )
                    ],
                    remediation=self.meta.remediation,
                )
            )
        findings.extend(
            self._slice_findings(model, prefix, alignment, stats, tags)
        )
        return findings

    def _slice_findings(
        self,
        model: IntegrityModel,
        prefix: str,
        alignment: Alignment,
        stats: DiffStats,
        tags: dict[str, dict[str, str]],
    ) -> list[Finding]:
        flagged = [
            s
            for s in stats.slices
            if s.n >= MIN_SLICE_ITEMS
            and s.regressions >= MIN_SLICE_REGRESSIONS
            and s.p_value < SLICE_ALPHA
            and s.net_regression_rate >= stats.net_regression_rate + SLICE_EXCESS
        ]
        # Worst first: largest net regression rate, then most regressions.
        flagged.sort(key=lambda s: (-s.net_regression_rate, -s.regressions, s.key, s.value))
        findings: list[Finding] = []
        for sl in flagged[:MAX_SLICE_FINDINGS]:
            worst_here = [
                flip
                for flip in stats.worst
                if tags.get(flip.task_id, {}).get(sl.key) == sl.value
            ][:MAX_WORST_SHOWN]
            evidence = [
                _coverage_evidence(alignment),
                (
                    f"slice {sl.key}={sl.value}: {sl.n} paired items, "
                    f"{sl.regressions} pass-to-fail vs {sl.gains} "
                    f"fail-to-pass (net regression "
                    f"{sl.net_regression_rate:.0%}, McNemar "
                    f"p={sl.p_value:.3g})"
                ),
                (
                    f"the run overall nets {stats.net_regression_rate:.0%} "
                    f"({stats.regressions} regressions, {stats.gains} gains "
                    f"over {stats.n_paired} paired items): this slice is "
                    f"at least {SLICE_EXCESS:.0%} worse, so the regression "
                    "is concentrated, not uniform"
                ),
                (
                    f"aggregate on paired items: baseline "
                    f"{stats.baseline_mean:.1%} -> candidate "
                    f"{stats.candidate_mean:.1%} "
                    f"({stats.mean_delta:+.1%}) -- the net hides the "
                    "concentration"
                ),
            ]
            if worst_here:
                evidence.append(
                    "worst in slice: "
                    + ", ".join(
                        f"{f.task_id} ({f.baseline_score:.2f} -> "
                        f"{f.candidate_score:.2f})"
                        for f in worst_here
                    )
                )
            findings.append(
                Finding(
                    id=self.meta.id,
                    title=(
                        f"{prefix}regression concentrated in {sl.key}="
                        f"{sl.value}: {sl.regressions} pass-to-fail vs "
                        f"{sl.gains} fail-to-pass (net regression "
                        f"{sl.net_regression_rate:.0%} of the slice)"
                    ),
                    severity=Severity.HIGH,
                    confidence=Confidence.MEDIUM,
                    description=self.meta.threat,
                    evidence=evidence,
                    locations=[
                        _diff_loc(
                            model,
                            f"attempts[] paired against the baseline, "
                            f"slice {sl.key}={sl.value} ({sl.n} items)",
                        )
                    ],
                    remediation=self.meta.remediation,
                )
            )
        hidden = len(flagged) - len(findings)
        if hidden > 0 and findings:
            findings[0].evidence.append(
                f"{hidden} further slice(s) meet the regression rule and "
                "are folded here: "
                + ", ".join(
                    f"{s.key}={s.value} (net {s.net_regression_rate:.0%})"
                    for s in flagged[MAX_SLICE_FINDINGS:]
                )
            )
        return findings

    def _global_evidence(
        self, alignment: Alignment, stats: DiffStats
    ) -> list[str]:
        evidence = [
            _coverage_evidence(alignment),
            (
                f"aggregate on paired items: baseline "
                f"{stats.baseline_mean:.1%} -> candidate "
                f"{stats.candidate_mean:.1%} ({stats.mean_delta:+.1%})"
            ),
            (
                f"flips: {stats.regressions} pass-to-fail, {stats.gains} "
                f"fail-to-pass, {stats.unchanged} unchanged -- the "
                "aggregate delta nets the two flip directions against "
                "each other"
            ),
            (
                f"McNemar exact test on {stats.regressions + stats.gains} "
                f"discordant pairs: p={stats.p_value:.3g} (gate: "
                f"p < {ALPHA} and net regression >= "
                f"{NET_REGRESSION_MIN:.0%} of paired items)"
            ),
        ]
        if stats.worst:
            evidence.append(
                "worst regressions: "
                + ", ".join(
                    f"{f.task_id} ({f.baseline_score:.2f} -> "
                    f"{f.candidate_score:.2f})"
                    for f in stats.worst[:MAX_WORST_SHOWN]
                )
            )
        return evidence

    def _unjudged(
        self,
        model: IntegrityModel,
        reason: str,
        alignment: Alignment | None = None,
    ) -> Finding:
        evidence = [reason]
        if alignment is not None:
            evidence.insert(0, _coverage_evidence(alignment))
        evidence.append(
            "Statistical signal: align the two artifacts on the same item "
            "set and model series before reading any delta between them."
        )
        return Finding(
            id=self.meta.id,
            title=f"Run comparison cannot be judged: {reason}",
            severity=Severity.MEDIUM,
            confidence=Confidence.LOW,
            description=self.meta.threat,
            evidence=evidence,
            locations=[
                _diff_loc(model, "attempts[] (paired comparison unjudged)")
            ],
            remediation=self.meta.remediation,
        )


CHECKS = [
    PairedRegressionCheck(),  # DIFF-001
]
