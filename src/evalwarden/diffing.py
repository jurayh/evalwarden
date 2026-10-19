"""Paired run comparison: what moved between two result sets, item by item.

Pure statistics over aligned per-item outcomes. No model imports, so the
estimators stay independently testable. Used by the DIFF lane.

Two runs of an eval can tie on aggregate score while hundreds of items
flip in opposite directions, and the flips can concentrate in one slice
of the item set. An aggregate delta nets that movement against itself
and reports calm. This module reports the movement instead: per-item
flips between a baseline and a candidate run over the items both runs
attempted, a paired significance test on the flips, and the same
accounting per slice of task metadata.

Exact definitions:
- An item outcome is a score in [0, 1]; an item passes at PASS_AT, the
  threshold the DATA lane uses for attempt correctness. A pair is
  (task_id, baseline score, candidate score) over one aligned item.
- A regression is a pair that passed in the baseline and fails in the
  candidate; a gain is the reverse. Pairs passing or failing on both
  sides are unchanged, however far their scores moved inside the band.
- Significance is McNemar's exact test on the discordant pairs (the
  flips): under the null that flips are symmetric noise, the regression
  count is Binomial(regressions + gains, 0.5), and the two-sided exact
  p-value sums the binomial tail at the observed split. Exact, paired,
  and fully deterministic -- a bootstrap would inject resampling noise
  into a CI gate, and an unpaired test would ignore that the same items
  were measured twice.
- Alignment is set intersection over item ids. Coverage is the paired
  count over the union of both sides' items: the fraction of the
  combined item set the comparison actually covers. Unpaired items are
  counted per side, never silently dropped and never diffed as if the
  other side had scored them zero.
- A slice is one (metadata key, value) group over the paired items. Its
  net regression rate is (regressions - gains) / slice size; the global
  rate is the same ratio over all paired items.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import comb

PASS_AT = 0.5  # item score at or above this counts as a pass

# (task_id, baseline score, candidate score) for one aligned item.
ItemPair = tuple[str, float, float]


def passed(score: float) -> bool:
    """Whether an item score counts as a pass (the DATA lane threshold)."""
    return score >= PASS_AT


def mcnemar_exact_p(regressions: int, gains: int) -> float:
    """Two-sided exact McNemar p-value for a flip split.

    Under the null, each discordant pair flips regression-ward with
    probability 0.5, so the smaller side's count is Binomial(n, 0.5)
    with n = regressions + gains. No discordant pairs is no evidence:
    p = 1.
    """
    n = regressions + gains
    if n == 0:
        return 1.0
    k = min(regressions, gains)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2.0 * tail)


@dataclass
class Alignment:
    """How two item sets line up: what paired, what each side kept alone."""

    paired_ids: list[str]  # sorted item ids present on both sides
    baseline_only: list[str]  # sorted ids only the baseline attempted
    candidate_only: list[str]  # sorted ids only the candidate attempted

    @property
    def n_paired(self) -> int:
        return len(self.paired_ids)

    @property
    def coverage(self) -> float:
        """Paired items over the union of both sides (0 when both empty)."""
        union = self.n_paired + len(self.baseline_only) + len(self.candidate_only)
        if union == 0:
            return 0.0
        return self.n_paired / union


def align(baseline_ids, candidate_ids) -> Alignment:
    """Align two item-id collections by identity."""
    base = set(baseline_ids)
    cand = set(candidate_ids)
    return Alignment(
        paired_ids=sorted(base & cand),
        baseline_only=sorted(base - cand),
        candidate_only=sorted(cand - base),
    )


@dataclass
class SliceStats:
    """Flip accounting for one (metadata key, value) group of pairs."""

    key: str
    value: str
    n: int
    regressions: int
    gains: int
    p_value: float

    @property
    def net_regression_rate(self) -> float:
        """(regressions - gains) / n: positive means the slice regressed."""
        if self.n == 0:
            return 0.0
        return (self.regressions - self.gains) / self.n


@dataclass
class Flip:
    """One pass-to-fail item, with the score movement behind the flip."""

    task_id: str
    baseline_score: float
    candidate_score: float

    @property
    def drop(self) -> float:
        return self.baseline_score - self.candidate_score


@dataclass
class DiffStats:
    """Flip accounting over one aligned pair of item-outcome sets."""

    n_paired: int
    regressions: int
    gains: int
    unchanged: int  # pairs with the same pass/fail outcome on both sides
    baseline_mean: float  # mean item score over paired items, baseline side
    candidate_mean: float  # mean item score over paired items, candidate side
    p_value: float  # McNemar exact p over the discordant pairs
    worst: list[Flip] = field(default_factory=list)  # regressions, worst first
    slices: list[SliceStats] = field(default_factory=list)

    @property
    def mean_delta(self) -> float:
        """Aggregate score movement over the paired items (candidate - baseline)."""
        return self.candidate_mean - self.baseline_mean

    @property
    def net_regression_rate(self) -> float:
        """(regressions - gains) / paired items: positive means net regression."""
        if self.n_paired == 0:
            return 0.0
        return (self.regressions - self.gains) / self.n_paired


def diff_stats(
    pairs: list[ItemPair],
    tags: dict[str, dict[str, str]] | None = None,
) -> DiffStats:
    """Compute flip statistics over aligned item pairs.

    ``tags`` maps task_id -> {metadata key: value} for slice accounting;
    items without a tag for a key simply do not join that key's slices.
    Slices are sorted by (key, value) and worst regressions by score drop
    descending with task_id as the tie-break, so every output is
    deterministic for a given input.
    """
    regressions = 0
    gains = 0
    unchanged = 0
    worst: list[Flip] = []
    base_total = 0.0
    cand_total = 0.0
    # (key, value) -> [n, regressions, gains]
    slice_cells: dict[tuple[str, str], list[int]] = {}
    for task_id, base_score, cand_score in pairs:
        base_total += base_score
        cand_total += cand_score
        base_pass = passed(base_score)
        cand_pass = passed(cand_score)
        if base_pass and not cand_pass:
            kind = 1  # regression
            regressions += 1
            worst.append(Flip(task_id, base_score, cand_score))
        elif cand_pass and not base_pass:
            kind = 2  # gain
            gains += 1
        else:
            kind = 0
            unchanged += 1
        if tags:
            for key, value in tags.get(task_id, {}).items():
                cell = slice_cells.setdefault((key, value), [0, 0, 0])
                cell[0] += 1
                if kind == 1:
                    cell[1] += 1
                elif kind == 2:
                    cell[2] += 1
    n = len(pairs)
    worst.sort(key=lambda flip: (-flip.drop, flip.task_id))
    slices = [
        SliceStats(
            key=key,
            value=value,
            n=cell[0],
            regressions=cell[1],
            gains=cell[2],
            p_value=mcnemar_exact_p(cell[1], cell[2]),
        )
        for (key, value), cell in sorted(slice_cells.items())
    ]
    return DiffStats(
        n_paired=n,
        regressions=regressions,
        gains=gains,
        unchanged=unchanged,
        baseline_mean=base_total / n if n else 0.0,
        candidate_mean=cand_total / n if n else 0.0,
        p_value=mcnemar_exact_p(regressions, gains),
        worst=worst,
        slices=slices,
    )
