"""Eval noise budgets: decompose repeated-run score variance by source.

Pure statistics over per-item scores from repeated runs. No model imports,
so the estimators stay independently testable. Used by the NOISE lane.

An eval score is a measurement, and every measurement has a noise budget.
Re-running an eval re-rolls three distinct dice: fresh generations (model
sampling), re-grades of frozen outputs (judge noise), and repeated tool or
environment executions (flakiness). A score change smaller than the noise
those dice produce is not an improvement; it is weather. This module
measures how much weather there is and which die casts most of it.

Exact definitions:
- Each observation is (task_id, score, generation, grade, environment).
  The three indices tag which run conditions produced the score: scores
  differing only in `grade` are re-grades of one frozen output, and so on.
- A source's variance component is the degrees-of-freedom-weighted pooled
  within-cell sample variance along its own axis: judge variance is the
  spread of re-grades inside (task, generation, environment) cells,
  sampling variance the spread of fresh generations inside (task, grade,
  environment) cells, environment variance the spread of environment
  repeats inside (task, generation, grade) cells. Each axis estimate
  therefore absorbs any interaction that moves scores along that axis --
  exactly what re-rolling that die costs.
- Total variance is the within-task variance of a single score. The
  interaction component is the residual total minus the three main
  effects, clamped at zero; it is only defined when all three axes are
  measured.
- A source is *measured* only when its axis takes at least two values and
  the pooled degrees of freedom reach MIN_AXIS_DF. An unmeasured source
  reports None, never a fabricated zero: with a single generation, the
  data says nothing about sampling noise.
- The noise floor assumes the reported score is the mean over the
  observed items and replicates. A component's contribution to that
  mean's variance divides by (items x that axis's replicate count), so
  Var(reported) = sampling/(I*G) + judge/(I*J) + environment/(I*E)
  + interaction/(I*G*J*E), summed over measured components only. The
  noise floor is the smallest absolute difference between two independent
  protocol-identical scores that clears noise at 95% confidence:
  NOISE_Z * sqrt(2 * Var(reported)). With unmeasured axes the floor is
  a lower bound on the true floor.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from math import sqrt

NOISE_Z = 1.96  # two-sided 95% for the noise floor
MIN_AXIS_DF = 10  # pooled df below this: the axis is unmeasured, not zero

# (task_id, score, generation_index, grade_index, environment_index)
ScoreTuple = tuple[str, float, int, int, int]

SAMPLING = "sampling"
JUDGE = "judge"
ENVIRONMENT = "environment"
INTERACTION = "interaction"
SOURCES = (SAMPLING, JUDGE, ENVIRONMENT)


@dataclass
class NoiseBudget:
    """Variance budget for one eval's repeated-run scores."""

    n_items: int
    n_scores: int
    levels: dict[str, int]  # replicates observed per axis (source -> count)
    components: dict[str, float | None]  # per-source variance; None = unmeasured
    interaction: float | None  # residual variance; None unless all axes measured
    total_variance: float  # within-task variance of a single score
    run_mean_variance: float  # variance of the reported (averaged) score
    noise_floor: float  # minimum detectable real delta at 95% confidence

    @property
    def measured_sources(self) -> list[str]:
        """Sources whose variance the data actually measured."""
        return [s for s in SOURCES if self.components.get(s) is not None]

    @property
    def unmeasured_sources(self) -> list[str]:
        """Sources the run conditions never varied: no estimate exists."""
        return [s for s in SOURCES if self.components.get(s) is None]

    @property
    def total_noise(self) -> float:
        """Variance attributed across measured components plus interaction."""
        total = sum(v for v in self.components.values() if v is not None)
        if self.interaction is not None:
            total += self.interaction
        return total

    def shares(self) -> dict[str, float]:
        """Fraction of total noise per component (measured ones, plus
        interaction when defined). Empty when total noise is zero."""
        total = self.total_noise
        if total <= 0.0:
            return {}
        out: dict[str, float] = {}
        for source in SOURCES:
            value = self.components.get(source)
            if value is not None:
                out[source] = value / total
        if self.interaction is not None:
            out[INTERACTION] = self.interaction / total
        return out

    def dominant(self) -> tuple[str, float] | None:
        """(source, share) of the largest budget component, or None."""
        shares = self.shares()
        if not shares:
            return None
        source = max(shares, key=lambda s: shares[s])
        return source, shares[source]


def _pooled_variance(cells: list[list[float]]) -> tuple[float, int]:
    """(variance, pooled df) across cells, each contributing its sample
    variance weighted by its degrees of freedom. Cells of size < 2 carry
    no spread and are skipped."""
    weighted = 0.0
    df = 0
    for cell in cells:
        n = len(cell)
        if n < 2:
            continue
        mean = sum(cell) / n
        ss = sum((x - mean) ** 2 for x in cell)
        weighted += ss
        df += n - 1
    if df == 0:
        return 0.0, 0
    return weighted / df, df


def _axis_variance(
    scores: list[ScoreTuple], axis: int
) -> tuple[float | None, int]:
    """Variance along one index axis (2=generation, 3=grade, 4=environment).

    Groups scores into cells that share every other index, then pools the
    within-cell variance over distinct axis values. Returns (None, levels)
    when the axis never varies or the pooled degrees of freedom fall below
    MIN_AXIS_DF: the source is then unmeasured, not zero.
    """
    levels = len({s[axis] for s in scores})
    if levels < 2:
        return None, levels
    cells: dict[tuple, list[float]] = defaultdict(list)
    for s in scores:
        key = (s[0],) + tuple(s[k] for k in (2, 3, 4) if k != axis)
        cells[key].append(s[1])
    variance, df = _pooled_variance(list(cells.values()))
    if df < MIN_AXIS_DF:
        return None, levels
    return variance, levels


def noise_budget(scores: list[ScoreTuple]) -> NoiseBudget | None:
    """Decompose repeated-run score variance into a noise budget.

    Returns None when there is nothing to estimate from: fewer than two
    items, or no item with a repeated score. Otherwise returns the budget,
    with each unmeasured source left as None.
    """
    if not scores:
        return None
    items = {s[0] for s in scores}
    if len(items) < 2:
        return None
    by_item: dict[str, list[float]] = defaultdict(list)
    for s in scores:
        by_item[s[0]].append(s[1])
    total_variance, _ = _pooled_variance(list(by_item.values()))
    if total_variance <= 0.0 and all(len(v) < 2 for v in by_item.values()):
        return None

    sampling, g_levels = _axis_variance(scores, 2)
    judge, j_levels = _axis_variance(scores, 3)
    environment, e_levels = _axis_variance(scores, 4)
    components: dict[str, float | None] = {
        SAMPLING: sampling,
        JUDGE: judge,
        ENVIRONMENT: environment,
    }
    levels = {SAMPLING: g_levels, JUDGE: j_levels, ENVIRONMENT: e_levels}

    interaction: float | None = None
    if all(v is not None for v in components.values()):
        interaction = max(
            0.0,
            total_variance - sum(v for v in components.values() if v is not None),
        )

    n_items = len(items)
    run_mean_variance = 0.0
    for source in SOURCES:
        value = components[source]
        if value is not None:
            run_mean_variance += value / (n_items * levels[source])
    if interaction is not None:
        run_mean_variance += interaction / (
            n_items * g_levels * j_levels * e_levels
        )
    noise_floor = NOISE_Z * sqrt(2.0 * run_mean_variance)

    return NoiseBudget(
        n_items=n_items,
        n_scores=len(scores),
        levels=levels,
        components=components,
        interaction=interaction,
        total_variance=total_variance,
        run_mean_variance=run_mean_variance,
        noise_floor=noise_floor,
    )


def claim_verdict(claimed_delta: float, budget: NoiseBudget) -> str:
    """Judge a claimed score change against the noise floor.

    "within_noise": the claim is below the floor -- and the floor only
    grows as unmeasured sources are added, so this verdict is safe even
    with partial data. "real": the claim clears the floor with every
    source measured. "cant_tell": the claim clears the floor computed
    from partial data, so the unmeasured sources decide it.
    """
    if abs(claimed_delta) < budget.noise_floor:
        return "within_noise"
    if budget.unmeasured_sources:
        return "cant_tell"
    return "real"
