"""Noise budget helpers: planted-variance recovery (the kill criterion).

Synthetic score tables with planted variance components: one where judge
noise is 60% of the variance, one where sampling dominates, one balanced.
The decomposition must recover the planted shares within tolerance and
attribute the dominant source correctly in at least 80% of seeded cases.
"""
from __future__ import annotations

import random
from math import sqrt

from evalwarden.noise import (
    ENVIRONMENT,
    JUDGE,
    SAMPLING,
    ScoreTuple,
    claim_verdict,
    noise_budget,
)

N_ITEMS = 40
N_GEN, N_GRADE, N_ENV = 3, 3, 2
TOLERANCE = 0.12  # recovered share within this of the planted share
SEEDS = range(10)


def plant_table(
    seed: int,
    var_sampling: float,
    var_judge: float,
    var_environment: float,
    var_residual: float,
    n_items: int = N_ITEMS,
    n_gen: int = N_GEN,
    n_grade: int = N_GRADE,
    n_env: int = N_ENV,
) -> list[ScoreTuple]:
    """Additive planted components: score = item base + sampling effect
    (per item x generation) + judge effect (per item x grade) +
    environment effect (per item x environment) + per-cell residual."""
    rng = random.Random(seed)
    scores: list[ScoreTuple] = []
    for i in range(n_items):
        task_id = f"t{i:03d}"
        base = rng.gauss(70.0, 4.0)
        samp = {g: rng.gauss(0.0, sqrt(var_sampling)) for g in range(n_gen)}
        judg = {j: rng.gauss(0.0, sqrt(var_judge)) for j in range(n_grade)}
        env = {e: rng.gauss(0.0, sqrt(var_environment)) for e in range(n_env)}
        for g in range(n_gen):
            for e in range(n_env):
                for j in range(n_grade):
                    resid = rng.gauss(0.0, sqrt(var_residual))
                    scores.append(
                        (
                            task_id,
                            base + samp[g] + judg[j] + env[e] + resid,
                            g,
                            j,
                            e,
                        )
                    )
    return scores


def _planted_shares(var_s: float, var_j: float, var_e: float, var_r: float):
    total = var_s + var_j + var_e + var_r
    return {SAMPLING: var_s / total, JUDGE: var_j / total, ENVIRONMENT: var_e / total}


# (var_sampling, var_judge, var_environment, var_residual), planted dominant
JUDGE_DOMINATED = (25.0, 60.0, 10.0, 5.0)
SAMPLING_DOMINATED = (60.0, 15.0, 15.0, 10.0)


def test_judge_dominated_table_recovered():
    planted = _planted_shares(*JUDGE_DOMINATED)
    dominant_ok = 0
    shares_ok = 0
    for seed in SEEDS:
        budget = noise_budget(plant_table(seed, *JUDGE_DOMINATED))
        assert budget is not None
        assert budget.measured_sources == [SAMPLING, JUDGE, ENVIRONMENT]
        dominant = budget.dominant()
        assert dominant is not None
        if dominant[0] == JUDGE:
            dominant_ok += 1
        shares = budget.shares()
        if all(abs(shares[s] - planted[s]) <= TOLERANCE for s in planted):
            shares_ok += 1
    # Kill criterion: >= 80% of seeded cases attribute and recover.
    assert dominant_ok >= 8, f"dominant judge in {dominant_ok}/10 seeds"
    assert shares_ok >= 8, f"shares within {TOLERANCE} in {shares_ok}/10 seeds"


def test_sampling_dominated_table_recovered():
    planted = _planted_shares(*SAMPLING_DOMINATED)
    dominant_ok = 0
    shares_ok = 0
    for seed in SEEDS:
        budget = noise_budget(plant_table(seed, *SAMPLING_DOMINATED))
        assert budget is not None
        dominant = budget.dominant()
        assert dominant is not None
        if dominant[0] == SAMPLING:
            dominant_ok += 1
        shares = budget.shares()
        if all(abs(shares[s] - planted[s]) <= TOLERANCE for s in planted):
            shares_ok += 1
    assert dominant_ok >= 8, f"dominant sampling in {dominant_ok}/10 seeds"
    assert shares_ok >= 8, f"shares within {TOLERANCE} in {shares_ok}/10 seeds"


def test_budget_shape():
    budget = noise_budget(plant_table(20261001, *JUDGE_DOMINATED))
    assert budget is not None
    assert budget.n_items == N_ITEMS
    assert budget.levels == {SAMPLING: N_GEN, JUDGE: N_GRADE, ENVIRONMENT: N_ENV}
    assert budget.interaction is not None and budget.interaction >= 0.0
    assert budget.total_variance > 0.0
    assert budget.noise_floor > 0.0
    shares = budget.shares()
    assert abs(sum(shares.values()) - 1.0) < 1e-9


def test_no_data_returns_none():
    assert noise_budget([]) is None
    assert noise_budget([("t1", 1.0, 0, 0, 0)]) is None  # one item
    # Two items but no repeats anywhere: nothing to estimate from.
    assert (
        noise_budget([("t1", 1.0, 0, 0, 0), ("t2", 2.0, 0, 0, 0)]) is None
    )


def test_identical_repeated_scores_give_zero_floor():
    scores = [
        (f"t{i}", 80.0, g, j, e)
        for i in range(5)
        for g in range(2)
        for j in range(2)
        for e in range(2)
    ]
    budget = noise_budget(scores)
    assert budget is not None
    assert budget.total_noise == 0.0
    assert budget.noise_floor == 0.0
    assert budget.shares() == {}
    assert budget.dominant() is None


def test_unmeasured_axes_report_none_not_zero():
    """Only generations vary (judge and environment frozen): the budget
    measures total replicate spread along one axis and refuses to
    attribute the rest."""
    rng = random.Random(7)
    scores: list[ScoreTuple] = []
    for i in range(20):
        base = rng.gauss(50.0, 3.0)
        for g in range(4):
            scores.append((f"t{i}", base + rng.gauss(0.0, 2.0), g, 0, 0))
    budget = noise_budget(scores)
    assert budget is not None
    assert budget.components[SAMPLING] is not None
    assert budget.components[JUDGE] is None
    assert budget.components[ENVIRONMENT] is None
    assert budget.interaction is None
    assert budget.unmeasured_sources == [JUDGE, ENVIRONMENT]
    assert budget.noise_floor > 0.0


def test_claim_verdicts():
    full = noise_budget(plant_table(3, *JUDGE_DOMINATED))
    assert full is not None and full.noise_floor > 0
    # A claim far below the floor is within noise; far above is real.
    assert claim_verdict(full.noise_floor / 10, full) == "within_noise"
    assert claim_verdict(full.noise_floor * 5, full) == "real"

    partial_scores: list[ScoreTuple] = []
    rng = random.Random(11)
    for i in range(20):
        base = rng.gauss(50.0, 3.0)
        for g in range(4):
            partial_scores.append((f"t{i}", base + rng.gauss(0.0, 0.1), g, 0, 0))
    partial = noise_budget(partial_scores)
    assert partial is not None and partial.unmeasured_sources
    # Clears the partial floor, but unmeasured sources decide it.
    assert claim_verdict(partial.noise_floor * 5, partial) == "cant_tell"
    # Below the partial floor is within noise regardless: the floor only
    # grows with more data.
    assert claim_verdict(partial.noise_floor / 10, partial) == "within_noise"
