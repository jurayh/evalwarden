"""Paired differ statistics: McNemar exact test, alignment, flip accounting.

The kill-criterion fixtures live in tests/test_diff.py; these tests pin
the estimators themselves on hand-computed cases.
"""
from __future__ import annotations

import pytest

from evalwarden.diffing import align, diff_stats, mcnemar_exact_p


def test_mcnemar_no_discordant_pairs_is_no_evidence():
    assert mcnemar_exact_p(0, 0) == 1.0


def test_mcnemar_one_sided_split():
    # All flips one way: p = 2 / 2^n.
    assert mcnemar_exact_p(30, 0) == pytest.approx(2 / 2**30)
    assert mcnemar_exact_p(0, 30) == pytest.approx(2 / 2**30)
    assert mcnemar_exact_p(1, 0) == 1.0


def test_mcnemar_known_split():
    # 12 vs 1 over 13 discordant pairs: 2 * (C(13,0) + C(13,1)) / 2^13.
    assert mcnemar_exact_p(12, 1) == pytest.approx(28 / 8192)


def test_mcnemar_symmetric_and_monotone():
    assert mcnemar_exact_p(24, 20) == mcnemar_exact_p(20, 24)
    assert 0.6 < mcnemar_exact_p(24, 20) < 0.7
    # A wider split at the same discordant volume is stronger evidence.
    assert mcnemar_exact_p(34, 10) < mcnemar_exact_p(24, 20)


def test_align_reports_paired_and_unpaired_per_side():
    alignment = align(["a", "b", "c"], ["b", "c", "d"])
    assert alignment.paired_ids == ["b", "c"]
    assert alignment.baseline_only == ["a"]
    assert alignment.candidate_only == ["d"]
    assert alignment.coverage == pytest.approx(2 / 4)


def test_align_empty_is_zero_coverage():
    alignment = align([], [])
    assert alignment.n_paired == 0
    assert alignment.coverage == 0.0


def test_diff_stats_counts_flips_and_means():
    pairs = [
        ("t1", 1.0, 0.0),  # regression
        ("t2", 0.0, 1.0),  # gain
        ("t3", 1.0, 1.0),  # unchanged pass
        ("t4", 0.0, 0.0),  # unchanged fail
        ("t5", 0.4, 0.6),  # score moved, outcome flipped fail -> pass: gain
    ]
    stats = diff_stats(pairs)
    assert stats.n_paired == 5
    assert stats.regressions == 1
    assert stats.gains == 2
    assert stats.unchanged == 2
    assert stats.baseline_mean == pytest.approx(2.4 / 5)
    assert stats.candidate_mean == pytest.approx(2.6 / 5)
    assert stats.net_regression_rate == pytest.approx(-1 / 5)


def test_diff_stats_worst_regressions_deterministic_order():
    pairs = [
        ("tb", 1.0, 0.0),  # drop 1.0
        ("ta", 1.0, 0.0),  # drop 1.0, ties broken by task_id
        ("tc", 0.9, 0.4),  # drop 0.5
        ("td", 1.0, 1.0),
    ]
    stats = diff_stats(pairs)
    assert [f.task_id for f in stats.worst] == ["ta", "tb", "tc"]


def test_diff_stats_slices_group_by_metadata():
    pairs = [
        ("t1", 1.0, 0.0),
        ("t2", 1.0, 0.0),
        ("t3", 0.0, 1.0),
        ("t4", 1.0, 1.0),  # no tags: joins no slice
    ]
    tags = {
        "t1": {"subject": "algebra"},
        "t2": {"subject": "algebra"},
        "t3": {"subject": "geometry"},
    }
    stats = diff_stats(pairs, tags)
    by_value = {(s.key, s.value): s for s in stats.slices}
    algebra = by_value[("subject", "algebra")]
    assert (algebra.n, algebra.regressions, algebra.gains) == (2, 2, 0)
    assert algebra.net_regression_rate == pytest.approx(1.0)
    geometry = by_value[("subject", "geometry")]
    assert (geometry.n, geometry.regressions, geometry.gains) == (1, 0, 1)
