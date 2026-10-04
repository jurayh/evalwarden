"""Tests for src/evalwarden/coverage.py: pure panel-coverage helpers.

These back JUDGE-009 (panel blind spots) and DATA-003 (elicitation
coverage). The check-level kill criterion lives in
tests/test_panel_coverage.py.
"""
from __future__ import annotations

from evalwarden.coverage import (
    JudgeCatch,
    ModeCatch,
    blind_spots,
    mode_catch_table,
    mode_item_counts,
    thin_modes,
)


def _mc(mode, n_items, *judges: JudgeCatch) -> ModeCatch:
    return ModeCatch(mode=mode, n_labeled_items=n_items, per_judge=list(judges))


def test_judge_catch_rate():
    assert JudgeCatch("a", 40, 30).rate == 0.75
    assert JudgeCatch("a", 0, 0).rate == 0.0


def test_mode_catch_table_counts_only_labeled_tagged_items():
    labels = {"t1": "sol-a", "t2": "sol-b", "t3": "sol-a"}
    mode_of = {"t1": "m1", "t2": "m1", "t4": "m1"}  # t3 untagged, t4 unlabeled
    verdicts = {
        "judge-a": {"t1": "sol-a", "t2": "sol-a", "t4": "sol-a"},
        "judge-b": {"t1": "sol-b"},
    }
    table = mode_catch_table(verdicts, labels, mode_of)
    assert set(table) == {"m1"}
    row = table["m1"]
    assert row.n_labeled_items == 2  # t1, t2 only
    by_judge = {jc.judge_id: jc for jc in row.per_judge}
    assert by_judge["judge-a"].n_labeled == 2
    assert by_judge["judge-a"].n_caught == 1  # t1 matches, t2 does not
    assert by_judge["judge-b"].n_labeled == 1
    assert by_judge["judge-b"].n_caught == 0


def test_mode_catch_table_judge_without_verdicts_absent_from_row():
    labels = {"t1": "sol-a"}
    mode_of = {"t1": "m1"}
    verdicts = {"judge-a": {"t1": "sol-a"}, "judge-b": {"t9": "sol-a"}}
    table = mode_catch_table(verdicts, labels, mode_of)
    assert [jc.judge_id for jc in table["m1"].per_judge] == ["judge-a"]


def test_blind_spots_all_judges_below_bar():
    table = {"blind-mode": _mc("blind-mode", 40, JudgeCatch("a", 40, 18), JudgeCatch("b", 40, 22))}
    spots = blind_spots(table)
    assert [s.mode for s in spots] == ["blind-mode"]


def test_blind_spots_one_catching_judge_covers_the_mode():
    table = {"ok-mode": _mc("ok-mode", 40, JudgeCatch("a", 40, 36), JudgeCatch("b", 40, 16))}
    assert blind_spots(table) == []


def test_blind_spots_rate_exactly_at_bar_is_a_catch():
    # The bar is "at or above catches", so 70% exactly covers the mode.
    table = {"edge-mode": _mc("edge-mode", 40, JudgeCatch("a", 40, 28), JudgeCatch("b", 40, 10))}
    assert blind_spots(table) == []


def test_blind_spots_thin_mode_stays_silent():
    # Fewer than min_mode_items labeled items: no blind-spot claim.
    table = {"rare-mode": _mc("rare-mode", 8, JudgeCatch("a", 8, 0), JudgeCatch("b", 8, 0))}
    assert blind_spots(table) == []


def test_blind_spots_needs_two_measurable_judges():
    # Judge b judged only 3 mode items: not measurable, so the panel claim
    # cannot rest on judge a alone.
    table = {"m": _mc("m", 40, JudgeCatch("a", 40, 10), JudgeCatch("b", 3, 0))}
    assert blind_spots(table) == []


def test_blind_spots_sparse_judge_does_not_create_blindness():
    # Judge b barely saw the mode and failed; judge a covers it. Covered.
    table = {"m": _mc("m", 40, JudgeCatch("a", 40, 38), JudgeCatch("b", 3, 0))}
    assert blind_spots(table) == []


def test_blind_spots_sorted_by_mode():
    table = {
        "z-mode": _mc("z-mode", 40, JudgeCatch("a", 40, 10), JudgeCatch("b", 40, 12)),
        "a-mode": _mc("a-mode", 40, JudgeCatch("a", 40, 10), JudgeCatch("b", 40, 12)),
    }
    assert [s.mode for s in blind_spots(table)] == ["a-mode", "z-mode"]


def test_mode_item_counts():
    mode_of = {"t1": "m1", "t2": "m1", "t3": "m2"}
    assert mode_item_counts(mode_of) == {"m1": 2, "m2": 1}


def test_thin_modes_thin_and_absent():
    counts = {"covered": 40, "thin-mode": 3, "mid": 29}
    thin, absent = thin_modes(counts, ["covered", "thin-mode", "mid", "ghost"], 30)
    assert thin == [("thin-mode", 3), ("mid", 29)]  # sorted by count
    assert absent == ["ghost"]


def test_thin_modes_exactly_at_minimum_is_not_thin():
    thin, absent = thin_modes({"m": 30}, [], 30)
    assert thin == [] and absent == []


def test_thin_modes_no_taxonomy_means_no_absent():
    thin, absent = thin_modes({"m": 2}, [], 30)
    assert thin == [("m", 2)]
    assert absent == []
