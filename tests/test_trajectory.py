"""Unit tests for the pure trajectory-integrity helpers."""
from __future__ import annotations

from evalwarden.trajectory import (
    canonical_call,
    find_loops,
    has_consumption_data,
    unused_outputs,
)


def test_canonical_call_ignores_arg_order():
    assert canonical_call("read", {"b": 1, "a": 2}) == canonical_call("read", {"a": 2, "b": 1})


def test_canonical_call_distinguishes_values_and_tools():
    assert canonical_call("read", {"a": 1}) != canonical_call("read", {"a": 2})
    assert canonical_call("read", {"a": 1}) != canonical_call("write", {"a": 1})


def test_canonical_call_ignores_prose_captions():
    a = canonical_call("Bash", {"command": "npm test", "description": "Run tests"})
    b = canonical_call("Bash", {"command": "npm test", "description": "Tests, again"})
    assert a == b
    c = canonical_call("Bash", {"command": "npm run build", "description": "Run tests"})
    assert a != c  # a changed semantic argument is still a different call


def test_canonical_call_survives_non_json_values():
    key = canonical_call("tool", {"x": object()})
    assert key.startswith("tool(")


def test_find_loops_empty():
    assert find_loops([]) == []


def test_find_loops_below_threshold_silent():
    calls = ["a", "b", "a", "c", "b"]  # max total 2, no consecutive runs
    assert find_loops(calls) == []


def test_find_loops_total_rule():
    calls = ["x", "a", "y", "a", "z", "a", "w", "a"]  # a x4 scattered
    hits = find_loops(calls)
    assert [(h.key, h.total) for h in hits] == [("a", 4)]


def test_find_loops_consecutive_rule():
    calls = ["x", "b", "b", "b", "y"]  # b x3 back-to-back, total 3 < 4
    hits = find_loops(calls)
    assert len(hits) == 1
    assert hits[0].key == "b" and hits[0].max_consecutive == 3


def test_find_loops_reports_first_appearance_order():
    calls = ["m", "n", "m", "m", "m", "m", "n", "n", "n", "n"]
    hits = find_loops(calls)
    assert [h.key for h in hits] == ["m", "n"]


def test_find_loops_custom_thresholds():
    calls = ["a", "a"]
    assert find_loops(calls, total_at=2) != []
    assert find_loops(calls, total_at=3) == []


def test_has_consumption_data():
    assert not has_consumption_data([])
    assert not has_consumption_data([[], [], []])
    assert has_consumption_data([[], ["s0"]])


def test_unused_outputs_basic():
    ids = ["s0", "s1", "s2", "s3"]
    consumes = [[], ["s0"], ["s1"], ["s0", "s2"]]
    # s3 is final (carved out); s0, s1, s2 all consumed
    assert unused_outputs(ids, consumes) == []


def test_unused_outputs_flags_unconsumed():
    ids = ["s0", "s1", "s2", "s3"]
    consumes = [[], [], ["s1"], ["s2"]]
    # s0 consumed by nothing; s3 final carved out
    assert unused_outputs(ids, consumes) == ["s0"]


def test_unused_outputs_final_step_carved_out():
    ids = ["s0", "s1"]
    consumes = [[], []]
    # s1 is final: never flagged even though nothing consumes it
    assert unused_outputs(ids, consumes) == ["s0"]


def test_unused_outputs_self_edge_does_not_count():
    ids = ["s0", "s1", "s2"]
    consumes = [["s0"], [], ["s1"]]
    # s0's only "consumer" is itself: still unused. s2 final carved out.
    assert unused_outputs(ids, consumes) == ["s0"]


def test_unused_outputs_empty():
    assert unused_outputs([], []) == []
