"""Clone helper tests: isomorphism, transport, declared rules, gap math."""
from __future__ import annotations

import pytest

from evalwarden.clones import (
    CloneableTask,
    clone_gap,
    derive_reference,
    generate_clone,
    parse_cloneable,
    parse_output_items,
    score,
    transport_output,
)

ENTRIES = [("Ana", 7), ("Boris", 3), ("Cleo", 9), ("Dario", 5), ("Mira", 12)]
THRESHOLD = 7  # qualifying: Ana, Cleo, Mira (roster order)


def _task(task_id="t1", output="Ana, Cleo, Mira", entries=ENTRIES, threshold=THRESHOLD):
    return CloneableTask(
        task_id=task_id,
        entries=tuple(entries),
        threshold=threshold,
        reference_items=tuple(derive_reference(entries, threshold)),
        output=output,
    )


def _metadata(entries=ENTRIES, threshold=THRESHOLD, family="selection"):
    return {
        "clone_family": family,
        "entries": [{"name": n, "value": v} for n, v in entries],
        "threshold": threshold,
    }


def test_derive_reference_roster_order_and_threshold_inclusive():
    assert derive_reference(ENTRIES, THRESHOLD) == ["Ana", "Cleo", "Mira"]
    assert derive_reference(ENTRIES, 13) == []


def test_parse_cloneable_accepts_consistent_task():
    task = parse_cloneable("t1", _metadata(), "Ana, Cleo, Mira", "Ana, Cleo, Mira")
    assert task is not None
    assert task.reference_items == ("Ana", "Cleo", "Mira")
    assert task.output == "Ana, Cleo, Mira"


@pytest.mark.parametrize(
    "metadata,target,output",
    [
        (_metadata(family="other"), "Ana, Cleo, Mira", "Ana, Cleo, Mira"),  # family
        ({"entries": []}, "Ana, Cleo, Mira", "Ana, Cleo, Mira"),  # no family tag
        (_metadata(), "Ana, Cleo, Mira", None),  # no cached output
        (_metadata(), None, "Ana, Cleo, Mira"),  # no target
        (_metadata(), ["Ana", "Cleo", "Mira"], "Ana, Cleo, Mira"),  # non-string target
        (_metadata(), "Cleo, Ana, Mira", "Ana, Cleo, Mira"),  # target order != roster
        (_metadata(), "Ana, Cleo", "Ana, Cleo, Mira"),  # target != derived reference
        (_metadata(entries=[("Ana", 7), ("Ana", 9)]), "Ana", "Ana"),  # dup names
        (_metadata(entries=[("Ana", "7")]), "Ana", "Ana"),  # non-numeric value
        (_metadata(entries=[]), "", "x"),  # no entries
        (None, "Ana, Cleo, Mira", "Ana, Cleo, Mira"),  # no metadata
    ],
)
def test_parse_cloneable_rejects_uncloneable(metadata, target, output):
    assert parse_cloneable("t1", metadata, target, output) is None


def test_clone_is_isomorphic():
    task = _task()
    clone = generate_clone(task)
    # Renaming is a bijection onto names the task does not use.
    assert set(clone.rename) == {n for n, _ in ENTRIES}
    assert not set(clone.rename.values()) & set(clone.rename)
    # The qualifying set is preserved exactly, carried through the rename.
    assert {clone.rename[n] for n in task.reference_items} == set(clone.reference_items())
    # Values and threshold share one positive scale factor.
    factor = clone.threshold / task.threshold
    assert factor in (2, 3)
    original_values = dict(task.entries)
    for name, value in clone.entries:
        original = next(o for o, r in clone.rename.items() if r == name)
        assert value == original_values[original] * factor


def test_clone_generation_is_deterministic():
    task = _task()
    assert generate_clone(task) == generate_clone(task)
    assert generate_clone(task, seed=7) == generate_clone(task, seed=7)


def test_transport_output_substitutes_names_preserving_order():
    rename = {"Ana": "Xenia", "Cleo": "Albin", "Mira": "Radek"}
    assert transport_output("Ana, Cleo, Mira", rename) == "Xenia, Albin, Radek"
    # Unknown items pass through untouched; the order is the model's, kept.
    assert transport_output("Cleo, Zed", rename) == "Albin, Zed"


def test_score_exact_string_is_order_sensitive():
    ref = ("Ana", "Cleo", "Mira")
    assert score("exact_string", "Ana, Cleo, Mira", ref)
    assert not score("exact_string", "Cleo, Ana, Mira", ref)
    assert not score("exact_string", "Ana, Cleo", ref)


def test_score_set_match_is_order_insensitive():
    ref = ("Ana", "Cleo", "Mira")
    assert score("set_match", "Mira, Ana, Cleo", ref)
    assert not score("set_match", "Ana, Cleo", ref)
    assert not score("set_match", "Ana, Cleo, Mira, Boris", ref)


def test_score_unknown_rule_raises():
    with pytest.raises(ValueError):
        score("regex", "Ana", ("Ana",))


def test_parse_output_items():
    assert parse_output_items(" Ana, Cleo ,, Mira ") == ["Ana", "Cleo", "Mira"]
    assert parse_output_items("") == []


def _fixture_tasks(n=12):
    """Correct outputs on varied rosters: the clone-gap test bed."""
    tasks = []
    for i in range(n):
        entries = [(f"N{i}{j}", v) for j, v in enumerate([4, 11, 6, 15, 9, 20])]
        reference = derive_reference(entries, 10)
        tasks.append(
            CloneableTask(
                task_id=f"t{i}",
                entries=tuple(entries),
                threshold=10,
                reference_items=tuple(reference),
                output=", ".join(reference),
            )
        )
    return tasks


def test_clone_gap_set_match_is_exactly_zero():
    stats = clone_gap(_fixture_tasks(), "set_match")
    assert stats is not None
    assert stats.gap == 0.0
    assert stats.flips == 0
    assert stats.clone_passes == stats.original_passes == stats.n_items


def test_clone_gap_exact_string_separates():
    stats = clone_gap(_fixture_tasks(), "exact_string")
    assert stats is not None
    assert stats.original_passes == stats.n_items
    assert stats.flips > 0
    assert stats.gap > 0.25
    assert stats.flips == stats.original_passes - stats.clone_passes
    assert set(stats.flipped_ids) <= {t.task_id for t in _fixture_tasks()}


def test_clone_gap_silent_without_support():
    assert clone_gap(_fixture_tasks(), "cosine_similarity") is None
    assert clone_gap([], "set_match") is None
