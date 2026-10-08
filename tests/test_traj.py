"""TRAJ-lane checks: exact loops (TRAJ-001) and unused outputs (TRAJ-002).

Kill-criterion proof: synthetic seeded trajectories with known injected
waste -- loops of varying length, unused outputs, clean efficient runs
mixed in. The analyzers must separate the wasteful classes cleanly; the
per-class detection numbers are asserted below (12 trajectories per class,
seeded RNG).
"""
from __future__ import annotations

import random

from evalwarden.checks.traj import LoopCheck, UnusedOutputCheck
from evalwarden.model import Confidence, Severity

from .conftest import make_attempt, make_model, make_step

SEED = 20261001
N_PER_CLASS = 12


def _chain(tag, calls, unused_idx=frozenset()):
    """Linear chain: step i consumes step i-1, except after unused indices."""
    spans = []
    for i, (tool, args) in enumerate(calls):
        sid = f"{tag}-{i:02d}"
        prev = f"{tag}-{i - 1:02d}"
        consumes = [prev] if i > 0 and (i - 1) not in unused_idx else []
        spans.append(make_step(step_id=sid, tool=tool, args=args, consumes=consumes))
    return spans


def _unique_calls(rng, n, start=0):
    return [(f"tool_{start + i}", {"seq": start + i, "salt": rng.randint(0, 10**6)}) for i in range(n)]


def _loop5(rng, tag):
    calls = _unique_calls(rng, 12)
    for i in (1, 4, 6, 8, 10):
        calls[i] = ("read_file", {"path": "notes.txt"})
    return _chain(tag, calls)


def _loop8(rng, tag):
    calls = _unique_calls(rng, 14)
    for i in (0, 2, 3, 5, 7, 9, 11, 13):
        calls[i] = ("grep", {"pattern": "todo"})
    return _chain(tag, calls)


def _spin3(rng, tag):
    calls = _unique_calls(rng, 10)
    for i in (4, 5, 6):
        calls[i] = ("list_dir", {"dir": "."})
    return _chain(tag, calls)


def _unused4(rng, tag):
    return _chain(tag, _unique_calls(rng, 12), unused_idx=frozenset({2, 5, 7, 9}))


def _clean(rng, tag):
    return _chain(tag, _unique_calls(rng, 10))


def _retry2(rng, tag):
    calls = _unique_calls(rng, 8)
    calls[2] = ("fetch_url", {"url": "https://example.com"})
    calls[6] = ("fetch_url", {"url": "https://example.com"})
    return _chain(tag, calls)


BUILDERS = {
    "LOOP5": _loop5,
    "LOOP8": _loop8,
    "SPIN3": _spin3,
    "UNUSED4": _unused4,
    "CLEAN": _clean,
    "RETRY2": _retry2,
}

# Expected (traj001_fires, traj002_fires) per class.
EXPECTED = {
    "LOOP5": (True, False),
    "LOOP8": (True, False),
    "SPIN3": (True, False),
    "UNUSED4": (False, True),
    "CLEAN": (False, False),
    "RETRY2": (False, False),
}


def _detection_numbers():
    rng = random.Random(SEED)
    numbers = {}
    for cls, build in BUILDERS.items():
        n1 = n2 = 0
        for k in range(N_PER_CLASS):
            spans = build(rng, f"{cls}-{k}")
            model = make_model(attempts=[make_attempt(task_id=f"{cls}-{k}", spans=spans)])
            n1 += 1 if LoopCheck().run(model) else 0
            n2 += 1 if UnusedOutputCheck().run(model) else 0
        numbers[cls] = (n1, n2)
    return numbers


def test_kill_criterion_separates_wasteful_from_clean():
    numbers = _detection_numbers()
    for cls, (n1, n2) in numbers.items():
        want1, want2 = EXPECTED[cls]
        assert (n1 == N_PER_CLASS) == want1, f"{cls}: TRAJ-001 fired {n1}/{N_PER_CLASS}"
        assert (n2 == N_PER_CLASS) == want2, f"{cls}: TRAJ-002 fired {n2}/{N_PER_CLASS}"


def test_loop_finding_is_deterministic_evidence():
    rng = random.Random(SEED)
    model = make_model(attempts=[make_attempt(task_id="t1", spans=_loop5(rng, "x"))])
    (finding,) = LoopCheck().run(model)
    assert finding.id == "TRAJ-001"
    # _loop5 repeats are scattered (longest run 1): reported, one
    # severity down -- scattered repeats can be deliberate re-verification.
    assert finding.severity == Severity.MEDIUM
    assert finding.confidence == Confidence.HIGH
    assert "read_file" in finding.evidence[0] and "x5" in finding.evidence[0]


def test_spin_finding_is_high():
    rng = random.Random(SEED)
    model = make_model(attempts=[make_attempt(task_id="t1", spans=_spin3(rng, "x"))])
    (finding,) = LoopCheck().run(model)
    assert finding.id == "TRAJ-001"
    assert finding.severity == Severity.HIGH  # 3x back-to-back is the stuck signature


def test_shell_repeats_with_recaptioned_descriptions_still_loop():
    # Real Claude Code shell args carry a free-text "description" that
    # changes between otherwise identical calls (Optimal Misbehavior
    # corpus). The caption is prose, not call semantics.
    calls = [
        ("Read", {"file_path": "a.py"}),
        ("PowerShell", {"command": "node verify.js", "description": "Run verify"}),
        ("Edit", {"file_path": "a.py", "old_string": "x", "new_string": "y"}),
        ("PowerShell", {"command": "node verify.js", "description": "Verify again"}),
        ("Read", {"file_path": "b.py"}),
        ("PowerShell", {"command": "node verify.js", "description": "Re-run verification"}),
        ("Grep", {"pattern": "todo"}),
        ("PowerShell", {"command": "node verify.js", "description": "Final verify pass"}),
    ]
    spans = _chain("cap", calls)
    model = make_model(attempts=[make_attempt(task_id="t1", spans=spans)])
    (finding,) = LoopCheck().run(model)
    assert finding.id == "TRAJ-001"
    assert "PowerShell" in finding.evidence[0] and "x4" in finding.evidence[0]


def test_unused_finding_names_steps():
    rng = random.Random(SEED)
    model = make_model(attempts=[make_attempt(task_id="t1", spans=_unused4(rng, "x"))])
    (finding,) = UnusedOutputCheck().run(model)
    assert finding.id == "TRAJ-002"
    assert finding.severity == Severity.HIGH
    assert finding.confidence == Confidence.HIGH
    assert "x-02" in finding.evidence[0]


def test_findings_are_per_attempt():
    rng = random.Random(SEED)
    attempts = [
        make_attempt(task_id="a", spans=_loop5(rng, "a")),
        make_attempt(task_id="b", spans=_loop8(rng, "b")),
        make_attempt(task_id="c", spans=_clean(rng, "c")),
    ]
    findings = LoopCheck().run(make_model(attempts=attempts))
    assert len(findings) == 2


def test_no_spans_stays_silent():
    model = make_model(attempts=[make_attempt(task_id="t1", spans=[])])
    assert LoopCheck().run(model) == []
    assert UnusedOutputCheck().run(model) == []


def test_no_consumption_data_stays_silent():
    rng = random.Random(SEED)
    spans = _chain("x", _unique_calls(rng, 8))
    for s in spans:
        s.consumes = []
    model = make_model(attempts=[make_attempt(task_id="t1", spans=spans)])
    assert UnusedOutputCheck().run(model) == []


def test_short_trajectory_stays_silent():
    # 4 spans, 2 effectively unused -- below the waste threshold and length guard
    spans = _chain("x", [("a", {}), ("b", {}), ("c", {}), ("d", {})],
                   unused_idx=frozenset({0, 1}))
    model = make_model(attempts=[make_attempt(task_id="t1", spans=spans)])
    assert UnusedOutputCheck().run(model) == []


def test_two_unused_outputs_stays_silent():
    rng = random.Random(SEED)
    spans = _chain("x", _unique_calls(rng, 8), unused_idx=frozenset({1, 4}))
    model = make_model(attempts=[make_attempt(task_id="t1", spans=spans)])
    assert UnusedOutputCheck().run(model) == []
