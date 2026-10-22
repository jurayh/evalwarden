"""TRAJ-003: stalled edit-test oscillation over phase outcomes.

Second spike (Oct 2026). The first TRAJ-003 spike died at its
feasibility gate: phases segment in the audit path, but no per-step
outcome existed to judge a test phase with. TrajectoryStep.outcome has
since landed, so the motif is buildable -- and bound by the same
precision controls as the first spike:

- planted stalls: repeated edit -> test cycles, changing arguments
  throughout (TRAJ-001 must stay silent on them), every test phase
  errored, no intervening explore/plan/review, no successful test
  afterward;
- honest lookalikes that must stay silent: TDD that eventually passes
  (including intermediate errored test calls), debugging with
  interleaved explore/review and changing hypotheses, a single
  full-suite rerun after the final edit;
- silence contracts: outcome-less test phases, coverage below the
  judged floor, and every shipped demo shape.
"""
from __future__ import annotations

from pathlib import Path

from evalwarden.adapters.inspect_ai import InspectAdapter
from evalwarden.checks import REGISTRY, get_check
from evalwarden.checks.traj import LoopCheck, StalledOscillationCheck
from evalwarden.engine import audit
from evalwarden.model import Confidence, Severity, TrajectoryStep
from evalwarden.trace_phases import segment_phases
from evalwarden.trajectory import find_stalled_oscillation, phase_outcomes

from .conftest import DEMO_AGENT_LOOP, DEMO_AGENT_ORDINARY, NATIVE_LOOP_LOG, make_attempt, make_model
from .trace_fixtures import (
    CLAUDE_CLEAN_SESSION,
    CLAUDE_LOOP_SESSION,
    CODEX_CLEAN_FILE,
    claude_clean_records,
    claude_loop_records,
    codex_clean_records,
    write_claude_project,
    write_codex_session,
)


def _spans(spec, tag="x"):
    """Build spans from a phase spec: [(label, outcome, n_spans), ...].

    outcome is one value for the whole phase or a per-span list.
    Arguments are unique per span, so nothing in these fixtures is an
    exact (tool, args) repeat.
    """
    spans: list[TrajectoryStep] = []
    counters: dict[str, int] = {}
    for label, outcome, n in spec:
        for j in range(n):
            counters[label] = counters.get(label, 0) + 1
            k = counters[label]
            oc = outcome[j] if isinstance(outcome, list) else outcome
            sid = f"{tag}-{label}-{k}"
            if label == "edit":
                spans.append(TrajectoryStep(
                    step_id=sid, tool="Edit",
                    args={"file_path": f"src/{tag}_mod{k}.py",
                          "old_string": f"old {tag} {k}",
                          "new_string": f"new {tag} {k}"},
                    outcome=oc))
            elif label == "test":
                spans.append(TrajectoryStep(
                    step_id=sid, tool="Bash",
                    args={"command": f"pytest -q tests/test_{tag}_{k}.py"},
                    outcome=oc))
            elif label == "explore":
                spans.append(TrajectoryStep(
                    step_id=sid, tool="Read",
                    args={"file_path": f"docs/{tag}_note{k}.md"}, outcome=oc))
            elif label == "plan":
                spans.append(TrajectoryStep(
                    step_id=sid, tool="TodoWrite",
                    args={"todos": f"step {tag} {k}"}, outcome=oc))
            elif label == "review":
                spans.append(TrajectoryStep(
                    step_id=sid, tool="Bash",
                    args={"command": f"git diff --stat -- src/{tag}_{k}.py"},
                    outcome=oc))
            else:  # other
                spans.append(TrajectoryStep(
                    step_id=sid, tool="CustomTool",
                    args={"q": f"{tag}-{k}"}, outcome=oc))
    return spans


def _model(spec, task_id="t1", tag="x"):
    spans = _spans(spec, tag)
    return spans, make_model(attempts=[make_attempt(task_id=task_id, spans=spans)])


def _labels(spans):
    return [seg.label for seg in segment_phases(spans)]


def _cycles(n, test_outcome="error"):
    spec = []
    for _ in range(n):
        spec += [("edit", "ok", 2), ("test", test_outcome, 2)]
    return spec


# ------------------------------------------------------------ planted


def test_planted_stall_fires_and_traj001_stays_silent():
    spans, model = _model([("explore", "ok", 2)] + _cycles(3))
    assert _labels(spans) == ["explore"] + ["edit", "test"] * 3
    (finding,) = StalledOscillationCheck().run(model)
    assert finding.id == "TRAJ-003"
    assert finding.severity == Severity.HIGH
    assert finding.confidence == Confidence.MEDIUM
    assert "3 edit-test cycles" in finding.title
    assert finding.evidence[0] == (
        "phases: explore > edit > test > edit > test > edit > test"
    )
    assert "recorded for 3 of 3 test phases" in finding.evidence[2]
    # Changing arguments throughout: the exact-loop check sees nothing.
    assert LoopCheck().run(model) == []


def test_planted_stall_counts_four_cycles():
    spans, model = _model(_cycles(4))
    (finding,) = StalledOscillationCheck().run(model)
    assert "4 edit-test cycles" in finding.title
    assert "4 consecutive edit > test cycles" in finding.evidence[1]


def test_earlier_successful_test_does_not_rescue_the_stall():
    # A passing setup check before the stall is a different episode;
    # the stall itself never resolves.
    spans, model = _model([("explore", "ok", 2), ("test", "ok", 2)] + _cycles(3))
    assert _labels(spans) == ["explore", "test"] + ["edit", "test"] * 3
    (finding,) = StalledOscillationCheck().run(model)
    assert finding.id == "TRAJ-003"
    assert "recorded for 4 of 4 test phases" in finding.evidence[2]


def test_stall_with_trailing_edit_still_fires():
    # The trajectory ends mid-fix, after the third errored cycle.
    spans, model = _model(_cycles(3) + [("edit", "ok", 2)])
    (finding,) = StalledOscillationCheck().run(model)
    assert finding.id == "TRAJ-003"


def test_stall_at_the_coverage_floor_fires():
    # 3 judged (errored) test phases and 3 unjudged ones: coverage is
    # exactly the 0.5 floor, so the motif may judge -- and the stall
    # run itself is fully judged.
    spec = _cycles(3) + [("explore", "ok", 2)]
    for _ in range(3):
        spec += [("edit", "ok", 2), ("test", None, 2)]
    spans, model = _model(spec)
    (finding,) = StalledOscillationCheck().run(model)
    assert finding.id == "TRAJ-003"
    assert "recorded for 3 of 6 test phases" in finding.evidence[2]


def test_findings_are_per_attempt():
    _, stall = _model(_cycles(3), task_id="stall", tag="s")
    _, clean = _model(
        [("edit", "ok", 2), ("test", "ok", 2)], task_id="clean", tag="c")
    model = make_model(attempts=[stall.attempts[0], clean.attempts[0]])
    findings = StalledOscillationCheck().run(model)
    assert len(findings) == 1
    assert "task stall" in findings[0].title


# ----------------------------------------------------- honest controls


def test_tdd_that_eventually_passes_stays_silent():
    spec = [
        ("explore", "ok", 2),
        ("test", "error", 2),   # the red phase: test call errors first
        ("edit", "ok", 2),
        ("test", "ok", 2),
        ("edit", "ok", 2),
        ("test", "error", 2),   # a later regression, errored call
        ("edit", "ok", 2),
        ("test", "ok", 2),      # resolved
    ]
    spans, model = _model(spec)
    assert _labels(spans) == [
        "explore", "test", "edit", "test", "edit", "test", "edit", "test"]
    assert StalledOscillationCheck().run(model) == []


def test_two_errored_cycles_then_a_pass_stays_silent():
    spec = _cycles(2) + [("edit", "ok", 2), ("test", "ok", 2)]
    spans, model = _model(spec)
    assert StalledOscillationCheck().run(model) == []


def test_debugging_with_interleaved_phases_stays_silent():
    # Errored tests throughout, but explore and review phases keep
    # intervening -- changing hypotheses, not a stall -- and the last
    # test phase succeeds.
    spec = [
        ("edit", "ok", 2), ("test", "error", 2),
        ("explore", "ok", 2),
        ("edit", "ok", 2), ("test", "error", 2),
        ("review", "ok", 2),
        ("edit", "ok", 2), ("test", "error", 2),
        ("explore", "ok", 2),
        ("edit", "ok", 2), ("test", "ok", 2),
    ]
    spans, model = _model(spec)
    assert StalledOscillationCheck().run(model) == []


def test_single_full_suite_rerun_after_final_edit_stays_silent():
    spec = [("explore", "ok", 2), ("edit", "ok", 2), ("test", "ok", 4)]
    spans, model = _model(spec)
    assert _labels(spans) == ["explore", "edit", "test"]
    assert StalledOscillationCheck().run(model) == []


def test_single_errored_test_phase_stays_silent():
    spans, model = _model([("edit", "ok", 2), ("test", "error", 2)])
    assert StalledOscillationCheck().run(model) == []


def test_resolved_stall_stays_silent():
    spec = _cycles(3) + [("edit", "ok", 2), ("test", "ok", 2)]
    spans, model = _model(spec)
    assert StalledOscillationCheck().run(model) == []


def test_plan_phase_breaks_the_run():
    spec = _cycles(2) + [("plan", "ok", 2)] + _cycles(2)
    spans, model = _model(spec)
    assert StalledOscillationCheck().run(model) == []


# -------------------------------------------------- silence contracts


def test_mixed_test_phase_is_not_errored():
    # One runner errored, another completed, in every test phase: the
    # phase is not an errored phase, so there is no stall to report.
    spec = []
    for _ in range(3):
        spec += [("edit", "ok", 2), ("test", ["error", "ok", "error"], 3)]
    spans, model = _model(spec)
    assert StalledOscillationCheck().run(model) == []


def test_outcomeless_test_phases_stay_silent():
    spans, model = _model(_cycles(3, test_outcome=None))
    assert StalledOscillationCheck().run(model) == []


def test_below_the_coverage_floor_stays_silent():
    # The same 3 errored cycles, but most test phases carry no outcome:
    # the unseen phases could hide the resolution.
    spec = _cycles(3)
    for _ in range(4):
        spec += [("edit", "ok", 2), ("test", None, 2)]
    spans, model = _model(spec)  # coverage 3/7 < 0.5
    assert StalledOscillationCheck().run(model) == []


def test_two_errored_cycles_stay_silent():
    spans, model = _model(_cycles(2))
    assert StalledOscillationCheck().run(model) == []


def test_no_spans_stays_silent():
    model = make_model(attempts=[make_attempt(task_id="t1", spans=[])])
    assert StalledOscillationCheck().run(model) == []


# ---------------------------------------------------- helper semantics


def test_phase_outcome_rule():
    spans = _spans([
        ("test", ["error", "error"], 2), ("edit", "ok", 2),
        ("test", ["error", "ok"], 2), ("edit", "ok", 2),
        ("test", [None, None], 2), ("edit", "ok", 2),
        ("test", ["ok", None], 2),
    ])
    segments = segment_phases(spans)
    outcomes = phase_outcomes(segments, spans)
    tests = [o for o in outcomes if o.label == "test"]
    assert [(o.judged, o.errored, o.successful) for o in tests] == [
        (True, True, False),    # every recorded call errored
        (True, False, True),    # mixed: a completion exists, not errored
        (False, False, False),  # nothing recorded: not evidence
        (True, False, True),    # one completion is enough to succeed
    ]


def test_helper_reports_the_longest_qualifying_run():
    spec = _cycles(3) + [("explore", "ok", 2)] + _cycles(4)
    spans = _spans(spec)
    segments = segment_phases(spans)
    stall = find_stalled_oscillation(segments, phase_outcomes(segments, spans))
    assert stall is not None
    assert stall.cycles == 4
    assert stall.test_segments == 7
    assert stall.judged_test_segments == 7


# ------------------------------------------- registry and real shapes


def test_registry_includes_traj003_after_traj002():
    ids = [c.meta.id for c in REGISTRY]
    assert get_check("TRAJ-003") is not None
    assert ids.index("TRAJ-003") == ids.index("TRAJ-002") + 1


def test_silent_on_inspect_demo_artifacts():
    for path in (DEMO_AGENT_LOOP, DEMO_AGENT_ORDINARY):
        adapter = InspectAdapter()
        model = adapter.normalize(adapter.collect(path))
        assert StalledOscillationCheck().run(model) == [], path


def test_agent_loop_is_traj001_territory_not_traj003():
    adapter = InspectAdapter()
    model = adapter.normalize(adapter.collect(DEMO_AGENT_LOOP))
    assert LoopCheck().run(model)  # the exact repeats fire there
    assert StalledOscillationCheck().run(model) == []


def test_silent_on_native_demo_log():
    model = audit(NATIVE_LOOP_LOG).model
    assert StalledOscillationCheck().run(model) == []


def test_silent_on_claude_demo_sessions(tmp_path: Path):
    clean = write_claude_project(
        tmp_path / "c", claude_clean_records(), CLAUDE_CLEAN_SESSION)
    loop = write_claude_project(
        tmp_path / "l", claude_loop_records(), CLAUDE_LOOP_SESSION)
    for path in (clean, loop):
        model = audit(path, adapter_name="claude-code").model
        assert StalledOscillationCheck().run(model) == [], path


def test_silent_on_codex_demo_session(tmp_path: Path):
    path = write_codex_session(tmp_path, codex_clean_records(), CODEX_CLEAN_FILE)
    model = audit(path).model
    assert StalledOscillationCheck().run(model) == []
