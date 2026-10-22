"""Deterministic trajectory-integrity metrics.

Pure functions over per-step call data. Checks call these; the math never
touches the integrity model, so it stays independently testable.

Two waste shapes, both deterministic:
- Loops (TRAJ-001): the same (tool, args) pair repeated. An agent that calls
  the identical tool with identical arguments again and again is stuck or
  retrying blindly -- either way the repeats are pure budget burn.
- Unused outputs (TRAJ-002): tool calls whose results no downstream step
  consumed. Work whose product went nowhere is waste by definition.
- Stalled phase patterns (TRAJ-003): edit -> test cycles whose test phases
  all errored, repeated without an intervening explore/plan/review phase
  and without a later successful test phase. The individual calls keep
  changing -- no exact loop -- while the work pattern goes nowhere. This
  shape needs per-step outcomes (TrajectoryStep.outcome): a phase pattern
  alone cannot separate a stall from honest debugging, because debugging
  walks the same edit -> test alternation with errored verifications on
  the way to a pass.

Exact definitions:
- canonical_call(tool, args): "tool({json, sorted keys})". Two calls are
  "the same" iff their canonical keys are byte-identical -- no fuzzy
  matching, by design. A single changed argument is a different call,
  with one exception: prose caption fields (see PROSE_ARG_KEYS) are
  dropped before comparison. Harnesses attach a human-written caption
  to shell calls ("Run the tests after the fix"); the caption describes
  the call, it is not part of what the call does, and real traces show
  identical commands captioned differently escaping detection while
  caption-free calls dominate the hits (Optimal Misbehavior corpus,
  Oct 2026).
- A loop hit is a canonical key occurring >= total_at times in one
  trajectory, or >= consecutive_at times back-to-back. The consecutive rule
  catches the classic spin (read, read, read); the total rule catches the
  diffuse retry loop scattered through a long run.
- unused_outputs(step_ids, consumes): ids never named as a consumed target
  by any *other* step, excluding the final step. The final tool output feeds
  the agent's answer, which is not a tool call and so cannot appear in the
  consumption graph; flagging it would manufacture waste. Self-edges (a step
  naming its own id) do not count as consumption. "Consumed" is per the
  recorded data flow only -- steps the adapter could not attribute are
  simply absent from the graph, which is why TRAJ-002 stays silent when no
  consumption edges are recorded at all.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import TrajectoryStep
    from .trace_phases import PhaseSegment


# Argument keys that carry prose about a call rather than call
# semantics. Claude Code's Bash/PowerShell inputs include a
# human-readable "description" of the command; the same command with a
# reworded caption is the same call.
PROSE_ARG_KEYS = {"description"}


def canonical_call(tool: str, args: dict) -> str:
    """One canonical string for a (tool, args) pair.

    Argument order does not matter; argument values do, except prose
    caption fields (PROSE_ARG_KEYS), which are dropped. Non-JSON values
    fall back to str() so adapters never crash the linter on odd types.
    """
    semantic = {k: v for k, v in args.items() if k not in PROSE_ARG_KEYS}
    return f"{tool}({json.dumps(semantic, sort_keys=True, default=str)})"


@dataclass
class Loop:
    """One repeated (tool, args) pair inside a trajectory."""

    key: str  # canonical_call output: "tool({args})"
    total: int  # occurrences in the whole trajectory
    max_consecutive: int  # longest back-to-back run


def find_loops(
    calls: list[str], total_at: int = 4, consecutive_at: int = 3
) -> list[Loop]:
    """Loop hits over an ordered list of canonical call keys.

    A key is a hit when it occurs >= total_at times overall, or
    >= consecutive_at times back-to-back. Hits are returned in order of
    first appearance.
    """
    totals: dict[str, int] = {}
    best_run: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    run_key: str | None = None
    run_len = 0
    for i, key in enumerate(calls):
        totals[key] = totals.get(key, 0) + 1
        if key not in first_seen:
            first_seen[key] = i
        if key == run_key:
            run_len += 1
        else:
            run_key, run_len = key, 1
        best_run[key] = max(best_run.get(key, 0), run_len)
    return [
        Loop(key=key, total=totals[key], max_consecutive=best_run[key])
        for key in sorted(first_seen, key=first_seen.__getitem__)
        if totals[key] >= total_at or best_run[key] >= consecutive_at
    ]


def has_consumption_data(consumes: list[list[str]]) -> bool:
    """True when the trajectory records any data-flow edge at all.

    Without a single recorded edge, "unused" is indistinguishable from
    "the adapter did not record data flow" -- the check must stay silent.
    """
    return any(len(c) > 0 for c in consumes)


def unused_outputs(step_ids: list[str], consumes: list[list[str]]) -> list[str]:
    """Step ids whose output no other step consumed, excluding the final step.

    `consumes[i]` names the step ids consumed BY step i. A step counts as
    used when some other step names it; the final step is carved out
    because its output feeds the agent's answer, not another tool call.
    """
    consumed: set[str] = set()
    for i, targets in enumerate(consumes):
        own = step_ids[i] if i < len(step_ids) else None
        for t in targets:
            if t != own:  # self-edges are not consumption
                consumed.add(t)
    final = step_ids[-1] if step_ids else None
    return [sid for sid in step_ids if sid != final and sid not in consumed]


# ------------------------------------------------------- phase patterns
#
# TRAJ-003 works one level up from calls: over the phase segmentation
# (trace_phases.segment_phases) joined with per-step outcomes. The join
# rule is the precision crux, so it is stated exactly:
#
# - A phase is *judged* when at least one of its spans carries an
#   outcome. Outcome-less phases (the adapter recorded no per-call
#   outcome) cannot be called errored or successful -- they are simply
#   not evidence either way.
# - A judged phase is *successful* when at least one span completed
#   ("ok"). A mixed phase -- one verification call errored, another
#   completed -- shows a partial signal (one runner crashed, another
#   ran), and reading the crash as the phase's verdict would manufacture
#   stalls out of ordinary multi-runner work.
# - A judged phase is *errored* when every outcome-bearing span in it
#   errored: at least one "error" and no "ok" at all.
#
# "Errored" still means only that the harness reported the calls as
# errored (TrajectoryStep.outcome's semantics rule); the helpers never
# look at output text.


@dataclass
class PhaseOutcome:
    """One phase segment joined with the outcomes of its spans."""

    label: str
    judged: bool  # at least one span carries an outcome
    successful: bool  # judged, and at least one span completed ("ok")
    errored: bool  # judged, and every outcome-bearing span errored


def phase_outcomes(
    segments: "list[PhaseSegment]", spans: "list[TrajectoryStep]"
) -> list[PhaseOutcome]:
    """Join phase segments to per-span outcomes, one record per segment."""
    out: list[PhaseOutcome] = []
    for seg in segments:
        outcomes = [
            s.outcome for s in spans[seg.start : seg.end] if s.outcome is not None
        ]
        out.append(
            PhaseOutcome(
                label=seg.label,
                judged=bool(outcomes),
                successful="ok" in outcomes,
                errored=bool(outcomes) and all(o == "error" for o in outcomes),
            )
        )
    return out


@dataclass
class StalledOscillation:
    """A run of edit -> test cycles whose test phases all errored."""

    cycles: int  # consecutive errored edit -> test cycles in the run
    first_segment: int  # segment index of the run's first edit phase
    last_segment: int  # segment index one past the run's last test phase
    test_segments: int  # test phases in the whole trajectory
    judged_test_segments: int  # test phases carrying at least one outcome


def find_stalled_oscillation(
    segments: "list[PhaseSegment]",
    outcomes: list[PhaseOutcome],
    min_cycles: int = 3,
    min_test_coverage: float = 0.5,
) -> StalledOscillation | None:
    """Find an unresolved run of errored edit -> test cycles, if any.

    A *cycle* is an edit segment immediately followed by an errored test
    segment. Cycles chain only while the alternation is unbroken -- any
    other phase (explore, plan, review, other) between two cycles ends
    the run, because an intervening exploration or review is exactly how
    honest debugging separates itself from a stall. A run counts only
    when no test phase from the run's first test onward is successful:
    a stall that later resolves is debugging that worked. Returns the
    longest qualifying run.

    Coverage gate: the motif judges only trajectories whose test phases
    mostly carry outcomes -- judged test phases must be at least
    ``min_test_coverage`` of all test phases. Below that, the unseen
    phases could hide the resolution, and "stalled" would be a claim
    about missing data rather than about the trajectory.
    """
    test_total = sum(1 for o in outcomes if o.label == "test")
    if test_total == 0:
        return None
    judged = sum(1 for o in outcomes if o.label == "test" and o.judged)
    if judged / test_total < min_test_coverage:
        return None

    best: StalledOscillation | None = None
    n = len(segments)
    i = 0
    while i + 1 < n:
        if not (
            segments[i].label == "edit"
            and segments[i + 1].label == "test"
            and outcomes[i + 1].errored
        ):
            i += 1
            continue
        start = i
        cycles = 0
        k = i
        while (
            k + 1 < n
            and segments[k].label == "edit"
            and segments[k + 1].label == "test"
            and outcomes[k + 1].errored
        ):
            cycles += 1
            k += 2
        resolved = any(
            o.label == "test" and o.successful for o in outcomes[start + 1 :]
        )
        if cycles >= min_cycles and not resolved:
            if best is None or cycles > best.cycles:
                best = StalledOscillation(
                    cycles=cycles,
                    first_segment=start,
                    last_segment=k,
                    test_segments=test_total,
                    judged_test_segments=judged,
                )
        i = k
    return best
