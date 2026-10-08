"""Deterministic trajectory-integrity metrics.

Pure functions over per-step call data. Checks call these; the math never
touches the integrity model, so it stays independently testable.

Two waste shapes, both deterministic:
- Loops (TRAJ-001): the same (tool, args) pair repeated. An agent that calls
  the identical tool with identical arguments again and again is stuck or
  retrying blindly -- either way the repeats are pure budget burn.
- Unused outputs (TRAJ-002): tool calls whose results no downstream step
  consumed. Work whose product went nowhere is waste by definition.

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
