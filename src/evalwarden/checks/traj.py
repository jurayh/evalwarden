"""Trajectory integrity checks: is the agent's tool use wasteful?

TRAJ-001 flags exact loops: the same (tool, args) pair repeated beyond a
threshold -- a stuck or blindly-retrying agent burning budget on identical
calls.
TRAJ-002 flags unused outputs: tool calls whose results no downstream step
consumed, per the recorded data-flow graph -- work whose product went
nowhere.

Precision-first: both checks stay silent when the trajectory data cannot
support the claim. TRAJ-001 needs spans at all; TRAJ-002 additionally needs
recorded consumption edges -- without them, "unused" is indistinguishable
from "the adapter did not record data flow", so the check says nothing.
"""
from __future__ import annotations

from ..model import Attempt, Confidence, Finding, IntegrityModel, Severity, SourceLocation
from ..trajectory import canonical_call, find_loops, has_consumption_data, unused_outputs
from .base import Check, CheckMeta

LOOP_TOTAL_AT = 4  # same (tool, args) >= 4x in one trajectory is a loop
LOOP_CONSECUTIVE_AT = 3  # >= 3x back-to-back is the classic spin
MIN_SPANS_UNUSED = 5  # shorter trajectories cannot show meaningful waste
UNUSED_AT = 3  # >= 3 outputs consumed by nothing downstream is waste
MAX_EVIDENCE = 5  # loops / unused steps shown per finding


def _traj_loc(model: IntegrityModel, excerpt: str) -> SourceLocation:
    return SourceLocation(file=model.artifact_file("trajectories.jsonl"), excerpt=excerpt)


def _short(key: str, limit: int = 80) -> str:
    return key if len(key) <= limit else key[: limit - 1] + "..."


class LoopCheck(Check):
    meta = CheckMeta(
        id="TRAJ-001",
        title="Trajectory repeats identical tool calls",
        threat=(
            "The same tool called with the same arguments, over and over, is "
            "an agent stuck in a loop or retrying blindly. Every repeat is "
            "budget spent learning nothing new."
        ),
        remediation=(
            "Add loop guards to the agent harness: cap identical retries, "
            "detect repeated (tool, args) pairs at runtime and break out, "
            "or surface the repeated failure to the operator instead of "
            "spinning."
        ),
    )

    def run(self, model: IntegrityModel) -> list[Finding]:
        findings: list[Finding] = []
        for attempt in model.attempts:
            if not attempt.spans:
                continue
            calls = [canonical_call(s.tool, s.args) for s in attempt.spans]
            hits = find_loops(calls, LOOP_TOTAL_AT, LOOP_CONSECUTIVE_AT)
            if not hits:
                continue
            # The spinniest hit leads. Severity leans on the longest
            # back-to-back run: consecutive identical calls are the
            # stuck-agent signature, while the same total scattered
            # across a long session can be deliberate re-verification
            # (playtest screenshots in the Optimal Misbehavior corpus
            # repeated 14x with a longest run of 2) -- still reported,
            # one severity down.
            ordered = sorted(
                hits, key=lambda h: (-h.max_consecutive, -h.total))
            severity = (
                Severity.HIGH
                if ordered[0].max_consecutive >= LOOP_CONSECUTIVE_AT
                else Severity.MEDIUM
            )
            findings.append(
                Finding(
                    id=self.meta.id,
                    title=(
                        f"Trajectory loops: {_short(ordered[0].key)} repeated "
                        f"{ordered[0].total}x in task {attempt.task_id}"
                    ),
                    severity=severity,
                    confidence=Confidence.HIGH,
                    description=self.meta.threat,
                    evidence=[
                        f"loop: {_short(h.key)} x{h.total} "
                        f"(longest run x{h.max_consecutive})"
                        for h in ordered[:MAX_EVIDENCE]
                    ],
                    locations=[
                        _traj_loc(
                            model,
                            f"attempt {attempt.task_id}: {len(hits)} looped call "
                            f"pair(s) over {len(attempt.spans)} spans",
                        )
                    ],
                    remediation=self.meta.remediation,
                )
            )
        return findings


class UnusedOutputCheck(Check):
    meta = CheckMeta(
        id="TRAJ-002",
        title="Tool outputs consumed by nothing downstream",
        threat=(
            "Tool calls whose results no later step used are work whose "
            "product went nowhere: the trajectory paid for computation and "
            "tokens, then ignored the answer."
        ),
        remediation=(
            "Tighten the agent's tool discipline: only call tools whose "
            "results the plan needs, and record data flow (which outputs "
            "each step consumes) so waste is visible instead of silent."
        ),
    )

    def run(self, model: IntegrityModel) -> list[Finding]:
        findings: list[Finding] = []
        for attempt in model.attempts:
            spans = attempt.spans
            if len(spans) < MIN_SPANS_UNUSED:
                continue
            consumes = [list(s.consumes) for s in spans]
            if not has_consumption_data(consumes):
                continue  # cannot tell unused from unrecorded: stay silent
            unused = unused_outputs([s.step_id for s in spans], consumes)
            if len(unused) < UNUSED_AT:
                continue
            by_id = {s.step_id: s for s in spans}
            findings.append(
                Finding(
                    id=self.meta.id,
                    title=(
                        f"{len(unused)} tool outputs went unused in task "
                        f"{attempt.task_id}"
                    ),
                    severity=Severity.HIGH,
                    confidence=Confidence.HIGH,
                    description=self.meta.threat,
                    evidence=[
                        f"unused: step {sid} ({by_id[sid].tool}) output "
                        f"consumed by nothing downstream"
                        for sid in unused[:MAX_EVIDENCE]
                    ],
                    locations=[
                        _traj_loc(
                            model,
                            f"attempt {attempt.task_id}: {len(unused)} unused "
                            f"outputs over {len(spans)} spans (recorded data flow)",
                        )
                    ],
                    remediation=self.meta.remediation,
                )
            )
        return findings


CHECKS: list[Check] = [
    LoopCheck(),  # TRAJ-001
    UnusedOutputCheck(),  # TRAJ-002
]
