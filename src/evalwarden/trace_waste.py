"""Trajectory waste analysis: TRAJ findings, priced.

Runs the same deterministic definitions as the TRAJ lane
(:mod:`evalwarden.trajectory`, thresholds from
:mod:`evalwarden.checks.traj`) over an imported model and prices the
waste in tokens and dollars:

- **Loop waste**: for each looped (tool, args) pair, every occurrence
  beyond the first is waste -- the first call could have taught the
  agent something; the identical repeats could not.
- **Unused-output waste**: steps TRAJ-002 would flag (recorded data
  flow says no other step consumed their output), excluding the final
  step, under the same guards -- at least 5 spans, at least 3 unused,
  and silence when no consumption edges are recorded at all. Coding
  agent formats record no consumption edges, so this half stays silent
  on Claude Code / Codex imports by design, exactly as TRAJ-002 does.

Pricing is exact, never estimated: a wasted call is priced from the
per-step tokens its adapter recorded for it (Claude Code usage per
assistant record; Codex per-turn token deltas). When any wasted call in
a group carries no recorded per-step tokens, the group is reported with
``priced=False`` and its token/dollar fields are ``None`` -- a shared
or missing total is never split or averaged to fake a price. Prices
use the same defaults as the CLI (USD per 1M tokens in/out) and are
always estimates of dollar cost, labeled as such by callers.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .checks.traj import LOOP_CONSECUTIVE_AT, LOOP_TOTAL_AT, MIN_SPANS_UNUSED, UNUSED_AT
from .model import IntegrityModel
from .trajectory import canonical_call, find_loops, has_consumption_data, unused_outputs

DEFAULT_PRICE_IN_PER_1M = 3.0
DEFAULT_PRICE_OUT_PER_1M = 15.0


@dataclass
class LoopWaste:
    """One looped call pair in one attempt, with its repeats priced."""

    task_id: str
    key: str  # canonical_call output
    total: int  # occurrences in the trajectory
    wasted_calls: int  # occurrences beyond the first
    tokens_in: int | None  # summed over the wasted calls, when all are priced
    tokens_out: int | None
    usd: float | None
    priced: bool  # False when some wasted call has no recorded per-step tokens


@dataclass
class UnusedWaste:
    """The unused-output waste of one attempt, when TRAJ-002 can assess it."""

    task_id: str
    step_ids: list[str]
    tokens_in: int | None
    tokens_out: int | None
    usd: float | None
    priced: bool


@dataclass
class WasteReport:
    loops: list[LoopWaste] = field(default_factory=list)
    unused: list[UnusedWaste] = field(default_factory=list)
    # Totals cover priced groups only; unpriced waste is counted in
    # unpriced_wasted_calls so the gap is visible, never silently zero.
    total_wasted_tokens_in: int = 0
    total_wasted_tokens_out: int = 0
    total_wasted_usd: float = 0.0
    unpriced_wasted_calls: int = 0
    price_in_per_1m: float = DEFAULT_PRICE_IN_PER_1M
    price_out_per_1m: float = DEFAULT_PRICE_OUT_PER_1M

    @property
    def total_wasted_calls(self) -> int:
        return sum(w.wasted_calls for w in self.loops) + sum(
            len(u.step_ids) for u in self.unused
        )


def _price(steps, price_in: float, price_out: float):
    """(tokens_in, tokens_out, usd) over steps, or (None, None, None).

    Exactness rule: every step must carry recorded per-step tokens.
    """
    if any(s.tokens_in is None or s.tokens_out is None for s in steps):
        return None, None, None
    tokens_in = sum(s.tokens_in for s in steps)
    tokens_out = sum(s.tokens_out for s in steps)
    usd = tokens_in / 1_000_000 * price_in + tokens_out / 1_000_000 * price_out
    return tokens_in, tokens_out, round(usd, 6)


def analyze_waste(
    model: IntegrityModel,
    price_in_per_1m: float = DEFAULT_PRICE_IN_PER_1M,
    price_out_per_1m: float = DEFAULT_PRICE_OUT_PER_1M,
) -> WasteReport:
    """Price the loop and unused-output waste in an imported model."""
    report = WasteReport(price_in_per_1m=price_in_per_1m, price_out_per_1m=price_out_per_1m)
    for attempt in model.attempts:
        spans = attempt.spans
        if not spans:
            continue
        # Loops: same thresholds as TRAJ-001.
        calls = [canonical_call(s.tool, s.args) for s in spans]
        for hit in find_loops(calls, LOOP_TOTAL_AT, LOOP_CONSECUTIVE_AT):
            occurrences = [s for s, key in zip(spans, calls) if key == hit.key]
            wasted = occurrences[1:]  # the first call is not waste
            tokens_in, tokens_out, usd = _price(wasted, price_in_per_1m, price_out_per_1m)
            entry = LoopWaste(
                task_id=attempt.task_id,
                key=hit.key,
                total=hit.total,
                wasted_calls=len(wasted),
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                usd=usd,
                priced=usd is not None,
            )
            report.loops.append(entry)
            if entry.priced:
                report.total_wasted_tokens_in += tokens_in or 0
                report.total_wasted_tokens_out += tokens_out or 0
                report.total_wasted_usd += usd or 0.0
            else:
                report.unpriced_wasted_calls += entry.wasted_calls
        # Unused outputs: same guards as TRAJ-002 (silent without data flow).
        if len(spans) >= MIN_SPANS_UNUSED:
            consumes = [list(s.consumes) for s in spans]
            if has_consumption_data(consumes):
                unused_ids = unused_outputs([s.step_id for s in spans], consumes)
                if len(unused_ids) >= UNUSED_AT:
                    by_id = {s.step_id: s for s in spans}
                    steps = [by_id[sid] for sid in unused_ids]
                    tokens_in, tokens_out, usd = _price(
                        steps, price_in_per_1m, price_out_per_1m)
                    entry = UnusedWaste(
                        task_id=attempt.task_id,
                        step_ids=list(unused_ids),
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                        usd=usd,
                        priced=usd is not None,
                    )
                    report.unused.append(entry)
                    if entry.priced:
                        report.total_wasted_tokens_in += tokens_in or 0
                        report.total_wasted_tokens_out += tokens_out or 0
                        report.total_wasted_usd += usd or 0.0
                    else:
                        report.unpriced_wasted_calls += len(entry.step_ids)
    report.total_wasted_usd = round(report.total_wasted_usd, 6)
    return report
