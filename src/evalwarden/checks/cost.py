"""Cost and efficiency checks.

COST-001: cost per success reporting (flags only when usage is missing).
COST-002: retry multiplier -- successes that cost many attempts each.
COST-003: wasted spend -- most of the budget burned on attempts that never passed.
COST-004: runaway attempt -- one attempt burning far more than the typical one.

"Agent cost per completed task is the only unit a finance review can act on."
All four checks are deterministic arithmetic over priced attempts (attempts
carrying tokens_in/tokens_out). Prices are estimates, configurable, and always
labeled as such in the report.
"""
from __future__ import annotations

from collections import defaultdict

from ..model import Attempt, Confidence, CostSummary, Finding, IntegrityModel, Severity, SourceLocation
from .base import Check, CheckMeta

RETRY_MULTIPLIER = 2.0  # avg tries per success above this is a finding
WASTED_SHARE = 0.5  # spend share on non-passing attempts above this is a finding
RUNAWAY_FACTOR = 10.0  # tokens_out beyond this multiple of the median is a finding
MIN_ATTEMPTS_RUNAWAY = 5  # attempts needed before naming a runaway


def _attempt_cost_usd(a: Attempt, price_in_per_1m: float, price_out_per_1m: float) -> float:
    return (a.tokens_in or 0) / 1_000_000 * price_in_per_1m + (a.tokens_out or 0) / 1_000_000 * price_out_per_1m


def _percentile(sorted_vals: list[int], q: float) -> int | None:
    """Nearest-rank percentile over a pre-sorted list. None when empty."""
    if not sorted_vals:
        return None
    rank = max(1, int(q * len(sorted_vals) + 0.5))
    return sorted_vals[min(rank, len(sorted_vals)) - 1]


def summarize_cost(
    attempts: list[Attempt],
    price_in_per_1m: float = 3.0,
    price_out_per_1m: float = 15.0,
    budget_per_task_usd: float | None = None,
) -> CostSummary | None:
    """Build the cost and efficiency summary. None when no attempt is priced."""
    priced = [a for a in attempts if a.tokens_in is not None and a.tokens_out is not None]
    if not priced:
        return None
    costs = [_attempt_cost_usd(a, price_in_per_1m, price_out_per_1m) for a in priced]
    estimated = sum(costs)
    successes = [a for a in priced if a.status == "pass"]

    # Per-task accounting: group attempts by task; a task is successful when any
    # of its attempts passed.
    by_task: dict[str, list[Attempt]] = defaultdict(list)
    for a in priced:
        by_task[a.task_id].append(a)
    task_cost = {
        task_id: sum(_attempt_cost_usd(a, price_in_per_1m, price_out_per_1m) for a in group)
        for task_id, group in by_task.items()
    }
    successful_tasks = [tid for tid, group in by_task.items() if any(a.status == "pass" for a in group)]
    tries_per_success = [sum(a.tries for a in by_task[tid]) for tid in successful_tasks]

    wasted = sum(c for a, c in zip(priced, costs) if a.status != "pass")

    outs = sorted(a.tokens_out or 0 for a in priced)
    max_out = max(outs)
    max_task = next(a.task_id for a in priced if (a.tokens_out or 0) == max_out)

    tasks_within_budget = None
    if budget_per_task_usd is not None:
        tasks_within_budget = sum(
            1
            for tid in successful_tasks
            if task_cost[tid] <= budget_per_task_usd
        )

    tool_calls = [a.tool_calls for a in successes]
    return CostSummary(
        attempts=len(priced),
        successes=len(successes),
        total_tokens_in=sum(a.tokens_in or 0 for a in priced),
        total_tokens_out=sum(a.tokens_out or 0 for a in priced),
        estimated_usd=round(estimated, 6),
        cost_per_success_usd=round(estimated / len(successes), 6) if successes else None,
        avg_tool_calls_per_success=round(sum(tool_calls) / len(tool_calls), 2) if tool_calls else None,
        price_in_per_1m=price_in_per_1m,
        price_out_per_1m=price_out_per_1m,
        avg_tries_per_success=round(sum(tries_per_success) / len(tries_per_success), 2)
        if tries_per_success
        else None,
        max_tries=max((a.tries for a in priced), default=None),
        wasted_usd=round(wasted, 6),
        wasted_share=round(wasted / estimated, 4) if estimated else 0.0,
        p50_tokens_out=_percentile(outs, 0.5),
        p90_tokens_out=_percentile(outs, 0.9),
        max_tokens_out=max_out,
        max_tokens_out_task=max_task,
        budget_per_task_usd=budget_per_task_usd,
        tasks_total=len(by_task) if budget_per_task_usd is not None else None,
        tasks_within_budget=tasks_within_budget,
    )


def _cost_location(model: IntegrityModel) -> SourceLocation:
    return SourceLocation(file=model.artifact_file("run.json"), excerpt="attempts[] tokens_in/tokens_out/tries")


class CostReportingCheck(Check):
    meta = CheckMeta(
        id="COST-001",
        title="Cost per success not reported",
        threat=(
            "A pass rate without cost hides the real trade-off: an agent that "
            "succeeds after 50 retries at 15x token burn is not the same system "
            "as one that succeeds first try. Without usage data, cost per "
            "completed task -- the unit a budget review can act on -- is unknown."
        ),
        remediation=(
            "Record tokens in/out, latency, tool calls, and retry counts per "
            "attempt in the run log, and report cost per successful task "
            "alongside the pass rate."
        ),
    )

    def __init__(
        self,
        price_in_per_1m: float = 3.0,
        price_out_per_1m: float = 15.0,
        budget_per_task_usd: float | None = None,
    ):
        self.price_in_per_1m = price_in_per_1m
        self.price_out_per_1m = price_out_per_1m
        self.budget_per_task_usd = budget_per_task_usd

    def with_options(self, **kwargs) -> "CostReportingCheck":
        return CostReportingCheck(
            price_in_per_1m=kwargs.get("price_in_per_1m", self.price_in_per_1m),
            price_out_per_1m=kwargs.get("price_out_per_1m", self.price_out_per_1m),
            budget_per_task_usd=kwargs.get("budget_per_task_usd", self.budget_per_task_usd),
        )

    def run(self, model: IntegrityModel) -> list[Finding]:
        summary = summarize_cost(
            model.attempts,
            self.price_in_per_1m,
            self.price_out_per_1m,
            self.budget_per_task_usd,
        )
        if summary is not None:
            model.cost_summary = summary
            return []
        if not model.attempts:
            # No runs at all: nothing to cost. The report shows this as a
            # coverage gap, not a finding.
            return []
        return [
            Finding(
                id=self.meta.id,
                title="Token usage not recorded: cost per success unknown",
                severity=Severity.MEDIUM,
                confidence=Confidence.HIGH,
                description=self.meta.threat,
                evidence=[
                    f"{len(model.attempts)} attempt(s) recorded, none with tokens_in/tokens_out.",
                    "Cost per completed task cannot be computed from this run log.",
                ],
                locations=[_cost_location(model)],
                remediation=self.meta.remediation,
            )
        ]


class RetryMultiplierCheck(Check):
    meta = CheckMeta(
        id="COST-002",
        title="Successes cost multiple attempts each",
        threat=(
            "A pass rate counts tasks; a budget counts attempts. When each "
            "success takes several tries on average, the reported pass rate "
            "understates the true cost per completed task -- and a best-of-n "
            "style run can look far stronger than the deployed system."
        ),
        remediation=(
            "Report tries per success alongside the pass rate, and cap retries "
            "per task so the measured cost matches the deployed configuration."
        ),
    )

    def run(self, model: IntegrityModel) -> list[Finding]:
        summary = model.cost_summary
        if summary is None or summary.avg_tries_per_success is None:
            return []
        if summary.avg_tries_per_success <= RETRY_MULTIPLIER:
            return []
        return [
            Finding(
                id=self.meta.id,
                title=f"Each success took {summary.avg_tries_per_success} attempts on average",
                severity=Severity.MEDIUM,
                confidence=Confidence.HIGH,
                description=self.meta.threat,
                evidence=[
                    f"{summary.successes} successful task(s) took "
                    f"{summary.avg_tries_per_success} tries per success on average "
                    f"(max {summary.max_tries} tries on one attempt).",
                    "The pass rate alone understates cost per completed task.",
                ],
                locations=[_cost_location(model)],
                remediation=self.meta.remediation,
            )
        ]


class WastedSpendCheck(Check):
    meta = CheckMeta(
        id="COST-003",
        title="Most spend produced no successful task",
        threat=(
            "When the majority of token spend goes to attempts that never pass, "
            "the headline cost per success is dominated by failure, not by the "
            "solver doing useful work. That pattern points at a broken "
            "solver, a broken task set, or both."
        ),
        remediation=(
            "Break down spend by attempt outcome before quoting cost per "
            "success; investigate tasks that burn budget without ever passing."
        ),
    )

    def run(self, model: IntegrityModel) -> list[Finding]:
        summary = model.cost_summary
        if summary is None:
            return []
        if summary.wasted_share <= WASTED_SHARE:
            return []
        return [
            Finding(
                id=self.meta.id,
                title=f"{summary.wasted_share:.0%} of spend burned on non-passing attempts",
                severity=Severity.MEDIUM,
                confidence=Confidence.HIGH,
                description=self.meta.threat,
                evidence=[
                    f"${summary.wasted_usd:.4f} of ${summary.estimated_usd:.4f} estimated "
                    f"({summary.wasted_share:.0%}) went to attempts that never passed.",
                    "Cost per success is dominated by failure, not by useful work.",
                ],
                locations=[_cost_location(model)],
                remediation=self.meta.remediation,
            )
        ]


class RunawayAttemptCheck(Check):
    meta = CheckMeta(
        id="COST-004",
        title="Runaway attempt burned far more than a typical one",
        threat=(
            "A single attempt burning an order of magnitude more output tokens "
            "than the median attempt is the signature of an unbounded loop: a "
            "solver stuck retrying, a tool call spiraling, or a run with no "
            "effective step or token cap. One such attempt can dominate the "
            "whole run's cost."
        ),
        remediation=(
            "Enforce per-attempt token and step caps in the harness, and "
            "investigate the flagged attempt's trace for the loop."
        ),
    )

    def run(self, model: IntegrityModel) -> list[Finding]:
        summary = model.cost_summary
        if summary is None:
            return []
        if summary.attempts < MIN_ATTEMPTS_RUNAWAY:
            return []
        typical = summary.p50_tokens_out
        if typical is None or typical == 0 or summary.max_tokens_out is None:
            return []
        if summary.max_tokens_out <= RUNAWAY_FACTOR * typical:
            return []
        return [
            Finding(
                id=self.meta.id,
                title=f"Attempt on {summary.max_tokens_out_task} burned "
                f"{summary.max_tokens_out:,} output tokens "
                f"({summary.max_tokens_out / typical:.0f}x the typical attempt)",
                severity=Severity.MEDIUM,
                confidence=Confidence.MEDIUM,
                description=self.meta.threat,
                evidence=[
                    f"Task {summary.max_tokens_out_task}: {summary.max_tokens_out:,} output "
                    f"tokens vs a typical {typical:,} across {summary.attempts} priced attempts.",
                    "A single unbounded attempt can dominate total run cost.",
                ],
                locations=[_cost_location(model)],
                remediation=self.meta.remediation,
            )
        ]


CHECKS: list[Check] = [
    CostReportingCheck(),  # COST-001
    RetryMultiplierCheck(),  # COST-002
    WastedSpendCheck(),  # COST-003
    RunawayAttemptCheck(),  # COST-004
]
