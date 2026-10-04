"""Panel coverage metrics: judge x sample coverage of failure modes.

Pure functions over per-judge verdicts, reference labels, and per-item
failure-mode tags. No model imports, so they are independently testable.
Used by JUDGE-009 (panel blind spots) and DATA-003 (elicitation coverage).

JUDGE-008 (ensemble.py) asks whether judges duplicate each other
(judge x judge). This module asks the orthogonal question: does the panel,
between its members, actually catch each failure mode the eval claims to
test (judge x sample)? A panel can have zero redundant judges and still be
blind to an entire failure mode if every judge fails on it the same way.

Exact definitions:
- A judge "catches" a failure mode when its verdicts on that mode's labeled
  items agree with the reference labels at a rate at or above the catch bar.
- A mode is a panel blind spot when at least `min_judges` judges have a
  measurable rate on it (each with at least `min_judge_items` labeled
  verdicts), the mode has at least `min_mode_items` labeled items overall,
  and every measurable judge falls below the catch bar.
- Elicitation coverage is the count side: how many items carry each mode
  tag at all, and which declared taxonomy modes have none.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class JudgeCatch:
    """One judge's catch rate on one failure mode."""

    judge_id: str
    n_labeled: int  # labeled items of the mode this judge rendered a verdict on
    n_caught: int  # of those, verdicts matching the reference label

    @property
    def rate(self) -> float:
        """Fraction of the judge's verdicts on the mode matching the label."""
        return self.n_caught / self.n_labeled if self.n_labeled else 0.0


@dataclass
class ModeCatch:
    """Per-mode catch table row: labeled item count and per-judge rates."""

    mode: str
    n_labeled_items: int  # distinct labeled items carrying this mode tag
    per_judge: list[JudgeCatch] = field(default_factory=list)

    @property
    def best_rate(self) -> float:
        """Highest catch rate over all judges (measurable or not)."""
        return max((jc.rate for jc in self.per_judge), default=0.0)

    def measurable(self, min_judge_items: int) -> list[JudgeCatch]:
        """Judges with enough labeled verdicts on this mode to have a rate."""
        return [jc for jc in self.per_judge if jc.n_labeled >= min_judge_items]


def mode_catch_table(
    verdicts: dict[str, dict[str, str]],
    labels: dict[str, str],
    mode_of: dict[str, str],
) -> dict[str, ModeCatch]:
    """Build the judge x mode catch table.

    verdicts maps judge_id -> task_id -> verdict; labels maps task_id ->
    reference label; mode_of maps task_id -> failure mode. Only items that
    carry both a mode tag and a reference label enter the table. A judge
    appears in a mode's row only if it rendered at least one verdict on that
    mode's labeled items.
    """
    labeled_by_mode: dict[str, set[str]] = {}
    for task_id, mode in mode_of.items():
        if task_id in labels:
            labeled_by_mode.setdefault(mode, set()).add(task_id)
    table: dict[str, ModeCatch] = {}
    for mode, items in labeled_by_mode.items():
        per_judge: list[JudgeCatch] = []
        for judge_id in sorted(verdicts):
            own = verdicts[judge_id]
            judged = [t for t in items if t in own]
            if not judged:
                continue
            caught = sum(1 for t in judged if own[t] == labels[t])
            per_judge.append(JudgeCatch(judge_id, len(judged), caught))
        table[mode] = ModeCatch(
            mode=mode, n_labeled_items=len(items), per_judge=per_judge
        )
    return table


def blind_spots(
    table: dict[str, ModeCatch],
    catch_bar: float = 0.70,
    min_mode_items: int = 20,
    min_judge_items: int = 10,
    min_judges: int = 2,
) -> list[ModeCatch]:
    """Modes no measurable judge catches, sorted by mode name.

    A mode qualifies when it has at least `min_mode_items` labeled items, at
    least `min_judges` judges with at least `min_judge_items` labeled
    verdicts on it, and every measurable judge's catch rate is below
    `catch_bar`. Modes failing any gate are skipped silently: thin evidence
    is DATA-003's claim, not a blind-spot claim.
    """
    out: list[ModeCatch] = []
    for mode in sorted(table):
        mc = table[mode]
        if mc.n_labeled_items < min_mode_items:
            continue
        measurable = mc.measurable(min_judge_items)
        if len(measurable) < min_judges:
            continue
        if all(jc.rate < catch_bar for jc in measurable):
            out.append(mc)
    return out


def mode_item_counts(mode_of: dict[str, str]) -> dict[str, int]:
    """Items per failure mode, over every tagged item (labeled or not)."""
    counts: dict[str, int] = {}
    for mode in mode_of.values():
        counts[mode] = counts.get(mode, 0) + 1
    return counts


def thin_modes(
    counts: dict[str, int],
    taxonomy: list[str],
    min_items: int = 30,
) -> tuple[list[tuple[str, int]], list[str]]:
    """(thin, absent): modes under `min_items`, and declared modes with none.

    Thin lists (mode, item count) sorted by count then name, so the thinnest
    coverage leads the evidence. Absent lists declared taxonomy modes with
    zero items, sorted by name. Without a taxonomy the absent list is empty:
    undeclared modes cannot be missed.
    """
    thin = sorted(
        ((mode, count) for mode, count in counts.items() if count < min_items),
        key=lambda mc: (mc[1], mc[0]),
    )
    absent = sorted(mode for mode in taxonomy if counts.get(mode, 0) == 0)
    return thin, absent
