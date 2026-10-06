"""Isomorphic-clone helpers: does a verifier's verdict survive a surface change?

Pure functions over plain task data. No model imports, so they are
independently testable. Used by the GRAD-003 check.

The spike covers one task family, "selection": a roster of named entries
with numeric scores, where the answer is the entries at or above a
threshold, listed in roster order. A clone is the same underlying problem
re-surfaced: entities renamed through a bijection, roster entries
reordered, and values (with the threshold) rescaled by a common positive
factor, which preserves the qualifying set exactly.

Re-scoring needs the verifier. A verifier is code, and this suite never
executes artifact code -- so what gets re-scored is the scoring rule the
artifact *declares* (grader.json's verifier.rule), restricted to a small
closed set implemented here:

- "exact_string": pass iff the output equals the reference rendered as
  ", ".join(reference items in stored order). Rewards the surface form.
- "set_match": pass iff the set of items parsed from the output equals
  the reference set. Order- and rendering-insensitive.

The cached output is a recorded artifact: the only transport applied to
it is the clone's renaming substitution. Its elements are never
re-ordered, re-sorted, or re-rendered -- that would fabricate a model
output the model never produced. A rule whose verdict depends on the
stored surface order therefore flips on clones; a rule checking the
underlying selection does not.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

FAMILY_SELECTION = "selection"

RULE_EXACT_STRING = "exact_string"
RULE_SET_MATCH = "set_match"
SUPPORTED_RULES = (RULE_EXACT_STRING, RULE_SET_MATCH)

# Alternate names clones rename entities to. Sampled per clone without
# replacement, excluding the task's own names.
_RENAME_POOL = [
    "Albin", "Beata", "Cyril", "Danuta", "Emil", "Frida", "Gustav", "Hana",
    "Ivor", "Jana", "Karel", "Lidia", "Milan", "Milos", "Otilia", "Radek",
    "Sanja", "Tibor", "Ursula", "Vaclav", "Wanda", "Xenia", "Yannick", "Zdenka",
]


@dataclass
class CloneableTask:
    """One selection task in cloneable form, with its cached output."""

    task_id: str
    entries: tuple[tuple[str, float], ...]  # (name, value) in roster order
    threshold: float
    reference_items: tuple[str, ...]  # qualifying names, roster order
    output: str  # cached model output, as recorded


@dataclass
class Clone:
    """An isomorphic re-surfacing of a task."""

    entries: tuple[tuple[str, float], ...]  # renamed, reordered, rescaled
    threshold: float
    rename: dict[str, str]  # original name -> clone name

    def reference_items(self) -> tuple[str, ...]:
        return tuple(derive_reference(self.entries, self.threshold))


def derive_reference(
    entries: tuple[tuple[str, float], ...] | list[tuple[str, float]],
    threshold: float,
) -> list[str]:
    """Qualifying entry names (value >= threshold), in roster order."""
    return [name for name, value in entries if value >= threshold]


def parse_output_items(output: str) -> list[str]:
    """The item names a comma-joined answer lists, in listed order."""
    return [part.strip() for part in output.split(",") if part.strip()]


def parse_cloneable(
    task_id: str,
    metadata: dict | None,
    target: str | list[str] | None,
    output: str | None,
) -> CloneableTask | None:
    """Build a CloneableTask from recorded artifact data, or None.

    A task is cloneable only when everything is recorded and consistent:
    family metadata with well-formed entries and threshold, a string
    target that matches the reference derived from the entries exactly
    (same items in the same roster order), and a cached output. Anything
    else is not evidence this check can use, so it is skipped -- never
    repaired or guessed at.
    """
    if not isinstance(metadata, dict) or metadata.get("clone_family") != FAMILY_SELECTION:
        return None
    if output is None or not isinstance(target, str):
        return None
    raw_entries = metadata.get("entries")
    threshold = metadata.get("threshold")
    if not isinstance(raw_entries, list) or not raw_entries:
        return None
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        return None
    entries: list[tuple[str, float]] = []
    seen: set[str] = set()
    for raw in raw_entries:
        if not isinstance(raw, dict):
            return None
        name, value = raw.get("name"), raw.get("value")
        if not isinstance(name, str) or not name or name in seen:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        seen.add(name)
        entries.append((name, float(value)))
    reference = derive_reference(entries, float(threshold))
    if not reference or parse_output_items(target) != reference:
        return None
    return CloneableTask(
        task_id=task_id,
        entries=tuple(entries),
        threshold=float(threshold),
        reference_items=tuple(reference),
        output=output,
    )


def generate_clone(task: CloneableTask, seed: int = 0) -> Clone:
    """The seeded isomorphic clone of a task.

    Deterministic: the RNG is seeded from the spike seed and the task id
    only, so every audit of the same artifact clones identically.
    Renaming is a bijection onto pool names the task does not use;
    reordering is a shuffle of the roster; rescaling multiplies every
    value and the threshold by one positive factor, preserving the
    qualifying set exactly.
    """
    rng = random.Random(f"evalwarden-grad003:{seed}:{task.task_id}")
    originals = [name for name, _ in task.entries]
    pool = [name for name in _RENAME_POOL if name not in originals]
    rename = dict(zip(originals, rng.sample(pool, len(originals))))
    factor = rng.choice([2, 3])
    cloned = [(rename[name], value * factor) for name, value in task.entries]
    rng.shuffle(cloned)
    return Clone(
        entries=tuple(cloned),
        threshold=task.threshold * factor,
        rename=rename,
    )


def transport_output(output: str, rename: dict[str, str]) -> str:
    """Carry a cached output through the clone's renaming.

    Item-wise substitution only: each listed name is replaced by its
    clone name, in the order the model listed them. The output is never
    re-ordered or re-rendered.
    """
    return ", ".join(rename.get(item, item) for item in parse_output_items(output))


def score(rule: str, output: str, reference_items: tuple[str, ...] | list[str]) -> bool:
    """One declared scoring rule applied to one output and reference."""
    if rule == RULE_EXACT_STRING:
        return output.strip() == ", ".join(reference_items)
    if rule == RULE_SET_MATCH:
        return set(parse_output_items(output)) == set(reference_items)
    raise ValueError(f"unsupported verifier rule: {rule!r}")


@dataclass
class CloneGap:
    n_items: int
    original_passes: int
    clone_passes: int
    gap: float  # original pass rate minus clone pass rate
    flips: int  # items passing on the original, failing on the clone
    flipped_ids: list[str] = field(default_factory=list)

    @property
    def original_rate(self) -> float:
        return self.original_passes / self.n_items if self.n_items else 0.0

    @property
    def clone_rate(self) -> float:
        return self.clone_passes / self.n_items if self.n_items else 0.0


def clone_gap(
    tasks: list[CloneableTask], rule: str, seed: int = 0
) -> CloneGap | None:
    """Re-score cached outputs on originals and clones; measure the gap.

    Returns None when the rule is not one this module implements (the
    artifact declared something the audit cannot re-score) or when there
    are no cloneable tasks. Both verdicts are computed by re-scoring --
    recorded attempt statuses are never trusted as the original verdict.
    """
    if rule not in SUPPORTED_RULES or not tasks:
        return None
    original_passes = 0
    clone_passes = 0
    flipped: list[str] = []
    for task in tasks:
        original_ok = score(rule, task.output, task.reference_items)
        clone = generate_clone(task, seed=seed)
        transported = transport_output(task.output, clone.rename)
        clone_ok = score(rule, transported, clone.reference_items())
        original_passes += original_ok
        clone_passes += clone_ok
        if original_ok and not clone_ok:
            flipped.append(task.task_id)
    n = len(tasks)
    return CloneGap(
        n_items=n,
        original_passes=original_passes,
        clone_passes=clone_passes,
        gap=(original_passes - clone_passes) / n,
        flips=len(flipped),
        flipped_ids=flipped,
    )
