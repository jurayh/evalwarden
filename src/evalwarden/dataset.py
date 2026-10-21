"""Dataset audit helpers: saturation and redundancy metrics.

Pure functions over per-item correctness and prompt texts. No model imports,
so they are independently testable. Used by the DATA-lane checks.

Saturation answers: do models still differ on these items, or is headroom
gone? Redundancy answers: how much of the set is near-duplicate filler?
"""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass, field


def normalized_tokens(text: str) -> list[str]:
    """The DATA lane's token normalization: lowercase alphanumeric runs."""
    return re.findall(r"[a-z0-9]+", text.lower())


def text_digest(text: str) -> str:
    """Stable digest of a text's normalized form (exact-overlap identity)."""
    return hashlib.sha256(" ".join(normalized_tokens(text)).encode()).hexdigest()


def shingle_containment(a: set[str], b: set[str]) -> float:
    """Share of the smaller shingle set that appears in the larger.

    Containment catches a short item embedded in a longer document, where
    Jaccard stays low because the union is dominated by the long side.
    """
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


@dataclass
class SaturationStats:
    n_items: int
    n_models: int
    models: list[str]
    # Fraction of items every model got right: these discriminate nothing.
    dead_fraction: float
    # Fraction of items no model got right.
    all_wrong_fraction: float
    # Fraction of items the models disagree on: these carry the signal.
    disputed_fraction: float
    # 1 - best model accuracy: room left at the ceiling.
    headroom: float
    per_model_accuracy: dict[str, float] = field(default_factory=dict)


def saturation_stats(scores: dict[str, dict[str, bool]]) -> SaturationStats | None:
    """Compute saturation over per-item correctness.

    scores maps model_id -> task_id -> correct. Only items attempted by every
    model count (the common set); returns None when fewer than two models
    are present, since saturation is about models no longer differing.
    """
    models = sorted(scores)
    if len(models) < 2:
        return None
    common = set.intersection(*(set(task_ids) for task_ids in scores.values()))
    n = len(common)
    if n == 0:
        return None
    dead = sum(1 for t in common if all(scores[m][t] for m in models))
    all_wrong = sum(1 for t in common if not any(scores[m][t] for m in models))
    disputed = n - dead - all_wrong
    accuracy = {m: sum(1 for t in common if scores[m][t]) / n for m in models}
    return SaturationStats(
        n_items=n,
        n_models=len(models),
        models=models,
        dead_fraction=dead / n,
        all_wrong_fraction=all_wrong / n,
        disputed_fraction=disputed / n,
        headroom=1.0 - max(accuracy.values()),
        per_model_accuracy=accuracy,
    )


def _shingles(text: str, k: int = 3) -> set[str]:
    toks = normalized_tokens(text)
    if not toks:
        return set()
    if len(toks) < k:
        return {" ".join(toks)}
    return {" ".join(toks[i : i + k]) for i in range(len(toks) - k + 1)}


@dataclass
class RedundancyStats:
    n_items: int
    near_duplicate_pairs: int
    # Fraction of items having at least one near-duplicate.
    duplicate_item_fraction: float
    # (id_a, id_b, similarity) for the most similar pairs, most similar first.
    examples: list[tuple[str, str, float]] = field(default_factory=list)


def near_duplicate_pairs(
    prompts: dict[str, str],
    threshold: float = 0.8,
    k: int = 3,
    max_examples: int = 5,
) -> list[tuple[str, str, float]]:
    """Find prompt pairs with shingle-Jaccard similarity at or above threshold.

    Uses an inverted index over shingles so only pairs sharing a shingle pay
    the exact Jaccard computation. Returns (id_a, id_b, similarity) sorted by
    similarity descending, id_a < id_b.
    """
    ids = sorted(prompts)
    shingled = {i: _shingles(prompts[i], k) for i in ids}
    index: dict[str, list[str]] = defaultdict(list)
    for i, shingles in shingled.items():
        for shingle in shingles:
            index[shingle].append(i)
    candidates: set[tuple[str, str]] = set()
    for docs in index.values():
        if 1 < len(docs) < 500:
            for x in range(len(docs)):
                for y in range(x + 1, len(docs)):
                    a, b = docs[x], docs[y]
                    candidates.add((a, b) if a < b else (b, a))
    hits: list[tuple[str, str, float]] = []
    for a, b in candidates:
        sa, sb = shingled[a], shingled[b]
        if not sa or not sb:
            continue
        similarity = len(sa & sb) / len(sa | sb)
        if similarity >= threshold:
            hits.append((a, b, similarity))
    hits.sort(key=lambda h: (-h[2], h[0], h[1]))
    return hits[:max_examples] if max_examples else hits


def redundancy_stats(
    prompts: dict[str, str],
    threshold: float = 0.8,
    max_examples: int = 5,
) -> RedundancyStats:
    """Summarize near-duplication across a prompt set."""
    ids = [i for i in sorted(prompts) if prompts[i].strip()]
    hits = near_duplicate_pairs(
        {i: prompts[i] for i in ids}, threshold=threshold, max_examples=0
    )
    dup_ids = {a for a, _, _ in hits} | {b for _, b, _ in hits}
    return RedundancyStats(
        n_items=len(ids),
        near_duplicate_pairs=len(hits),
        duplicate_item_fraction=len(dup_ids) / len(ids) if ids else 0.0,
        examples=hits[:max_examples],
    )
