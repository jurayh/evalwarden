"""Train/eval leakage: overlap between a training file and an eval set.

DATA-004's evidence layer. The eval side comes from the normalized
model (task prompts and targets, plus per-item outcomes); the training
side comes from a file the auditor supplies alongside the artifact --
fine-tune corpora are not eval artifacts, so they never enter the
integrity model. This module loads that file strictly and computes the
overlap deterministically and lexically: no embeddings, no model calls.

Two tiers, calibrated on seeded fixtures (planted token-level edits vs
same-topic controls sharing instruction boilerplate; see the DATA-004
tests):

- exact: the normalized texts are identical (digest match);
- near-duplicate: shingle-Jaccard >= NEAR_JACCARD or shingle
  containment >= NEAR_CONTAINMENT against the best-matching training
  row, computed over *distinctive* shingles: shingles appearing in
  more than half the eval items are the eval's own scaffolding
  (shared instructions, item templates), not any item's identity, and
  are excluded from both sides. Without that exclusion, templated
  eval items cross-match each other at Jaccard 0.45 / containment
  0.62 on scaffolding alone; with it, same-template cross-matches top
  out at 0.28 / 0.44. Calibrated edit classes (adjacent swaps,
  sentence reorders, substitutions, filler edits) land at Jaccard
  >= 0.43 / containment >= 0.63, topical controls at <= 0.02 / 0.04
  -- the thresholds sit in that gap on both metrics. Combined
  swap-plus-substitution edits at the extreme, and heavy multi-edit
  paraphrase generally, fall below both thresholds: lexical detection
  cannot see them, and the check says so rather than guessing.

Precision over recall throughout: every claimed leak names the matched
pair (training row id, eval item id, both scores) so a human can verify
it, and topical similarity alone never fires.
"""
from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from pathlib import Path

from .dataset import _shingles, shingle_containment, text_digest

NEAR_JACCARD = 0.45  # shingle-Jaccard at or above this is a near-duplicate
NEAR_CONTAINMENT = 0.60  # shingle containment at or above this is one too
MAX_POSTING = 500  # shingles in more training rows than this are boilerplate
SCAFFOLD_DF = 0.5  # shingles in more than this share of eval items are scaffolding
SCAFFOLD_MIN_ITEMS = 8  # below this many eval items, document frequency means nothing


class TrainingFileError(ValueError):
    """A training file is malformed; the message names the exact place."""


@dataclass
class TrainingRecord:
    id: str
    prompt: str
    target: str | None = None


def item_text(prompt: str, target: str | list[str] | None) -> str:
    """The comparable text of an item: prompt plus target where present.

    Both sides of the comparison build their text this way, so a target
    present on only one side changes the text on that side only -- an
    asymmetric target cannot manufacture an exact match.
    """
    if target is None:
        return prompt
    target_text = target if isinstance(target, str) else " ".join(target)
    return f"{prompt}\n{target_text}" if target_text else prompt


def _record_id(raw: object, default: str, where: str) -> str:
    if raw is None:
        return default
    if not isinstance(raw, str) or not raw.strip():
        raise TrainingFileError(f"{where}: 'id' must be a non-empty string")
    return raw


def _record_target(raw: object, where: str) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list) and all(isinstance(item, str) for item in raw):
        return " ".join(raw)
    raise TrainingFileError(f"{where}: 'target' must be a string or an array of strings")


def _check_ids(records: list[TrainingRecord], source: str) -> list[TrainingRecord]:
    seen: set[str] = set()
    for record in records:
        if record.id in seen:
            raise TrainingFileError(f"{source}: duplicate training row id {record.id!r}")
        seen.add(record.id)
    if not records:
        raise TrainingFileError(f"{source}: no training records found")
    return records


def load_training_jsonl(text: str, source: str) -> list[TrainingRecord]:
    """Parse a training JSONL file: one object per line, prompt required."""
    records: list[TrainingRecord] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        where = f"{source} line {line_number}"
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TrainingFileError(
                f"{where}: invalid JSON at column {exc.colno}: {exc.msg}"
            ) from exc
        if not isinstance(record, dict):
            raise TrainingFileError(f"{where}: record must be a JSON object")
        prompt = record.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise TrainingFileError(f"{where}: 'prompt' must be a non-empty string")
        records.append(
            TrainingRecord(
                id=_record_id(record.get("id"), f"train-{line_number}", where),
                prompt=prompt,
                target=_record_target(record.get("target"), where),
            )
        )
    return _check_ids(records, source)


def load_training_csv(text: str, source: str) -> list[TrainingRecord]:
    """Parse a training CSV file: a prompt column is required."""
    if text.startswith("\ufeff"):
        text = text[1:]
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise TrainingFileError(f"{source} line 1: CSV header row is required")
    headers = [name.strip() if isinstance(name, str) else "" for name in reader.fieldnames]
    if "prompt" not in headers:
        raise TrainingFileError(f"{source} line 1: a 'prompt' column is required")
    reader.fieldnames = headers
    records: list[TrainingRecord] = []
    for raw_row in reader:
        row_number = reader.line_num
        if None in raw_row:
            raise TrainingFileError(f"{source} row {row_number}: too many fields for the header")
        row = {
            key: (value.strip() if isinstance(value, str) else "")
            for key, value in raw_row.items()
        }
        if not any(row.values()):
            continue
        where = f"{source} row {row_number}"
        prompt = row.get("prompt", "")
        if not prompt:
            raise TrainingFileError(f"{where}: 'prompt' must be a non-empty string")
        records.append(
            TrainingRecord(
                id=_record_id(row.get("id") or None, f"train-{row_number}", where),
                prompt=prompt,
                target=row.get("target") or None,
            )
        )
    return _check_ids(records, source)


def load_training_file(path: str | Path) -> list[TrainingRecord]:
    """Load a training file (.jsonl or .csv) with strict validation."""
    target = Path(path)
    if not target.is_file():
        raise TrainingFileError(f"no such training file: {target}")
    try:
        text = target.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise TrainingFileError(f"{target}: training file must be UTF-8 text") from exc
    except OSError as exc:
        raise TrainingFileError(f"could not read {target}: {exc}") from exc
    suffix = target.suffix.lower()
    if suffix == ".jsonl":
        return load_training_jsonl(text, target.name)
    if suffix == ".csv":
        return load_training_csv(text, target.name)
    raise TrainingFileError(
        f"{target}: unsupported training file format {suffix!r} (use .jsonl or .csv)"
    )


@dataclass
class LeakageMatch:
    eval_id: str
    train_id: str
    tier: str  # "exact" | "near"
    jaccard: float
    containment: float
    passed: bool  # the eval item's recorded outcome


@dataclass
class LeakageStats:
    n_eval_items: int
    n_training_rows: int
    total_passes: int  # eval items (with comparable text) that passed
    exact: list[LeakageMatch] = field(default_factory=list)
    near: list[LeakageMatch] = field(default_factory=list)

    @property
    def leaked_ids(self) -> list[str]:
        return [m.eval_id for m in self.exact + self.near]

    @property
    def contaminated_wins(self) -> int:
        return sum(1 for m in self.exact + self.near if m.passed)

    @property
    def overlap_rate(self) -> float:
        return len(self.leaked_ids) / self.n_eval_items if self.n_eval_items else 0.0

    @property
    def contaminated_share(self) -> float:
        """Contaminated wins as a share of all passed items (0 with no passes)."""
        return self.contaminated_wins / self.total_passes if self.total_passes else 0.0


def _similarity(a: set[str], b: set[str]) -> tuple[float, float]:
    if not a or not b:
        return 0.0, 0.0
    jaccard = len(a & b) / len(a | b)
    return jaccard, shingle_containment(a, b)


def leakage_stats(
    eval_texts: dict[str, str],
    training: list[TrainingRecord],
    passed_ids: set[str],
) -> LeakageStats:
    """Compute exact and near-duplicate overlap of eval items in training rows.

    eval_texts maps eval item id -> comparable text (see item_text);
    passed_ids holds the eval items with a passing recorded outcome.
    Each eval item reports at most one match: its exact twin if one
    exists, else its most similar training row when that row crosses
    the near-duplicate threshold.
    """
    train_texts = {r.id: item_text(r.prompt, r.target) for r in training}
    train_digests: dict[str, str] = {}
    train_shingles: dict[str, set[str]] = {}
    for train_id, text in train_texts.items():
        train_digests.setdefault(text_digest(text), train_id)
        train_shingles[train_id] = _shingles(text)

    # Scaffolding discount for the near tier: shingles shared by most
    # of the eval set are its template, not an item's identity. The
    # discount set comes from the eval corpus alone and is removed from
    # both sides, so the comparison runs over the same vocabulary.
    scaffold: set[str] = set()
    if len(eval_texts) >= SCAFFOLD_MIN_ITEMS:
        frequency: dict[str, int] = {}
        for text in eval_texts.values():
            for shingle in _shingles(text):
                frequency[shingle] = frequency.get(shingle, 0) + 1
        scaffold = {
            shingle
            for shingle, count in frequency.items()
            if count / len(eval_texts) > SCAFFOLD_DF
        }
    train_distinctive = {
        train_id: shingles - scaffold for train_id, shingles in train_shingles.items()
    }
    index: dict[str, list[str]] = {}
    for train_id, shingles in train_distinctive.items():
        for shingle in shingles:
            index.setdefault(shingle, []).append(train_id)

    total_passes = sum(1 for eval_id in eval_texts if eval_id in passed_ids)
    exact: list[LeakageMatch] = []
    near: list[LeakageMatch] = []
    for eval_id in sorted(eval_texts):
        text = eval_texts[eval_id]
        passed = eval_id in passed_ids
        twin = train_digests.get(text_digest(text))
        if twin is not None:
            jaccard, containment = _similarity(_shingles(text), train_shingles[twin])
            exact.append(LeakageMatch(eval_id, twin, "exact", jaccard, containment, passed))
            continue
        eval_shingles = _shingles(text) - scaffold
        if not eval_shingles:
            continue
        candidates: set[str] = set()
        for shingle in eval_shingles:
            posting = index.get(shingle, [])
            if 0 < len(posting) <= MAX_POSTING:
                candidates.update(posting)
        best: LeakageMatch | None = None
        for train_id in candidates:
            jaccard, containment = _similarity(eval_shingles, train_distinctive[train_id])
            if jaccard < NEAR_JACCARD and containment < NEAR_CONTAINMENT:
                continue
            match = LeakageMatch(eval_id, train_id, "near", jaccard, containment, passed)
            if best is None or (match.jaccard, match.containment) > (best.jaccard, best.containment):
                best = match
        if best is not None:
            near.append(best)
    exact.sort(key=lambda m: m.eval_id)
    near.sort(key=lambda m: (-m.jaccard, -m.containment, m.eval_id))
    return LeakageStats(
        n_eval_items=len(eval_texts),
        n_training_rows=len(training),
        total_passes=total_passes,
        exact=exact,
        near=near,
    )
