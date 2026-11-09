"""JSON reporter: the machine-readable diff document.

The document is a public contract, identified by the top-level `schema` /
`schema_version` fields, the same shape as the audit findings contract
(`evalwarden.findings` 1.0):

    {
      "schema": "evalwarden.diff",
      "schema_version": "1.0",
      "tool": {"name": "evalwarden", "version": "..."},
      "baseline_eval_id": ..., "candidate_eval_id": ...,
      "verdict": "BLOCKED" | "PASS",
      "score": 0-100 (diagnostic only),
      "comparisons": [ <one record per paired model series>, ... ],
      "notes": [ series that existed on one side only ],
      "findings": [ <finding record>, ... ],   # see reporters.records
      "unsupported": [ fields a reader did not recognise ]
    }

A comparison record carries, losslessly, everything the text `diff`
report prints for that series: the alignment (paired count, the
baseline-only and candidate-only item ids, coverage), the aggregate
means and delta, the flip accounting (counts, McNemar p, net rate, and
the item ids behind every regression and every gain), and every slice
with its own counts, p-value, and net rate. Numbers are the engine's
own floats, unrounded: formatting them with the text reporter's format
specs reproduces the text output exactly.

`parse_diff_document` is the strict reader for the contract. It
follows the canonical importer's validation precedent (see
`canonical.py`): the `schema` / `schema_version` pair must match
exactly, missing or mistyped fields raise a path-named
`CanonicalValidationError`, and unknown fields are not rejected but
named in the parsed document's `unsupported` list. The small
validators below mirror canonical's messages deliberately, so the two
versioned contracts fail in the same words.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

from .. import __version__
from ..canonical import CanonicalValidationError
from ..checks.diff import SeriesComparison
from ..diffing import Flip, SliceStats
from ..engine import DiffResult
from ..model import Confidence, Severity
from .records import finding_record

SCHEMA_NAME = "evalwarden.diff"
SCHEMA_VERSION = "1.0"

_VERDICTS = ("BLOCKED", "PASS")
_SEVERITIES = {s.value for s in Severity}
_CONFIDENCES = {c.value for c in Confidence}


# -- document construction -------------------------------------------------


def _flip_record(flip: Flip) -> dict:
    """One flipped item: the ids and scores behind the flip."""
    return {
        "task_id": flip.task_id,
        "baseline_score": flip.baseline_score,
        "candidate_score": flip.candidate_score,
    }


def _slice_record(sl: SliceStats) -> dict:
    return {
        "key": sl.key,
        "value": sl.value,
        "n": sl.n,
        "regressions": sl.regressions,
        "gains": sl.gains,
        "p_value": sl.p_value,
        "net_regression_rate": sl.net_regression_rate,
    }


def _comparison_record(comparison: SeriesComparison) -> dict:
    """One paired series: alignment, aggregate, flips, slices.

    `unjudged_reason` appears only when the series could not be
    judged, the same conditional-key rule `records.location_record`
    uses for `line` and `excerpt`: only facts actually recorded.
    """
    stats = comparison.stats
    alignment = comparison.alignment
    record: dict = {
        "label": comparison.label,
        "judgeable": comparison.judgeable,
    }
    if not comparison.judgeable:
        record["unjudged_reason"] = comparison.unjudged_reason
    record.update(
        {
            "alignment": {
                "n_paired": alignment.n_paired,
                "baseline_only": list(alignment.baseline_only),
                "candidate_only": list(alignment.candidate_only),
                "coverage": alignment.coverage,
            },
            "aggregate": {
                "baseline_mean": stats.baseline_mean,
                "candidate_mean": stats.candidate_mean,
                "mean_delta": stats.mean_delta,
            },
            "flips": {
                "regressions": stats.regressions,
                "gains": stats.gains,
                "unchanged": stats.unchanged,
                "p_value": stats.p_value,
                "net_regression_rate": stats.net_regression_rate,
                "regression_items": [_flip_record(f) for f in stats.worst],
                "gain_items": [_flip_record(f) for f in stats.gain_items],
            },
            "slices": [_slice_record(s) for s in stats.slices],
        }
    )
    return record


def diff_document(result: DiffResult) -> dict:
    """Build the versioned JSON diff document for a diff result."""
    return {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "tool": {"name": "evalwarden", "version": __version__},
        "baseline_eval_id": result.baseline.eval_id,
        "candidate_eval_id": result.candidate.eval_id,
        "verdict": result.verdict,
        "score": result.score,
        "comparisons": [_comparison_record(c) for c in result.comparisons],
        "notes": list(result.notes),
        "findings": [finding_record(f) for f in result.findings],
        "unsupported": [],
    }


def render_diff_json(result: DiffResult) -> str:
    """Render a diff result as the JSON diff document."""
    return json.dumps(diff_document(result), indent=2)


# -- strict reader ----------------------------------------------------------


@dataclass
class DiffFlip:
    """One flipped item as parsed from a diff document."""

    task_id: str
    baseline_score: float
    candidate_score: float

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "baseline_score": self.baseline_score,
            "candidate_score": self.candidate_score,
        }


@dataclass
class DiffSlice:
    """One slice's flip accounting as parsed from a diff document."""

    key: str
    value: str
    n: int
    regressions: int
    gains: int
    p_value: float
    net_regression_rate: float

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "value": self.value,
            "n": self.n,
            "regressions": self.regressions,
            "gains": self.gains,
            "p_value": self.p_value,
            "net_regression_rate": self.net_regression_rate,
        }


@dataclass
class DiffComparison:
    """One paired series as parsed from a diff document."""

    label: str
    judgeable: bool
    unjudged_reason: str | None
    n_paired: int
    baseline_only: list[str]
    candidate_only: list[str]
    coverage: float
    baseline_mean: float
    candidate_mean: float
    mean_delta: float
    regressions: int
    gains: int
    unchanged: int
    p_value: float
    net_regression_rate: float
    regression_items: list[DiffFlip]
    gain_items: list[DiffFlip]
    slices: list[DiffSlice]

    def to_dict(self) -> dict:
        record: dict = {
            "label": self.label,
            "judgeable": self.judgeable,
        }
        if not self.judgeable:
            record["unjudged_reason"] = self.unjudged_reason
        record.update(
            {
                "alignment": {
                    "n_paired": self.n_paired,
                    "baseline_only": list(self.baseline_only),
                    "candidate_only": list(self.candidate_only),
                    "coverage": self.coverage,
                },
                "aggregate": {
                    "baseline_mean": self.baseline_mean,
                    "candidate_mean": self.candidate_mean,
                    "mean_delta": self.mean_delta,
                },
                "flips": {
                    "regressions": self.regressions,
                    "gains": self.gains,
                    "unchanged": self.unchanged,
                    "p_value": self.p_value,
                    "net_regression_rate": self.net_regression_rate,
                    "regression_items": [f.to_dict() for f in self.regression_items],
                    "gain_items": [f.to_dict() for f in self.gain_items],
                },
                "slices": [s.to_dict() for s in self.slices],
            }
        )
        return record


@dataclass
class DiffDocument:
    """A parsed `evalwarden.diff` document, with unknown fields named.

    `to_dict` rebuilds the document exactly: parsing a document this
    module rendered and rebuilding it returns the same dict, which is
    the round-trip the contract promises.
    """

    baseline_eval_id: str
    candidate_eval_id: str
    verdict: str
    score: int
    tool_name: str
    tool_version: str
    comparisons: list[DiffComparison]
    notes: list[str]
    findings: list[dict]
    unsupported: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "tool": {"name": self.tool_name, "version": self.tool_version},
            "baseline_eval_id": self.baseline_eval_id,
            "candidate_eval_id": self.candidate_eval_id,
            "verdict": self.verdict,
            "score": self.score,
            "comparisons": [c.to_dict() for c in self.comparisons],
            "notes": list(self.notes),
            "findings": [dict(f) for f in self.findings],
            "unsupported": list(self.unsupported),
        }


def _fail(path: str, message: str) -> None:
    raise CanonicalValidationError(f"{path}: {message}")


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _object(value, path: str) -> dict:
    if not isinstance(value, dict):
        _fail(path, "must be a JSON object")
    return value


def _list(value, path: str) -> list:
    if not isinstance(value, list):
        _fail(path, "must be an array")
    return value


def _string(value, path: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        qualifier = "" if allow_empty else "non-empty "
        _fail(path, f"must be a {qualifier}string")
    return value


def _boolean(value, path: str) -> bool:
    if not isinstance(value, bool):
        _fail(path, "must be a boolean")
    return value


def _integer(
    value, path: str, *, minimum: int | None = None, maximum: int | None = None
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(path, "must be an integer")
    if minimum is not None and value < minimum:
        _fail(path, f"must be at least {minimum}")
    if maximum is not None and value > maximum:
        _fail(path, f"must be at most {maximum}")
    return value


def _number(
    value, path: str, *, minimum: float | None = None, maximum: float | None = None
) -> float:
    if not _is_number(value) or not math.isfinite(float(value)):
        _fail(path, "must be a finite number")
    number = float(value)
    if minimum is not None and number < minimum:
        _fail(path, f"must be at least {minimum:g}")
    if maximum is not None and number > maximum:
        _fail(path, f"must be at most {maximum:g}")
    return number


def _string_list(value, path: str, *, allow_empty_strings: bool = False) -> list[str]:
    values = _list(value, path)
    return [
        _string(item, f"{path}[{index}]", allow_empty=allow_empty_strings)
        for index, item in enumerate(values)
    ]


def _unknown_fields(
    value: dict, known: set[str], path: str, unsupported: list[str]
) -> None:
    for key in value:
        if key not in known:
            unsupported.append(
                f"{path}.{key} (not part of {SCHEMA_NAME} {SCHEMA_VERSION}; ignored)"
            )


_FINDING_KEYS = {
    "check_id",
    "severity",
    "confidence",
    "title",
    "message",
    "evidence",
    "locations",
    "remediation",
    "fingerprint",
}


def _parse_finding(raw, path: str, unsupported: list[str]) -> dict:
    """One finding record, validated against the shared record shape."""
    data = _object(raw, path)
    _unknown_fields(data, _FINDING_KEYS, path, unsupported)
    record: dict = {}
    for key in ("check_id", "title", "message"):
        if key not in data:
            _fail(f"{path}.{key}", "is required")
        record[key] = _string(data[key], f"{path}.{key}")
    for key in ("severity", "confidence"):
        if key not in data:
            _fail(f"{path}.{key}", "is required")
        record[key] = _string(data[key], f"{path}.{key}")
    if record["severity"] not in _SEVERITIES:
        _fail(f"{path}.severity", f"must be one of {sorted(_SEVERITIES)}")
    if record["confidence"] not in _CONFIDENCES:
        _fail(f"{path}.confidence", f"must be one of {sorted(_CONFIDENCES)}")
    for key in ("remediation", "fingerprint"):
        if key not in data:
            _fail(f"{path}.{key}", "is required")
        record[key] = _string(data[key], f"{path}.{key}", allow_empty=True)
    if "evidence" not in data:
        _fail(f"{path}.evidence", "is required")
    record["evidence"] = _string_list(data["evidence"], f"{path}.evidence")
    if "locations" not in data:
        _fail(f"{path}.locations", "is required")
    locations = _list(data["locations"], f"{path}.locations")
    parsed_locations = []
    for index, raw_loc in enumerate(locations):
        loc_path = f"{path}.locations[{index}]"
        loc = _object(raw_loc, loc_path)
        _unknown_fields(loc, {"file", "line", "excerpt"}, loc_path, unsupported)
        if "file" not in loc:
            _fail(f"{loc_path}.file", "is required")
        entry: dict = {"file": _string(loc["file"], f"{loc_path}.file")}
        if "line" in loc:
            entry["line"] = _integer(loc["line"], f"{loc_path}.line", minimum=0)
        if "excerpt" in loc:
            entry["excerpt"] = _string(
                loc["excerpt"], f"{loc_path}.excerpt", allow_empty=True
            )
        parsed_locations.append(entry)
    record["locations"] = parsed_locations
    return record


def _parse_flip(raw, path: str, unsupported: list[str]) -> DiffFlip:
    data = _object(raw, path)
    _unknown_fields(data, {"task_id", "baseline_score", "candidate_score"}, path, unsupported)
    return DiffFlip(
        task_id=_string(data.get("task_id"), f"{path}.task_id"),
        baseline_score=_number(
            data.get("baseline_score"), f"{path}.baseline_score", minimum=0.0, maximum=1.0
        ),
        candidate_score=_number(
            data.get("candidate_score"), f"{path}.candidate_score", minimum=0.0, maximum=1.0
        ),
    )


def _parse_slice(raw, path: str, unsupported: list[str]) -> DiffSlice:
    data = _object(raw, path)
    _unknown_fields(
        data,
        {"key", "value", "n", "regressions", "gains", "p_value", "net_regression_rate"},
        path,
        unsupported,
    )
    return DiffSlice(
        key=_string(data.get("key"), f"{path}.key"),
        value=_string(data.get("value"), f"{path}.value"),
        n=_integer(data.get("n"), f"{path}.n", minimum=0),
        regressions=_integer(data.get("regressions"), f"{path}.regressions", minimum=0),
        gains=_integer(data.get("gains"), f"{path}.gains", minimum=0),
        p_value=_number(data.get("p_value"), f"{path}.p_value", minimum=0.0, maximum=1.0),
        net_regression_rate=_number(
            data.get("net_regression_rate"),
            f"{path}.net_regression_rate",
            minimum=-1.0,
            maximum=1.0,
        ),
    )


def _parse_comparison(raw, path: str, unsupported: list[str]) -> DiffComparison:
    data = _object(raw, path)
    _unknown_fields(
        data,
        {"label", "judgeable", "unjudged_reason", "alignment", "aggregate", "flips", "slices"},
        path,
        unsupported,
    )
    label = _string(data.get("label"), f"{path}.label")
    judgeable = _boolean(data.get("judgeable"), f"{path}.judgeable")
    unjudged_reason = None
    if "unjudged_reason" in data:
        unjudged_reason = _string(
            data["unjudged_reason"], f"{path}.unjudged_reason", allow_empty=True
        )
    alignment = _object(data.get("alignment"), f"{path}.alignment")
    _unknown_fields(
        alignment, {"n_paired", "baseline_only", "candidate_only", "coverage"}, f"{path}.alignment", unsupported
    )
    n_paired = _integer(alignment.get("n_paired"), f"{path}.alignment.n_paired", minimum=0)
    baseline_only = _string_list(
        alignment.get("baseline_only"), f"{path}.alignment.baseline_only"
    )
    candidate_only = _string_list(
        alignment.get("candidate_only"), f"{path}.alignment.candidate_only"
    )
    coverage = _number(
        alignment.get("coverage"), f"{path}.alignment.coverage", minimum=0.0, maximum=1.0
    )
    aggregate = _object(data.get("aggregate"), f"{path}.aggregate")
    _unknown_fields(
        aggregate, {"baseline_mean", "candidate_mean", "mean_delta"}, f"{path}.aggregate", unsupported
    )
    baseline_mean = _number(
        aggregate.get("baseline_mean"), f"{path}.aggregate.baseline_mean", minimum=0.0, maximum=1.0
    )
    candidate_mean = _number(
        aggregate.get("candidate_mean"), f"{path}.aggregate.candidate_mean", minimum=0.0, maximum=1.0
    )
    mean_delta = _number(
        aggregate.get("mean_delta"), f"{path}.aggregate.mean_delta", minimum=-1.0, maximum=1.0
    )
    flips = _object(data.get("flips"), f"{path}.flips")
    _unknown_fields(
        flips,
        {"regressions", "gains", "unchanged", "p_value", "net_regression_rate", "regression_items", "gain_items"},
        f"{path}.flips",
        unsupported,
    )
    regressions = _integer(flips.get("regressions"), f"{path}.flips.regressions", minimum=0)
    gains = _integer(flips.get("gains"), f"{path}.flips.gains", minimum=0)
    unchanged = _integer(flips.get("unchanged"), f"{path}.flips.unchanged", minimum=0)
    p_value = _number(flips.get("p_value"), f"{path}.flips.p_value", minimum=0.0, maximum=1.0)
    net_rate = _number(
        flips.get("net_regression_rate"), f"{path}.flips.net_regression_rate", minimum=-1.0, maximum=1.0
    )
    regression_items = [
        _parse_flip(item, f"{path}.flips.regression_items[{i}]", unsupported)
        for i, item in enumerate(_list(flips.get("regression_items"), f"{path}.flips.regression_items"))
    ]
    gain_items = [
        _parse_flip(item, f"{path}.flips.gain_items[{i}]", unsupported)
        for i, item in enumerate(_list(flips.get("gain_items"), f"{path}.flips.gain_items"))
    ]
    slices = [
        _parse_slice(item, f"{path}.slices[{i}]", unsupported)
        for i, item in enumerate(_list(data.get("slices"), f"{path}.slices"))
    ]
    return DiffComparison(
        label=label,
        judgeable=judgeable,
        unjudged_reason=unjudged_reason,
        n_paired=n_paired,
        baseline_only=baseline_only,
        candidate_only=candidate_only,
        coverage=coverage,
        baseline_mean=baseline_mean,
        candidate_mean=candidate_mean,
        mean_delta=mean_delta,
        regressions=regressions,
        gains=gains,
        unchanged=unchanged,
        p_value=p_value,
        net_regression_rate=net_rate,
        regression_items=regression_items,
        gain_items=gain_items,
        slices=slices,
    )


def parse_diff_document(data) -> DiffDocument:
    """Strictly parse a dict as an `evalwarden.diff` 1.0 document.

    Missing or mistyped fields raise `CanonicalValidationError` naming
    the exact `$.path`; a wrong `schema` / `schema_version` is rejected
    the same way; unknown fields are named in the returned document's
    `unsupported` list, following the canonical importer's precedent.
    """
    unsupported: list[str] = []
    root = _object(data, "$")
    _unknown_fields(
        root,
        {
            "schema",
            "schema_version",
            "tool",
            "baseline_eval_id",
            "candidate_eval_id",
            "verdict",
            "score",
            "comparisons",
            "notes",
            "findings",
            "unsupported",
        },
        "$",
        unsupported,
    )
    if root.get("schema") != SCHEMA_NAME:
        _fail("$.schema", f"must be {SCHEMA_NAME!r}")
    if root.get("schema_version") != SCHEMA_VERSION:
        _fail(
            "$.schema_version",
            f"must be {SCHEMA_VERSION!r} (this reader understands only that version)",
        )
    tool = _object(root.get("tool"), "$.tool")
    _unknown_fields(tool, {"name", "version"}, "$.tool", unsupported)
    tool_name = _string(tool.get("name"), "$.tool.name")
    if tool_name != "evalwarden":
        _fail("$.tool.name", "must be 'evalwarden'")
    tool_version = _string(tool.get("version"), "$.tool.version")
    baseline_eval_id = _string(root.get("baseline_eval_id"), "$.baseline_eval_id")
    candidate_eval_id = _string(root.get("candidate_eval_id"), "$.candidate_eval_id")
    verdict = _string(root.get("verdict"), "$.verdict")
    if verdict not in _VERDICTS:
        _fail("$.verdict", "must be 'BLOCKED' or 'PASS'")
    score = _integer(root.get("score"), "$.score", minimum=0, maximum=100)
    comparisons = [
        _parse_comparison(item, f"$.comparisons[{i}]", unsupported)
        for i, item in enumerate(_list(root.get("comparisons"), "$.comparisons"))
    ]
    notes = _string_list(root.get("notes"), "$.notes")
    findings = [
        _parse_finding(item, f"$.findings[{i}]", unsupported)
        for i, item in enumerate(_list(root.get("findings"), "$.findings"))
    ]
    declared_unsupported = _string_list(
        root.get("unsupported", []), "$.unsupported", allow_empty_strings=True
    )
    unsupported = [*declared_unsupported, *unsupported]
    return DiffDocument(
        baseline_eval_id=baseline_eval_id,
        candidate_eval_id=candidate_eval_id,
        verdict=verdict,
        score=score,
        tool_name=tool_name,
        tool_version=tool_version,
        comparisons=comparisons,
        notes=notes,
        findings=findings,
        unsupported=unsupported,
    )
