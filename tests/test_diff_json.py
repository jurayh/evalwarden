"""The versioned diff JSON contract: `evalwarden.diff` 1.0.

Acceptance proof, field for field: on the three DIFF-001 fixtures --
(a) aggregate tie with concentrated flips, (b) symmetric noise, (c)
broad improvement plus a real regression slice -- plus the calibration
guard case (significant but immaterial net), the JSON document's
numbers must match the text output's numbers exactly, and the
document must round-trip through the strict reader. The text and JSON
renderers read the same engine values, so the check is that neither
renderer transforms them: JSON floats are the engine floats bit for
bit, and formatting them with the text reporter's specs reproduces
the text strings verbatim.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from evalwarden.canonical import CanonicalValidationError
from evalwarden.cli import app
from evalwarden.diffing import diff_stats
from evalwarden.engine import diff_models
from evalwarden.reporters.diff_report import (
    diff_document,
    parse_diff_document,
    render_diff_json,
)

from .test_diff import (
    _model,
    _pair_a,
    _pair_b,
    _pair_c,
    _write_inspect_dir,
)

runner = CliRunner()


def _pair_calibration():
    """The gate-calibration pair: 4000 items, flips 231/169 (p ~ 0.002)
    but net regression 1.55%, below the 2% materiality floor: PASS."""
    n = 4000
    ids = [f"t{i:04d}" for i in range(n)]
    base_out = {tid: i < 2000 for i, tid in enumerate(ids)}
    cand_out = dict(base_out)
    for tid in ids[:231]:
        cand_out[tid] = False
    for tid in ids[2000:2169]:
        cand_out[tid] = True
    return _model("base", base_out), _model("cand", cand_out)


def _dirs_from_pair(tmp_path: Path, name: str, baseline, candidate):
    """Write a model pair as Inspect-JSON dirs, the way the CLI loads them."""

    def outcomes(model):
        return {a.task_id: a.status == "pass" for a in model.attempts}

    def subjects(model):
        return {t.id: t.metadata.get("subject", "") for t in model.tasks}

    subject_of = {**subjects(baseline), **subjects(candidate)}
    base_dir = _write_inspect_dir(
        tmp_path / f"{name}-base", baseline.eval_id, outcomes(baseline), subject_of
    )
    cand_dir = _write_inspect_dir(
        tmp_path / f"{name}-cand", candidate.eval_id, outcomes(candidate), subject_of
    )
    return base_dir, cand_dir


def _text_and_json(tmp_path: Path, name: str, baseline, candidate):
    """(text output, parsed JSON doc) for one pair through the CLI."""
    base_dir, cand_dir = _dirs_from_pair(tmp_path, name, baseline, candidate)
    text = runner.invoke(app, ["diff", str(base_dir), str(cand_dir)])
    as_json = runner.invoke(app, ["diff", str(base_dir), str(cand_dir), "--format", "json"])
    assert text.exit_code == as_json.exit_code, (text.output, as_json.output)
    return text.output, json.loads(as_json.output)


# -- contract shape -----------------------------------------------------------


def test_fixture_a_document_shape():
    baseline, candidate = _pair_a()
    doc = diff_document(diff_models(baseline, candidate))
    assert doc["schema"] == "evalwarden.diff"
    assert doc["schema_version"] == "1.0"
    assert doc["tool"]["name"] == "evalwarden"
    assert doc["tool"]["version"]
    assert doc["baseline_eval_id"] == "base"
    assert doc["candidate_eval_id"] == "cand"
    assert doc["verdict"] == "BLOCKED"
    assert doc["score"] == 90  # one HIGH finding: 100 - 10
    assert doc["notes"] == []
    assert doc["unsupported"] == []
    assert len(doc["comparisons"]) == 1
    comp = doc["comparisons"][0]
    assert comp["label"] == "the unnamed model"
    assert comp["judgeable"] is True
    assert "unjudged_reason" not in comp  # only recorded when unjudged
    assert comp["alignment"] == {
        "n_paired": 240,
        "baseline_only": [],
        "candidate_only": [],
        "coverage": 1.0,
    }
    assert comp["aggregate"] == {
        "baseline_mean": 0.8,
        "candidate_mean": 0.8,
        "mean_delta": 0.0,
    }
    flips = comp["flips"]
    assert flips["regressions"] == 36
    assert flips["gains"] == 36
    assert flips["unchanged"] == 168
    assert flips["p_value"] == 1.0
    assert flips["net_regression_rate"] == 0.0
    # Every flip carries its item id, both directions.
    assert {f["task_id"] for f in flips["regression_items"]} == {
        f"algebra-{i:02d}" for i in range(18, 48)
    } | {f"{s}-{i:02d}" for s in ("geometry", "calculus", "statistics") for i in (0, 1)}
    assert {f["task_id"] for f in flips["gain_items"]} == {
        f"{s}-{i:02d}" for s in ("geometry", "calculus", "statistics") for i in range(48, 60)
    }
    assert all(
        f["baseline_score"] == 1.0 and f["candidate_score"] == 0.0
        for f in flips["regression_items"]
    )
    assert all(
        f["baseline_score"] == 0.0 and f["candidate_score"] == 1.0
        for f in flips["gain_items"]
    )
    algebra = next(s for s in comp["slices"] if s["value"] == "algebra")
    assert algebra["key"] == "subject"
    assert algebra["value"] == "algebra"
    assert algebra["n"] == 60
    assert algebra["regressions"] == 30
    assert algebra["gains"] == 0
    assert algebra["p_value"] == 2 / 2**30  # McNemar exact for a 30/0 split
    assert algebra["net_regression_rate"] == 0.5
    assert f"{algebra['p_value']:.3g}" == "1.86e-09"
    assert len(comp["slices"]) == 4  # every slice is carried, not just flagged ones
    assert len(doc["findings"]) == 1
    assert doc["findings"][0]["check_id"] == "DIFF-001"


def test_fixture_b_document_passes_with_zero_findings():
    baseline, candidate = _pair_b()
    doc = diff_document(diff_models(baseline, candidate))
    assert doc["verdict"] == "PASS"
    assert doc["score"] == 100
    assert doc["findings"] == []
    flips = doc["comparisons"][0]["flips"]
    assert (flips["regressions"], flips["gains"]) == (24, 20)
    assert len(flips["regression_items"]) == 24
    assert len(flips["gain_items"]) == 20
    assert f"{flips['p_value']:.3g}" == "0.652"


def test_fixture_c_document_reports_gains_and_blocks():
    baseline, candidate = _pair_c()
    doc = diff_document(diff_models(baseline, candidate))
    assert doc["verdict"] == "BLOCKED"
    assert doc["score"] == 90
    comp = doc["comparisons"][0]
    flips = comp["flips"]
    assert (flips["regressions"], flips["gains"]) == (15, 46)
    assert comp["aggregate"]["mean_delta"] == pytest.approx(31 / 240)
    algebra = next(s for s in comp["slices"] if s["value"] == "algebra")
    assert (algebra["regressions"], algebra["gains"]) == (12, 1)
    assert f"{algebra['p_value']:.3g}" == "0.00342"
    assert algebra["net_regression_rate"] == pytest.approx(11 / 60)
    assert len(doc["findings"]) == 1
    # The gains are on the record, in the slice finding's own evidence.
    assert any("46 gains" in e for f in doc["findings"] for e in f["evidence"])


def test_calibration_guard_document_passes():
    baseline, candidate = _pair_calibration()
    doc = diff_document(diff_models(baseline, candidate))
    assert doc["verdict"] == "PASS"
    assert doc["findings"] == []
    flips = doc["comparisons"][0]["flips"]
    assert (flips["regressions"], flips["gains"]) == (231, 169)
    assert flips["p_value"] < 0.01
    assert flips["net_regression_rate"] == pytest.approx(62 / 4000)
    assert len(flips["regression_items"]) == 231
    assert len(flips["gain_items"]) == 169


# -- exact match: JSON numbers are the engine numbers ------------------------


@pytest.mark.parametrize("pair", [_pair_a, _pair_b, _pair_c, _pair_calibration])
def test_json_numbers_are_the_engine_numbers_bit_for_bit(pair):
    """No rounding, no re-derivation: the document carries the floats
    the gate computed, so the gate and the document cannot disagree."""
    baseline, candidate = pair()
    result = diff_models(baseline, candidate)
    doc = diff_document(result)
    comp = doc["comparisons"][0]
    stats = result.comparisons[0].stats
    alignment = result.comparisons[0].alignment
    assert comp["alignment"]["n_paired"] == alignment.n_paired
    assert comp["alignment"]["coverage"] == alignment.coverage
    assert comp["aggregate"]["baseline_mean"] == stats.baseline_mean
    assert comp["aggregate"]["candidate_mean"] == stats.candidate_mean
    assert comp["aggregate"]["mean_delta"] == stats.mean_delta
    assert comp["flips"]["p_value"] == stats.p_value
    assert comp["flips"]["net_regression_rate"] == stats.net_regression_rate
    assert [f["task_id"] for f in comp["flips"]["regression_items"]] == [
        f.task_id for f in stats.worst
    ]
    assert [f["task_id"] for f in comp["flips"]["gain_items"]] == [
        f.task_id for f in stats.gain_items
    ]
    for doc_slice, stat_slice in zip(comp["slices"], stats.slices):
        assert doc_slice["p_value"] == stat_slice.p_value
        assert doc_slice["net_regression_rate"] == stat_slice.net_regression_rate


# -- exact match: text strings are the JSON numbers, formatted --------------


@pytest.mark.parametrize(
    "pair,name", [(_pair_a, "a"), (_pair_b, "b"), (_pair_c, "c"), (_pair_calibration, "cal")]
)
def test_text_strings_are_the_json_numbers_formatted(tmp_path: Path, pair, name):
    """Every number the text report prints, formatted from the JSON
    document's own values, appears verbatim in the text output."""
    baseline, candidate = pair()
    text, doc = _text_and_json(tmp_path, name, baseline, candidate)
    comp = doc["comparisons"][0]
    alignment = comp["alignment"]
    aggregate = comp["aggregate"]
    flips = comp["flips"]
    assert (
        f"alignment: {alignment['n_paired']} paired items "
        f"({len(alignment['baseline_only'])} baseline-only, "
        f"{len(alignment['candidate_only'])} candidate-only; "
        f"coverage {alignment['coverage']:.0%})"
    ) in text
    assert (
        f"aggregate on paired items: {aggregate['baseline_mean']:.1%} -> "
        f"{aggregate['candidate_mean']:.1%} ({aggregate['mean_delta']:+.1%})"
    ) in text
    assert (
        f"flips: {flips['regressions']} pass-to-fail, "
        f"{flips['gains']} fail-to-pass, {flips['unchanged']} unchanged; "
        f"McNemar p={flips['p_value']:.3g}; net regression "
        f"{flips['net_regression_rate']:+.1%}"
    ) in text
    assert f"Verdict: {doc['score']} / 100 {doc['verdict']}" in text
    for sl in comp["slices"]:
        # A slice's numbers reach the text only when the slice is
        # flagged (as finding evidence); unflagged slices have no text
        # rendering to match, and the engine-equality test above covers
        # their numbers instead.
        if not any(
            f"{sl['key']}={sl['value']}" in f["title"] for f in doc["findings"]
        ):
            continue
        assert (
            f"slice {sl['key']}={sl['value']}: {sl['n']} paired items, "
            f"{sl['regressions']} pass-to-fail vs {sl['gains']} "
            f"fail-to-pass (net regression "
            f"{sl['net_regression_rate']:.0%}, McNemar "
            f"p={sl['p_value']:.3g})"
        ) in text
    # Findings are lossless: every evidence line and title in the JSON
    # document is printed by the text report, verbatim.
    for finding in doc["findings"]:
        assert finding["title"] in text
        for ev in finding["evidence"]:
            assert f"    evidence: {ev}" in text


# -- engine: gain items are retained, largest rise first ---------------------


def test_diff_stats_retains_gain_items_sorted_by_rise():
    pairs = [
        ("t1", 0.0, 0.6),  # gain, rise 0.6
        ("t2", 0.0, 1.0),  # gain, rise 1.0
        ("t3", 0.4, 0.9),  # gain, rise 0.5
        ("t4", 1.0, 0.0),  # regression
        ("t5", 1.0, 1.0),  # unchanged
    ]
    stats = diff_stats(pairs)
    assert stats.gains == 3
    assert [(f.task_id, f.rise) for f in stats.gain_items] == [
        ("t2", 1.0),
        ("t1", 0.6),
        ("t3", pytest.approx(0.5)),
    ]
    assert [f.task_id for f in stats.worst] == ["t4"]


# -- round-trip through the strict reader --------------------------------------


@pytest.mark.parametrize("pair", [_pair_a, _pair_b, _pair_c, _pair_calibration])
def test_document_round_trips_through_the_strict_reader(pair):
    baseline, candidate = pair()
    result = diff_models(baseline, candidate)
    doc = diff_document(result)
    parsed = parse_diff_document(json.loads(render_diff_json(result)))
    assert parsed.to_dict() == doc


def test_unjudged_comparison_round_trips_with_its_reason():
    ids = [f"t{i:02d}" for i in range(20)]
    baseline = _model("base", {tid: True for tid in ids})
    candidate = _model("cand", {tid: False for tid in ids})
    result = diff_models(baseline, candidate)
    assert result.verdict == "PASS"
    doc = diff_document(result)
    comp = doc["comparisons"][0]
    assert comp["judgeable"] is False
    assert "minimum 30" in comp["unjudged_reason"]
    parsed = parse_diff_document(copy.deepcopy(doc))
    assert parsed.to_dict() == doc
    assert parsed.comparisons[0].unjudged_reason == comp["unjudged_reason"]


# -- strict reader: rejection and naming ---------------------------------------


def _doc_a():
    baseline, candidate = _pair_a()
    return diff_document(diff_models(baseline, candidate))


def test_wrong_schema_version_is_rejected_naming_the_path():
    doc = _doc_a()
    doc["schema_version"] = "2.0"
    with pytest.raises(CanonicalValidationError, match=r"\$\.schema_version"):
        parse_diff_document(doc)


def test_wrong_schema_is_rejected_naming_the_path():
    doc = _doc_a()
    doc["schema"] = "evalwarden.findings"
    with pytest.raises(CanonicalValidationError, match=r"\$\.schema"):
        parse_diff_document(doc)


def test_missing_field_is_rejected_naming_the_path():
    doc = _doc_a()
    del doc["comparisons"]
    with pytest.raises(CanonicalValidationError, match=r"\$\.comparisons"):
        parse_diff_document(doc)


def test_missing_nested_field_is_rejected_naming_the_path():
    doc = _doc_a()
    del doc["comparisons"][0]["flips"]["p_value"]
    with pytest.raises(
        CanonicalValidationError, match=r"\$\.comparisons\[0\]\.flips\.p_value"
    ):
        parse_diff_document(doc)


def test_non_finite_number_is_rejected():
    doc = _doc_a()
    doc["comparisons"][0]["flips"]["p_value"] = float("nan")
    with pytest.raises(
        CanonicalValidationError, match=r"\$\.comparisons\[0\]\.flips\.p_value"
    ):
        parse_diff_document(doc)


def test_bad_verdict_and_score_are_rejected():
    doc = _doc_a()
    doc["verdict"] = "MAYBE"
    with pytest.raises(CanonicalValidationError, match=r"\$\.verdict"):
        parse_diff_document(doc)
    doc = _doc_a()
    doc["score"] = 101
    with pytest.raises(CanonicalValidationError, match=r"\$\.score"):
        parse_diff_document(doc)


def test_non_object_root_is_rejected():
    with pytest.raises(CanonicalValidationError, match=r"\$: must be a JSON object"):
        parse_diff_document(["not", "a", "document"])


def test_unknown_fields_are_named_not_rejected():
    doc = _doc_a()
    doc["bogus"] = 1
    doc["comparisons"][0]["flips"]["surprise"] = "hi"
    parsed = parse_diff_document(doc)
    assert parsed.unsupported == [
        "$.bogus (not part of evalwarden.diff 1.0; ignored)",
        "$.comparisons[0].flips.surprise (not part of evalwarden.diff 1.0; ignored)",
    ]
    # Named fields survive a further round trip.
    assert parse_diff_document(parsed.to_dict()).unsupported == parsed.unsupported


def test_declared_unsupported_is_preserved():
    doc = _doc_a()
    doc["unsupported"] = ["$.old (not part of evalwarden.diff 1.0; ignored)"]
    parsed = parse_diff_document(doc)
    assert parsed.unsupported == doc["unsupported"]
    assert parsed.to_dict() == doc


# -- CLI -----------------------------------------------------------------------


def test_cli_diff_json_mode(tmp_path: Path):
    baseline, candidate = _pair_a()
    base_dir, cand_dir = _dirs_from_pair(tmp_path, "cli-a", baseline, candidate)
    result = runner.invoke(app, ["diff", str(base_dir), str(cand_dir), "--format", "json"])
    assert result.exit_code == 1, result.output
    assert result.output.startswith("{")  # the document owns stdout
    doc = json.loads(result.output)
    assert doc["schema"] == "evalwarden.diff"
    assert doc["verdict"] == "BLOCKED"
    assert doc["score"] == 90


def test_cli_diff_json_passes_on_noise_control(tmp_path: Path):
    baseline, candidate = _pair_b()
    base_dir, cand_dir = _dirs_from_pair(tmp_path, "cli-b", baseline, candidate)
    result = runner.invoke(app, ["diff", str(base_dir), str(cand_dir), "--format", "json"])
    assert result.exit_code == 0, result.output
    doc = json.loads(result.output)
    assert doc["verdict"] == "PASS"
    assert doc["findings"] == []


def test_cli_diff_json_honors_fail_on(tmp_path: Path):
    # Fixture (a) blocks on a HIGH finding; --fail-on error needs an
    # ERROR, so the same document exits 0 under that policy.
    baseline, candidate = _pair_a()
    base_dir, cand_dir = _dirs_from_pair(tmp_path, "cli-fo", baseline, candidate)
    result = runner.invoke(
        app, ["diff", str(base_dir), str(cand_dir), "--format", "json", "--fail-on", "error"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["verdict"] == "BLOCKED"


def test_cli_diff_bad_format_exits_2(tmp_path: Path):
    baseline, candidate = _pair_b()
    base_dir, cand_dir = _dirs_from_pair(tmp_path, "cli-bad", baseline, candidate)
    result = runner.invoke(
        app, ["diff", str(base_dir), str(cand_dir), "--format", "yaml"]
    )
    assert result.exit_code == 2
    assert "--format must be one of text|json" in result.output


def test_cli_diff_text_default_is_unchanged(tmp_path: Path):
    baseline, candidate = _pair_a()
    base_dir, cand_dir = _dirs_from_pair(tmp_path, "cli-text", baseline, candidate)
    result = runner.invoke(app, ["diff", str(base_dir), str(cand_dir)])
    assert result.exit_code == 1, result.output
    assert not result.output.startswith("{")
    assert "240 paired items" in result.output
    assert "80.0% -> 80.0% (+0.0%)" in result.output
    assert "36 pass-to-fail, 36 fail-to-pass" in result.output
