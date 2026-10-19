"""DIFF-001 (paired regression versus a baseline run).

Kill-criterion proof at the check level, on seeded pairs whose outcomes
are planted item by item:
(a) a pair whose aggregate scores tie exactly while pass-to-fail flips
    concentrate in one subject slice -- the differ must flag the slice
    and block even though the global flip split is perfectly symmetric;
(b) a control pair whose flips are symmetric noise -- the differ must
    produce no findings and pass;
(c) a pair with broad improvement plus a small real regression slice --
    the gains are reported and the verdict is still driven by the
    regression rule.
Plus the alignment contract (coverage floors, series pairing) and a
pair of real Inspect-JSON artifacts loaded through the adapter, which
is the ground truth that pairing and slice metadata work on what
adapters actually produce.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from evalwarden.checks import REGISTRY
from evalwarden.checks.diff import PairedRegressionCheck, series_outcomes
from evalwarden.cli import app
from evalwarden.engine import diff_models, diff_pair, load_model
from evalwarden.model import Attempt, Confidence, IntegrityModel, Severity, TaskSample

from .conftest import make_model

SUBJECTS = ["algebra", "geometry", "calculus", "statistics"]
PER_SUBJECT = 60


def _ids(subject: str, n: int = PER_SUBJECT) -> list[str]:
    return [f"{subject}-{i:02d}" for i in range(n)]


def _model(
    eval_id: str,
    outcomes: dict[str, bool],
    model_id: str | None = None,
    subject_of: dict[str, str] | None = None,
) -> IntegrityModel:
    tasks = [
        TaskSample(
            id=tid,
            prompt=f"prompt for {tid}",
            metadata={"subject": subject_of[tid]} if subject_of else {},
        )
        for tid in outcomes
    ]
    attempts = [
        Attempt(
            task_id=tid,
            status="pass" if ok else "fail",
            score=1.0 if ok else 0.0,
            model_id=model_id,
        )
        for tid, ok in outcomes.items()
    ]
    model = make_model(attempts=attempts, tasks=tasks)
    model.eval_id = eval_id
    return model


def _subject_map(subjects: list[str], n: int = PER_SUBJECT) -> dict[str, str]:
    return {tid: s for s in subjects for tid in _ids(s, n)}


def _outcomes(pass_rule) -> dict[str, bool]:
    """Outcomes over all subjects: pass_rule(subject, index) -> bool."""
    return {
        tid: pass_rule(subject, i)
        for subject in SUBJECTS
        for i, tid in enumerate(_ids(subject))
    }


# -- (a) aggregate tie, concentrated regression ---------------------------


def _pair_a() -> tuple[IntegrityModel, IntegrityModel]:
    subject_of = _subject_map(SUBJECTS)
    baseline = _model("base", _outcomes(lambda s, i: i < 48), subject_of=subject_of)

    def cand_rule(s: str, i: int) -> bool:
        if s == "algebra":
            return i < 18  # 30 of 48 passes flip to fail, no gains
        return 2 <= i or i >= 48  # 2 regressions, all 12 fails flip to pass

    candidate = _model("cand", _outcomes(cand_rule), subject_of=subject_of)
    return baseline, candidate


def test_a_aggregates_tie_exactly():
    baseline, candidate = _pair_a()
    stats = PairedRegressionCheck(baseline).comparisons(candidate)[0][0].stats
    assert stats.baseline_mean == pytest.approx(0.8)
    assert stats.candidate_mean == pytest.approx(0.8)
    assert stats.mean_delta == pytest.approx(0.0)
    # The movement the delta hides: 36 flips each way.
    assert stats.regressions == 36
    assert stats.gains == 36
    assert stats.p_value == 1.0


def test_a_slice_flagged_and_blocks():
    baseline, candidate = _pair_a()
    result = diff_models(baseline, candidate)
    assert result.verdict == "BLOCKED"
    assert len(result.findings) == 1
    finding = result.findings[0]
    assert finding.id == "DIFF-001"
    assert finding.severity is Severity.HIGH
    assert finding.confidence is Confidence.MEDIUM
    assert "subject=algebra" in finding.title
    assert any("worst in slice" in e for e in finding.evidence)


# -- (b) symmetric noise control ------------------------------------------


def _pair_b() -> tuple[IntegrityModel, IntegrityModel]:
    subject_of = _subject_map(SUBJECTS)
    baseline = _model("base", _outcomes(lambda s, i: i < 45), subject_of=subject_of)
    # Per subject: 6 pass-to-fail (idx 0-5), 5 fail-to-pass (idx 45-49).
    candidate = _model(
        "cand",
        _outcomes(lambda s, i: 6 <= i < 50),
        subject_of=subject_of,
    )
    return baseline, candidate


def test_b_noise_control_stays_silent_and_passes():
    baseline, candidate = _pair_b()
    stats = PairedRegressionCheck(baseline).comparisons(candidate)[0][0].stats
    assert (stats.regressions, stats.gains) == (24, 20)
    result = diff_models(baseline, candidate)
    assert result.findings == []
    assert result.verdict == "PASS"


# -- (c) broad improvement plus a real regression slice --------------------


def _pair_c() -> tuple[IntegrityModel, IntegrityModel]:
    subject_of = _subject_map(SUBJECTS)
    baseline = _model("base", _outcomes(lambda s, i: i < 40), subject_of=subject_of)

    def cand_rule(s: str, i: int) -> bool:
        if s == "algebra":
            return 12 <= i <= 40  # 12 regressions, 1 gain
        return 1 <= i < 55  # 1 regression, 15 gains

    candidate = _model("cand", _outcomes(cand_rule), subject_of=subject_of)
    return baseline, candidate


def test_c_improvement_reported_but_regression_slice_drives_verdict():
    baseline, candidate = _pair_c()
    stats = PairedRegressionCheck(baseline).comparisons(candidate)[0][0].stats
    assert (stats.regressions, stats.gains) == (15, 46)
    assert stats.mean_delta == pytest.approx(31 / 240)
    result = diff_models(baseline, candidate)
    assert result.verdict == "BLOCKED"
    assert len(result.findings) == 1
    finding = result.findings[0]
    assert "subject=algebra" in finding.title
    # The gains are on the record, in the slice finding's own evidence.
    assert any("46 gains" in e for e in finding.evidence)


# -- gate calibration: significance and materiality are both required ------


def test_global_regression_blocks_when_significant_and_material():
    subject_of = _subject_map(SUBJECTS)
    baseline = _model("base", _outcomes(lambda s, i: i < 48), subject_of=subject_of)
    # Uniform regression everywhere: 12 pass-to-fail per subject, no gains.
    candidate = _model("cand", _outcomes(lambda s, i: i < 36), subject_of=subject_of)
    result = diff_models(baseline, candidate)
    assert result.verdict == "BLOCKED"
    titles = [f.title for f in result.findings]
    assert any("candidate regressed" in t for t in titles)
    global_finding = next(f for f in result.findings if "candidate regressed" in f.title)
    assert global_finding.severity is Severity.HIGH
    assert global_finding.confidence is Confidence.HIGH
    assert any("worst regressions" in e for e in global_finding.evidence)


def test_significant_but_immaterial_net_does_not_block():
    # 4000 paired items; flips split 231/169 (McNemar p ~ 0.002) but the
    # net regression is 1.55% of paired items: below the 2% materiality
    # floor, so significance alone does not close the gate.
    n = 4000
    ids = [f"t{i:04d}" for i in range(n)]
    base_out = {tid: i < 2000 for i, tid in enumerate(ids)}
    cand_out = dict(base_out)
    for tid in ids[:231]:
        cand_out[tid] = False  # regressions out of baseline passes
    for tid in ids[2000:2169]:
        cand_out[tid] = True  # gains out of baseline fails
    baseline = _model("base", base_out)
    candidate = _model("cand", cand_out)
    stats = PairedRegressionCheck(baseline).comparisons(candidate)[0][0].stats
    assert (stats.regressions, stats.gains) == (231, 169)
    assert stats.p_value < 0.01
    assert stats.net_regression_rate == pytest.approx(62 / 4000)
    result = diff_models(baseline, candidate)
    assert result.findings == []
    assert result.verdict == "PASS"


# -- alignment contract -----------------------------------------------------


def test_too_few_paired_items_is_advisory_not_a_verdict():
    ids = [f"t{i:02d}" for i in range(20)]
    baseline = _model("base", {tid: True for tid in ids})
    candidate = _model("cand", {tid: False for tid in ids})
    result = diff_models(baseline, candidate)
    assert result.verdict == "PASS"
    assert len(result.findings) == 1
    finding = result.findings[0]
    assert finding.id == "DIFF-001"
    assert finding.severity is Severity.MEDIUM
    assert finding.confidence is Confidence.LOW
    assert "cannot be judged" in finding.title


def test_low_alignment_coverage_is_advisory_not_a_verdict():
    base_ids = [f"t{i:03d}" for i in range(100)]
    baseline = _model("base", {tid: True for tid in base_ids})
    candidate = _model("cand", {tid: False for tid in base_ids[:40]})
    comparison = PairedRegressionCheck(baseline).comparisons(candidate)[0][0]
    assert comparison.stats.n_paired == 40
    assert comparison.alignment.coverage == pytest.approx(0.4)
    assert not comparison.judgeable
    result = diff_models(baseline, candidate)
    assert result.verdict == "PASS"
    assert len(result.findings) == 1
    assert "coverage" in result.findings[0].title


def test_no_pairable_series_is_advisory():
    ids = _ids("algebra", 40)
    baseline = _model("base", {tid: True for tid in ids}, model_id="m1")
    extra = _model("base", {tid: True for tid in ids}, model_id="m2")
    baseline.attempts += extra.attempts
    candidate = _model("cand", {tid: True for tid in ids}, model_id="m3")
    result = diff_models(baseline, candidate)
    assert result.verdict == "PASS"
    assert len(result.findings) == 1
    assert "no model series pairs" in result.findings[0].title


def test_plain_audit_stays_silent():
    # DIFF-001 is registered, but under a single-artifact audit there is
    # no baseline: every registered check runs, and DIFF-001 says nothing.
    _, candidate = _pair_a()
    diff_findings = [
        f for check in REGISTRY for f in check.run(candidate) if f.id == "DIFF-001"
    ]
    assert diff_findings == []


# -- pairing semantics ------------------------------------------------------


def test_series_outcomes_average_repeated_attempts():
    attempts = [
        Attempt(task_id="t1", status="pass", score=1.0),
        Attempt(task_id="t1", status="fail", score=0.0),
        Attempt(task_id="t2", status="pass", score=None),
        Attempt(task_id="t3", status="error", score=None),
    ]
    model = make_model(attempts=attempts, tasks=[])
    series = series_outcomes(model)
    assert series[None] == {"t1": 0.5, "t2": 1.0, "t3": 0.0}


def test_epoch_averaged_flips_count_as_regressions():
    ids = _ids("algebra", 40)
    tasks = [
        TaskSample(id=t, prompt="p", metadata={"subject": "algebra"}) for t in ids
    ]
    # Baseline: two epochs per item scoring 1.0 and 0.0 -> item score
    # 0.5, a pass. Candidate: both epochs fail -> item score 0.0.
    baseline = make_model(
        attempts=[
            Attempt(task_id=tid, status=status, score=score)
            for tid in ids
            for status, score in (("pass", 1.0), ("fail", 0.0))
        ],
        tasks=tasks,
    )
    candidate = make_model(
        attempts=[
            Attempt(task_id=tid, status="fail", score=0.0)
            for tid in ids
            for _ in range(2)
        ],
        tasks=tasks,
    )
    comparison = PairedRegressionCheck(baseline).comparisons(candidate)[0][0]
    assert comparison.stats.regressions == 40
    assert diff_models(baseline, candidate).verdict == "BLOCKED"


def test_finetune_rename_pairs_across_model_names():
    ids = _ids("algebra", 40)
    base_out = {tid: i < 32 for i, tid in enumerate(ids)}
    cand_out = {tid: i < 17 for i, tid in enumerate(ids)}  # 15 regressions
    baseline = _model("base", base_out, model_id="base-7b")
    candidate = _model("cand", cand_out, model_id="base-7b-ft")
    result = diff_models(baseline, candidate)
    assert result.verdict == "BLOCKED"
    assert len(result.findings) == 1
    assert "base-7b -> base-7b-ft" in result.findings[0].title


def test_multi_model_artifacts_pair_per_model():
    ids = _ids("algebra", 40)

    def two_model_model(eval_id: str, m2_pass_below: int) -> IntegrityModel:
        attempts = []
        for model_id, threshold in (("m1", 32), ("m2", m2_pass_below)):
            attempts += [
                Attempt(
                    task_id=tid,
                    status="pass" if i < threshold else "fail",
                    score=1.0 if i < threshold else 0.0,
                    model_id=model_id,
                )
                for i, tid in enumerate(ids)
            ]
        model = make_model(
            attempts=attempts,
            tasks=[TaskSample(id=t, prompt="p") for t in ids],
        )
        model.eval_id = eval_id
        return model

    baseline = two_model_model("base", 32)
    candidate = two_model_model("cand", 17)  # m2 regresses, m1 unchanged
    comparisons, notes = PairedRegressionCheck(baseline).comparisons(candidate)
    assert len(comparisons) == 2
    assert notes == []
    result = diff_models(baseline, candidate)
    assert result.verdict == "BLOCKED"
    assert len(result.findings) == 1
    assert result.findings[0].title.startswith("m2: ")


# -- real adapter pair (the kill-criterion ground truth) --------------------


def _write_inspect_dir(
    path: Path, eval_id: str, outcomes: dict[str, bool], subject_of: dict[str, str]
) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    dataset = {
        "schema_version": "evalwarden-artifact-v1",
        "eval_id": eval_id,
        "tasks": [
            {
                "id": tid,
                "prompt": f"prompt for {tid}",
                "metadata": {"subject": subject_of[tid]},
            }
            for tid in outcomes
        ],
    }
    (path / "dataset.json").write_text(json.dumps(dataset), encoding="utf-8")
    (path / "grader.json").write_text(json.dumps({"kind": "script"}), encoding="utf-8")
    (path / "environment.json").write_text(json.dumps({"env": {}}), encoding="utf-8")
    run = {
        "attempts": [
            {
                "task_id": tid,
                "status": "pass" if ok else "fail",
                "score": 1.0 if ok else 0.0,
            }
            for tid, ok in outcomes.items()
        ]
    }
    (path / "run.json").write_text(json.dumps(run), encoding="utf-8")
    return path


def test_real_inspect_pair_aligns_and_slices_on_adapter_metadata(tmp_path: Path):
    ids_a = _ids("algebra", 20)
    ids_b = _ids("geometry", 20)
    subject_of = {**{t: "algebra" for t in ids_a}, **{t: "geometry" for t in ids_b}}
    base_out = {tid: True for tid in ids_a + ids_b}
    cand_out = dict(base_out)
    for tid in ids_a[:12]:
        cand_out[tid] = False  # 12 of 20 algebra items regress
    base_dir = _write_inspect_dir(tmp_path / "base", "real-base", base_out, subject_of)
    cand_dir = _write_inspect_dir(tmp_path / "cand", "real-cand", cand_out, subject_of)

    baseline = load_model(base_dir)
    candidate = load_model(cand_dir)
    assert baseline.adapter_name == "inspect"
    # The adapter carries task metadata into the model: slices can use it.
    assert {t.metadata["subject"] for t in candidate.tasks} == {"algebra", "geometry"}

    comparison = PairedRegressionCheck(baseline).comparisons(candidate)[0][0]
    assert comparison.alignment.n_paired == 40
    assert comparison.alignment.coverage == 1.0
    assert comparison.stats.regressions == 12
    slice_keys = {(s.key, s.value) for s in comparison.stats.slices}
    assert ("subject", "algebra") in slice_keys

    result = diff_models(baseline, candidate)
    assert result.verdict == "BLOCKED"
    assert any("subject=algebra" in f.title for f in result.findings)


def test_real_inspect_pair_noise_control_passes(tmp_path: Path):
    ids = _ids("algebra", 40)
    subject_of = {t: "algebra" for t in ids}
    base_out = {tid: True for tid in ids}
    cand_out = dict(base_out)
    for tid in ids[:6]:
        cand_out[tid] = False
    # Six flips one way only would be a real (if small) regression signal
    # at p=0.03 -- above ALPHA, so still a pass; the point here is the
    # adapter pair runs end to end.
    base_dir = _write_inspect_dir(tmp_path / "base", "real-base", base_out, subject_of)
    cand_dir = _write_inspect_dir(tmp_path / "cand", "real-cand", cand_out, subject_of)
    result, policy_failed = diff_pair(base_dir, cand_dir)
    assert result.comparisons[0].alignment.coverage == 1.0
    assert not policy_failed


# -- CLI ---------------------------------------------------------------------

runner = CliRunner()


def test_cli_checks_and_explain_list_diff():
    checks = runner.invoke(app, ["checks"])
    assert checks.exit_code == 0
    assert "DIFF-001" in checks.output
    assert "Run comparison" in checks.output
    explain = runner.invoke(app, ["explain", "DIFF-001"])
    assert explain.exit_code == 0
    assert "Paired regression versus a baseline run" in explain.output


def test_cli_diff_blocks_on_regression_pair(tmp_path: Path):
    ids_a = _ids("algebra", 20)
    ids_b = _ids("geometry", 20)
    subject_of = {**{t: "algebra" for t in ids_a}, **{t: "geometry" for t in ids_b}}
    base_out = {tid: True for tid in ids_a + ids_b}
    cand_out = dict(base_out)
    for tid in ids_a[:12]:
        cand_out[tid] = False
    base_dir = _write_inspect_dir(tmp_path / "base", "cli-base", base_out, subject_of)
    cand_dir = _write_inspect_dir(tmp_path / "cand", "cli-cand", cand_out, subject_of)
    result = runner.invoke(app, ["diff", str(base_dir), str(cand_dir)])
    assert result.exit_code == 1, result.output
    assert "BLOCKED" in result.output
    assert "DIFF-001" in result.output
    assert "40 paired items" in result.output


def test_cli_diff_passes_on_symmetric_pair(tmp_path: Path):
    ids = _ids("algebra", 40)
    subject_of = {t: "algebra" for t in ids}
    base_out = {tid: i < 34 for i, tid in enumerate(ids)}
    cand_out = dict(base_out)
    for tid in ids[:6]:
        cand_out[tid] = False  # 6 regressions
    for tid in ids[34:40]:
        cand_out[tid] = True  # 6 gains: perfectly symmetric
    base_dir = _write_inspect_dir(tmp_path / "base", "cli-base", base_out, subject_of)
    cand_dir = _write_inspect_dir(tmp_path / "cand", "cli-cand", cand_out, subject_of)
    result = runner.invoke(app, ["diff", str(base_dir), str(cand_dir)])
    assert result.exit_code == 0, result.output
    assert "PASS" in result.output
    assert "6 pass-to-fail, 6 fail-to-pass" in result.output
