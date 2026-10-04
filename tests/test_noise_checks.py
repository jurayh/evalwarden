"""NOISE-001 (dominant noise source) and NOISE-002 (claimed delta verdict).

Kill-criterion proof at the check level: seeded tables with a planted
dominant source must make NOISE-001 name that source; a balanced table
must stay silent; a clean low-noise control with a planted zero delta
must come back within-noise under NOISE-002. Seeded share recovery is
proven in tests/test_noise.py.
"""
from __future__ import annotations

from evalwarden.checks.noise import ClaimedDeltaCheck, DominantNoiseCheck
from evalwarden.model import Confidence, RunItemScore, Severity

from .conftest import make_model
from .test_noise import JUDGE_DOMINATED, SAMPLING_DOMINATED, plant_table

SEED = 20261004


def _noise_model(tuples, claimed_delta=None, n_items=None):
    scores = [
        RunItemScore(
            task_id=t[0], score=t[1], generation_index=t[2],
            grade_index=t[3], environment_index=t[4],
        )
        for t in tuples
    ]
    if n_items is not None:
        keep = {f"t{i:03d}" for i in range(n_items)}
        scores = [s for s in scores if s.task_id in keep]
    model = make_model()
    model.run_scores = scores
    model.claimed_delta = claimed_delta
    return model


def test_noise001_names_judge_when_judge_dominates():
    model = _noise_model(plant_table(SEED, *JUDGE_DOMINATED))
    findings = DominantNoiseCheck().run(model)
    assert len(findings) == 1
    finding = findings[0]
    assert finding.id == "NOISE-001"
    assert finding.severity is Severity.MEDIUM
    assert finding.confidence is Confidence.MEDIUM
    assert "judge" in finding.title
    assert any("noise floor" in e for e in finding.evidence)


def test_noise001_names_sampling_when_sampling_dominates():
    model = _noise_model(plant_table(SEED, *SAMPLING_DOMINATED))
    findings = DominantNoiseCheck().run(model)
    assert len(findings) == 1
    assert "sampling" in findings[0].title


def test_noise001_silent_on_balanced_budget():
    model = _noise_model(plant_table(SEED, 20.0, 20.0, 20.0, 2.0))
    assert DominantNoiseCheck().run(model) == []


def test_noise001_silent_without_decomposition():
    """One measured axis is total variance, not attribution."""
    model = _noise_model(plant_table(SEED, *JUDGE_DOMINATED), n_items=40)
    for score in model.run_scores:
        score.grade_index = 0
        score.environment_index = 0
    assert DominantNoiseCheck().run(model) == []


def test_noise001_silent_when_thin_or_absent():
    thin = _noise_model(plant_table(SEED, *JUDGE_DOMINATED), n_items=5)
    assert DominantNoiseCheck().run(thin) == []
    assert DominantNoiseCheck().run(make_model()) == []


def test_noise002_zero_delta_on_clean_control_is_within_noise():
    """Clean low-noise control: a planted zero improvement is weather."""
    clean = plant_table(SEED, 0.04, 0.04, 0.04, 0.01)
    model = _noise_model(clean, claimed_delta=0.0)
    findings = ClaimedDeltaCheck().run(model)
    assert len(findings) == 1
    assert findings[0].id == "NOISE-002"
    assert "inside the noise floor" in findings[0].title
    assert findings[0].confidence is Confidence.MEDIUM


def test_noise002_small_claim_inside_floor_fires():
    model = _noise_model(plant_table(SEED, *JUDGE_DOMINATED), claimed_delta=0.5)
    findings = ClaimedDeltaCheck().run(model)
    assert len(findings) == 1
    assert "inside the noise floor" in findings[0].title


def test_noise002_real_claim_stays_silent():
    model = _noise_model(plant_table(SEED, *JUDGE_DOMINATED), claimed_delta=25.0)
    assert ClaimedDeltaCheck().run(model) == []


def test_noise002_cant_tell_with_unmeasured_axis():
    """Claim clears the floor from generation replicates alone, but judge
    and environment noise were never varied: undecidable, honestly."""
    tuples = plant_table(SEED, *JUDGE_DOMINATED)
    for i, t in enumerate(tuples):
        tuples[i] = (t[0], t[1], t[2], 0, 0)
    model = _noise_model(tuples, claimed_delta=25.0)
    findings = ClaimedDeltaCheck().run(model)
    assert len(findings) == 1
    assert "cannot be judged" in findings[0].title
    assert findings[0].confidence is Confidence.LOW


def test_noise002_cant_tell_without_repeats():
    single = [(f"t{i:03d}", 70.0 + i, 0, 0, 0) for i in range(20)]
    model = _noise_model(single, claimed_delta=1.0)
    findings = ClaimedDeltaCheck().run(model)
    assert len(findings) == 1
    assert "cannot be judged" in findings[0].title


def test_noise002_silent_without_claim_or_data():
    model = _noise_model(plant_table(SEED, *JUDGE_DOMINATED))
    assert ClaimedDeltaCheck().run(model) == []
    empty = make_model()
    empty.claimed_delta = 1.0
    assert ClaimedDeltaCheck().run(empty) == []


def test_model_wiring_defaults():
    from evalwarden.model import IntegrityModel

    model = IntegrityModel(eval_id="e", adapter_name="t", adapter_version="0")
    assert model.run_scores == []
    assert model.claimed_delta is None
    record = RunItemScore(task_id="t1", score=0.5)
    assert (record.generation_index, record.grade_index, record.environment_index) == (0, 0, 0)
