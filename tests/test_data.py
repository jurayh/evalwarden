"""DATA-lane checks: saturation (DATA-001) and redundancy (DATA-002).

The kill-criterion proof ran on real data (MMLU via the HELM MMLU suite
v1.0.0, full test set, flagship 3-model panel, vs GPQA paper baselines):
dead fraction 0.636 vs 0.071, about 9x separation (re-verified Oct 2026;
the original HELM Classic attribution was a mislabel). These
tests pin the metric logic offline with hand-built matrices shaped like the
real outcomes, so CI stays hermetic.
"""
from __future__ import annotations

from evalwarden.checks.data import RedundancyCheck, SaturationCheck
from evalwarden.dataset import (
    near_duplicate_pairs,
    redundancy_stats,
    saturation_stats,
)
from evalwarden.model import Attempt, TaskSample

from .conftest import make_model


def _matrix(dead: int, disputed: int, all_wrong: int, models=("a", "b", "c")):
    """Build model -> task -> correct with the given item-type counts."""
    scores: dict[str, dict[str, bool]] = {m: {} for m in models}
    i = 0
    for _ in range(dead):
        for m in models:
            scores[m][f"t{i}"] = True
        i += 1
    for _ in range(disputed):
        for k, m in enumerate(models):
            scores[m][f"t{i}"] = k == (i % len(models))
        i += 1
    for _ in range(all_wrong):
        for m in models:
            scores[m][f"t{i}"] = False
        i += 1
    return scores


def _attempts_from_matrix(scores: dict[str, dict[str, bool]]) -> list[Attempt]:
    attempts = []
    for model, tasks in scores.items():
        for task_id, correct in tasks.items():
            attempts.append(
                Attempt(
                    task_id=task_id,
                    status="pass" if correct else "fail",
                    model_id=model,
                )
            )
    return attempts


def test_saturation_stats_saturated_matrix():
    stats = saturation_stats(_matrix(dead=80, disputed=15, all_wrong=5))
    assert stats is not None
    assert stats.dead_fraction == 0.80
    assert stats.disputed_fraction == 0.15
    assert stats.n_models == 3
    assert stats.n_items == 100


def test_saturation_stats_discriminating_matrix():
    stats = saturation_stats(_matrix(dead=7, disputed=54, all_wrong=39))
    assert stats is not None
    assert stats.dead_fraction == 0.07
    assert stats.disputed_fraction == 0.54
    assert stats.headroom > 0.5


def test_saturation_stats_needs_two_models():
    assert saturation_stats({"only": {"t1": True}}) is None
    assert saturation_stats({}) is None


def test_saturation_check_fires_on_saturated():
    # Shaped like the real MMLU outcome (dead ~0.64).
    model = make_model(attempts=_attempts_from_matrix(_matrix(67, 27, 6)))
    findings = SaturationCheck().run(model)
    assert len(findings) == 1
    assert findings[0].id == "DATA-001"
    assert "67%" in findings[0].title


def test_saturation_check_silent_on_discriminating():
    # Shaped like the real GPQA outcome (dead ~0.07).
    model = make_model(attempts=_attempts_from_matrix(_matrix(7, 54, 39)))
    assert SaturationCheck().run(model) == []


def test_saturation_check_silent_single_model():
    attempts = [
        Attempt(task_id=f"t{i}", status="pass", model_id="solo") for i in range(40)
    ]
    assert SaturationCheck().run(make_model(attempts=attempts)) == []


def test_saturation_check_silent_without_model_ids():
    attempts = [Attempt(task_id=f"t{i}", status="pass") for i in range(40)]
    assert SaturationCheck().run(make_model(attempts=attempts)) == []


def test_saturation_check_silent_below_min_items():
    model = make_model(attempts=_attempts_from_matrix(_matrix(20, 5, 0)))
    assert SaturationCheck().run(model) == []


def test_near_duplicate_pairs_finds_verbatim_and_near():
    prompts = {
        "a": "What is the capital of France? Answer with the city name.",
        "b": "What is the capital of France? Answer with the city name.",
        "c": "What is the capital of France? Answer with the city name!",
        "d": "Explain photosynthesis in three sentences for a child.",
    }
    hits = near_duplicate_pairs(prompts)
    pairs = {(a, b) for a, b, _ in hits}
    assert ("a", "b") in pairs
    assert ("a", "c") in pairs or ("b", "c") in pairs
    assert not any("d" in (a, b) for a, b in pairs)
    assert hits[0][2] == 1.0  # verbatim pair is most similar


def test_redundancy_stats_fraction():
    prompts = {f"t{i}": f"Completely different question number {i} about topic {i}."
               for i in range(18)}
    prompts["dup1"] = "What is the capital of France? Answer with the city name."
    prompts["dup2"] = "What is the capital of France? Answer with the city name."
    stats = redundancy_stats(prompts)
    assert stats.n_items == 20
    assert stats.duplicate_item_fraction == 0.10
    assert stats.near_duplicate_pairs >= 1


def test_redundancy_check_fires():
    tasks = [
        TaskSample(id=f"t{i}", prompt=f"Unique question {i} on unrelated topic {i}.")
        for i in range(18)
    ]
    tasks.append(TaskSample(id="d1", prompt="What is the capital of France? Name the city."))
    tasks.append(TaskSample(id="d2", prompt="What is the capital of France? Name the city."))
    findings = RedundancyCheck().run(make_model(tasks=tasks))
    assert len(findings) == 1
    assert findings[0].id == "DATA-002"


def test_redundancy_check_silent_on_clean_set():
    tasks = [
        TaskSample(id=f"t{i}", prompt=f"Unique question {i} on unrelated topic {i} words.")
        for i in range(25)
    ]
    assert RedundancyCheck().run(make_model(tasks=tasks)) == []


def test_redundancy_check_silent_below_min_tasks():
    tasks = [TaskSample(id="d1", prompt="Same prompt here."),
             TaskSample(id="d2", prompt="Same prompt here.")]
    assert RedundancyCheck().run(make_model(tasks=tasks)) == []
