"""JUDGE-009 (panel blind spots) and DATA-003 (elicitation coverage).

Kill-criterion proof: a seeded synthetic panel with two planted blind
spots (every judge below the 70% catch bar on verbosity-gaming and
sycophancy), two covered modes as controls, a 3-item thin mode, and a
declared-but-absent taxonomy mode. JUDGE-009 must recover the planted
blind spots with zero false alarms on the covered modes; DATA-003 must
flag the thin and absent modes and nothing else.
"""
from __future__ import annotations

import random

from evalwarden.checks.data import ElicitationCoverageCheck
from evalwarden.checks.judge import PanelBlindSpotCheck
from evalwarden.model import Confidence, Grader, Judgment, Severity, TaskSample

from .conftest import make_judge_grader, make_judgment, make_model

SEED = 20261003
JUDGES = ("judge-a", "judge-b", "judge-c")


def _flip(label: str) -> str:
    return "sol-b" if label == "sol-a" else "sol-a"


def _plant_panel(seed: int, mode_specs: dict) -> tuple[list[TaskSample], dict, list[Judgment]]:
    """mode_specs: mode -> (n_items, {judge_id: catch_rate}).

    Builds tasks, reference labels, and judgments whose per-(judge, mode)
    agreement with the labels is exactly the requested catch rate.
    """
    rng = random.Random(seed)
    tasks: list[TaskSample] = []
    labels: dict[str, str] = {}
    judgments: list[Judgment] = []
    for mode, (n_items, rates) in mode_specs.items():
        caught: dict[str, set[int]] = {
            jid: set(rng.sample(range(n_items), round(rate * n_items)))
            for jid, rate in rates.items()
        }
        for i in range(n_items):
            task_id = f"{mode}-{i:03d}"
            label = "sol-a" if i % 2 == 0 else "sol-b"
            tasks.append(
                TaskSample(id=task_id, prompt=f"{mode} item {i}", failure_mode=mode)
            )
            labels[task_id] = label
            for jid, hits in caught.items():
                verdict = label if i in hits else _flip(label)
                judgments.append(
                    make_judgment(task_id=task_id, winner=verdict, judge_id=jid)
                )
    return tasks, labels, judgments


def _panel_model(mode_specs: dict, taxonomy: list[str] | None = None):
    tasks, labels, judgments = _plant_panel(SEED, mode_specs)
    model = make_model(
        grader=make_judge_grader(reference_labels=labels),
        tasks=tasks,
        judgments=judgments,
    )
    model.failure_mode_taxonomy = list(taxonomy or [])
    return model


PLANTED_MODES = {
    # Planted blind spots: every judge below the 70% bar.
    "verbosity-gaming": (40, {j: r for j, r in zip(JUDGES, (0.45, 0.50, 0.40))}),
    "sycophancy": (40, {j: r for j, r in zip(JUDGES, (0.55, 0.45, 0.50))}),
    # Covered controls: at least one judge catches the mode.
    "instruction-following": (40, {j: r for j, r in zip(JUDGES, (0.90, 0.60, 0.55))}),
    "factual-recall": (40, {j: r for j, r in zip(JUDGES, (0.85, 0.85, 0.85))}),
    # Thin mode: 3 eliciting items (DATA-003's claim, not JUDGE-009's).
    "rare-typo-mode": (3, {j: 1.0 for j in JUDGES}),
}
PLANTED_TAXONOMY = [*PLANTED_MODES, "tool-misuse"]  # tool-misuse: declared, absent


def _finding_text(findings) -> str:
    return " ".join([f.title for f in findings] + [e for f in findings for e in f.evidence])


def test_kill_criterion_judge009_recovers_planted_blind_spots():
    model = _panel_model(PLANTED_MODES, PLANTED_TAXONOMY)
    findings = PanelBlindSpotCheck().run(model)
    assert len(findings) == 1
    text = _finding_text(findings)
    planted = {"verbosity-gaming", "sycophancy"}
    recovered = {m for m in planted if m in text}
    assert recovered == planted, f"recovered {recovered}, planted {planted}"
    assert len(recovered) / len(planted) >= 0.80
    # Zero false alarms on the covered controls.
    for covered in ("instruction-following", "factual-recall"):
        assert covered not in text
    # The 3-item mode is below JUDGE-009's evidence floor.
    assert "rare-typo-mode" not in text
    f = findings[0]
    assert f.id == "JUDGE-009"
    assert f.severity is Severity.MEDIUM
    assert f.confidence is Confidence.MEDIUM


def test_kill_criterion_data003_flags_thin_and_absent_only():
    model = _panel_model(PLANTED_MODES, PLANTED_TAXONOMY)
    findings = ElicitationCoverageCheck().run(model)
    assert len(findings) == 1
    text = _finding_text(findings)
    assert "rare-typo-mode" in text  # 3 items < 30
    assert "tool-misuse" in text  # declared, no items
    # Well-elicited modes stay unflagged.
    for covered in ("verbosity-gaming", "sycophancy", "instruction-following", "factual-recall"):
        assert covered not in text
    f = findings[0]
    assert f.id == "DATA-003"
    assert f.severity is Severity.MEDIUM
    assert f.confidence is Confidence.MEDIUM


def test_clean_panel_with_full_coverage_is_silent():
    clean = {
        "mode-x": (40, {j: r for j, r in zip(JUDGES, (0.85, 0.80, 0.75))}),
        "mode-y": (35, {j: r for j, r in zip(JUDGES, (0.60, 0.90, 0.65))}),
    }
    model = _panel_model(clean, ["mode-x", "mode-y"])
    assert PanelBlindSpotCheck().run(model) == []
    assert ElicitationCoverageCheck().run(model) == []


def test_judge009_silent_without_reference_labels():
    model = _panel_model(PLANTED_MODES, PLANTED_TAXONOMY)
    model.grader = make_judge_grader(reference_labels={})
    assert PanelBlindSpotCheck().run(model) == []


def test_judge009_silent_without_failure_mode_tags():
    tasks, labels, judgments = _plant_panel(SEED, PLANTED_MODES)
    untagged = [TaskSample(id=t.id, prompt=t.prompt) for t in tasks]
    model = make_model(
        grader=make_judge_grader(reference_labels=labels),
        tasks=untagged,
        judgments=judgments,
    )
    assert PanelBlindSpotCheck().run(model) == []
    assert ElicitationCoverageCheck().run(model) == []


def test_judge009_silent_for_single_judge():
    specs = {"blind-mode": (40, {"judge-a": 0.40})}
    model = _panel_model(specs)
    assert PanelBlindSpotCheck().run(model) == []


def test_judge009_silent_for_unattributed_judgments():
    tasks, labels, judgments = _plant_panel(SEED, PLANTED_MODES)
    for j in judgments:
        j.judge_id = None
    model = make_model(
        grader=make_judge_grader(reference_labels=labels),
        tasks=tasks,
        judgments=judgments,
    )
    assert PanelBlindSpotCheck().run(model) == []


def test_judge009_silent_for_pointwise_only_judgments():
    # Pointwise top scores are not comparable to reference winners; treating
    # them as verdicts would manufacture blind spots, so stay silent.
    tasks, labels, _ = _plant_panel(SEED, PLANTED_MODES)
    pointwise = [
        Judgment(
            task_id=t.id,
            scores={"sol-a": 5.0, "sol-b": 1.0},
            judge_id=jid,
        )
        for t in tasks
        for jid in JUDGES
    ]
    model = make_model(
        grader=make_judge_grader(reference_labels=labels, protocol="pointwise"),
        tasks=tasks,
        judgments=pointwise,
    )
    assert PanelBlindSpotCheck().run(model) == []


def test_judge009_ignores_non_judge_graders():
    model = _panel_model(PLANTED_MODES, PLANTED_TAXONOMY)
    model.grader = Grader(kind="script", reference_labels=model.grader.reference_labels)
    assert PanelBlindSpotCheck().run(model) == []


def test_data003_silent_when_tagged_items_below_gate():
    # A taxonomy exists, but only 10 tagged items: cannot distinguish thin
    # modes from an adapter that recorded almost no tags.
    specs = {"only-mode": (10, {j: 0.9 for j in JUDGES})}
    model = _panel_model(specs, ["only-mode", "ghost-mode"])
    assert ElicitationCoverageCheck().run(model) == []


def test_data003_thin_modes_flagged_without_taxonomy():
    specs = {
        "well-covered": (35, {j: 0.9 for j in JUDGES}),
        "thin-mode": (5, {j: 0.9 for j in JUDGES}),
    }
    model = _panel_model(specs)  # no taxonomy declared
    findings = ElicitationCoverageCheck().run(model)
    assert len(findings) == 1
    text = _finding_text(findings)
    assert "thin-mode" in text
    assert "well-covered" not in text
    assert "no items" not in text  # no taxonomy: no absent-mode claims
