"""GRAD-003 tests: declared-rule clone audit.

Kill criterion (end-to-end, over the seeded demo fixtures): the planted
shortcut verifier (declared rule exact_string) is flagged with a large
clone gap, and the honest verifier (set_match, same tasks, same cached
outputs) stays silent -- its fixture audits fully clean.
"""
from __future__ import annotations

from pathlib import Path

from evalwarden.adapters.inspect_ai import InspectAdapter
from evalwarden.checks.grader import GAP_AT, MIN_CLONE_ITEMS, CloneGapCheck
from evalwarden.engine import audit
from evalwarden.model import Attempt, Confidence, Grader, Severity, TaskSample

from .conftest import DEMO_CLONE_HONEST, DEMO_CLONE_SHORTCUT, make_model

check = CloneGapCheck()
adapter = InspectAdapter()

ENTRIES = [
    {"name": "Ana", "value": 7},
    {"name": "Boris", "value": 3},
    {"name": "Cleo", "value": 9},
    {"name": "Dario", "value": 5},
    {"name": "Mira", "value": 12},
]


def _selection_task(task_id: str, output: str | None) -> TaskSample:
    return TaskSample(
        id=task_id,
        prompt="roster",
        target="Ana, Cleo, Mira",
        metadata={
            "clone_family": "selection",
            "entries": ENTRIES,
            "threshold": 7,
        },
    )


def _model(rule: str | None, n_tasks: int = MIN_CLONE_ITEMS, output="Ana, Cleo, Mira"):
    tasks = [_selection_task(f"t{i}", output) for i in range(n_tasks)]
    attempts = [
        Attempt(task_id=t.id, status="pass", score=1.0, output=output) for t in tasks
    ]
    return make_model(
        grader=Grader(verifier_rule=rule), tasks=tasks, attempts=attempts
    )


def test_exact_string_rule_flagged():
    findings = check.run(_model("exact_string"))
    # Every item is isomorphic, so the verdicts hinge on roster order
    # surviving the clone shuffle; with MIN_CLONE_ITEMS items seeded the
    # same way as the spike fixtures, the gap clears the bar.
    assert len(findings) == 1
    finding = findings[0]
    assert finding.id == "GRAD-003"
    assert finding.severity == Severity.HIGH
    assert finding.confidence == Confidence.MEDIUM
    assert f"at or above {GAP_AT:.2f}" in " ".join(finding.evidence)


def test_set_match_rule_not_flagged():
    assert check.run(_model("set_match")) == []


def test_silent_without_declared_rule():
    assert check.run(_model(None)) == []


def test_silent_for_unsupported_rule():
    assert check.run(_model("llm_judge")) == []


def test_silent_below_min_clone_items():
    assert check.run(_model("exact_string", n_tasks=MIN_CLONE_ITEMS - 1)) == []


def test_silent_without_cached_outputs():
    assert check.run(_model("exact_string", output=None)) == []


def test_silent_for_uncloneable_tasks():
    model = _model("exact_string")
    for task in model.tasks:
        task.metadata = {}
    assert check.run(model) == []


# ------------------------------------------------------------ adapter wiring


def _normalize(path: Path):
    return adapter.normalize(adapter.collect(path))


def test_adapter_carries_declared_rule_and_cached_outputs():
    shortcut = _normalize(DEMO_CLONE_SHORTCUT)
    honest = _normalize(DEMO_CLONE_HONEST)
    assert shortcut.grader.verifier_rule == "exact_string"
    assert honest.grader.verifier_rule == "set_match"
    assert all(a.output is not None for a in shortcut.attempts)
    # The two fixtures differ only in the declared rule.
    assert [t.metadata for t in shortcut.tasks] == [t.metadata for t in honest.tasks]
    assert [a.output for a in shortcut.attempts] == [a.output for a in honest.attempts]


def test_adapter_defaults_rule_and_output_to_absent(tmp_path: Path):
    import json

    (tmp_path / "dataset.json").write_text(json.dumps({"tasks": [{"id": "t1", "prompt": "p"}]}))
    (tmp_path / "grader.json").write_text(json.dumps({"kind": "script", "verifier": {"path": "v.py"}}))
    (tmp_path / "run.json").write_text(json.dumps({"attempts": [{"task_id": "t1", "status": "pass"}]}))
    model = _normalize(tmp_path)
    assert model.grader.verifier_rule is None
    assert model.attempts[0].output is None


# ------------------------------------------------- kill criterion, end to end


def test_kill_criterion_shortcut_flagged_honest_clean():
    shortcut = audit(DEMO_CLONE_SHORTCUT)
    assert [f.id for f in shortcut.findings] == ["GRAD-003"]
    assert shortcut.verdict == "BLOCKED"
    finding = shortcut.findings[0]
    assert "0.54" in finding.title  # gap 13/24 on the seeded fixture set
    assert "13 item(s) flipped" in " ".join(finding.evidence)

    honest = audit(DEMO_CLONE_HONEST)
    assert honest.findings == []
    assert honest.verdict == "PASS"
