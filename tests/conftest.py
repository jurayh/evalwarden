"""Shared fixtures for the evalwarden test suite."""
from __future__ import annotations

from pathlib import Path

import pytest

import evalwarden
from evalwarden.model import Attempt, Environment, Grader, IntegrityModel, Judgment, TaskSample, TrajectoryStep

REPO_ROOT = Path(__file__).resolve().parent.parent


def _demo_path(name: str) -> Path:
    """Demo fixtures ship inside the installed package; fall back to a checkout."""
    packaged = Path(evalwarden.__file__).resolve().parent / "demo" / name
    if packaged.is_dir():
        return packaged
    checkout = REPO_ROOT / "demo" / name
    assert checkout.is_dir(), f"demo fixture missing: {name}"
    return checkout


DEMO_LEAKY = _demo_path("leaky")
DEMO_HARDENED = _demo_path("hardened")
DEMO_JUDGE_BAD = _demo_path("judge_bad")
DEMO_JUDGE_CLEAN = _demo_path("judge_clean")
DEMO_JUDGE_PANEL = _demo_path("judge_panel")
DEMO_AGENT_LOOP = _demo_path("agent_loop")
DEMO_AGENT_ORDINARY = _demo_path("agent_ordinary")
DEMO_COST_WASTEFUL = _demo_path("cost_wasteful")
DEMO_COST_CLEAN = _demo_path("cost_clean")
DEMO_PROMPTFOO_BAD = _demo_path("promptfoo_bad")
DEMO_PROMPTFOO_CLEAN = _demo_path("promptfoo_clean")


def make_model(
    env_vars: dict[str, str] | None = None,
    mounts: list | None = None,
    grader: Grader | None = None,
    attempts: list[Attempt] | None = None,
    judgments: list[Judgment] | None = None,
    tasks: list[TaskSample] | None = None,
) -> IntegrityModel:
    return IntegrityModel(
        eval_id="test-eval",
        adapter_name="test",
        adapter_version="0.0.0",
        tasks=tasks if tasks is not None else [TaskSample(id="t1", prompt="do the thing")],
        environment=Environment(env_vars=env_vars or {}, mounts=mounts or []),
        grader=grader or Grader(),
        attempts=attempts or [],
        judgments=judgments or [],
    )


def make_judge_grader(**kwargs) -> Grader:
    defaults = dict(
        kind="judge",
        judge_model="judge-7b",
        protocol="pairwise",
        counterbalanced=True,
        temperature=0.0,
        repeats=3,
        rubric_criteria=["helpfulness"],
        scale_anchors={"1": "bad", "5": "good"},
        reference_labels={f"t{i:02d}": "sol-a" for i in range(1, 13)},
    )
    defaults.update(kwargs)
    return Grader(**defaults)


def make_judgment(task_id="t01", winner="sol-a", order=("a", "b"),
                 la=600, lb=400, repeat_index=0, scores=None, confidence=None,
                 judge_id=None) -> Judgment:
    return Judgment(
        task_id=task_id,
        candidates=["sol-a", "sol-b"],
        presentation_order=[f"sol-{c}" for c in order],
        winner=winner,
        scores=scores or {},
        lengths={"sol-a": la, "sol-b": lb},
        repeat_index=repeat_index,
        confidence=confidence,
        judge_id=judge_id,
    )


def make_step(step_id="s00", tool="read_file", args=None, consumes=None,
              output=None) -> TrajectoryStep:
    return TrajectoryStep(
        step_id=step_id,
        tool=tool,
        args=dict(args) if args is not None else {},
        output=output,
        consumes=list(consumes) if consumes is not None else [],
    )


def make_attempt(task_id="t1", status="pass", spans=None, **kwargs) -> Attempt:
    return Attempt(
        task_id=task_id,
        status=status,
        spans=list(spans) if spans is not None else [],
        **kwargs,
    )


@pytest.fixture
def leaky_dir() -> Path:
    assert DEMO_LEAKY.is_dir(), "demo/leaky fixture missing"
    return DEMO_LEAKY


@pytest.fixture
def hardened_dir() -> Path:
    assert DEMO_HARDENED.is_dir(), "demo/hardened fixture missing"
    return DEMO_HARDENED
