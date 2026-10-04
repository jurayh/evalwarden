"""Rule registry. v0.3 ships deterministic, high-precision checks only.

Sequencing rule: earn trust with deterministic evidence before adding
probabilistic signals. Every finding carries a confidence label; a linter that
cries contamination on a clean eval is worse than no auditor.
"""
from __future__ import annotations

from .base import Check
from .cost import CHECKS as COST_CHECKS
from .data import CHECKS as DATA_CHECKS
from .env_leakage import EnvLeakageCheck
from .grader import EmptyPathCheck, VerifierWritableCheck
from .judge import CHECKS as JUDGE_CHECKS
from .noise import CHECKS as NOISE_CHECKS
from .traj import CHECKS as TRAJ_CHECKS

REGISTRY: list[Check] = [
    EnvLeakageCheck(),  # ENV-001
    VerifierWritableCheck(),  # GRAD-001
    EmptyPathCheck(),  # GRAD-002
    *COST_CHECKS,  # COST-001 .. COST-004
    *JUDGE_CHECKS,  # JUDGE-001 .. JUDGE-009
    *DATA_CHECKS,  # DATA-001 .. DATA-003
    *TRAJ_CHECKS,  # TRAJ-001 .. TRAJ-002
    *NOISE_CHECKS,  # NOISE-001 .. NOISE-002
]

BY_ID: dict[str, Check] = {check.meta.id: check for check in REGISTRY}


def get_check(check_id: str) -> Check | None:
    return BY_ID.get(check_id.upper())
