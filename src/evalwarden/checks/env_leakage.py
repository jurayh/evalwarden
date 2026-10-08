"""ENV-001: eval-detection signals or leaked state visible to the agent.

If the agent can observe that it is being evaluated -- run IDs, task IDs,
agent tokens -- or can read answer-bearing material, it can condition its
behavior on the measurement instead of the task. The score then measures the
leak, not the capability.

Precision note: exact known-bad names are ERROR/high-confidence. Heuristic
token matches are HIGH/medium-confidence and must name the variable so a
human can dismiss them. Token matching uses word boundaries so that, e.g.,
INTERVAL is never flagged for containing EVAL.
"""
from __future__ import annotations

import re

from ..model import Confidence, Finding, IntegrityModel, Severity, SourceLocation
from .base import Check, CheckMeta

# Names that are, by themselves, evidence the agent can detect evaluation.
# Mirrors the Vivaria env-var leakage class (RUN_ID, TASK_ID, AGENT_TOKEN, ...).
EXACT_BAD_NAMES = frozenset(
    {
        "RUN_ID",
        "TASK_ID",
        "AGENT_TOKEN",
        "AGENT_ID",
        "EVAL_ID",
        "EVAL_RUN_ID",
        "EVAL_NAME",
    }
)

# Heuristic tokens: suspicious, but a human should confirm. Matched on token
# boundaries only.
HEURISTIC_TOKENS = ("TASK_ID", "EVAL", "BENCHMARK", "GOLD", "ORACLE", "ANSWER", "SOLUTION", "REFERENCE")
_TOKEN_RE = re.compile(r"(?:^|_)(%s)(?:_|$)" % "|".join(HEURISTIC_TOKENS))

# Path segments that indicate answer-bearing material.
ANSWER_PATH_SEGMENTS = frozenset(
    {"gold", "golds", "golden", "answer", "answers", "oracle", "solution", "solutions", "reference", "expected"}
)


def _heuristic_hit(name: str) -> str | None:
    m = _TOKEN_RE.search(name.upper())
    return m.group(1) if m else None


class EnvLeakageCheck(Check):
    meta = CheckMeta(
        id="ENV-001",
        title="Eval-detection signal or leaked state visible to the agent",
        threat=(
            "If the agent can observe run IDs, task IDs, or agent tokens -- or read "
            "answer-bearing material such as gold patches -- it can condition its "
            "behavior on the measurement instead of the task. The resulting score "
            "measures the leak, not the capability."
        ),
        remediation=(
            "Remove eval-detection variables from the agent's environment (keep them "
            "harness-side only), mount gold/answer material so the agent cannot read "
            "it, and re-run."
        ),
    )

    def run(self, model: IntegrityModel) -> list[Finding]:
        findings: list[Finding] = []
        for name in model.environment.env_vars:
            upper = name.upper()
            if upper in EXACT_BAD_NAMES:
                findings.append(
                    Finding(
                        id=self.meta.id,
                        title=f"Eval-detection variable visible to agent: {name}",
                        severity=Severity.ERROR,
                        confidence=Confidence.HIGH,
                        description=self.meta.threat,
                        evidence=[
                            f"environment variable {name!r} is visible to the agent/solver.",
                            "Known eval-detection signal: an agent can branch on this value "
                            "(e.g. look up a gold patch by task ID).",
                        ],
                        locations=[SourceLocation(file=model.artifact_file("environment.json"), excerpt=f"env.{name}")],
                        remediation=self.meta.remediation,
                    )
                )
                continue
            hit = _heuristic_hit(name)
            if hit:
                findings.append(
                    Finding(
                        id=self.meta.id,
                        title=f"Suspicious environment variable visible to agent: {name}",
                        severity=Severity.HIGH,
                        confidence=Confidence.MEDIUM,
                        description=self.meta.threat,
                        evidence=[
                            f"environment variable {name!r} is visible to the agent/solver.",
                            f"Name contains the token {hit!r}, which often marks eval-detection "
                            "or answer-bearing state. Confirm whether the agent needs this; "
                            "rename or hide it if not.",
                        ],
                        locations=[SourceLocation(file=model.artifact_file("environment.json"), excerpt=f"env.{name}")],
                        remediation=self.meta.remediation,
                    )
                )
        for mount in model.environment.mounts:
            if mount.agent_access not in ("read", "write"):
                continue
            # Tokenize the path on separators and on _ - . so that
            # "gold_map.json" matches "gold" but "goldenretriever.py" does not.
            tokens = set()
            for segment in mount.path.lower().split("/"):
                tokens.update(re.split(r"[_\-.]+", segment))
            leaked = sorted(tokens & ANSWER_PATH_SEGMENTS)
            if leaked:
                findings.append(
                    Finding(
                        id=self.meta.id,
                        title=f"Answer-bearing material mounted where the agent can read it: {mount.path}",
                        severity=Severity.ERROR,
                        confidence=Confidence.HIGH,
                        description=(
                            "The agent has read access to files whose path indicates reference "
                            "answers (gold patches, oracles, expected outputs). A solver can "
                            "submit these directly instead of solving the task."
                        ),
                        evidence=[
                            f"mount {mount.path!r} is readable by the agent (mode={mount.mode}).",
                            f"Path segment(s) {leaked} indicate answer-bearing material.",
                        ],
                        locations=[SourceLocation(file=model.artifact_file("environment.json"), excerpt=f"mounts[] -> {mount.path}")],
                        remediation=self.meta.remediation,
                    )
                )
        return findings


CHECKS: list[Check] = [
    EnvLeakageCheck(),  # ENV-001
]
