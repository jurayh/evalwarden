"""Normalized integrity model: the framework-neutral description of an evaluation.

Adapters translate harness-specific artifacts into this model. Checks only ever
see this model, never harness internals. That boundary is what keeps this tool
a linter instead of another eval framework.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum


class Severity(str, Enum):
    """How much a finding undermines the reported score."""

    ERROR = "error"  # Direct evidence the score can be invalid.
    HIGH = "high"  # A concrete exploit path exists.
    MEDIUM = "medium"  # Weakens trust; needs context to judge.
    LOW = "low"  # Hygiene note.


class Confidence(str, Enum):
    """Every finding carries a confidence label. Precision over recall."""

    HIGH = "high"  # Deterministic, directly observed evidence.
    MEDIUM = "medium"  # Strong heuristic signal; corroborate before blocking.
    LOW = "low"  # Weak/indirect signal; advisory only.


SEVERITY_ORDER = [Severity.ERROR, Severity.HIGH, Severity.MEDIUM, Severity.LOW]


@dataclass
class SourceLocation:
    file: str
    line: int | None = None
    excerpt: str | None = None

    def render(self) -> str:
        base = self.file
        if self.line is not None:
            base += f":{self.line}"
        if self.excerpt:
            base += f" -- {self.excerpt}"
        return base


@dataclass
class Finding:
    id: str  # Stable rule ID, e.g. "ENV-001".
    title: str
    severity: Severity
    confidence: Confidence
    description: str  # Why it matters.
    evidence: list[str] = field(default_factory=list)
    locations: list[SourceLocation] = field(default_factory=list)
    remediation: str = ""
    fingerprint: str = ""  # Stable hash for baselining across runs.

    def __post_init__(self) -> None:
        if not self.fingerprint:
            key = "|".join(
                [
                    self.id,
                    self.title,
                    ",".join(sorted(loc.file for loc in self.locations)),
                ]
            )
            self.fingerprint = hashlib.sha256(key.encode()).hexdigest()[:16]


@dataclass
class Mount:
    path: str
    mode: str = "ro"  # "ro" | "rw"
    agent_access: str = "read"  # "none" | "read" | "write"


@dataclass
class Environment:
    # Names of env vars visible to the agent/solver. Values are NEVER stored:
    # secret values must not end up in reports.
    env_vars: dict[str, str] = field(default_factory=dict)  # name -> "<redacted>"
    mounts: list[Mount] = field(default_factory=list)


@dataclass
class TaskSample:
    id: str
    prompt: str
    # The sample's target (answer key) and, for choice tasks, its choices,
    # when the harness records them. Adapters preserve them verbatim; no
    # check treats a target as a human reference label on its own.
    target: str | list[str] | None = None
    choices: list[str] | None = None
    # Failure mode this item elicits, e.g. "verbosity-gaming". Adapters set
    # it when the eval tags items by failure mode; the JUDGE-009 / DATA-003
    # coverage checks read it and stay silent without tags.
    failure_mode: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class Grader:
    kind: str = "script"  # "script" | "judge" (model-as-judge)
    verifier_path: str | None = None
    verifier_writable_by_agent: bool = False
    # The scoring rule the artifact declares its verifier implements, e.g.
    # "exact_string" | "set_match". A verifier is code and this suite never
    # executes artifact code; what an artifact can honestly offer is the
    # rule it declares, which the clone audit (GRAD-003) re-scores.
    # None when the artifact declares no rule: the audit then stays silent.
    verifier_rule: str | None = None
    accepts_empty_output: bool = False
    tests: list[str] = field(default_factory=list)
    # Model-judge configuration. Populated only when kind == "judge".
    judge_model: str | None = None
    judge_family: str | None = None  # model family, for self-preference analysis
    protocol: str | None = None  # "pairwise" | "pointwise" | "listwise"
    counterbalanced: bool | None = None  # pairwise: was presentation order randomized?
    temperature: float | None = None
    repeats: int = 1  # judgments collected per item
    rubric_criteria: list[str] = field(default_factory=list)
    scale_anchors: dict[str, str] = field(default_factory=dict)  # anchor -> description
    # Reference (human) labels the judge is validated against: task_id -> label.
    # For pairwise judgments the label is the winning candidate id; for
    # pointwise it is the expected score or pass/fail as a string.
    reference_labels: dict[str, str] = field(default_factory=dict)


@dataclass
class Judgment:
    """One recorded model-judge verdict, structured.

    Raw judgment text (rationales, chain-of-thought) is NEVER stored here:
    adapters redact it at the boundary. Only the structured verdict survives,
    which is all the integrity checks need.
    """

    task_id: str
    candidates: list[str] = field(default_factory=list)  # canonical (sorted) ids
    presentation_order: list[str] = field(default_factory=list)  # as shown to the judge
    winner: str | None = None  # pairwise: winning candidate id
    scores: dict[str, float] = field(default_factory=dict)  # candidate -> score
    lengths: dict[str, int] = field(default_factory=dict)  # candidate -> chars
    repeat_index: int = 0
    # Stated probability the verdict is correct, 0..1. None when the harness
    # does not record confidence (adapters populate it when available).
    confidence: float | None = None
    # Which judge in a panel produced this verdict. None for single-judge
    # evals (adapters populate it when the harness runs a multi-judge panel;
    # used by JUDGE-008 ensemble redundancy).
    judge_id: str | None = None


@dataclass
class TrajectoryStep:
    """One tool call inside an agent trajectory, structured.

    Adapters populate these from harness traces (JSONL spans). Raw
    rationales and full outputs are redacted/truncated at the boundary:
    only the structured call shape and the data-flow edges survive, which
    is all the trajectory checks need.
    """

    step_id: str  # unique within the attempt's trajectory
    tool: str  # tool name, e.g. "read_file"
    args: dict = field(default_factory=dict)  # normalized JSON-able arguments
    output: str | None = None  # redacted/truncated tool result (kept for future checks)
    consumes: list[str] = field(default_factory=list)  # step_ids whose outputs this step used


@dataclass
class Attempt:
    task_id: str
    status: str  # "pass" | "fail" | "error" | "incomplete"
    score: float | None = None
    tool_calls: int = 0
    actions: list[str] = field(default_factory=list)
    tokens_in: int | None = None
    tokens_out: int | None = None
    latency_s: float | None = None
    tries: int = 1
    empty_submission: bool = False
    # The attempt's cached output text, when the artifact records it.
    # Most adapters do not retain raw outputs; the clone audit (GRAD-003)
    # reads this and stays silent for attempts that carry none.
    output: str | None = None
    # Which model produced this attempt. Adapters set it when the eval ran
    # more than one model; the DATA-lane checks group by it. None means the
    # eval ran a single unnamed model.
    model_id: str | None = None
    # Trajectory spans, when the harness records tool-call traces. Adapters
    # populate them from JSONL spans; empty when the harness keeps no trace.
    # The TRAJ-lane checks read these and stay silent without them.
    spans: list[TrajectoryStep] = field(default_factory=list)


@dataclass
class CostSummary:
    attempts: int
    successes: int
    total_tokens_in: int
    total_tokens_out: int
    estimated_usd: float
    cost_per_success_usd: float | None
    avg_tool_calls_per_success: float | None
    price_in_per_1m: float
    price_out_per_1m: float
    # Efficiency breakdown. All derived deterministically from priced attempts;
    # "priced" means the attempt carried tokens_in/tokens_out.
    avg_tries_per_success: float | None = None  # mean tries over successful tasks
    max_tries: int | None = None  # max tries observed on any priced attempt
    wasted_usd: float = 0.0  # estimated spend on attempts that did not pass
    wasted_share: float = 0.0  # wasted_usd / estimated_usd, 0 when total is 0
    p50_tokens_out: int | None = None
    p90_tokens_out: int | None = None
    max_tokens_out: int | None = None
    max_tokens_out_task: str | None = None  # task_id behind the max burn
    # Budget-capped success: per-task spend vs an optional per-task budget.
    budget_per_task_usd: float | None = None
    tasks_total: int | None = None
    tasks_within_budget: int | None = None  # tasks passing with total cost <= budget


@dataclass
class RunItemScore:
    """Per-item score from one repeated run of the eval.

    Repeated runs are how an eval's noise becomes measurable: each score
    records which run conditions were re-rolled to produce it.
    generation_index varies over fresh generations (model sampling);
    grade_index varies over re-grades of a frozen output (judge noise);
    environment_index varies over repeated tool/environment executions
    (flakiness). Scores sharing an index tuple are the same condition;
    adapters leave all indices at 0 when the harness records no repeats,
    and the NOISE-lane checks then report total variance only -- or stay
    silent when there is nothing to estimate from.
    """

    task_id: str
    score: float
    generation_index: int = 0  # fresh-generation replicate (sampling noise)
    grade_index: int = 0  # re-grade of a frozen output (judge noise)
    environment_index: int = 0  # repeated environment execution (flakiness)


@dataclass
class IntegrityModel:
    eval_id: str
    adapter_name: str
    adapter_version: str
    tasks: list[TaskSample] = field(default_factory=list)
    # Failure modes the eval declares it covers (its elicitation taxonomy).
    # Adapters populate it from the eval definition when one exists; DATA-003
    # flags declared modes that no item elicits. Empty when the eval declares
    # no taxonomy: undeclared modes cannot be missed.
    failure_mode_taxonomy: list[str] = field(default_factory=list)
    environment: Environment = field(default_factory=Environment)
    grader: Grader = field(default_factory=Grader)
    attempts: list[Attempt] = field(default_factory=list)
    judgments: list[Judgment] = field(default_factory=list)
    # Per-item scores from repeated runs, when the harness re-runs the eval
    # under recorded conditions. Empty when the eval ran once: the NOISE
    # lane then has nothing to decompose and stays silent.
    run_scores: list[RunItemScore] = field(default_factory=list)
    # Mean score change the eval claims over a prior run or baseline
    # (new minus old, in score points), when it claims one. Adapters set it
    # from the eval's own reporting; NOISE-002 judges it against the noise
    # floor measured from run_scores. None when no improvement is claimed.
    claimed_delta: float | None = None
    # Harness fields the adapter saw but could not translate. Shown in the
    # report as coverage gaps, never silently dropped.
    unsupported: list[str] = field(default_factory=list)
    # Artifact file -> sha256, for the reproducibility manifest.
    digests: dict[str, str] = field(default_factory=dict)
    cost_summary: CostSummary | None = None
    # Maps canonical artifact filenames (the ones checks cite in evidence) to
    # the actual filenames the adapter read, e.g.
    # {"environment.json": "promptfooconfig.yaml"}. Lets adapters keep check
    # evidence pointers honest without the checks knowing harness layouts.
    file_aliases: dict[str, str] = field(default_factory=dict)

    def artifact_file(self, canonical: str) -> str:
        """The real filename behind a canonical artifact name in evidence."""
        return self.file_aliases.get(canonical, canonical)
