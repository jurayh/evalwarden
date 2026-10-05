# evalwarden

**A linter for evals, not another eval framework.**

Benchmark scores get cited as proof of capability: which model to deploy, which paper to accept, which agent to buy. Almost nobody audits the benchmarks themselves. A solver can read the task ID from the environment, look up the gold answer, and report 100%. A grader can be writable by the agent it grades. A model judge can be uncalibrated, biased, and self-contradictory — while two-thirds of the dataset is dead weight every model already answers. The score looks fine. The score is meaningless.

evalwarden audits the measurement system around a score: the dataset, the evidence boundary, the grader, the run records, and the cost. It never runs your evals and never changes your harness. It reads your eval artifacts and tells you whether the score can be trusted, with file-level evidence for every finding.

![evalwarden HTML integrity report](docs/images/report-screenshot.png)

## Quickstart

```bash
pip install evalwarden
evalwarden demo
```

That audits a deliberately broken benchmark and writes `evalwarden-demo-report.html`. Open it in a browser. No model key required.

## The flagship demo: caught red-handed

A tiny synthetic coding benchmark reports **3/3 PASS**. The solver earned none of it: it reads `TASK_ID` from the environment, looks up the answer in `gold_map.json`, and submits the gold patch. The auditor flags the exact leak channels, each with file-level evidence:

- `ENV-001` — `TASK_ID`, `RUN_ID`, `AGENT_TOKEN` visible to the agent
- `ENV-001` — `gold_map.json` mounted where the agent can read it
- `GRAD-001` — the verifier is writable by the agent

Result: **0/100 BLOCKED**. Then the `hardened` fixture shows the fix: opaque IDs, no gold mounted, read-only verifier, a genuine solver. Result: **100/100 PASS**. Same benchmark, same auditor, before and after.

```bash
evalwarden demo --fixture leaky     # the cheat: 0/100 BLOCKED
evalwarden demo --fixture hardened  # the fix: 100/100 PASS
```

## The check catalog

Deterministic, high-precision checks. A linter that cries contamination on a clean eval is worse than no auditor, so every finding carries a **confidence** label and clean evals produce zero findings.

**ENV — the evidence boundary**

| ID | Check | Severity |
|----|-------|----------|
| ENV-001 | Eval-detection signal or leaked state visible to the agent | Error |

**GRAD — the grader**

| ID | Check | Severity |
|----|-------|----------|
| GRAD-001 | Verifier writable by the agent | Error |
| GRAD-002 | Grader grants credit without completion | Error |

**COST — the run records**

| ID | Check | Severity |
|----|-------|----------|
| COST-001 | Cost per success not reported (usage data missing) | Medium |
| COST-002 | Successes cost multiple attempts each (retry multiplier) | Medium |
| COST-003 | Most spend burned on attempts that never passed | Medium |
| COST-004 | Runaway attempt burned far more than a typical one | Medium |

**JUDGE — the model judge**

| ID | Check | Severity |
|----|-------|----------|
| JUDGE-001 | Model judge lacks validation (no labels, unanchored rubric, hot single-sample) | High |
| JUDGE-002 | Pairwise order not counterbalanced | High |
| JUDGE-003 | Judge contradicts itself on repeated judgments | High |
| JUDGE-004 | Judge disagrees with reference labels | High |
| JUDGE-005 | Position bias: presentation order predicts the winner | Medium |
| JUDGE-006 | Verbosity bias: longer answers win disproportionately | Medium |
| JUDGE-007 | Judge confidence miscalibrated (stated confidence does not track accuracy) | Medium |
| JUDGE-008 | Redundant judges in panel (a judge never differs from the rest) | Medium |
| JUDGE-009 | Panel blind spots: failure modes no judge in the panel catches | Medium |

**DATA — the dataset itself**

| ID | Check | Severity |
|----|-------|----------|
| DATA-001 | Dataset looks saturated (most items answered correctly by every model) | Medium |
| DATA-002 | Dataset contains near-duplicate items | Low |
| DATA-003 | Failure modes under-elicited (too few eliciting items, or a declared mode with none) | Medium |

**TRAJ — the agent's trajectory**

| ID | Check | Severity |
|----|-------|----------|
| TRAJ-001 | Trajectory repeats identical tool calls (exact loops) | High |
| TRAJ-002 | Tool outputs consumed by nothing downstream | High |

**NOISE — the noise budget**

| ID | Check | Severity |
|----|-------|----------|
| NOISE-001 | One noise source dominates the eval budget (sampling, judge, or environment) | Medium |
| NOISE-002 | Claimed score change inside the noise floor, or unjudgeable at this protocol | Medium |

`evalwarden explain COST-004` prints any check's threat model, evidence, and fix.

## How evalwarden differs

**Eval frameworks run evals. evalwarden audits them.** Inspect AI, Promptfoo, and lm-eval-harness execute tasks and compute scores. evalwarden never executes anything: read-only adapters translate harness artifacts into a framework-neutral integrity model, and checks only ever see that model. That boundary is what keeps this a linter instead of another framework.

**Leaderboards rank models. evalwarden grades the tests.** A leaderboard tells you who won on a benchmark. evalwarden tells you whether the benchmark was worth winning on — whether the dataset still discriminates, the judge is calibrated, and the evidence boundary held.

**One-off audit notebooks don't run in CI. These checks do.** A notebook audit is a snapshot that rots. evalwarden's checks are deterministic rules with policy exit codes (`0` passes, `1` findings cross `--fail-on`), so the same audit runs on every commit.

**Precision over recall, always.** A false contamination accusation is worse than a missed issue. Checks stay silent when the evidence is thin, every finding carries a confidence label, and every card says **"Diagnostic, not a certification"** — the report says "no blocking findings observed under this policy," never "certified safe."

## Proof on real benchmarks

The linter runs against real public benchmarks, translated mechanically from their published definitions: pinned sources, no invented traces, provenance committed alongside. Cards live in [`examples/report-cards/real/`](examples/report-cards/real/):

| Benchmark | Result |
|-----------|--------|
| [SWE-bench Verified](examples/report-cards/real/swe-bench-verified-via-inspect-evals.html) (via inspect_evals) | **100/100 PASS, zero findings.** Gold patches, test patches, and grading stay harness-side where the agent cannot reach them. It does not manufacture problems. |
| [HealthBench](examples/report-cards/real/healthbench-via-inspect-evals.html) (via inspect_evals) | **85/100 BLOCKED.** The model judge ships with no calibration set in the eval definition itself. (The authors validated their grader in the paper through a separate meta-eval task — the linter flags precisely the gap: anyone auditing the artifact alone cannot verify the judge.) |
| MMLU vs GPQA (DATA-001 saturation) | **67% of MMLU items are dead** — answered correctly by every model tested — vs **7% on GPQA**. Two-thirds of the slice measures nothing; the check says so with a number. |

The [reading guide](examples/report-cards/real/README.md) walks through the cards, the translation, and the scope limits.

## Evidence registry

Report cards live on as versioned, independently regenerable evidence in the [Evalwarden Evidence Registry](https://github.com/jurayh/evidence-registry): each entry is an evidence pack binding an executable audit to pinned public inputs and a one-command regeneration, so a stranger can verify the card.

## How it works

```
eval artifact/ ──▶ adapter (inspect | promptfoo) ──▶ integrity model ──▶ checks ──▶ report
     read-only, offline                          data · boundary · grader · runs
```

Two adapters ship: `inspect` for Inspect-style eval artifact directories, and `promptfoo` for Promptfoo's `promptfooconfig.yaml` plus the JSON export from `promptfoo eval --output results.json`. Both are strictly read-only and offline; variable names are kept for analysis while secret values never enter the normalized model. A new check is one module plus one registration line; a new reporter is one module plus one import.

## CLI

```
evalwarden audit <eval-artifact> [--adapter auto|inspect|promptfoo] [--output report.html]
                              [--format text|json|sarif] [--json findings.json] [--fail-on high]
                              [--price-in 3.0] [--price-out 15.0]
                              [--budget-per-task USD]
evalwarden demo [--fixture leaky|hardened|judge_bad|judge_clean|cost_wasteful|cost_clean|promptfoo_bad|promptfoo_clean]
             [--output evalwarden-demo-report.html] [--budget-per-task USD]
evalwarden report-card <eval-artifact> [--output card.html]
evalwarden report-cards <eval...> [--fixtures a,b] [--output-dir cards/]
evalwarden explain <CHECK-ID>
```

Exit codes: `0` policy passes, `1` findings cross `--fail-on`, `2` the audit could not complete. The same policy runs locally and as a CI gate.

## Reports

Self-contained HTML (inline CSS, no JavaScript, no remote assets), terminal output, JSON findings, SARIF, and report cards. Secret values are never stored — only variable *names* enter the model. Integrity scores are diagnostic, not a certification.

## Machine-readable output and CI

`--format` selects what `audit` writes to stdout. It is presentation only: the exit codes (`0` pass, `1` findings cross `--fail-on`, `2` audit could not complete) are identical in every format, so the same command gates a CI job locally and remotely.

```bash
evalwarden audit evals/my-eval --format json     # JSON findings document
evalwarden audit evals/my-eval --format sarif    # SARIF 2.1.0 log
evalwarden audit evals/my-eval --json findings.json  # same JSON, to a file
```

In the machine formats, stdout carries only the document; status notes go to stderr.

### JSON schema

The JSON document is a versioned public contract (`schema: evalwarden.findings`, `schema_version: 1.0`):

| Field | Meaning |
|-------|---------|
| `tool` | `{"name": "evalwarden", "version": ...}` |
| `eval_id`, `adapter` | The audited eval and the adapter that read it |
| `verdict`, `score` | `PASS`/`BLOCKED` and the 0–100 diagnostic score |
| `findings[]` | One record per finding: `check_id`, `severity`, `confidence`, `title`, `message`, `evidence`, `locations` (`file`, optional `line`/`excerpt`), `remediation`, and a stable `fingerprint` for baselining across runs |
| `unsupported` | Artifact fields the adapter saw but could not translate (coverage gaps, never silently dropped) |

### SARIF

`--format sarif` emits a SARIF 2.1.0 log: every registered check becomes a rule (descriptions and help are the same text `evalwarden explain` prints), and every finding becomes a result — `error`/`high` severity maps to SARIF `error`, `medium` to `warning`, `low` to `note`. Locations point at the audited artifact file (prefix the path you passed to `audit`; pass a repo-relative path in CI so code scanning can resolve them). Findings with no file get a logical location naming the eval — a file or line is never invented.

### GitHub Action

The repo ships a composite Action (`action.yml`) that installs evalwarden from PyPI, runs the audit, uploads SARIF to code scanning, and then fails the job with the audit's exit code:

```yaml
name: eval-audit
on: [pull_request]
permissions:
  contents: read
  security-events: write  # required for the SARIF upload
jobs:
  evalwarden:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: jurayh/evalwarden@v0.9.0
        with:
          path: evals/my-eval   # eval artifact directory
          # format: sarif       # sarif | json | text (default sarif)
          # version: 0.9.0      # PyPI version pin (default: latest)
          # fail-on: high       # error | high | medium | low
```

Findings appear as code-scanning alerts on the PR. The SARIF upload runs even when the audit fails its policy, so a blocking finding never hides the evidence; the job then fails with the audit's exit code, also exposed as the `exit-code` output.

## Non-goals

Running or scheduling evaluations, replacing task/solver/scorer APIs, trace observability, generic red-teaming, public leaderboards, declaring any benchmark contamination-free.

## Development

```bash
pip install -e ".[dev]"
pytest
```

The test suite is the product's credibility: every rule has positive, negative, and precision fixtures (clean evals must *not* be flagged), the adapter has a read-only contract test, and the fixtures run end to end.
