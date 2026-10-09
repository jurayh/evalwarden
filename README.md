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

## What an audit looks like

Point it at a real Inspect AI log. This agent called the same tool with the same arguments five times in a row (packaged fixture: `src/evalwarden/demo/inspect_native_loop/log.eval`, generated with Inspect AI):

```console
$ evalwarden audit src/evalwarden/demo/inspect_native_loop/log.eval
evalwarden audit: evalwarden-native-loop
adapter: inspect 0.3.0
Integrity: 90 / 100 BLOCKED (0 errors, 1 high, 0 medium, 0 low)
claim blocked by: TRAJ-001

H TRAJ-001 [high|confidence:high] Trajectory loops: lookup({"query": "status"}) repeated 5x in task loop-1
    evidence: loop: lookup({"query": "status"}) x5 (longest run x5)
    at: log.eval -- attempt loop-1: 1 looped call pair(s) over 6 spans
```

Three ways in, depending on what you already have:

- **Native Inspect logs:** audit a real `.eval` file directly, no translation step. [Native Inspect `.eval` logs](#native-inspect-eval-logs)
- **Any other harness:** emit one canonical JSON, JSONL, or CSV file instead of writing an adapter. [Universal importer](#universal-importer)
- **In CI:** JSON and SARIF output, plus a GitHub Action that uploads findings to code scanning. [Machine-readable output and CI](#machine-readable-output-and-ci)

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
| GRAD-003 | Verifier verdicts depend on surface form (clone gap) | High |

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
eval artifact/ ──▶ adapter (inspect | promptfoo | universal) ──▶ integrity model ──▶ checks ──▶ report
     read-only, offline                          data · boundary · grader · runs
```

Three adapters ship: `inspect` for Inspect eval artifacts — both a native `.eval` log (see below) and the Inspect-style JSON artifact directory — `promptfoo` for Promptfoo's `promptfooconfig.yaml` plus the JSON export from `promptfoo eval --output results.json`, and `universal` for the canonical Evalwarden input formats described below. All are strictly read-only and offline; variable names are kept for analysis while secret values never enter the normalized model. A new check is one module plus one registration line; a new reporter is one module plus one import.

## Native Inspect `.eval` logs

Point `audit` (or `report-card` / `report-cards`) straight at a real Inspect AI log — no manual translation first:

```bash
evalwarden audit logs/2026-01-01T00-00-00_my-task_abc123.eval
```

A directory holding exactly one `.eval` file works too; a directory with several logs is refused, because each log is its own eval (batch them with `report-cards` instead). The archive is parsed in memory with the standard library only, nothing is extracted to disk, and the log is never modified. Malformed, unfinished, or unsupported-version logs fail with a clear error rather than a partial model.

What a native log feeds, honestly:

- **Tasks and attempts.** Samples become tasks with prompts, targets, and choices preserved, plus explicit sample metadata only — `failure_mode` appears only when the sample declares it. Each sample epoch becomes an attempt with its score, token usage, and latency as recorded. No USD cost is ever fabricated; COST checks price recorded tokens only when you pass `--price-in` / `--price-out`.
- **Trajectories (TRAJ-001 yes, TRAJ-002 no).** Tool calls become spans with the native call id, tool name, arguments, and the tool result truncated at the adapter's 512-char boundary, so repeated-call loops fire TRAJ-001. Inspect records no downstream consumption edges, so TRAJ-002 stays silent on native logs by design.
- **Noise (NOISE-001/002).** Repeated epochs of the log's single headline score become run scores with the epoch as the generation index, so sampling noise is measurable. Judge and environment axes are never fabricated, and mixed multi-scorer logs yield no noise series. `claimed_delta` stays absent unless the log's metadata declares it.
- **Judges (partial).** Scorer results map to scores and, for map-valued (per-candidate) scores, judgments with the scorer's name. Reference labels appear only when the log makes them explicit (a declared `reference_labels` structure or a score's `reference_label` metadata) — a target is an answer key, not a human label, and is never treated as one.
- **Not fed.** A native log carries no environment record (visible env vars, mounts), so ENV-001 has nothing to audit, and no declared failure-mode taxonomy, so panel-coverage checks (JUDGE-009, DATA-003) stay silent unless the log declares one. One consequence of the model: epochs land as repeated attempts per task, so COST-002 reads an `epochs: 3` protocol as 3 tries per success — the same reading repeated run entries get on the JSON path.

## Universal importer

Any harness can integrate by emitting one canonical file — no bespoke adapter and no separate import command:

```bash
evalwarden audit my-eval.evalwarden.json
evalwarden audit my-eval.jsonl
evalwarden audit scores.csv
```

The formats are versioned public contracts. Validation is strict: malformed JSON names the JSON line/column, JSONL errors name the record line and field, and CSV errors name the row and field. Unknown-but-harmless fields are reported under `unsupported`; a malformed file never imports a partial model.

### Canonical JSON (`evalwarden.model` 1.0)

One document mirrors the integrity model:

```json
{
  "schema": "evalwarden.model",
  "schema_version": "1.0",
  "eval_id": "my-eval",
  "failure_mode_taxonomy": ["verbosity-gaming"],
  "environment": {"env_vars": ["API_TOKEN"], "mounts": []},
  "grader": {"kind": "script", "verifier_rule": "set_match"},
  "tasks": [{"id": "t1", "prompt": "...", "target": "answer", "choices": null,
             "failure_mode": null, "metadata": {}}],
  "attempts": [{"task_id": "t1", "status": "pass", "score": 1.0,
                "spans": []}],
  "judgments": [],
  "run_scores": [],
  "claimed_delta": null,
  "unsupported": []
}
```

Top-level fields are `schema`, `schema_version`, `eval_id`, `failure_mode_taxonomy`, `environment`, `grader`, `tasks`, `attempts`, `judgments`, `run_scores`, `claimed_delta`, and `unsupported`. Nested objects use the integrity model's field names: tasks carry `id`, `prompt`, `target`, `choices`, `failure_mode`, and `metadata`; attempts carry status, score, usage, output, `model_id`, and nested trajectory `spans`; judgments carry verdicts plus optional `confidence` and `judge_id`; run scores carry `task_id`, `score`, and the `generation_index` / `grade_index` / `environment_index` condition tags. The grader object is flat (`kind`, verifier fields, judge fields, `scale_anchors`, and `reference_labels`). Environment values are never imported: `environment.env_vars` is a list of variable *names* only.

In Python, frameworks can skip files entirely:

```python
from evalwarden.model import IntegrityModel

model = IntegrityModel.from_canonical_json(text)       # or from_canonical_document(doc)
document = model.to_canonical_document()               # serialize back to the contract
```

`evalwarden.canonical` also exposes `model_from_jsonl` and `model_from_csv` for the other two faces.

### JSONL record stream

One JSON object per line, tagged with `type`. The first non-blank record must be `eval` (carrying `schema`, `schema_version`, `eval_id`, optional `failure_mode_taxonomy`, `claimed_delta`, and `unsupported`):

| Record type | Carries |
|---|---|
| `environment` | `env_vars` (names only) and `mounts`; at most one |
| `grader` | The flat grader fields; at most one |
| `task` | One task object |
| `attempt` | One attempt object; may carry nested `spans` |
| `judgment` | One judgment object |
| `run_score` | One repeated-run score with its condition indices |
| `span` | One trajectory step plus `task_id` and optional `attempt_index` (zero-based attempt occurrence for that task, default 0); may appear before or after its attempt |

Canonical JSON and JSONL have the same reach: with the corresponding fields present, they can feed every lane — ENV (environment/mounts), GRAD (grader, cached outputs, clone metadata), COST (attempt usage), JUDGE (grader and judgments), DATA (tasks, failure modes, model-tagged attempts), TRAJ (spans, including recorded `consumes` edges), and NOISE (run scores and `claimed_delta`). Fields the stream does not state stay absent; nothing is inferred.

### CSV score table

The CSV contract is deliberately minimal:

| Column | Required | Meaning |
|---|---|---|
| `task_id` | yes | Item the score belongs to |
| `score` | yes | Finite numeric score |
| `status` | no | `pass` / `fail` / `error` / `incomplete`; attempts are created only from rows that state it — a score alone does not state pass/fail |
| `model_id` | no | Model that produced the row (enables DATA-001 with at least two models) |
| `run_id` | no | Run label, mapped to `generation_index` in first-seen order |
| `run_index` / `generation_index` | no | Fresh-generation condition index |
| `grade_index` | no | Re-grade condition index |
| `environment_index` | no | Environment-repeat condition index |

Run scores are created only when a run or condition column states repeated-run structure, and only for a single model series — one NOISE series cannot honestly represent several models, so multi-model tables report that omission under `unsupported` instead. CSV cannot carry prompts, grader or environment records, judgments, trajectories, failure modes, claimed deltas, or data-flow edges, so it can honestly support score-based DATA-001 and NOISE-001 verdicts (and COST-001's "usage not recorded" signal when attempts exist), while checks needing the missing records stay silent and the coverage output says why.

## CLI

```
evalwarden audit <eval-artifact> [--adapter auto|inspect|promptfoo|universal] [--output report.html]
                              [--format text|json|sarif] [--json findings.json] [--fail-on high]
                              [--price-in 3.0] [--price-out 15.0]
                              [--budget-per-task USD]
evalwarden demo [--fixture leaky|hardened|judge_bad|judge_clean|cost_wasteful|cost_clean|promptfoo_bad|promptfoo_clean]
             [--output evalwarden-demo-report.html] [--budget-per-task USD]
evalwarden report-card <eval-artifact> [--output card.html]
evalwarden report-cards <eval...> [--fixtures a,b] [--output-dir cards/]
evalwarden trace [sessions-path] [--demo] [--since 7d] [--project NAME]
              [--agent all|claude-code|codex] [--output trace-report.html]
evalwarden checks
evalwarden explain <CHECK-ID>
```

Exit codes: `0` policy passes, `1` findings cross `--fail-on`, `2` the audit could not complete. The same policy runs locally and as a CI gate.

Shell completion: `evalwarden --install-completion` installs tab completion for the current shell.

## Trace observatory

`evalwarden trace` reads the Claude Code and Codex sessions already on
your disk (auto-discovered under `~/.claude/projects` and
`~/.codex/sessions`, or `--demo` for built-in sessions) and renders one
local page: every session segmented into explore / plan / edit / test /
review phases, repeated tool calls priced in dollars from the usage the
harness actually recorded, and a "what this page cannot see" section
naming what the format does not record. No account, no rerun, no
upload. Both importers are validated against real public session
corpora with exact token reconciliation.

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
          path: evals/my-eval   # eval artifact directory or native Inspect .eval log
          # format: sarif       # sarif | json | text (default sarif)
          # version: 0.9.0      # PyPI version pin (default: latest)
          # fail-on: high       # error | high | medium | low
```

Findings appear as code-scanning alerts on the PR. The SARIF upload runs even when the audit fails its policy, so a blocking finding never hides the evidence; the job then fails with the audit's exit code, also exposed as the `exit-code` output.

### Pre-commit

The repo ships a [pre-commit](https://pre-commit.com) hook definition, so the same audit can run before every commit:

```yaml
# .pre-commit-config.yaml
- repo: https://github.com/jurayh/evalwarden
  rev: v0.13.1
  hooks:
    - id: evalwarden-audit
      args: [evals/my-eval]   # your eval artifact directory or native Inspect .eval log
```

The hook runs `evalwarden audit` on the path you pass in `args` and fails the commit when findings cross the policy (default `--fail-on high`; add `--fail-on` to `args` to tune it).

## Non-goals

Running or scheduling evaluations, replacing task/solver/scorer APIs, trace observability, generic red-teaming, public leaderboards, declaring any benchmark contamination-free.

## Development

```bash
pip install -e ".[dev]"
pytest
```

The test suite is the product's credibility: every rule has positive, negative, and precision fixtures (clean evals must *not* be flagged), the adapter has a read-only contract test, and the fixtures run end to end.

Contributing a check, an adapter, or an evidence pack? Start with [CONTRIBUTING.md](CONTRIBUTING.md).
