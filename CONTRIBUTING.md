# Contributing to evalwarden

evalwarden is a linter for evals: it audits the measurement system around a
benchmark score, read-only, with deterministic checks and no model calls.
Contributions are welcome in three shapes — a new check, a framework
integration, and an evidence pack. This file is the on-ramp for all three.

## Development setup

```bash
git clone https://github.com/jurayh/evalwarden.git
cd evalwarden
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Requires Python 3.11+. Then:

```bash
pytest                       # the full suite; it must stay green
evalwarden demo              # audit the packaged demo fixtures
evalwarden audit path/to/eval            # audit an artifact directory
evalwarden audit logs/run.eval           # or a native Inspect .eval log
evalwarden explain JUDGE-001             # what a check is and why it fires
```

The `dev` extra installs only pytest. Runtime dependencies (typer, jinja2,
pyyaml) serve the CLI and reporters; checks and helpers are standard-library
only, and a contribution must not add another runtime dependency.

## Adding a check

The pattern every existing check follows:

1. **Pure helpers first.** Put the metric math in its own module under
   `src/evalwarden/` (see `trajectory.py`, `calibration.py`, `ensemble.py`,
   `coverage.py`, `noise.py`, `dataset.py`, `clones.py`). Helpers take plain
   data in and return plain data out — no imports from `evalwarden.model` or
   `evalwarden.checks` — so they are unit-testable in isolation.
2. **One check module.** The check class lives in its lane module under
   `src/evalwarden/checks/` (`judge.py`, `data.py`, `traj.py`, ...). It
   subclasses `Check`, sets `meta`, and implements `run(model)`:

   ```python
   class MyCheck(Check):
       meta = CheckMeta(
           id="LANE-00X",
           title="Plain-English title, shown everywhere the check appears",
           threat="Why this failure mode makes a score untrustworthy.",
           remediation="What the eval author should change.",
       )

       def run(self, model: IntegrityModel) -> list[Finding]:
           ...
   ```

   `CheckMeta` is load-bearing: `evalwarden explain <ID>`, the report cards,
   and the SARIF rule text all read it. Pick the next free ID in the lane's
   numbering.
3. **One registration line.** Add the check to the lane module's `CHECKS`
   list; `src/evalwarden/checks/__init__.py` unpacks those lists into
   `REGISTRY`. Nothing else in the engine changes.
4. **Thresholds are named constants** at the top of the check module, each
   with a comment saying why that value (see `checks/traj.py`).

Conventions that are not optional:

- **Silence beats a weak claim.** If the artifacts do not carry the fields
  your check needs, or the sample is too small to support the claim, return
  no findings. Several checks also require minimum evidence counts before
  they may fire at all. A missing signal is reported as a coverage gap, never
  as a finding.
- **Severity and confidence are separate axes.** Severity is how much a
  finding undermines the score (error / high / medium / low); confidence is
  how direct the evidence is (high / medium / low). Deterministic,
  directly-observed evidence earns HIGH severity and HIGH confidence;
  statistical signals typically land at MEDIUM / MEDIUM. Findings deduct
  from the integrity score by severity (25 / 10 / 5 / 1), so grade honestly.
- **Tests include a kill criterion.** Every check ships with a seeded
  separation test: fixtures with the failure planted at known rates, plus
  clean controls, where the check must fire on the planted cases and stay
  silent on the controls (see `tests/test_traj.py`, `tests/test_grad003.py`,
  `tests/test_panel_coverage.py`). Helper unit tests go in
  `tests/test_<helper>.py`, check tests in `tests/test_<lane>.py`, and clean
  fixtures must pass unaudited — a false alarm on a clean eval is a failing
  test, not a judgment call.

Spike first: if a proposed check cannot separate planted from clean on
seeded data, it does not get built. Say so in the issue or PR instead of
widening the scope until it fires.

## Adding an adapter or integrating a framework

Prefer not writing an adapter at all. Any harness can integrate by emitting
one of the universal importer formats, documented in the README:

- canonical JSON — a single `evalwarden.model` 1.0 document,
- JSONL — a stream of typed records (task, attempt, judgment, run_score,
  span, ...),
- CSV — a flat score table (task_id + score required).

`evalwarden audit file.jsonl` works on all three today, and the same parsers
are importable in-process (`evalwarden.canonical`: `model_from_jsonl`,
`model_from_csv`, `IntegrityModel.from_canonical_json`).

Write a real adapter (a module in `src/evalwarden/adapters/` implementing
the `Adapter` protocol — `detect` / `collect` / `normalize` — registered
with `@register` and imported by `engine.py`) only when a format carries
structure the importer formats cannot. An adapter must guarantee:

- **Read-only.** It never mutates its inputs. Adapter tests hash the input
  files before and after an audit and require identical hashes.
- **Offline and location-preserving.** No network in default mode; findings
  cite the files the evidence actually lives in.
- **No fabricated fields.** Populate only what the source artifact states.
  Never infer consumption edges, reference labels, failure modes, costs, or
  judge identity — an absent field keeps the dependent checks silent, which
  is the correct outcome.
- **Unsupported content is reported**, in the model's `unsupported` list,
  never silently dropped.
- **Versioned.** The adapter pins its own version and declares which schema
  versions of the source format it understands; anything else fails with a
  clear `AuditError`, not a partial model.

## Adding an evidence pack

The [Evidence Registry](https://github.com/jurayh/evidence-registry)
(gallery: https://jurayh.github.io/evidence-registry/) holds reproducible
audits of real benchmarks, each regenerable by a stranger from public,
pinned inputs. Packs live in that repo, not this one. Its conventions, in
brief: every input pinned by version and content hash, a builder script in
`builders/`, `verify_pack.py` passing end to end, and every card framed as
what it is — diagnostic, not a certification — with claims scoped to the
exact artifact audited.

## House rules

- **Precision over recall.** A false contamination accusation is worse than
  a missed issue. When in doubt, the check stays silent.
- **Deterministic, standard-library-only checks.** No model calls, no
  network, no randomness without a fixed seed in tests.
- **Claims stay scoped to the audited artifact.** A finding describes the
  artifact that was read, not the benchmark's authors, the framework, or
  any other version of either.
- **Tests green before a change counts.** New behavior lands with its
  tests; a red suite is not a reviewable state.

## Pull requests

Include:

- what changed and why, in plain language,
- for a new check, the separation numbers: planted rates, recovered gap /
  detection counts, and the clean-control result,
- for an adapter, the read-only evidence (hash test) and what the format
  deliberately does not populate,
- the test result (suite count before and after).

Small, single-purpose PRs review fastest. If a change touches the canonical
model or the findings/JSON contracts, call that out explicitly — both are
public contracts other tools already depend on.
