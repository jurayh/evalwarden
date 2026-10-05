"""CLI: evalwarden audit | demo | explain. A linter-style interface.

Exit codes: 0 = policy passes, 1 = findings cross the --fail-on threshold,
2 = the audit could not complete.
"""
from __future__ import annotations

from pathlib import Path

import typer

import evalwarden
from .adapters import AuditError
from .checks import BY_ID
from .engine import AuditResult, audit_with_policy
from .reporters import render_html, render_json, render_sarif, render_terminal
from .reporters.report_card import CardEntry, render_index, render_report_card, slugify

app = typer.Typer(
    help="A linter for agent evaluations. Not another eval framework.",
    no_args_is_help=True,
)


@app.command()
def audit(
    path: Path = typer.Argument(..., help="Path to the eval artifact directory or a native Inspect .eval log."),
    adapter: str = typer.Option("auto", help="Adapter to use: auto, inspect, promptfoo."),
    output: Path = typer.Option(
        Path("evalwarden-report.html"), help="Where to write the self-contained HTML report."
    ),
    output_format: str = typer.Option(
        "text",
        "--format",
        help="Findings format written to stdout: text, json, sarif. Presentation only; exit codes are unchanged.",
    ),
    json_output: Path | None = typer.Option(
        None, "--json", help="Also write machine-readable JSON findings to this file."
    ),
    fail_on: str = typer.Option(
        "high", help="Minimum finding severity that fails the audit: error|high|medium|low."
    ),
    price_in: float = typer.Option(3.0, help="Estimated USD per 1M input tokens."),
    price_out: float = typer.Option(15.0, help="Estimated USD per 1M output tokens."),
    budget_per_task: float | None = typer.Option(
        None,
        "--budget-per-task",
        help="Optional per-task budget in USD: report how many tasks passed within it.",
    ),
) -> None:
    """Audit an eval artifact and write an integrity report."""
    if fail_on not in ("error", "high", "medium", "low"):
        typer.echo("error: --fail-on must be one of error|high|medium|low", err=True)
        raise typer.Exit(2)
    if output_format not in ("text", "json", "sarif"):
        typer.echo("error: --format must be one of text|json|sarif", err=True)
        raise typer.Exit(2)
    try:
        result, policy_failed = audit_with_policy(
            path,
            adapter_name=adapter,
            fail_on=fail_on,
            price_in_per_1m=price_in,
            price_out_per_1m=price_out,
            budget_per_task_usd=budget_per_task,
        )
    except AuditError as exc:
        typer.echo(f"error: audit could not complete: {exc}", err=True)
        raise typer.Exit(2)
    # Machine formats own stdout: status notes go to stderr so the document
    # on stdout stays parseable. Exit codes are identical in every format.
    machine = output_format != "text"
    if output_format == "json":
        typer.echo(render_json(result))
    elif output_format == "sarif":
        # Locations are artifact-relative; for a single-file artifact (a
        # native .eval log) the base is its parent directory.
        sarif_base = str(path.parent) if path.is_file() else str(path)
        typer.echo(render_sarif(result, base_uri=sarif_base))
    else:
        typer.echo(render_terminal(result))
    output.write_text(render_html(result, evalwarden.__version__), encoding="utf-8")
    typer.echo(f"Report: {output}", err=machine)
    if json_output is not None:
        json_output.write_text(render_json(result), encoding="utf-8")
        typer.echo(f"JSON: {json_output}", err=machine)
    raise typer.Exit(1 if policy_failed else 0)


def _demo_dir(fixture: str = "leaky") -> Path:
    here = Path(evalwarden.__file__).resolve().parent  # .../evalwarden
    candidates = [
        here / "demo" / fixture,  # installed package data (pip install .)
        here.parent.parent / "demo" / fixture,  # editable install, repo root
        Path.cwd() / "demo" / fixture,  # launched from a checkout
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise AuditError(f"demo fixture not found (expected demo/{fixture}/ next to the repo root)")


_DEMO_BEATS = {
    "leaky": [
        "beat 1 -- run: the cheating solver reports 3/3 PASS (see demo/leaky/run.json).",
        "beat 2 -- audit: the linter shows why that score is invalid.",
        "",
        "beat 3 -- harden: see demo/hardened/ for the fixed eval (opaque IDs, no gold access).",
    ],
    "hardened": [
        "beat 1 -- run: the genuine solver passes 3/3 (see demo/hardened/run.json).",
        "beat 2 -- audit: no blocking findings; the score stands.",
    ],
    "judge_bad": [
        "beat 1 -- run: a model judge scores 25 pairwise comparisons (see demo/judge_bad/judge_run.json).",
        "beat 2 -- audit: the linter finds an unvalidated judge, an AB-only protocol,",
        "         self-contradicting repeats, 48% reference agreement, and position/verbosity bias.",
        "",
        "beat 3 -- harden: see demo/judge_clean/ for the validated judge (all findings clear).",
    ],
    "judge_clean": [
        "beat 1 -- run: a validated model judge scores 25 pairwise comparisons.",
        "beat 2 -- audit: counterbalanced, calibrated, consistent -- no findings; the score stands.",
    ],
    "cost_wasteful": [
        "beat 1 -- run: an agent burns retries, fails expensively, and one attempt loops",
        "         (see demo/cost_wasteful/run.json).",
        "beat 2 -- audit: the linter prices the waste -- retry multiplier, failed spend,",
        "         and a runaway attempt (COST-002, COST-003, COST-004).",
        "",
        "beat 3 -- harden: see demo/cost_clean/ for the same tasks solved in one try each.",
    ],
    "cost_clean": [
        "beat 1 -- run: the same tasks, solved first try at modest token cost.",
        "beat 2 -- audit: no cost findings; cost per success is what the pass rate implies.",
    ],
    "promptfoo_bad": [
        "beat 1 -- run: a Promptfoo support-ticket classifier eval, config plus results.json.",
        "beat 2 -- audit: TASK_ID and RUN_ID planted in env (ENV-001), and llm-rubric "
        "assertions graded with no labeled calibration set and no anchored scale (JUDGE-001).",
    ],
    "promptfoo_clean": [
        "beat 1 -- run: the same Promptfoo eval, innocuous env, deterministic assertions only.",
        "beat 2 -- audit: zero findings; token usage and latency fully reported.",
    ],
}


@app.command()
def demo(
    output: Path = typer.Option(
        Path("evalwarden-demo-report.html"), help="Where to write the demo HTML report."
    ),
    fixture: str = typer.Option(
        "leaky", help="Demo fixture: leaky, hardened, judge_bad, judge_clean, cost_wasteful, cost_clean, promptfoo_bad, promptfoo_clean."
    ),
    budget_per_task: float | None = typer.Option(
        None,
        "--budget-per-task",
        help="Optional per-task budget in USD: report how many tasks passed within it.",
    ),
) -> None:
    """Run a demo: audit a deliberately broken (or fixed) eval fixture."""
    if fixture not in _DEMO_BEATS:
        typer.echo(
            f"error: unknown fixture {fixture!r} (known: {', '.join(sorted(_DEMO_BEATS))})",
            err=True,
        )
        raise typer.Exit(2)
    try:
        fixture_dir = _demo_dir(fixture)
        result, policy_failed = audit_with_policy(fixture_dir, budget_per_task_usd=budget_per_task)
    except AuditError as exc:
        typer.echo(f"error: demo could not run: {exc}", err=True)
        raise typer.Exit(2)
    for beat in _DEMO_BEATS[fixture]:
        typer.echo(beat)
    typer.echo("")
    typer.echo(render_terminal(result))
    output.write_text(render_html(result, evalwarden.__version__), encoding="utf-8")
    typer.echo(f"Report: {output}")
    if policy_failed:
        typer.echo("demo fixture has findings by design; the exit code stays 0 (use `audit` for policy gates)")


@app.command()
def explain(check_id: str = typer.Argument(..., help="Check ID, e.g. ENV-001.")) -> None:
    """Explain a check: the threat, what evidence it needs, how to fix it."""
    check = BY_ID.get(check_id.upper())
    if check is None:
        known = ", ".join(sorted(BY_ID))
        typer.echo(f"unknown check {check_id!r} (known: {known})", err=True)
        raise typer.Exit(2)
    meta = check.meta
    typer.echo(f"{meta.id}: {meta.title}")
    typer.echo("")
    typer.echo(f"threat: {meta.threat}")
    typer.echo("")
    typer.echo(f"fix: {meta.remediation}")


def _audit_or_exit(path: Path, budget_per_task: float | None = None) -> AuditResult:
    try:
        result, _ = audit_with_policy(path, budget_per_task_usd=budget_per_task)
    except AuditError as exc:
        typer.echo(f"error: audit could not complete: {exc}", err=True)
        raise typer.Exit(2)
    return result


@app.command("report-card")
def report_card(
    path: Path = typer.Argument(..., help="Path to the eval artifact directory or a native Inspect .eval log."),
    output: Path = typer.Option(
        Path("evalwarden-report-card.html"), help="Where to write the report card."
    ),
) -> None:
    """Write a shareable one-page integrity report card for an eval."""
    result = _audit_or_exit(path)
    output.write_text(render_report_card(result, evalwarden.__version__), encoding="utf-8")
    typer.echo(f"Report card: {output} ({result.score}/100 {result.verdict})")


@app.command("report-cards")
def report_cards(
    paths: list[Path] = typer.Argument(
        None, help="Eval artifact directories to card (or use --fixtures)."
    ),
    fixtures: str = typer.Option(
        "", "--fixtures", help="Comma-separated built-in demo fixture names."
    ),
    output_dir: Path = typer.Option(
        Path("report-cards"), help="Directory for the cards and index.html."
    ),
    budget_per_task: float | None = typer.Option(
        None,
        "--budget-per-task",
        help="Optional per-task budget in USD for the cost breakdown.",
    ),
) -> None:
    """Write report cards for several evals, plus an index page listing them."""
    targets: list[Path] = list(paths or [])
    for name in [f.strip() for f in fixtures.split(",") if f.strip()]:
        if name not in _DEMO_BEATS:
            typer.echo(
                f"error: unknown fixture {name!r} (known: {', '.join(sorted(_DEMO_BEATS))})",
                err=True,
            )
            raise typer.Exit(2)
        try:
            targets.append(_demo_dir(name))
        except AuditError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(2)
    if not targets:
        typer.echo("error: give eval paths or --fixtures", err=True)
        raise typer.Exit(2)

    output_dir.mkdir(parents=True, exist_ok=True)
    entries: list[CardEntry] = []
    seen: set[str] = set()
    for target in targets:
        result = _audit_or_exit(target, budget_per_task)
        stem = slugify(result.model.eval_id)
        # Keep filenames unique when two evals slug to the same stem.
        suffix, candidate = 2, stem
        while candidate in seen:
            candidate = f"{stem}-{suffix}"
            suffix += 1
        seen.add(candidate)
        filename = f"{candidate}.html"
        (output_dir / filename).write_text(
            render_report_card(result, evalwarden.__version__), encoding="utf-8"
        )
        entries.append(
            CardEntry(
                eval_id=result.model.eval_id,
                verdict=result.verdict,
                score=result.score,
                filename=filename,
                generated_at=_utcnow(),
            )
        )
        typer.echo(f"  {filename}: {result.model.eval_id} ({result.score}/100 {result.verdict})")
    (output_dir / "index.html").write_text(
        render_index(entries, evalwarden.__version__), encoding="utf-8"
    )
    typer.echo(f"Index: {output_dir / 'index.html'}")


def _utcnow() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


if __name__ == "__main__":
    app()
