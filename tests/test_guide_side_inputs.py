"""Guide side-inputs section: the shipped example files, the guide text,
and the real CLI outputs must agree with each other, so the guide cannot
rot silently. The three older paths are pinned here too: the guide's
header claims every output on the page was captured from the installed
version, so the test re-runs them and compares line for line.
"""
from __future__ import annotations

import html
import re
import shutil
from pathlib import Path

from typer.testing import CliRunner

import evalwarden
from evalwarden.cli import app

ROOT = Path(__file__).resolve().parent.parent
GUIDE = ROOT / "docs/guide.html"
EXAMPLES = ROOT / "docs/examples/side-inputs"
DEMO_LOG = ROOT / "src/evalwarden/demo/inspect_native_loop/log.eval"

runner = CliRunner()

# The guide elides findings it has already shown in full elsewhere with a
# single line; the comparison expands that line back out of the real run.
PATH3_ELISION = (
    "… two more findings in the full output: COST-001 (medium, token usage "
    "not recorded, as in Path 2) and JUDGE-001 again (medium, the rubric "
    "has no explicit scale anchors)"
)
SIDE_ELISION = (
    "… one more finding in the full output: COST-001 (medium, token usage "
    "not recorded, as in Path 2)"
)


def _guide_blocks() -> dict[str, list[str]]:
    """fname -> [plain-text <pre> bodies] in guide order."""
    text = GUIDE.read_text(encoding="utf-8")
    blocks = re.findall(
        r'<span class="fname">([^<]*)</span></div>\s*<pre>(.*?)</pre>', text, re.S
    )

    def strip(s: str) -> str:
        return html.unescape(re.sub(r"<[^>]+>", "", s)).rstrip("\n")

    out: dict[str, list[str]] = {}
    for fname, body in blocks:
        out.setdefault(fname, []).append(strip(body))
    return out


def _elide_cost_block(lines: list[str]) -> list[str]:
    """Replace the COST-001 finding block with the guide's elision line."""
    kept: list[str] = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("M COST-001"):
            while i < len(lines) and lines[i].strip() != "":
                i += 1
            kept.append(SIDE_ELISION)
            continue
        kept.append(lines[i])
        i += 1
    return kept


def _elide_path3_tail(lines: list[str]) -> list[str]:
    """Replace the COST-001 + JUDGE-001-medium findings with the elision."""
    start = next(i for i, l in enumerate(lines) if l.startswith("M COST-001"))
    return lines[:start] + [PATH3_ELISION] + [""] + lines[-3:]


def _normalize_report_line(lines: list[str]) -> list[str]:
    """The guide runs with the default --output, so its Report line reads
    the relative name; the tests pass an explicit temp path. The report
    path is a run detail, not audit content: normalize it before comparing.
    """
    return [
        "Report: evalwarden-report.html" if line.startswith("Report: ") else line
        for line in lines
    ]


def _run_audit(args: list[str]) -> str:
    result = runner.invoke(app, ["audit", *args])
    return result.output


def test_guide_header_names_the_installed_version():
    text = GUIDE.read_text(encoding="utf-8")
    assert f"evalwarden {evalwarden.__version__}" in text


def test_example_files_are_printed_verbatim_in_the_guide():
    blocks = _guide_blocks()
    for fname in ("train.jsonl", "leak_eval.jsonl", "spec.json", "guessed.jsonl"):
        content = (EXAMPLES / fname).read_text(encoding="utf-8").rstrip("\n")
        assert content in blocks.get(fname, []), fname


def test_traj004_example_fires_and_is_silent_without_spec(tmp_path: Path):
    shutil.copy(EXAMPLES / "guessed.jsonl", tmp_path / "guessed.jsonl")
    shutil.copy(EXAMPLES / "spec.json", tmp_path / "spec.json")
    out = tmp_path / "report.html"
    result = runner.invoke(
        app,
        ["audit", str(tmp_path / "guessed.jsonl"), "--spec-file",
         str(tmp_path / "spec.json"), "--output", str(out)],
    )
    assert result.exit_code == 1, result.output
    assert "TRAJ-004" in result.output
    assert "Agent second-guesses the spec" in result.output
    assert "On invalid input, raise ValueError." in result.output
    assert "the test checks parse returns None on bad input" in result.output
    assert "return None" in result.output
    silent = runner.invoke(
        app, ["audit", str(tmp_path / "guessed.jsonl"), "--output", str(out)]
    )
    assert silent.exit_code == 0, silent.output
    assert "TRAJ-004" not in silent.output


def test_data004_example_fires_both_tiers_and_is_silent_without_file(
    tmp_path: Path,
):
    shutil.copy(EXAMPLES / "leak_eval.jsonl", tmp_path / "leak_eval.jsonl")
    shutil.copy(EXAMPLES / "train.jsonl", tmp_path / "train.jsonl")
    out = tmp_path / "report.html"
    result = runner.invoke(
        app,
        ["audit", str(tmp_path / "leak_eval.jsonl"), "--training-file",
         str(tmp_path / "train.jsonl"), "--output", str(out)],
    )
    assert result.exit_code == 1, result.output
    assert "Train/eval leakage (exact): 2 contaminated win(s)" in result.output
    assert "Train/eval leakage (near-duplicate): 1 contaminated win(s)" in result.output
    assert "ft-t03-swap" in result.output
    silent = runner.invoke(
        app, ["audit", str(tmp_path / "leak_eval.jsonl"), "--output", str(out)]
    )
    assert silent.exit_code == 0, silent.output
    assert "DATA-004" not in silent.output


def test_guide_side_input_blocks_match_fresh_runs(tmp_path: Path):
    blocks = _guide_blocks()["terminal — auditing with a side input"]
    assert len(blocks) == 2
    cases = [
        ("leak_eval.jsonl", "--training-file", "train.jsonl", "DATA-004"),
        ("guessed.jsonl", "--spec-file", "spec.json", "TRAJ-004"),
    ]
    for artifact, flag, side, check_id in cases:
        shutil.copy(EXAMPLES / artifact, tmp_path / artifact)
        shutil.copy(EXAMPLES / side, tmp_path / side)
        result = runner.invoke(
            app,
            ["audit", str(tmp_path / artifact), flag, str(tmp_path / side),
             "--output", str(tmp_path / "report.html")],
        )
        assert result.exit_code == 1, result.output
        shown = _normalize_report_line(
            _elide_cost_block(result.output.rstrip("\n").splitlines())
        )
        block = next(b for b in blocks if check_id in b)
        lines = block.splitlines()
        assert lines[0].startswith("$ "), lines[0]
        assert lines[1:] == shown, check_id


def test_guide_paths_reproduce(tmp_path: Path):
    blocks = _guide_blocks()
    # Path 2: the CSV is printed in full; run it and compare line for line.
    scores = tmp_path / "scores.csv"
    scores.write_text(blocks["scores.csv"][0] + "\n", encoding="utf-8")
    result = runner.invoke(
        app, ["audit", str(scores), "--output", str(tmp_path / "r2.html")]
    )
    assert result.exit_code == 0, result.output
    shown = blocks["terminal — auditing a score table"][0].splitlines()[1:]
    assert shown == _normalize_report_line(
        result.output.rstrip("\n").splitlines()
    )
    # Path 3: the JSONL is printed in full; the guide elides the last two
    # findings, so the comparison expands the elision line.
    judged = tmp_path / "judged.jsonl"
    judged.write_text(blocks["judged.jsonl"][0] + "\n", encoding="utf-8")
    result = runner.invoke(
        app, ["audit", str(judged), "--output", str(tmp_path / "r3.html")]
    )
    assert result.exit_code == 1, result.output
    shown = blocks["terminal — auditing a JSONL stream"][0].splitlines()[1:]
    fresh = _normalize_report_line(
        _elide_path3_tail(result.output.rstrip("\n").splitlines())
    )
    assert shown == fresh
    # Path 1: the packaged demo log; audit output and report-card line.
    log = tmp_path / "log.eval"
    shutil.copy(DEMO_LOG, log)
    result = runner.invoke(
        app, ["audit", str(log), "--output", str(tmp_path / "r1.html")]
    )
    assert result.exit_code == 1, result.output
    shown = blocks["terminal — auditing a native Inspect log"][0].splitlines()[1:]
    assert shown == _normalize_report_line(
        result.output.rstrip("\n").splitlines()
    )
    card = runner.invoke(
        app, ["report-card", str(log), "--output", str(tmp_path / "card.html")]
    )
    assert card.exit_code == 0, card.output
    shown = blocks["terminal — writing a report card"][0].splitlines()[1:]
    got = card.output.rstrip("\n").splitlines()
    got = [
        "Report card: evalwarden-report-card.html (90/100 BLOCKED)"
        if line.startswith("Report card: ")
        else line
        for line in got
    ]
    assert shown == got
