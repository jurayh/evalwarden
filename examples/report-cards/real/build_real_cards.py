#!/usr/bin/env python3
"""Build evalwarden audit artifacts from real public benchmark definitions.

This script translates PUBLIC, downloadable eval definitions into evalwarden's
artifact format (dataset.json / environment.json / grader.json) WITHOUT
inventing anything:

- Only facts present in the public sources are encoded. Anything the
  definition does not say is left out, and the omission is documented in
  PROVENANCE.json next to each artifact.
- No traces are fabricated: there is no run.json. Checks that need run
  records (COST-002..004, GRAD-002's attempt leg, JUDGE-003..006) simply do
  not fire, and COST-001 treats "no runs at all" as a coverage gap, not a
  finding.
- Answer-bearing material (gold patches, test patches, full rubrics) is
  EXCLUDED from the translated artifact even though it is public: the
  integrity question is whether the AGENT can see it, and the task
  definitions keep it harness-side. The exclusion is documented.

Pinned sources (all public, no auth):
- inspect_evals @ 244e43cc924d1de5a78dad7db259bbd4471c97e7
  (https://github.com/UKGovernmentBEIS/inspect_evals)
- SWE-bench_Verified: princeton-nlp/SWE-bench_Verified
  rev c104f840cc67f8b6eec6f759ebc8b2693d585d4a (pinned in the task source)
- HealthBench: https://openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_eval.jsonl
  (URL pinned in the task source)
- WritingBench: benchmark_all.jsonl inside the inspect_evals repo at the pinned
  commit (fetched via raw.githubusercontent.com at that commit)
- MMLU: cais/mmlu (config "all", test split) rev
  c30699e8356da336a370243923dbaf21066bb9fe (revision pinned in the task
  source); prompt template MultipleChoiceTemplate.SINGLE_ANSWER from
  UKGovernmentBEIS/inspect_ai src/inspect_ai/solver/_multiple_choice.py

Usage:
    python3 build_real_cards.py --work-dir /tmp/real_work
    # then:
    evalwarden report-cards /tmp/real_work/swe-bench-verified /tmp/real_work/healthbench \
        --output-dir examples/report-cards/real/

Re-running the script reproduces the artifacts byte-for-byte as long as the
pinned sources are unchanged.
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

SCHEMA_VERSION = "evalwarden-artifact-v1"
N_SAMPLES = 12

INSPECT_EVALS_COMMIT = "244e43cc924d1de5a78dad7db259bbd4471c97e7"
SWE_BENCH_DATASET = "princeton-nlp/SWE-bench_Verified"
SWE_BENCH_REVISION = "c104f840cc67f8b6eec6f759ebc8b2693d585d4a"
HEALTHBENCH_URL = (
    "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/"
    "2025-05-07-06-14-12_oss_eval.jsonl"
)
# Exact input prompt from inspect_evals/swe_bench/swe_bench.py
SWE_INPUT_PROMPT = "Please solve the following coding issue:\n\n{issue_text}"
# Docker image template from inspect_evals/swe_bench/swe_bench.py
SWE_IMAGE_TEMPLATE = "ghcr.io/epoch-research/swe-bench.eval.{arch}.{id}:latest"


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "evalwarden-card-builder/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_jsonl_head(url: str, want_lines: int, chunk_bytes: int = 2_000_000) -> list[str]:
    """Fetch the head of a JSONL file in small ranged chunks until enough
    complete lines are collected. Returns the complete lines."""
    buf = b""
    offset = 0
    for _ in range(12):  # cap: 12 chunks x 2MB
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "evalwarden-card-builder/1.0",
                "Range": f"bytes={offset}-{offset + chunk_bytes - 1}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                data = resp.read()
        except Exception as exc:  # noqa: BLE001 - network flakiness, not fatal yet
            raise RuntimeError(f"download failed at offset {offset}: {exc}") from exc
        if not data:
            break
        buf += data
        offset += len(data)
        text = buf.decode("utf-8", errors="replace")
        complete = [ln for ln in text.splitlines() if ln.strip().endswith("}")]
        if len(complete) >= want_lines:
            return complete
        if len(data) < chunk_bytes:
            break  # EOF
    text = buf.decode("utf-8", errors="replace")
    return [ln for ln in text.splitlines() if ln.strip().endswith("}")]


def build_swe_bench(work: Path) -> Path:
    """Translate the inspect_evals swe_bench (SWE-bench Verified) task definition."""
    out = work / "swe-bench-verified"
    out.mkdir(parents=True, exist_ok=True)

    data = _get_json(
        f"https://datasets-server.huggingface.co/rows?dataset={SWE_BENCH_DATASET}"
        f"&config=default&split=test&offset=0&length={N_SAMPLES}"
    )
    rows = [r["row"] for r in data["rows"]]
    assert len(rows) == N_SAMPLES, f"expected {N_SAMPLES} rows, got {len(rows)}"

    tasks = []
    for r in rows:
        tasks.append(
            {
                "id": r["instance_id"],
                "prompt": SWE_INPUT_PROMPT.format(issue_text=r["problem_statement"]),
                "metadata": {
                    "repo": r["repo"],
                    "version": r["version"],
                    "base_commit": r["base_commit"],
                    # Template only: the task resolves {id}/{arch} per sample at
                    # definition time (swe_bench.py resolve_image_name).
                    "image_name_template": SWE_IMAGE_TEMPLATE,
                    "sandbox": "docker",
                },
            }
        )

    (out / "dataset.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "eval_id": "SWE-bench Verified via inspect_evals",
                "tasks": tasks,
                "notes": (
                    f"Definition sample: {N_SAMPLES} of 500 instances from "
                    f"{SWE_BENCH_DATASET} rev {SWE_BENCH_REVISION[:12]}. "
                    "Gold patch, test patch, and FAIL_TO_PASS/PASS_TO_PASS "
                    "contents are public in the dataset but EXCLUDED here: "
                    "they live in harness-side sample metadata (TaskState), "
                    "which the react agent's sandbox tools cannot read. "
                    "hints_text likewise stays metadata-side and is not in "
                    "the agent prompt."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (out / "environment.json").write_text(
        json.dumps(
            {
                "env": {},
                "mounts": [],
                "notes": (
                    "The task definition (swe_bench.py, solvers.py) declares "
                    "no agent-visible environment variables and no mounts. "
                    "The agent (swe_bench_agent_with_inspect_tool_support: "
                    "bash/python/text_editor) runs inside a Docker container "
                    "holding the repo at base_commit. The container image "
                    "name embeds the instance_id, but the definition "
                    "provides no mechanism exposing it inside the container "
                    "(no docker socket, no env passthrough). An oracle "
                    "solver (swe_bench_oracle_solver) ships in solvers.py for "
                    "pipeline validation only; it runs harness-side and "
                    "applies the gold patch from TaskState metadata, which "
                    "the evaluated agent cannot reach."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (out / "grader.json").write_text(
        json.dumps(
            {
                "kind": "script",
                "verifier": {
                    "path": (
                        "swe_bench_scorer: writes eval script into the sandbox "
                        "and runs FAIL_TO_PASS/PASS_TO_PASS via the upstream "
                        "swebench harness (get_eval_report)"
                    ),
                    "writable_by_agent": False,
                },
                "tests": ["FAIL_TO_PASS", "PASS_TO_PASS"],
                "notes": (
                    "Scoring is delegated to the upstream swebench harness. "
                    "The test patch is applied by the SCORER after the agent "
                    "submits; the agent never sees the graded tests and has "
                    "no write path to the scoring. Gradeability requires "
                    "both test-output markers (the harness ignores the eval "
                    "script's exit code by design)."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (out / "PROVENANCE.json").write_text(
        json.dumps(
            {
                "benchmark": "SWE-bench Verified",
                "task_definition": "inspect_evals/swe_bench @ " + INSPECT_EVALS_COMMIT,
                "task_files_read": ["swe_bench.py", "solvers.py", "scorers.py"],
                "dataset": SWE_BENCH_DATASET,
                "dataset_revision": SWE_BENCH_REVISION,
                "dataset_split": "test",
                "sample": f"first {N_SAMPLES} rows via HuggingFace datasets-server",
                "translation": "mechanical: prompt = DEFAULT_INPUT_PROMPT, metadata = public non-answer fields",
                "excluded": [
                    "patch (gold patch): public but harness-side; agent cannot read it",
                    "test_patch: applied by scorer post-submission; agent never sees it",
                    "FAIL_TO_PASS / PASS_TO_PASS test bodies: same as test_patch",
                    "hints_text: metadata-side, not in the agent prompt",
                ],
                "no_traces": True,
                "generated_by": "examples/report-cards/real/build_real_cards.py",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"swe-bench-verified: {len(tasks)} tasks -> {out}")
    return out


def build_healthbench(work: Path) -> Path:
    """Translate the inspect_evals healthbench task definition (default args)."""
    out = work / "healthbench"
    out.mkdir(parents=True, exist_ok=True)

    # Pull the head of the JSONL in small chunks until N_SAMPLES complete
    # examples are collected (examples vary widely in size).
    lines = _get_jsonl_head(HEALTHBENCH_URL, N_SAMPLES)
    examples = [json.loads(ln) for ln in lines[:N_SAMPLES]]
    assert len(examples) == N_SAMPLES, f"expected {N_SAMPLES} examples, got {len(examples)}"

    tasks = []
    criterion_texts: list[str] = []
    for ex in examples:
        user_msg = next(m["content"] for m in ex["prompt"] if m["role"] == "user")
        rubrics = ex["rubrics"]
        tasks.append(
            {
                "id": ex["prompt_id"],
                "prompt": user_msg,
                "metadata": {
                    "n_rubric_items": len(rubrics),
                    "total_positive_points": sum(r["points"] for r in rubrics if r["points"] > 0),
                    "total_negative_points": sum(r["points"] for r in rubrics if r["points"] < 0),
                    "axes": sorted({t.split(":", 1)[1] for r in rubrics for t in r["tags"] if t.startswith("axis:")}),
                    "example_tags": ex.get("example_tags", []),
                },
            }
        )
        for r in rubrics:
            if len(criterion_texts) < 4:
                criterion_texts.append(r["criterion"])

    (out / "dataset.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "eval_id": "HealthBench via inspect_evals",
                "tasks": tasks,
                "notes": (
                    f"Definition sample: {N_SAMPLES} of 5000 examples from the "
                    "HealthBench full split (URL pinned in the task source). "
                    "The solver sees only the conversation. Rubric criterion "
                    "texts are grader-side and summarized here as counts/axes; "
                    "four verbatim criteria are carried in grader.json as the "
                    "rubric_criteria sample."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (out / "environment.json").write_text(
        json.dumps(
            {
                "env": {},
                "mounts": [],
                "notes": (
                    "The solver is plain `generate` (no tools, no sandbox). "
                    "The task definition (healthbench.py) declares no "
                    "agent-visible environment. Rubrics are grader-side; the "
                    "solver never sees them."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (out / "grader.json").write_text(
        json.dumps(
            {
                "kind": "judge",
                "judge": {
                    # Defaults from the healthbench() task signature.
                    "model": "openai/gpt-4o-mini",
                    "protocol": "pointwise",
                    "temperature": 0.0,
                    "repeats": 1,
                    "rubric_criteria": criterion_texts,
                    # No scale_anchors: the rubric is a boolean checklist
                    # (criteria_met per item x points), not a scalar scale.
                    "scale_anchors": {},
                },
                "notes": (
                    "Judge defaults from healthbench() in healthbench.py. "
                    "Each (conversation, rubric item) pair is graded "
                    "criteria_met=true/false by the judge model per "
                    "GRADER_TEMPLATE (scorer.py); the score is the "
                    "points-weighted sum. No labeled calibration set ships "
                    "in the eval definition: judge validation lives in the "
                    "separate healthbench_meta_eval maintenance task "
                    "(29,511 physician-graded items, macro F1), not in this "
                    "artifact. The rubric is a boolean checklist with point "
                    "weights, not a scalar scale with anchored levels."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (out / "PROVENANCE.json").write_text(
        json.dumps(
            {
                "benchmark": "HealthBench (OpenAI, 2025)",
                "task_definition": "inspect_evals/healthbench @ " + INSPECT_EVALS_COMMIT,
                "task_files_read": ["healthbench.py", "scorer.py", "dataset.py", "eval.yaml"],
                "dataset_url": HEALTHBENCH_URL,
                "dataset_split": "full (5000 examples per eval.yaml)",
                "sample": f"first {N_SAMPLES} JSONL records",
                "judge_defaults_source": "healthbench() task signature (judge_model, judge_temperature)",
                "grader_template": "GRADER_TEMPLATE in scorer.py",
                "judge_validation_note": (
                    "The benchmark authors validate the grader in the paper "
                    "via the healthbench_meta_eval subset; that validation "
                    "is NOT part of the eval definition artifact audited here."
                ),
                "no_traces": True,
                "generated_by": "examples/report-cards/real/build_real_cards.py",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"healthbench: {len(tasks)} tasks -> {out}")
    return out


# Exact judge defaults from writingbench() / multi_scorer_wrapper() in
# writingbench.py at the pinned commit (read by a human; the machine link is
# this file's content hash).
WB_URL = (
    "https://raw.githubusercontent.com/UKGovernmentBEIS/inspect_evals/"
    "244e43cc924d1de5a78dad7db259bbd4471c97e7/src/inspect_evals/writingbench/"
    "benchmark_all.jsonl"
)
WB_JUDGE_MODEL = "anthropic/claude-3-5-haiku-latest"
WB_GRADE_PATTERN = r'"score"\s*:\s*(10|[1-9])'
WB_SCALE_ANCHORS = {
    "1-2": "Low score description: Critical deficiencies and major issues that prevent adequate functionality.",
    "3-4": "Below average score description: Lacking with noticeable shortcomings that impact overall effectiveness and require improvement.",
    "5-6": "Average score description: Adequate but not exemplary, Baseline performance that meets essential requirements. Most models may achieve this score.",
    "7-8": "Above average score description: Strong performance characterized by competent execution, though minor refinements are needed to achieve excellence.",
    "9-10": "High score description: Exceptional performance with all aspects optimally addressed, demonstrating superior effectiveness and quality without any flaws.",
}


def build_writingbench(work: Path) -> Path:
    """Translate the inspect_evals writingbench task definition (default args)."""
    out = work / "writingbench"
    out.mkdir(parents=True, exist_ok=True)

    lines = _get_jsonl_head(WB_URL, N_SAMPLES)
    records = [json.loads(ln) for ln in lines[:N_SAMPLES]]
    assert len(records) == N_SAMPLES, f"expected {N_SAMPLES} records, got {len(records)}"

    tasks = []
    for rec in records:
        checklist = rec["checklist"]
        tasks.append(
            {
                "id": str(rec["index"]),
                "prompt": rec["query"],
                "metadata": {
                    "domain1": rec["domain1"],
                    "domain2": rec["domain2"],
                    "n_checklist_items": len(checklist),
                    "checklist_names": [item["name"] for item in checklist],
                },
            }
        )
    # One verbatim checklist item, carried as grader-side evidence that the
    # per-criterion anchors exist (each item also carries its own 1-2..9-10
    # level texts beyond the generic scoring rules).
    criteria_sample = records[0]["checklist"][0]

    (out / "dataset.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "eval_id": "WritingBench via inspect_evals",
                "tasks": tasks,
                "notes": (
                    f"Definition sample: {N_SAMPLES} of 1000 queries from "
                    "benchmark_all.jsonl (URL pinned in the task source; the "
                    "file lives in the inspect_evals repo at the pinned "
                    "commit). The solver sees only the query. Each record "
                    "carries a 5-item checklist of grading criteria in its "
                    "metadata; those criterion texts are grader-side and "
                    "summarized here as names, with one verbatim item carried "
                    "in grader.json as the rubric_criteria sample."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (out / "environment.json").write_text(
        json.dumps(
            {
                "env": {},
                "mounts": [],
                "notes": (
                    "The solver is plain `generate` (no tools, no sandbox) "
                    "with GenerateConfig(top_p=0.8, top_k=20, temperature=0.7, "
                    "max_tokens=16000). The task definition (writingbench.py) "
                    "declares no agent-visible environment. The checklist is "
                    "sample metadata used only in the scoring template; the "
                    "solver never sees it."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (out / "grader.json").write_text(
        json.dumps(
            {
                "kind": "judge",
                "judge": {
                    # Defaults from writingbench() and multi_scorer_wrapper().
                    "model": WB_JUDGE_MODEL,
                    "protocol": "pointwise",
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "top_k": 20,
                    "max_tokens": 2048,
                    "repeats": 1,
                    "n_criteria_per_sample": 5,
                    "aggregation": "mean",
                    "grade_pattern": WB_GRADE_PATTERN,
                    "scale_anchors": WB_SCALE_ANCHORS,
                    "rubric_criteria": criteria_sample,
                },
                "notes": (
                    "Five model_graded_qa scorers via multi_scorer, one per "
                    "checklist item, reduced by mean. The scoring template "
                    "carries the generic 1-10 anchored level descriptions "
                    "plus the per-item criterion text; each checklist item "
                    "additionally ships its own 1-2..9-10 level texts. Judge "
                    "completions that do not match the grade pattern are "
                    "dropped from the per-sample criterion mean rather than "
                    "zeroed (changelog 3-A); a sample whose criteria all fail "
                    "is unscored, visible only via unscored_samples. No "
                    "labeled calibration set ships in the eval definition: "
                    "the README's Validation Notes cover pipeline execution "
                    "(end-to-end runs, metric math, multilingual handling, "
                    "parse-failure surfacing), not judge-vs-human agreement."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (out / "PROVENANCE.json").write_text(
        json.dumps(
            {
                "benchmark": "WritingBench (2025)",
                "task_definition": "inspect_evals/writingbench @ " + INSPECT_EVALS_COMMIT,
                "task_files_read": ["writingbench.py", "eval.yaml", "README.md"],
                "dataset_url": WB_URL,
                "dataset_split": "benchmark_all.jsonl (1000 queries per eval.yaml)",
                "sample": f"first {N_SAMPLES} JSONL records via ranged head fetch",
                "judge_defaults_source": (
                    "writingbench() task signature (judge_model) and "
                    "multi_scorer_wrapper() (grade pattern, judge GenerateConfig, "
                    "5 scorers, mean reducer); anchored level texts from "
                    "create_scoring_prompt()"
                ),
                "excluded": [
                    "checklist criterion texts (criteria_description + per-level anchors): public in the dataset but grader-side; the solver prompt carries only the query",
                ],
                "no_traces": True,
                "generated_by": "examples/report-cards/real/build_real_cards.py",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"writingbench: {len(tasks)} tasks -> {out}")
    return out


# Exact input prompt template: MultipleChoiceTemplate.SINGLE_ANSWER from
# inspect_ai (src/inspect_ai/solver/_multiple_choice.py), based on
# openai/simple-evals mmlu_eval.py. Read from UKGovernmentBEIS/inspect_ai
# main on 2026-09-29; the task resolves it via MMLU_MULTISHOT_QUESTION_TEMPLATE.
MMLU_PROMPT_TEMPLATE = (
    "Answer the following multiple choice question. The entire content of "
    "your response should be of the following format: 'ANSWER: $LETTER' "
    "(without quotes) where LETTER is one of {letters}.\n"
    "\n"
    "{question}\n"
    "\n"
    "{choices}"
)
MMLU_DATASET = "cais/mmlu"
MMLU_CONFIG = "all"
MMLU_SPLIT = "test"
# Revision pinned in the task source (MMLU_REVISION in mmlu.py).
MMLU_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"
MMLU_ROWS_URL = (
    f"https://datasets-server.huggingface.co/rows?dataset={MMLU_DATASET}"
    f"&config={MMLU_CONFIG}&split={MMLU_SPLIT}&offset=0&length={N_SAMPLES}"
)


def _format_mmlu_prompt(question: str, choices: list[str]) -> str:
    # Letters and choice lines verbatim from format_mmlu_question /
    # format_mmlu_choices in mmlu.py.
    letters = ",".join(chr(ord("a") + i) for i in range(len(choices)))
    choice_lines = "\n".join(
        f"({chr(ord('a') + i)}) {c}" for i, c in enumerate(choices)
    )
    return MMLU_PROMPT_TEMPLATE.format(
        letters=letters, question=question, choices=choice_lines
    )


def build_mmlu(work: Path) -> Path:
    """Translate the inspect_evals mmlu_0_shot task definition (default args)."""
    out = work / "mmlu"
    out.mkdir(parents=True, exist_ok=True)

    data = _get_json(MMLU_ROWS_URL)
    rows = [r["row"] for r in data["rows"]]
    assert len(rows) == N_SAMPLES, f"expected {N_SAMPLES} rows, got {len(rows)}"

    tasks = []
    for i, r in enumerate(rows):
        tasks.append(
            {
                "id": f"mmlu-test-{i:05d}",
                "prompt": _format_mmlu_prompt(r["question"], r["choices"]),
                "metadata": {
                    "subject": r["subject"],
                    "n_choices": len(r["choices"]),
                },
            }
        )

    (out / "dataset.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "eval_id": "MMLU via inspect_evals",
                "tasks": tasks,
                "notes": (
                    f"Definition sample: {N_SAMPLES} of 14042 test questions "
                    f"(57 subjects) from {MMLU_DATASET} config '{MMLU_CONFIG}' "
                    f"rev {MMLU_REVISION[:12]}. The solver sees only the "
                    "formatted multiple-choice question. The answer index is "
                    "public in the dataset but EXCLUDED here: it is the "
                    "harness-side sample target that choice() grades against, "
                    "which the agent never sees. Sample = first 12 rows in "
                    "dataset order via the datasets-server rows API; the "
                    "task default (mmlu_0_shot) shuffles with seed 42 after "
                    "dedup — the sample illustrates the instrument, not a "
                    "particular eval draw."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (out / "environment.json").write_text(
        json.dumps(
            {
                "env": {},
                "mounts": [],
                "notes": (
                    "The solver is multiple_choice over plain `generate` (no "
                    "tools, no sandbox): GenerateConfig(temperature=0.0), "
                    "non-CoT max tokens default GPT_5_MIN_TOKENS (16) with a "
                    "model-dependent floor (get_max_tokens in mmlu.py). The "
                    "task definition (mmlu.py) declares no agent-visible "
                    "environment variables and no mounts."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (out / "grader.json").write_text(
        json.dumps(
            {
                "kind": "script",
                "verifier": {
                    "path": (
                        "inspect_ai.scorer.choice: parses the model's selected "
                        "letter from the completion and compares it to the "
                        "harness-side target letter"
                    ),
                    "writable_by_agent": False,
                },
                "tests": ["letter-match (A/B/C/D)"],
                "notes": (
                    "choice() grades the selected letter against the sample "
                    "target, which lives in harness-side sample metadata. "
                    "Scoring runs harness-side after the agent submits; the "
                    "agent never sees the target and has no write path to the "
                    "scoring. Completions with no parseable letter score as "
                    "incorrect (0.0); there is no empty-output credit path. "
                    "No judge model is involved: JUDGE-001..006 do not apply."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (out / "PROVENANCE.json").write_text(
        json.dumps(
            {
                "benchmark": "MMLU (Hendrycks et al., 2020)",
                "task_definition": "inspect_evals/mmlu @ " + INSPECT_EVALS_COMMIT,
                "task_files_read": ["mmlu.py", "eval.yaml"],
                "prompt_template_source": (
                    "MultipleChoiceTemplate.SINGLE_ANSWER from "
                    "UKGovernmentBEIS/inspect_ai "
                    "src/inspect_ai/solver/_multiple_choice.py (read "
                    "2026-09-29; based on openai/simple-evals mmlu_eval.py); "
                    "choice letters formatted per format_mmlu_choices in mmlu.py"
                ),
                "dataset": MMLU_DATASET,
                "dataset_config": MMLU_CONFIG,
                "dataset_split": MMLU_SPLIT,
                "dataset_revision": MMLU_REVISION,
                "sample": (
                    f"first {N_SAMPLES} rows via HuggingFace datasets-server "
                    "(offset 0, length 12)"
                ),
                "translation": (
                    "mechanical: prompt = SINGLE_ANSWER template over "
                    "question + lettered choices; metadata = subject only"
                ),
                "excluded": [
                    "answer (answer index 0-3): public in the dataset but "
                    "harness-side; it is the sample target that choice() "
                    "grades against"
                ],
                "no_traces": True,
                "generated_by": "examples/report-cards/real/build_real_cards.py",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"mmlu: {len(tasks)} tasks -> {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True, help="Where to write the artifacts.")
    parser.add_argument(
        "--only",
        choices=["swe-bench-verified", "healthbench", "writingbench", "mmlu"],
        default=None,
        help="Build just one artifact (default: all).",
    )
    args = parser.parse_args()
    work = Path(args.work_dir)
    builders = {
        "swe-bench-verified": build_swe_bench,
        "healthbench": build_healthbench,
        "writingbench": build_writingbench,
        "mmlu": build_mmlu,
    }
    for name, fn in builders.items():
        if args.only is None or args.only == name:
            fn(work)
    print("done. Next: evalwarden audit <dir> to review, then evalwarden report-cards <dirs>.")


if __name__ == "__main__":
    main()
