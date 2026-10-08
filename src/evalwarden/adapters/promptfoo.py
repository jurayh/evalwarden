"""Promptfoo adapter (v0.5).

Reads a Promptfoo eval artifact directory:

    eval-artifact/
      promptfooconfig.yaml   (or .yml) -- prompts, providers, tests, assertions, env
      results.json           (optional) -- JSON export from `promptfoo eval -o results.json`

Minimum viable input per the spec: promptfooconfig.yaml plus the JSON export.
A config-only audit is valid but incomplete (no runs to analyze). When only
results.json is present, the config embedded in its envelope is used and the
substitution is reported explicitly.

Field names follow promptfoo's documented output schema (OutputFile envelope
with results.version == 3; per-result success/score/error, vars,
response.output, response.tokenUsage.{prompt,completion,total}, latencyMs,
gradingResult.{pass,score,reason,componentResults}). Anything else lands in
`unsupported`, never silently dropped.

This is a read-only translation layer: it parses the config and the recorded
results. It never executes an eval.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..model import (
    Attempt,
    Confidence,
    Environment,
    Grader,
    IntegrityModel,
    TaskSample,
)
from . import AuditError, register
from ._io import digest_file, read_json

ADAPTER_NAME = "promptfoo"
ADAPTER_VERSION = "0.6.0"
# Promptfoo's documented results envelope version (OutputFile.results.version).
RESULTS_VERSION = 3

CONFIG_NAMES = ("promptfooconfig.yaml", "promptfooconfig.yml")
RESULTS_NAME = "results.json"

# Top-level config keys the adapter understands. Everything else is reported
# in `unsupported` so coverage gaps are explicit, not silent.
KNOWN_CONFIG_KEYS = {
    "description",
    "env",
    "prompts",
    "providers",
    "defaultTest",
    "scenarios",
    "tests",
    "metadata",
    "outputPath",
    "writeLatestResults",
    "sharing",
}


def _envelope_has_config(results_path: Path) -> bool:
    """Cheap peek for detect(): does the results envelope carry a config dict?"""
    try:
        envelope = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(envelope, dict) and isinstance(envelope.get("config"), dict)


def _read_yaml(path: Path) -> Any:
    try:
        import yaml
    except ImportError as exc:
        raise AuditError(
            "promptfoo adapter needs PyYAML to parse promptfooconfig.yaml "
            "(pip install 'evalwarden' again, or pip install pyyaml)"
        ) from exc
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AuditError(f"missing required file: {path}") from exc
    except yaml.YAMLError as exc:
        raise AuditError(f"invalid YAML in {path}: {exc}") from exc


def _config_path(path: Path) -> Path | None:
    for name in CONFIG_NAMES:
        candidate = path / name
        if candidate.is_file():
            return candidate
    return None


@register
class PromptfooAdapter:
    name = ADAPTER_NAME
    version = ADAPTER_VERSION

    def detect(self, path: Path) -> Confidence:
        if not path.is_dir():
            return Confidence.LOW
        has_config = _config_path(path) is not None
        results_path = path / RESULTS_NAME
        has_results = results_path.is_file()
        if has_config and has_results:
            return Confidence.HIGH
        if has_config:
            # Config-only audit: valid but incomplete (no runs to analyze).
            return Confidence.MEDIUM
        if has_results and _envelope_has_config(results_path):
            # No promptfooconfig.yaml, but the results envelope carries the
            # config it was produced from.
            return Confidence.MEDIUM
        return Confidence.LOW

    def collect(self, path: Path) -> dict:
        """Read-only: files are opened for reading and never modified."""
        bundle: dict[str, Any] = {"root": path}
        digests: dict[str, str] = {}
        config_path = _config_path(path)
        if config_path is not None:
            bundle["config"] = _read_yaml(config_path)
            bundle["config_file"] = config_path.name
            digests[config_path.name] = digest_file(config_path)
        results_path = path / RESULTS_NAME
        if results_path.is_file():
            bundle["results"] = read_json(results_path)
            digests[RESULTS_NAME] = digest_file(results_path)
        bundle["digests"] = digests
        return bundle

    def normalize(self, bundle: dict) -> IntegrityModel:
        root: Path = bundle["root"]
        config = bundle.get("config") or {}
        config_file = bundle.get("config_file", "promptfooconfig.yaml")
        envelope = bundle.get("results") or {}
        inner = envelope.get("results") or {}

        if not isinstance(config, dict):
            raise AuditError(f"{config_file}: top-level mapping expected")

        unsupported: list[str] = []
        if "config" not in bundle and envelope.get("config") is not None:
            # Graceful degradation: no promptfooconfig.yaml, but the results
            # envelope carries the eval config it was produced from.
            config = envelope["config"]
            unsupported.append(
                f"{RESULTS_NAME}: promptfooconfig.yaml absent; "
                "using the config embedded in the results envelope"
            )

        version = inner.get("version")
        if "results" in bundle and version != RESULTS_VERSION:
            raise AuditError(
                f"{RESULTS_NAME}: unsupported results version {version!r} "
                f"(this adapter understands version {RESULTS_VERSION})"
            )

        eval_id = (
            config.get("description")
            or envelope.get("evalId")
            or root.name
        )

        tasks = _normalize_tests(config, unsupported)
        env_vars = {
            str(name): "<redacted>"  # values are never stored; names are the signal
            for name in (config.get("env") or {})
        }
        grader = _normalize_grader(config)
        attempts = _normalize_attempts(inner)

        for key in config:
            if key not in KNOWN_CONFIG_KEYS:
                unsupported.append(f"{config_file}:{key}")
        if config.get("scenarios"):
            unsupported.append(
                f"{config_file}:scenarios (scenario-generated tests not expanded)"
            )
        for test in _inline_tests(config):
            for key in test:
                if key not in {
                    "description", "vars", "assert", "options",
                    "threshold", "metadata",
                }:
                    unsupported.append(f"{config_file}:tests[]:{key}")

        return IntegrityModel(
            eval_id=str(eval_id),
            adapter_name=self.name,
            adapter_version=self.version,
            tasks=tasks,
            environment=Environment(env_vars=env_vars, mounts=[]),
            grader=grader,
            attempts=attempts,
            unsupported=unsupported,
            digests=dict(bundle.get("digests", {})),
            # Checks cite canonical artifact names; point them at the real files.
            file_aliases={
                "environment.json": config_file,
                "grader.json": config_file,
                "judge_run.json": config_file,
                "run.json": RESULTS_NAME,
            },
        )


def _inline_tests(config: dict) -> list[dict]:
    """Tests declared inline in the config. file:// and other references are
    left for `unsupported`: expanding them would pull arbitrary files into the
    audit, and the declared config is the contract under review."""
    tests = config.get("tests") or []
    if isinstance(tests, dict):
        tests = [tests]
    return [t for t in tests if isinstance(t, dict)]


def _normalize_tests(config: dict, unsupported: list[str]) -> list[TaskSample]:
    tasks: list[TaskSample] = []
    raw_tests = config.get("tests")
    if isinstance(raw_tests, str):
        unsupported.append(f"tests: {raw_tests} (external test file not expanded)")
        return tasks
    if isinstance(raw_tests, list) and any(
        isinstance(t, str) for t in raw_tests
    ):
        unsupported.append("tests[]: file:// references not expanded")
    for i, test in enumerate(_inline_tests(config)):
        vars_ = test.get("vars") or {}
        assertions = [str(a.get("type", "?")) for a in (test.get("assert") or []) if isinstance(a, dict)]
        tasks.append(
            TaskSample(
                id=str(test.get("description") or f"test-{i}"),
                prompt=str(test.get("description") or ""),
                metadata={
                    "vars": sorted(str(k) for k in vars_),
                    "assertions": assertions,
                },
            )
        )
    return tasks


def _assertions(config: dict) -> list[dict]:
    """All assertions: shared defaultTest ones plus per-test ones."""
    found: list[dict] = []
    default = config.get("defaultTest") or {}
    if isinstance(default, dict):
        found.extend(a for a in (default.get("assert") or []) if isinstance(a, dict))
    for test in _inline_tests(config):
        found.extend(a for a in (test.get("assert") or []) if isinstance(a, dict))
    return found


def _provider_id(ref: Any) -> str | None:
    if isinstance(ref, str):
        return ref
    if isinstance(ref, dict) and ref.get("id"):
        return str(ref["id"])
    return None


def _grader_provider(config: dict) -> tuple[str | None, float | None]:
    """The model grading llm-rubric assertions: per-test options.provider wins,
    then defaultTest.options.provider. Returns (provider_id, temperature)."""
    for test in _inline_tests(config):
        options = test.get("options") or {}
        if isinstance(options, dict) and options.get("provider") is not None:
            return _provider_id(options["provider"]), _provider_temperature(options["provider"])
    default = config.get("defaultTest") or {}
    options = default.get("options") or {} if isinstance(default, dict) else {}
    if isinstance(options, dict) and options.get("provider") is not None:
        return _provider_id(options["provider"]), _provider_temperature(options["provider"])
    return None, None


def _provider_temperature(ref: Any) -> float | None:
    if isinstance(ref, dict):
        cfg = ref.get("config") or {}
        if isinstance(cfg, dict) and isinstance(cfg.get("temperature"), (int, float)):
            return float(cfg["temperature"])
    return None


def _normalize_grader(config: dict) -> Grader:
    assertions = _assertions(config)
    types = sorted({str(a.get("type", "?")) for a in assertions})
    is_judge = "llm-rubric" in types
    grader = Grader(
        kind="judge" if is_judge else "script",
        tests=types,
    )
    if is_judge:
        provider_id, temperature = _grader_provider(config)
        grader.judge_model = provider_id
        grader.temperature = temperature
        # Each llm-rubric value is the rubric; promptfoo has no anchored scale
        # levels, which JUDGE-001 reports as an unanchored rubric.
        grader.rubric_criteria = [
            str(a["value"]) for a in assertions
            if str(a.get("type")) == "llm-rubric" and a.get("value")
        ]
    return grader


def _normalize_attempts(inner: dict) -> list[Attempt]:
    attempts: list[Attempt] = []
    rows = inner.get("results") or []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        error = row.get("error")
        success = row.get("success")
        status = "error" if error else ("pass" if success else "fail")
        response = row.get("response") or {}
        usage = response.get("tokenUsage") or {}
        latency_ms = row.get("latencyMs")
        output = response.get("output")
        attempts.append(
            Attempt(
                task_id=str(row.get("description") or f"test-{row.get('testIdx', i)}"),
                status=status,
                score=_to_float(row.get("score")),
                tokens_in=_to_int(usage.get("prompt")),
                tokens_out=_to_int(usage.get("completion")),
                latency_s=(float(latency_ms) / 1000.0) if isinstance(latency_ms, (int, float)) else None,
                # An explicitly empty model output scored as a pass is the
                # GRAD-002 signal; anything else is just a short answer.
                empty_submission=isinstance(output, str) and output == "" and status == "pass",
            )
        )
    return attempts


def _to_float(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _to_int(value: Any) -> int | None:
    return int(value) if isinstance(value, (int, float)) else None
