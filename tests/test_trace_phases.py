"""Phase segmentation: classifier table, smoothing, planted boundaries."""
from __future__ import annotations

from pathlib import Path

import pytest

from evalwarden.adapters.claude_code import ClaudeCodeAdapter
from evalwarden.adapters.codex import CodexAdapter
from evalwarden.model import TrajectoryStep
from evalwarden.trace_phases import classify_step, segment_phases

from .trace_fixtures import (
    CLAUDE_CLEAN_SESSION,
    CODEX_LOOP_FILE,
    CODEX_LOOP_PHASES,
    claude_clean_records,
    codex_loop_records,
    write_claude_project,
    write_codex_session,
)


def _step(tool: str, args: dict | None = None, step_id: str = "s") -> TrajectoryStep:
    return TrajectoryStep(step_id=step_id, tool=tool, args=args or {})


@pytest.mark.parametrize("tool,args,label", [
    ("Read", {"file_path": "x"}, "explore"),
    ("Grep", {"pattern": "teh"}, "explore"),
    ("Glob", {"pattern": "**/*.py"}, "explore"),
    ("Edit", {"file_path": "x"}, "edit"),
    ("Write", {"file_path": "x"}, "edit"),
    ("apply_patch", {"input": "*** Begin Patch"}, "edit"),
    ("update_plan", {"plan": []}, "plan"),
    ("TodoWrite", {"todos": []}, "plan"),
    ("Task", {"prompt": "scan"}, "explore"),
    ("Bash", {"command": "npm test"}, "test"),
    ("Bash", {"command": "pytest -q tests/"}, "test"),
    ("Bash", {"command": "npm run build"}, "test"),
    ("Bash", {"command": "cat src/foo.py"}, "explore"),
    ("Bash", {"command": "grep -rn pytest src"}, "explore"),  # searching, not running
    ("Bash", {"command": "sed -n '1,120p' src/cli.py"}, "explore"),
    ("Bash", {"command": "sed -i 's/a/b/' src/cli.py"}, "edit"),
    ("Bash", {"command": "git diff"}, "review"),
    ("Bash", {"command": "git status --short"}, "review"),
    ("Bash", {"command": "git log --oneline -5"}, "review"),
    ("shell", {"command": ["pytest", "-q"]}, "test"),
    ("shell", {"command": ["ls", "-R", "src"]}, "explore"),
    ("shell", {"command": ["git", "diff"]}, "review"),
    ("shell", {"command": ["sed", "-n", "1,120p", "src/cli.py"]}, "explore"),
    ("mystery_tool", {}, "other"),
    ("Bash", {"command": "cowsay hi"}, "other"),
    # Real shapes from the Optimal Misbehavior corpus (Oct 2026):
    # PowerShell is Claude Code's shell on Windows.
    ("PowerShell", {"command": "node verify.js"}, "test"),
    ("PowerShell", {"command": "Get-Content SPEC.md"}, "explore"),
    ("PowerShell", {"command": "Set-Content out.txt hello"}, "edit"),
    ("PowerShell", {"command": "git log --oneline -n 20; git status"}, "review"),
    # Chained commands: the meaningful segment names the phase.
    ("Bash", {"command": 'cd "C:/repo" && node verify.js'}, "test"),
    ("Bash", {"command": 'cd "C:/repo" && git diff --stat'}, "review"),
    ("Bash", {"command": 'cd "C:/repo" && git status && npm test'}, "test"),
    ("Bash", {"command": "ls -la && wc -l SPEC.md BUILD_PLAN.md"}, "explore"),
    # Operators inside quotes are text, not separators.
    ("Bash", {"command": 'node -e "a && b"'}, "other"),
    # Codex's real tool names (AletheiaResearch corpus, Oct 2026).
    ("exec_command", {"cmd": "pytest -q"}, "test"),
    ("exec_command", {"cmd": "ls -la"}, "explore"),
    ("exec_command", {"cmd": "git diff --stat"}, "review"),
    ("view_image", {"path": "/tmp/shot.png"}, "explore"),
    # Browser-automation MCP tools are playtest verification.
    ("mcp__Claude_Browser__computer", {"action": "screenshot"}, "test"),
    ("mcp__Claude_Browser__javascript_tool", {"action": "javascript_exec"}, "test"),
    ("mcp__some_other_server__thing", {}, "other"),
])
def test_classify_step(tool, args, label):
    assert classify_step(tool, args) == label


def _labels(segments):
    return [(s.label, s.start, s.end) for s in segments]


def test_codex_loop_recovers_planted_phases(tmp_path: Path):
    path = write_codex_session(tmp_path, codex_loop_records(), CODEX_LOOP_FILE)
    adapter = CodexAdapter()
    model = adapter.normalize(adapter.collect(path))
    (attempt,) = model.attempts
    segments = segment_phases(attempt.spans)
    assert _labels(segments) == CODEX_LOOP_PHASES
    # Change points are the segment starts after the first.
    assert [s.start for s in segments[1:]] == [3, 4, 6, 10]
    assert segments[2].step_ids == ["call_5", "call_6"]  # the edit phase


def test_claude_clean_phases(tmp_path: Path):
    path = write_claude_project(tmp_path, claude_clean_records(), CLAUDE_CLEAN_SESSION)
    adapter = ClaudeCodeAdapter()
    model = adapter.normalize(adapter.collect(path))
    main = model.attempts[0]
    assert _labels(segment_phases(main.spans)) == [
        ("explore", 0, 2), ("edit", 2, 3), ("test", 3, 4)]


def test_single_step_blip_is_smoothed():
    spans = [_step("Read", step_id="a"), _step("Edit", step_id="b"),
             _step("Read", step_id="c"), _step("Grep", step_id="d")]
    assert _labels(segment_phases(spans)) == [("explore", 0, 4)]


def test_blip_at_boundary_is_not_smoothed():
    # A single edit at the end has no right flank: it stays its own phase.
    spans = [_step("Read", step_id="a"), _step("Read", step_id="b"),
             _step("Edit", step_id="c")]
    assert _labels(segment_phases(spans)) == [("explore", 0, 2), ("edit", 2, 3)]


def test_empty_and_other():
    assert segment_phases([]) == []
    spans = [_step("mystery_tool", step_id="a"), _step("mystery_tool", step_id="b")]
    assert _labels(segment_phases(spans)) == [("other", 0, 2)]
