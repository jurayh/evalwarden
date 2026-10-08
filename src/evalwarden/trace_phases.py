"""Phase segmentation of tool-use trajectories.

The signature analysis of the trace observatory: segment one session's
ordered tool-call sequence into phases -- explore, plan, edit, test,
review -- with explicit change points. The method is deliberately
simple, deterministic, and inspectable:

1. **Classify** each step with the fixed rule table in
   :func:`classify_step`. A step's label comes from its tool name and,
   for shell tools, from the command text. Every label can be traced
   to one named rule; there is no model call and no hidden state.
2. **Smooth** single-step blips: a one-step run whose neighbours on
   both sides share a different label takes the neighbours' label.
   One stray ``git status`` inside an edit run does not split the
   phase. Smoothing repeats until stable, so the result does not
   depend on scan order.
3. **Run-length encode** the smoothed labels. Each maximal run is a
   :class:`PhaseSegment`; its boundaries are the change points.

Labels: ``explore`` (reading, searching, listing), ``plan`` (planning
tools), ``edit`` (file mutation), ``test`` (tests, builds, linters --
the verification bucket), ``review`` (inspecting the resulting diff /
repo state), and ``other`` (anything the rule table does not claim --
kept as its own segment rather than absorbed into a neighbour, so the
segmentation never invents a phase the rules cannot justify).
"""
from __future__ import annotations

import shlex
from dataclasses import dataclass, field

from .model import TrajectoryStep

PHASES = ("explore", "plan", "edit", "test", "review", "other")

_EXPLORE_TOOLS = {
    "read", "read_file", "view", "view_image", "grep", "glob", "ls",
    "list_dir", "find", "search", "webfetch", "websearch", "fetch", "cat",
    "notebookread",
}
_EDIT_TOOLS = {
    "edit", "edit_file", "write", "write_file", "notebookedit",
    "apply_patch", "create_file", "multiedit",
}
_PLAN_TOOLS = {"todowrite", "todo_write", "update_plan", "plan", "enterplanmode"}
_DELEGATE_TOOLS = {"task", "agent", "subagent", "spawn_agent"}
_SHELL_TOOLS = {
    "bash", "shell", "exec", "run_command", "terminal", "execute_command",
    "powershell",  # Claude Code's shell on Windows
    "exec_command",  # Codex's shell tool (AletheiaResearch corpus, Oct 2026)
}

# Shell command heads (first token, basename) that read state.
_READ_COMMANDS = {
    "ls", "cat", "head", "tail", "grep", "rg", "find", "pwd", "wc", "tree",
    "file", "stat", "which", "echo",  # echo of a value reads/prints state; `echo > file` is caught earlier as a redirect
}
# Command fragments that mark verification work (tests / builds / lint).
_TEST_MARKERS = (
    "pytest", "jest", "vitest", "unittest", "go test", "cargo test",
    "npm test", "yarn test", "pnpm test", "npm run test", "tox", "rspec",
    "npm run build", "yarn build", "pnpm build", "cargo build", "go build",
    "make", "tsc", "ruff", "flake8", "mypy", "eslint", "lint",
    "verify",  # verify scripts (`node verify.js`, `npm run verify`) are this bucket
)
_REVIEW_COMMANDS = {"diff", "status", "log", "show", "blame"}  # after `git`

# PowerShell cmdlets (Claude Code's Windows shell): reads inspect state,
# content cmdlets mutate files.
_PS_READ_COMMANDS = {
    "get-childitem", "get-content", "select-string", "test-path",
    "get-item", "get-location", "get-command",
}
_PS_EDIT_COMMANDS = {
    "set-content", "add-content", "out-file", "new-item", "copy-item",
    "move-item", "rename-item", "remove-item",
}


def _command_text(args: dict) -> str:
    """The shell command in a step's args, as one string ("" when absent)."""
    command = args.get("command")
    if isinstance(command, list):
        return " ".join(str(part) for part in command)
    if isinstance(command, str):
        return command
    for key in ("cmd", "input"):
        value = args.get(key)
        if isinstance(value, str):
            return value
    return ""


def _split_chain(command: str) -> list[str]:
    """Split a command on top-level ``&&``, ``||`` and ``;`` operators.

    Quote-aware: operators inside single or double quotes are text, not
    separators (a ``node -e "a && b"`` probe is one command).
    """
    parts, buf = [], []
    quote = None
    i = 0
    while i < len(command):
        ch = command[i]
        if quote:
            buf.append(ch)
            if ch == quote and command[i - 1] != "\\":
                quote = None
            i += 1
        elif ch in ("'", '"'):
            quote = ch
            buf.append(ch)
            i += 1
        elif command.startswith(("&&", "||"), i):
            parts.append("".join(buf))
            buf = []
            i += 2
        elif ch == ";":
            parts.append("".join(buf))
            buf = []
            i += 1
        else:
            buf.append(ch)
            i += 1
    parts.append("".join(buf))
    return [p.strip() for p in parts if p.strip()]


def _classify_shell(command: str) -> str:
    """Phase of a shell command, possibly a chain of commands.

    Real sessions chain: ``cd "C:/repo" && node verify.js`` does its
    meaningful work in the second segment, and a head-only rule sees
    only the ``cd``. A chain is classified per segment and combined by
    priority -- test, then review, then edit, then explore -- so the
    segment that does the session's verifiable work names the phase.
    A single command goes straight to :func:`_classify_shell_segment`.
    """
    segments = _split_chain(command)
    if len(segments) <= 1:
        return _classify_shell_segment(command)
    labels = [_classify_shell_segment(s) for s in segments]
    for label in ("test", "review", "edit", "explore"):
        if label in labels:
            return label
    return "other"


def _classify_shell_segment(command: str) -> str:
    """Phase of one unchained shell command, by first matching rule.

    Read commands win over test markers, so ``grep -rn pytest src`` is
    exploration (searching for a string), not a test run; test markers
    only claim commands whose head is not a read command.
    """
    text = " ".join(command.split())
    if not text:
        return "other"
    lowered = text.lower()
    # In-place editors mutate files even though `sed` can also print.
    if "sed -i" in lowered or "perl -i" in lowered:
        return "edit"
    try:
        tokens = shlex.split(text)
    except ValueError:
        tokens = text.split()
    if not tokens:
        return "other"
    head = tokens[0].rsplit("/", 1)[-1].rsplit("\\", 1)[-1].lower()
    if head in _PS_READ_COMMANDS:
        return "explore"
    if head in _PS_EDIT_COMMANDS:
        return "edit"
    if head == "git" and len(tokens) > 1 and tokens[1].lower() in _REVIEW_COMMANDS:
        return "review"
    if head == "sed":
        # `sed -n '1,120p'` prints (a read); other sed forms stay other
        # (in-place edits were claimed above).
        return "explore" if "-n" in tokens else "other"
    if head in _READ_COMMANDS:
        return "explore"
    if any(marker in lowered for marker in _TEST_MARKERS):
        return "test"
    return "other"


def classify_step(tool: str, args: dict) -> str:
    """The phase label of one step, from the fixed rule table.

    Rules, in order: planning tools -> ``plan``; delegation tools ->
    ``explore`` (a sub-agent is dispatched to go and look); known edit
    tools -> ``edit``; known read/search tools -> ``explore``; shell
    tools defer to :func:`_classify_shell` on the command text;
    browser-automation MCP tools -> ``test`` (driving the built thing
    in a browser is playtest verification); anything else -> ``other``.
    """
    name = tool.strip().lower()
    if name in _PLAN_TOOLS or ("plan" in name and "explain" not in name):
        return "plan"
    if name in _DELEGATE_TOOLS:
        return "explore"
    if name in _EDIT_TOOLS or "patch" in name:
        return "edit"
    if name in _EXPLORE_TOOLS:
        return "explore"
    if name in _SHELL_TOOLS:
        return _classify_shell(_command_text(args))
    if name.startswith("mcp__") and "browser" in name:
        return "test"
    return "other"


@dataclass
class PhaseSegment:
    """One maximal run of same-phase steps: ``spans[start:end]``."""

    label: str
    start: int  # index of the first step, inclusive
    end: int  # index one past the last step (the change point)
    step_ids: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)

    @property
    def length(self) -> int:
        return self.end - self.start


def _smooth(labels: list[str]) -> list[str]:
    """Relabel single-step runs flanked by the same different label.

    Repeats until stable; each pass reads the previous pass's labels, so
    the outcome is order-independent and deterministic.
    """
    labels = list(labels)
    changed = True
    while changed:
        changed = False
        smoothed = list(labels)
        for i in range(1, len(labels) - 1):
            if labels[i] != labels[i - 1] and labels[i - 1] == labels[i + 1]:
                smoothed[i] = labels[i - 1]
        if smoothed != labels:
            labels = smoothed
            changed = True
    return labels


def segment_phases(spans: list[TrajectoryStep]) -> list[PhaseSegment]:
    """Segment one trajectory's spans into phases with change points."""
    if not spans:
        return []
    labels = _smooth([classify_step(s.tool, s.args) for s in spans])
    segments: list[PhaseSegment] = []
    for i, (span, label) in enumerate(zip(spans, labels)):
        if segments and segments[-1].label == label:
            segments[-1].end = i + 1
            segments[-1].step_ids.append(span.step_id)
            segments[-1].tools.append(span.tool)
        else:
            segments.append(PhaseSegment(
                label=label, start=i, end=i + 1,
                step_ids=[span.step_id], tools=[span.tool],
            ))
    return segments


def segment_attempts(model) -> dict[str, list[PhaseSegment]]:
    """Phases per attempt in an integrity model, keyed by task id."""
    return {a.task_id: segment_phases(a.spans) for a in model.attempts if a.spans}
