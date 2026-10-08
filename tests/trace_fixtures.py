"""Faithful reconstructed Claude Code / Codex session fixtures.

The record shapes follow the documented schemas in the 2026-10-05
trace-tooling scan (``~/workspace/research_notes/coding-agent-trace-tooling-
scan-20261005-2233/report.md`` section 1):

- Claude Code: JSONL, one record per line, types ``user`` / ``assistant`` /
  ``queue-operation`` / ``attachment`` / ``last-prompt``; assistant records
  carry ``message.usage`` per record; tool calls are ``tool_use`` content
  blocks, results come back as ``tool_result`` blocks on user records;
  sub-agents live in ``<session-id>/subagents/agent-<id>.jsonl``.
- Codex: JSONL lines ``{timestamp, type, payload}``; ``response_item`` is
  the authoritative stream, ``event_msg`` a parallel duplicate;
  ``event_msg``/``token_count`` carries *cumulative* totals; the session
  id is the rollout filename UUID, not ``session_meta``'s id.

Expected totals below are hand-computed from the usage constants used to
build the records (the arithmetic is shown next to each constant), never
derived from adapter output.
"""
from __future__ import annotations

import json
from pathlib import Path

# ---------------------------------------------------------------- Claude Code

CLAUDE_CLEAN_SESSION = "11111111-1111-4111-8111-111111111111"
CLAUDE_LOOP_SESSION = "22222222-2222-4222-8222-222222222222"
CLAUDE_SUBAGENT_ID = "a1b2"

# Clean: assistant usage (in, out, cache_create, cache_read) per record:
#   Read turn (100, 20, 10, 50), Grep (200, 30, 0, 150), Edit (300, 40, 20, 200),
#   Bash (150, 25, 0, 100), final text (120, 60, 0, 80)
# in  = 100+200+300+150+120 = 870 ; out = 20+30+40+25+60 = 175
# cache_create = 10+0+20+0+0 = 30 ; cache_read = 50+150+200+100+80 = 580
CLAUDE_CLEAN_TOTALS = {
    "tokens_in": 870,
    "tokens_out": 175,
    "cache_creation": 30,
    "cache_read": 580,
    "spans": 4,
}
# Sub-agent: Grep turn (80, 15, 0, 0), final text (50, 10, 0, 0) -> 130 / 25
CLAUDE_SUBAGENT_TOTALS = {"tokens_in": 130, "tokens_out": 25, "spans": 1}
# Loop: Read (100, 20), Bash npm test x5 (200, 30) each, Read bar (150, 20),
# final text (100, 50). Cache all zero.
# in = 100 + 5*200 + 150 + 100 = 1350 ; out = 20 + 5*30 + 20 + 50 = 240
CLAUDE_LOOP_TOTALS = {"tokens_in": 1350, "tokens_out": 240, "spans": 7}
# Loop waste: the 4 repeats beyond the first Bash turn: 4*(200 in, 30 out).
CLAUDE_LOOP_WASTE = {"calls": 4, "tokens_in": 800, "tokens_out": 120}


def _claude_base(session: str, uuid: str, parent: str | None, ts: str, sidechain: bool = False) -> dict:
    return {
        "parentUuid": parent,
        "isSidechain": sidechain,
        "userType": "external",
        "cwd": "/repo",
        "sessionId": session,
        "version": "2.0.1",
        "gitBranch": "main",
        "uuid": uuid,
        "timestamp": ts,
    }


def _claude_user(session, uuid, parent, ts, content, sidechain=False, **extra) -> dict:
    rec = _claude_base(session, uuid, parent, ts, sidechain)
    rec.update({"type": "user", "message": {"role": "user", "content": content}, **extra})
    return rec


def _claude_tool_result(session, uuid, parent, ts, tool_use_id, text, sidechain=False) -> dict:
    return _claude_user(
        session, uuid, parent, ts,
        [{"type": "tool_result", "tool_use_id": tool_use_id, "content": text}],
        sidechain=sidechain,
    )


def _claude_assistant(session, uuid, parent, ts, blocks, usage, stop="tool_use", sidechain=False) -> dict:
    rec = _claude_base(session, uuid, parent, ts, sidechain)
    rec.update({
        "type": "assistant",
        "message": {
            "model": "claude-sonnet-4-5",
            "id": f"msg_{uuid}",
            "type": "message",
            "role": "assistant",
            "content": blocks,
            "stop_reason": stop,
            "usage": {
                "input_tokens": usage[0],
                "output_tokens": usage[1],
                "cache_creation_input_tokens": usage[2],
                "cache_read_input_tokens": usage[3],
                "service_tier": "standard",
            },
        },
    })
    return rec


def _tool_use(block_id: str, name: str, inp: dict) -> dict:
    return {"type": "tool_use", "id": block_id, "name": name, "input": inp}


def claude_clean_records() -> list[dict]:
    s = CLAUDE_CLEAN_SESSION
    return [
        _claude_user(s, "u1", None, "2026-10-01T09:00:00.000Z",
                     "Explore the repo and fix the typo in README, then check it renders."),
        {"type": "queue-operation", "operation": "enqueue", "timestamp": "2026-10-01T09:00:00.100Z",
         "sessionId": s, "content": "Explore the repo and fix the typo in README"},
        _claude_assistant(s, "a1", "u1", "2026-10-01T09:00:20.000Z",
                          [{"type": "thinking", "thinking": "start with the README"},
                           _tool_use("toolu_01", "Read", {"file_path": "README.md"})],
                          (100, 20, 10, 50)),
        _claude_tool_result(s, "u2", "a1", "2026-10-01T09:00:25.000Z", "toolu_01",
                              "# Title\nteh typo here\n"),
        _claude_assistant(s, "a2", "u2", "2026-10-01T09:00:40.000Z",
                          [_tool_use("toolu_02", "Grep", {"pattern": "teh", "path": "."})],
                          (200, 30, 0, 150)),
        _claude_tool_result(s, "u3", "a2", "2026-10-01T09:00:45.000Z", "toolu_02",
                              "README.md:2: teh typo here"),
        _claude_assistant(s, "a3", "u3", "2026-10-01T09:01:00.000Z",
                          [{"type": "text", "text": "Found it. Fixing now."},
                           _tool_use("toolu_03", "Edit", {"file_path": "README.md",
                                                           "old_string": "teh", "new_string": "the"})],
                          (300, 40, 20, 200)),
        _claude_tool_result(s, "u4", "a3", "2026-10-01T09:01:05.000Z", "toolu_03",
                              "The file README.md has been updated."),
        _claude_assistant(s, "a4", "u4", "2026-10-01T09:01:30.000Z",
                          [_tool_use("toolu_04", "Bash", {"command": "npm run build",
                                                           "description": "Build to check rendering"})],
                          (150, 25, 0, 100)),
        _claude_tool_result(s, "u5", "a4", "2026-10-01T09:01:40.000Z", "toolu_04", "build ok"),
        {"type": "attachment", "uuid": "att1", "parentUuid": "a4", "sessionId": s,
         "timestamp": "2026-10-01T09:01:41.000Z",
         "attachment": {"type": "file", "filename": "notes.txt"}},
        _claude_assistant(s, "a5", "u5", "2026-10-01T09:02:00.000Z",
                          [{"type": "text", "text": "Fixed teh -> the and the build passes."}],
                          (120, 60, 0, 80), stop="end_turn"),
        {"type": "last-prompt", "lastPrompt": "Explore the repo and fix the typo in README",
         "leafUuid": "a5", "sessionId": s},
    ]


def claude_subagent_records() -> list[dict]:
    s = CLAUDE_CLEAN_SESSION
    return [
        _claude_user(s, "su1", None, "2026-10-01T09:00:30.000Z",
                     "Scan the docs for TODO markers.", sidechain=True, agentId=CLAUDE_SUBAGENT_ID),
        _claude_assistant(s, "sa1", "su1", "2026-10-01T09:00:50.000Z",
                          [_tool_use("toolu_s1", "Grep", {"pattern": "TODO", "path": "docs"})],
                          (80, 15, 0, 0), sidechain=True),
        _claude_tool_result(s, "su2", "sa1", "2026-10-01T09:00:55.000Z", "toolu_s1",
                            "docs/guide.md:10: TODO", sidechain=True),
        _claude_assistant(s, "sa2", "su2", "2026-10-01T09:01:10.000Z",
                          [{"type": "text", "text": "One TODO found."}],
                          (50, 10, 0, 0), stop="end_turn", sidechain=True),
    ]


def claude_loop_records() -> list[dict]:
    s = CLAUDE_LOOP_SESSION
    records = [
        _claude_user(s, "u1", None, "2026-10-01T10:00:00.000Z", "Make the test suite pass."),
        _claude_assistant(s, "a1", "u1", "2026-10-01T10:00:20.000Z",
                          [_tool_use("toolu_01", "Read", {"file_path": "src/foo.py"})],
                          (100, 20, 0, 0)),
        _claude_tool_result(s, "u2", "a1", "2026-10-01T10:00:25.000Z", "toolu_01", "def foo(): ..."),
    ]
    parent = "u2"
    for i in range(5):  # the planted loop: identical Bash call five times
        a_uuid, u_uuid = f"b{i + 1}", f"r{i + 1}"
        records.append(_claude_assistant(
            s, a_uuid, parent, f"2026-10-01T10:0{i + 1}:00.000Z",
            [_tool_use(f"toolu_b{i + 1}", "Bash", {"command": "npm test",
                                                    "description": "Run tests"})],
            (200, 30, 0, 0)))
        records.append(_claude_tool_result(
            s, u_uuid, a_uuid, f"2026-10-01T10:0{i + 1}:30.000Z", f"toolu_b{i + 1}",
            "1 failing test: test_checkout"))
        parent = u_uuid
    records += [
        _claude_assistant(s, "a9", parent, "2026-10-01T10:07:00.000Z",
                          [_tool_use("toolu_09", "Read", {"file_path": "src/bar.py"})],
                          (150, 20, 0, 0)),
        _claude_tool_result(s, "u9", "a9", "2026-10-01T10:07:10.000Z", "toolu_09", "def bar(): ..."),
        _claude_assistant(s, "a10", "u9", "2026-10-01T10:08:00.000Z",
                          [{"type": "text", "text": "Still failing; needs a different fix."}],
                          (100, 50, 0, 0), stop="end_turn"),
    ]
    return records


def write_claude_project(root: Path, records: list[dict], session: str,
                         subagent_records: list[dict] | None = None) -> Path:
    """Write one session file (plus optional sub-agent sidechain) under root."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{session}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    if subagent_records is not None:
        sub_dir = root / session / "subagents"
        sub_dir.mkdir(parents=True, exist_ok=True)
        (sub_dir / f"agent-{CLAUDE_SUBAGENT_ID}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in subagent_records), encoding="utf-8")
        (sub_dir / f"agent-{CLAUDE_SUBAGENT_ID}.meta.json").write_text(
            json.dumps({"agentId": CLAUDE_SUBAGENT_ID,
                        "description": "Scan the docs for TODO markers."}),
            encoding="utf-8")
    return path


# --------------------------------------------------------------------- Codex

CODEX_CLEAN_FILE = "rollout-2026-10-01T09-00-00-33333333-3333-4333-8333-333333333333.jsonl"
CODEX_CLEAN_SESSION = "33333333-3333-4333-8333-333333333333"
CODEX_LOOP_FILE = "rollout-2026-10-02T10-30-00-44444444-4444-4444-8444-444444444444.jsonl"
CODEX_LOOP_SESSION = "44444444-4444-4444-8444-444444444444"
CODEX_LINEAGE_ID = "00000000-0000-4000-8000-000000000000"  # session_meta id: NOT the session id

# Clean cumulative token_count snapshots (input, cached, output, reasoning):
#   s1 (1000, 400, 100, 40) -> s2 (2500, 900, 260, 90) -> s2 repeated verbatim
#   -> s3 (4200, 1500, 500, 150)
# Attempt totals are the final cumulative values: in 4200, out 500.
# Deltas: (1000, 100), (1500, 160), (0, 0), (1700, 240); they sum to the totals.
CODEX_CLEAN_TOTALS = {"tokens_in": 4200, "tokens_out": 500, "cached": 1500,
                      "reasoning": 150, "spans": 4}
# Loop cumulative snapshots (input, cached, output, reasoning):
#   (2000,500,300,100) (5000,1200,900,300) then one per pytest turn
#   (5600,1400,1050,350) (6200,1600,1200,400) (6800,1800,1350,450)
#   (7400,1900,1500,475) and final (9000,2000,1700,500).
# Attempt totals: in 9000, out 1700. Each pytest turn's delta is (600, 150),
# so the 3 repeats beyond the first cost 3*(600 in, 150 out) = 1800 / 450.
CODEX_LOOP_TOTALS = {"tokens_in": 9000, "tokens_out": 1700, "cached": 2000,
                     "reasoning": 500, "spans": 12}
CODEX_LOOP_WASTE = {"calls": 3, "tokens_in": 1800, "tokens_out": 450}
# Planted phase structure of the loop fixture, as (label, start, end) spans:
CODEX_LOOP_PHASES = [("explore", 0, 3), ("plan", 3, 4), ("edit", 4, 6),
                     ("test", 6, 10), ("review", 10, 12)]


def _codex(ts: str, type_: str, payload: dict) -> dict:
    return {"timestamp": ts, "type": type_, "payload": payload}


def _codex_meta(ts: str) -> dict:
    return _codex(ts, "session_meta", {
        "id": CODEX_LINEAGE_ID, "timestamp": ts, "cwd": "/repo",
        "originator": "codex_cli_rs", "cli_version": "0.42.0",
        "source": "cli", "model_provider": "openai",
    })


def _codex_token_count(ts: str, total: tuple[int, int, int, int]) -> dict:
    inp, cached, out, reasoning = total
    return _codex(ts, "event_msg", {"type": "token_count", "info": {
        "total_token_usage": {
            "input_tokens": inp, "cached_input_tokens": cached,
            "output_tokens": out, "reasoning_output_tokens": reasoning,
            "total_tokens": inp + out,
        },
        "last_token_usage": {
            "input_tokens": inp, "cached_input_tokens": cached,
            "output_tokens": out, "reasoning_output_tokens": reasoning,
            "total_tokens": inp + out,
        },
    }})


def _codex_call(ts: str, call_id: str, name: str, arguments: dict) -> dict:
    return _codex(ts, "response_item", {"type": "function_call", "name": name,
                                        "call_id": call_id,
                                        "arguments": json.dumps(arguments)})


def _codex_call_output(ts: str, call_id: str, output: str) -> dict:
    return _codex(ts, "response_item", {"type": "function_call_output",
                                        "call_id": call_id, "output": output})


def _codex_patch(ts: str, call_id: str, patch: str) -> dict:
    return _codex(ts, "response_item", {"type": "custom_tool_call", "name": "apply_patch",
                                        "call_id": call_id, "input": patch,
                                        "status": "completed"})


def _codex_patch_output(ts: str, call_id: str) -> dict:
    return _codex(ts, "response_item", {"type": "custom_tool_call_output",
                                        "call_id": call_id, "output": "Success. Updated the file."})


def codex_clean_records() -> list[dict]:
    return [
        _codex_meta("2026-10-01T09:00:00.000Z"),
        _codex("2026-10-01T09:00:01.000Z", "event_msg", {"type": "task_started", "turn_id": "t1"}),
        _codex("2026-10-01T09:00:01.100Z", "turn_context",
               {"turn_id": "t1", "model": "gpt-5-codex", "cwd": "/repo"}),
        _codex("2026-10-01T09:00:02.000Z", "response_item",
               {"type": "message", "role": "user",
                "content": [{"type": "input_text", "text": "Add a --verbose flag to the CLI and test it."}]}),
        # event_msg duplicate of the same user message: must never double-count.
        _codex("2026-10-01T09:00:02.100Z", "event_msg",
               {"type": "user_message", "message": "Add a --verbose flag to the CLI and test it."}),
        _codex("2026-10-01T09:00:10.000Z", "response_item", {"type": "reasoning", "summary": []}),
        _codex_call("2026-10-01T09:00:11.000Z", "call_1", "shell", {"command": ["ls", "src"]}),
        _codex_call_output("2026-10-01T09:00:12.000Z", "call_1", "cli.py\nmain.py\n"),
        _codex_call("2026-10-01T09:00:20.000Z", "call_2", "shell",
                    {"command": ["sed", "-n", "1,120p", "src/cli.py"]}),
        _codex_call_output("2026-10-01T09:00:21.000Z", "call_2", "import argparse\n..."),
        _codex_token_count("2026-10-01T09:00:22.000Z", (1000, 400, 100, 40)),
        _codex("2026-10-01T09:00:30.000Z", "response_item",
               {"type": "message", "role": "assistant",
                "content": [{"type": "output_text", "text": "Found the CLI entry. Patching it."}]}),
        _codex("2026-10-01T09:00:30.100Z", "event_msg",
               {"type": "agent_message", "message": "Found the CLI entry. Patching it."}),
        _codex_patch("2026-10-01T09:00:31.000Z", "call_3",
                     "*** Begin Patch\n*** Update File: src/cli.py\n"
                     "+    parser.add_argument('--verbose')\n*** End Patch"),
        _codex_patch_output("2026-10-01T09:00:32.000Z", "call_3"),
        _codex("2026-10-01T09:00:32.100Z", "event_msg",
               {"type": "patch_apply_end", "call_id": "call_3", "success": True}),
        _codex_token_count("2026-10-01T09:00:33.000Z", (2500, 900, 260, 90)),
        # Verbatim repeat of the same cumulative snapshot: delta is zero.
        _codex_token_count("2026-10-01T09:00:33.500Z", (2500, 900, 260, 90)),
        _codex_call("2026-10-01T09:01:00.000Z", "call_4", "shell", {"command": ["pytest", "-q"]}),
        _codex_call_output("2026-10-01T09:01:05.000Z", "call_4", "3 passed"),
        _codex_token_count("2026-10-01T09:01:06.000Z", (4200, 1500, 500, 150)),
        _codex("2026-10-01T09:01:10.000Z", "response_item",
               {"type": "message", "role": "assistant",
                "content": [{"type": "output_text", "text": "Added --verbose; tests pass."}]}),
        _codex("2026-10-01T09:01:11.000Z", "event_msg", {"type": "task_complete", "turn_id": "t1"}),
    ]


def codex_loop_records() -> list[dict]:
    records = [
        _codex_meta("2026-10-02T10:30:00.000Z"),
        _codex("2026-10-02T10:30:01.000Z", "event_msg", {"type": "task_started", "turn_id": "t1"}),
        _codex("2026-10-02T10:30:01.100Z", "turn_context",
               {"turn_id": "t1", "model": "gpt-5-codex", "cwd": "/repo"}),
        _codex("2026-10-02T10:30:02.000Z", "response_item",
               {"type": "message", "role": "user",
                "content": [{"type": "input_text",
                             "text": "Investigate the flaky checkout test, fix it, and verify."}]}),
        _codex("2026-10-02T10:30:02.100Z", "event_msg",
               {"type": "user_message",
                "message": "Investigate the flaky checkout test, fix it, and verify."}),
        # explore: spans 0-2
        _codex_call("2026-10-02T10:31:00.000Z", "call_1", "shell", {"command": ["ls", "-R", "src"]}),
        _codex_call_output("2026-10-02T10:31:01.000Z", "call_1", "src/checkout.py\nsrc/cart.py\n"),
        _codex_call("2026-10-02T10:32:00.000Z", "call_2", "shell", {"command": ["cat", "src/checkout.py"]}),
        _codex_call_output("2026-10-02T10:32:01.000Z", "call_2", "def checkout(): ..."),
        _codex_call("2026-10-02T10:33:00.000Z", "call_3", "shell",
                    {"command": ["grep", "-rn", "retry", "src"]}),
        _codex_call_output("2026-10-02T10:33:01.000Z", "call_3", "src/checkout.py:12: retry(3)"),
        _codex_token_count("2026-10-02T10:33:02.000Z", (2000, 500, 300, 100)),
        # plan: span 3
        _codex_call("2026-10-02T10:34:00.000Z", "call_4", "update_plan",
                    {"plan": [{"step": "Reproduce the flake", "status": "in_progress"},
                              {"step": "Fix the retry", "status": "pending"}]}),
        _codex_call_output("2026-10-02T10:34:01.000Z", "call_4", "plan updated"),
        # edit: spans 4-5
        _codex_patch("2026-10-02T10:35:00.000Z", "call_5",
                     "*** Begin Patch\n*** Update File: src/checkout.py\n"
                     "-    retry(3)\n+    retry(3, backoff=2)\n*** End Patch"),
        _codex_patch_output("2026-10-02T10:35:01.000Z", "call_5"),
        _codex_patch("2026-10-02T10:36:00.000Z", "call_6",
                     "*** Begin Patch\n*** Update File: src/checkout.py\n"
                     "+    # backoff avoids the flaky window\n*** End Patch"),
        _codex_patch_output("2026-10-02T10:36:01.000Z", "call_6"),
        _codex_token_count("2026-10-02T10:36:02.000Z", (5000, 1200, 900, 300)),
    ]
    # test: spans 6-9, the planted loop (identical pytest call, own turn each)
    snapshots = [(5600, 1400, 1050, 350), (6200, 1600, 1200, 400),
                 (6800, 1800, 1350, 450), (7400, 1900, 1500, 475)]
    for i, snap in enumerate(snapshots):
        cid = f"call_{7 + i}"
        records.append(_codex_call(f"2026-10-02T10:4{i}:00.000Z", cid, "shell",
                                   {"command": ["pytest", "-q", "tests/test_checkout.py"]}))
        records.append(_codex_call_output(f"2026-10-02T10:4{i}:30.000Z", cid,
                                          "1 failed, 2 passed"))
        records.append(_codex_token_count(f"2026-10-02T10:4{i}:31.000Z", snap))
    records += [
        # review: spans 10-11
        _codex_call("2026-10-02T10:44:00.000Z", "call_11", "shell", {"command": ["git", "diff"]}),
        _codex_call_output("2026-10-02T10:44:01.000Z", "call_11", "diff --git a/src/checkout.py"),
        _codex_call("2026-10-02T10:44:30.000Z", "call_12", "shell",
                    {"command": ["git", "status", "--short"]}),
        _codex_call_output("2026-10-02T10:44:31.000Z", "call_12", " M src/checkout.py"),
        _codex_token_count("2026-10-02T10:45:00.000Z", (9000, 2000, 1700, 500)),
        _codex("2026-10-02T10:45:01.000Z", "response_item",
               {"type": "message", "role": "assistant",
                "content": [{"type": "output_text", "text": "Fixed the retry; test still flaky."}]}),
        _codex("2026-10-02T10:45:02.000Z", "event_msg", {"type": "task_complete", "turn_id": "t1"}),
    ]
    return records


def write_codex_session(root: Path, records: list[dict], filename: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / filename
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path
