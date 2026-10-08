# Trace observatory demo fixtures

Faithful reconstructions of Claude Code and Codex session files (see
tests/trace_fixtures.py for how they were built and the hand-computed
token totals). `evalwarden trace --demo` reads this directory: the
Claude project holds a clean session, its sub-agent sidechain, and a
session with a planted 5x `npm test` loop; the Codex tree holds a clean
rollout and one with a planted 4x `pytest` loop inside a full
explore/plan/edit/test/review phase structure.
