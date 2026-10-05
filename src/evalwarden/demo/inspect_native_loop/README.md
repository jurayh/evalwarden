# inspect_native_loop — a real Inspect AI `.eval` log with a tool loop

`log.eval` is a native Inspect AI log (schema version 2), generated with
Inspect AI 0.3.276 — not hand-built. It was produced by running a real
Inspect task (`evalwarden-native-loop`) with the built-in `mockllm/model`,
a ReAct solver, a `lookup` tool, and the `exact` scorer:

- `loop-1` calls `lookup({"query": "status"})` five times in a row before
  submitting — a genuine repeated tool loop. Its sample metadata carries
  `failure_mode: "tool-loop"`.
- `plain-1` / `plain-2` each look up once and submit. `plain-2`'s lookup
  returns a >512-char result, which the adapter truncates at its boundary.

Auditing it natively (`evalwarden audit log.eval`) fires TRAJ-001 on the
loop. TRAJ-002 stays silent: native ToolEvent records carry no downstream
consumption edges, and the adapter never infers them.
