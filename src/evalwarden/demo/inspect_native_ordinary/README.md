# inspect_native_ordinary — a real Inspect AI `.eval` log, nothing planted

`log.eval` is a native Inspect AI log (schema version 2), generated with
Inspect AI 0.3.276 — not hand-built. It was produced by running a real
Inspect task (`evalwarden-native-ordinary`) with the built-in
`mockllm/model`, a ReAct solver, a `lookup` tool, and the `exact` scorer:
12 samples, each run for 3 epochs, every sample answered correctly.

Auditing it natively (`evalwarden audit log.eval`) produces no TRAJ or
NOISE findings, while the model is fully populated: 12 tasks (targets and
choices preserved), 36 attempts with spans and token usage, and 36
run-score rows (one `exact` score per sample per epoch, epoch as the
generation index). One honest consequence of the model: epochs land as
repeated attempts per task, so COST-002 reads the 3-epoch protocol as
3 tries per success — the same reading the JSON adapter path gives
repeated run entries.
