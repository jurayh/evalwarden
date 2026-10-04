# Agent-loop fixture

An ordinary agent eval whose run records include tool-call trajectories. One attempt (t1) calls the identical failing test command five times in a row and leaves most of its tool outputs unconsumed: TRAJ-001 (exact loop) and TRAJ-002 (unused outputs) fire. No judge or repeated-run data is recorded, so the JUDGE, NOISE, and coverage lanes stay silent.
