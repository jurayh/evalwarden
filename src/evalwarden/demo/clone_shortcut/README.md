# Clone-shortcut fixture

24 selection tasks (roster of scored names; answer = everyone at or above
a threshold) with cached model outputs: 22 correct, 2 wrong. The grader
declares the `exact_string` scoring rule -- a pass requires the output to
equal the stored answer string exactly.

GRAD-003 re-scores the cached outputs on isomorphic clones (renamed
entities, reordered rosters, rescaled values) and finds a large clone
gap: the verdicts depend on the answer's surface form, not on task
success. Audits BLOCKED by GRAD-003.
