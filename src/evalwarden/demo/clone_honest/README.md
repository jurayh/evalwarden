# Clone-honest fixture

The same 24 selection tasks and cached outputs as `clone_shortcut`.
The grader declares the `set_match` scoring rule -- a pass requires the
parsed selection to equal the reference set, in any order.

GRAD-003 re-scores the cached outputs on isomorphic clones and finds no
gap: the verdicts check the underlying selection, so they survive
renamed entities, reordered rosters, and rescaled values. Audits PASS.
