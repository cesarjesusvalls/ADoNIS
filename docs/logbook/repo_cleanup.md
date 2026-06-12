# Logbook: repo tidy-up (started 2026-06-12)

Goal: prune investigation pollution, restore the original architecture goals
(NamedTuples, inheritance, modularity, per-module validation), before further physics work.
The 245 MeV cascade residual is PARKED (see `cascade_transport_residual.md`).

## #1 — audit (2026-06-12)

Three parallel read-only audits (docs, package architecture, scripts/outputs):
- docs: 4 superseded (res_investigation_log, res_WQ2_investigation_findings,
  cc0pi_fsi_investigation, REPRODUCTION_PLAN); parked logbook = cascade_transport_residual.md;
  phases/ + reference docs healthy. STATUS.md stale (pre-ADoNIS "diffpi", Phase 3 "NOT STARTED").
- architecture vs goals: NamedTuples/pytrees largely consistent (exception: Sample = raw dict);
  ABC hierarchy present (Channel/NuclearModel/FluxModel/FSIModel); no circular imports;
  constants re-hardcoded in res_xsec.py, form_factors.py, cascade.py, oset_xsec.py;
  legacy reference impls mixed into xsec/; adonis/data/oracle holds script-like processing;
  32 tests incl. autodiff==FD gates.
- clutter: 73 tracked scripts (durable), 14 untracked (10 `_`-debug + 3+1 cc0pi_tune_*);
  scratch/ = distilled prototypes; PNGs already gitignored (0 tracked); _oracle_out 5.9 GB
  gitignored (absrun/ = 1.8 GB; disk cleanup deferred).

User decisions: delete superseded docs (git history retains); commit cc0pi_tune_*, delete
`_`-debug scripts; do all 5 refactors test-gated; no _oracle_out cleanup now.

## #2 — docs tidy (2026-06-12)

- Deleted: docs/{REPRODUCTION_PLAN,res_investigation_log,res_WQ2_investigation_findings,cc0pi_fsi_investigation}.md
- Moved docs/cascade_transport_residual.md -> docs/logbook/ (parked, KEEP)
- Fixed living refs: README.md layout line, docs/phases/README.md plan anchor,
  res_amps2_frame_fix.md superseded-note, paper_figures/{free_nucleon,diagnostic}_WQ2.py docstrings
- Rewrote docs/STATUS.md (was stale "diffpi" Phase-1/2 text)
- Baseline test run launched before any changes: /tmp/adonis_baseline_tests.log
