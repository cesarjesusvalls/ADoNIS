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

## #3 — baseline + scripts tidy (2026-06-12)

- Baseline (pre-change): 31 passed, 1 failed (-x run). Failure = stale assertion in
  tests/test_fold_fs.py::test_final_state_onshell expecting on-shell pion at 138.04;
  the pion-mass convention fix puts pions on-shell at mpi0=134.98 (kin_m_pi).
  Fixed the assertion (862ac1e). Full suite without -x = 87 tests.
- Committed cc0pi_tune_{t2k,adonis,nominal}.py (003854a); deleted 10 untracked
  _-prefixed one-off debug scripts; removed tracked scratch/ (afcde37).

## #4 — refactor 1: constants (2026-06-12, commit 2b3aaa1)

- Canonical module = adonis/constants.py (Constants.hh precise + Particles.yml MASS_PDG_*
  + DCC aliases); adonis/xsec/constants.py -> re-export shim (21 importers untouched).
- Particles.yml verified from ../Achilles/data/Particles.yml: p 938.27, n 939.57,
  pi+ 139.571, pi0 134.977, mu 105.7.
- Replaced value-identical literals in backend, res_xsec, form_factors, flux, oset_xsec,
  anl_xsec, llewellyn_smith. nuclear/{qe_inclusive,inclusive_1pi}: 938.919 / 1/137.036 ->
  precise mN / alpha (ACHILLES uses precise; rel change <=3e-7).
- DELIBERATELY left local: fsi/mb/anl_xsec.py HBARC=197.32; fsi/oset_xsec.py
  NORM_FACTOR=197.327^2*10 (verbatim fNormFactor); fsi/cascade.py M_PI=139.57 — NOTE:
  Particles.yml is 139.571, so 139.57 may be a ~1e-5 infidelity in the pion energy-loss
  on-shell step; not touched because the 245 MeV residual is parked. llewellyn_smith
  COSTHC=0.97373 differs from Constants.hh Vud=0.97367 (provenance uncheched, left as-is).
- Gate: full suite 87 passed.

## #5 — refactors 2/3/5: Sample schema, xsec "legacy", prints (2026-06-12)

- Sample stays dict at runtime (deliberate, jit-friendly; some entries are static objects).
  Schema now ENFORCED at the seam: Channel.sample_fields (exact key set) +
  core/sample.check_sample; Generator validates every proposal. DCCSinglePion declares
  its 19 keys. Verified: generate unchanged; missing-key raises KeyError.
- Audit claims corrected on inspection:
  * adonis/data/oracle/ = importable parser library used by 10+ scripts and documented
    in README layout — NOT stray scripts. Kept in the package (refactor 4 dropped).
  * xsec/ has no legacy pile: dirac.py is the LIVE Fortran-path QE current; hadronic_qe.py
    (0 importers) is the bit-exact C++ Weyl QESpectral port — kept with a docstring NOTE.
  * all 34 print() in the package are __main__/CLI output or verbose-gated live fit
    progress (flush=True) — required visible behavior; prints->logging dropped as no-op.
