# ADoNIS TODO

Living task list.  Verified status in brackets: [V]=code-confirmed, [?]=needs a check.

## BFS / legacy-engine leftovers (vestigial since the pool became the SOLE cascade engine)
The BFS classes/functions were deleted ([[always-pool-never-bfs]]), but several BFS-era constants/params
survive unused.  Confirmed (grep): 0 uses in the pool (`cascade_full.py`); they appear only in their
definitions, a `generate.py` consistency-check/banner, configs, and diagnostic scripts.
- [ ] **`_N_RECOIL`** (cascade_discrete.py:93) — legacy top-K knockout-recoil cap.  Pool spawns 1 knockout
      /step (bounded by M / M_out), never uses it.  Remove the constant + `workflow/config.py:n_recoil`
      + the `generate.py` consistency-check (lines ~106-108) + the banner field (~156) + `ADONIS_N_RECOIL`
      plumbing (scripts/adonis_generate.py).  [V vestigial]
- [ ] **`_MAX_SEG`** (cascade_discrete.py:86) — legacy "interaction-driven kernel" per-particle attempt
      cap.  No use beyond the def.  Remove.  [V]
- [ ] **`_K_BR`, `_K_SLAB_REC`** (cascade_discrete.py:89,91) — legacy compressed-record slot counts.  Only
      referenced by scripts/test_cascade_caps.py (display) — not by the pool or the tuning rec_caps (which
      pass literal Kp,Kn).  Remove the constants (+ the test reference).  [V]
- [ ] **`max_gen`** param — accepted by `cascade_nucleus`/`cascade_carbon`/`run_fsi`/`_cascade_pool` but
      the pool has no generations -> it does NOTHING (only a docstring mention).  Drop the param + update
      ~10 call sites (grep-verify).  [V vestigial passthrough]
- [ ] Scrub the ~32 remaining `BFS|legacy|propagate_discrete|*_segment` references (mostly stale
      docstrings/comments) so the codebase reads pool-only.  [? mostly comments]
- [ ] After removal: `grep -rn "_N_RECOIL|_MAX_SEG|_K_BR|_K_SLAB_REC|max_gen|BFS" adonis/ scripts/` clean.

## Cat-3 fixed-buffer caps (audit jax_forced_divergences.md §3)
- [ ] **M_out** (out-buffer, escaped finals) is HARDCODED 24 in `_cascade_pool` (lines ~417,422).  The
      ~0.5% Ar overflow at P=12 combined sofl(stack)+oofl(out); for Ar the out-buffer likely dominates.
      Disentangle (scripts/test_cascade_buffers.py, running) -> if oofl: make M_out a param + bump for Ar.
- [ ] **max_steps** — particles still alive when the step loop ends are dropped SILENTLY (no counter).
      Check Ar (~1000 steps): instrument an alive-at-end count or compare out vs a 2x-max_steps run.
- [ ] **`_KSLAB=3`** (fast_xsec K) — run scripts/test_kslab_ab.py (fast_xsec True=K3 vs False=all nucleons,
      Ar); if channel fractions differ >1%, bump K or default fast_xsec=False.
- [x] log_cap: safe (Ar max 32 segments/event; default lowered 64->48, 1.5x margin).
- [ ] Commit the P=12->16 (cascade_nucleus/run_fsi/gen_cc_engine_rich) + log_cap 64->48 default changes
      once the disentangle/P-scan confirms the right P (and whether M_out also needs bumping).

## Audit follow-ups (docs/logbook/jax_forced_divergences.md)
- [ ] bilinear amplitude interp still ACTIVE in NON-blueprint paths (inclusive_1pi.py:27,
      sigma_enu.py:204/249/347) -> verify closure-cancels vs needs-spline.  (User: separate session.)
- [ ] Quantify the bilinear dsigma/dW tail deviation explicitly so the magnitude is on record.
- [ ] Promote remaining [R] audit items to [V] (second read).

## Cascade-vertex/segment matrix
- [ ] Full-stats matrix: the C run was stopped at 3 seeds (per request).  Re-render the 3-seed matrix;
      run Ar.  Watch whether the soft QE-p transmit/elastic cluster firms or washes out (was 1.9 @seed1
      -> dropped @seed2).
- [x] sigma_nn_ndelta near-threshold integration FIXED (NN->NNpi cell 3.32->0.42).
- [x] In-engine segment logger + chi2/ndf heatmap + overlay/ratio plots.

## Done this session (for reference)
- [x] Cascade-vertex/segment matrix pipeline (gen + ACHILLES VERTEXDUMP + driver), G4-like logger.
- [x] DCC bilinear default -> spline (W-shape footgun); RES confirmed spline.
- [x] SF importance sampler validated (peak-resolved + correlation, after fixing a test aliasing bug).
- [x] 4.2 straight-line: NOT a divergence (PotentialProp:False -> ACHILLES also straight-line).
- [x] cascade_carbon/setup_carbon -> cascade_nucleus/setup_nucleus rename.
