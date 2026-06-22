# Pool → differentiable core: unification log

**Goal.** Make the pool cascade the single core cascade of ADoNIS — faithful forward (its physics/perf
fixes) AND differentiable via the blueprint's kind-1 walk/weight split. Pool replaces BFS + the legacy
`DiscreteCascadeFSI`/`DiscreteNucleonFSI` and becomes the blueprint engine, preserving the blueprint
*pattern* (frozen θ-independent walk + pure reweight + replica banks + autodiff==FD).

**Decisions (user).** (1) single joint engine; (2) mirror exactly `sabs`/`sscat`; (3) delete dead paths
LAST, commit+push at every stage including before cleanup.

## Reweight spec to mirror (legacy)
- `pion_branch_reweight(brec=(bc,sa,ss,si,nh), sabs,sscat)`: per pion hit, branch-ratio LR with
  `p_abs=sabs·sa/D, p_scat=sscat·ss/D, p_conv=si/D`, `D=sabs·sa+sscat·ss+si`. (Pion hit-probability is
  NOT reweighted in the legacy — branch split only.)
- `nucleon_scat_reweight(srec=(hit,a_nom,ns), sscat)`: per nucleon candidate step, hit-prob LR
  `p=exp(-a_nom)` nominal vs `exp(-a_nom/sscat)`; `where(hit, p_k/p_n, (1-p_k)/(1-p_n))`.
- Pool `_pion_step` already returns `(has_hit, bcode, sa_j, ss_j, si_j)`; `_nucleon_step` returns
  `(has_hit, perp2_c, sig_c)` → `a_nom=π·perp2_c/(sig_c·MB_TO_FM2)` for candidate steps (perp2_c<1e6).

## Design
Accumulate per-event record buffers in the `run_cascade_pool` while-loop carry (NOT threaded through
compaction): the stepper surfaces per-slot per-step stats; each step scatters interacting slots into
fixed-capacity event buffers via a running per-event write-index (scratch-slot trick + overflow count).
`pool_fsi_reweight(record, sabs, sscat) = pion_branch_reweight × nucleon_scat_reweight`.

## Stages (each commit+push)
1. Pool emits kind-1 FSI records + `pool_fsi_reweight`. Gate: θ=(1,1)≡1 bit-exact; forward byte-identical.
2. autodiff==FD closure on `pool_fsi_reweight`. NOTE (corrected): the legacy walk is FULLY θ-independent
   (`_propagate_discrete` branch decision uses nominal `p_abs=sa/sig`; θ enters only via `w_fsi=
   pion_branch_reweight`). The pool walk is likewise θ-independent (Stage 1), so "reweight==in-walk" is
   automatic by construction (the engine multiplies event w by `pool_fsi_reweight(record,θ)` at apply,
   exactly as legacy multiplies by w_fsi). Threading θ into the walk would VIOLATE kind-1 — not done.
   Stage 2 = the autodiff==FD gradient gate only.
3. Single pool-backed FSI engine + migrate blueprint (cc0pi_tune*, cc0pi_disaggregated, cc1pi_fig_tki,
   gen_t2k_cc0pi/cc1pi_adonis) + reweight tests. Gate: cc0pi_tune via pool ≈ legacy within stats; ADoNIS
   vs ACHILLES forward unchanged. COMMIT (pre-cleanup checkpoint).
4. Cleanup: remove BFS forward + legacy classes + propagate_discrete; engine switch; extract cascade_real
   utilities, drop RealCascadeFSI/cascade_exact/dead tests. Gate: full suite green.

## DECISION: all-Gaussian (this session)
Use the Gaussian interaction probability everywhere (ADoNIS + ACHILLES), not Cylinder — see
[[all-gaussian-cascade-probability]]. Cylinder is a deterministic geometric hit (no kind-1 sigma
reweight); Gaussian is stochastic -> exact differentiable reweight, and both integrate to the same
total sigma. Apples-to-apples requires REGENERATING the ACHILLES T2K references in Gaussian (they used
Cylinder). Transition: finish unification (Gaussian) -> flip gen configs cylinder:false -> regen
ACHILLES+ADoNIS in Gaussian -> re-validate (48-cell matrices, lead #2, 0p).

Self-review fixes (applied before Stage 3 commit): rec emission is now OPT-IN at stepper build
(make_pool_stepper with_rec=) so forward generation pays zero overhead; build_replica_pool uses
max_steps=600 as an early-exit CEILING (removes the unverified legacy-260 truncation risk at no cost),
Kn 256->64, and asserts pool stack/out overflow==0. Open self-review item: the pool reweights ALL
nucleon scatters (more complete than legacy leading+1) -> the FIT BFP legitimately differs from legacy;
the Stage 3 gate is the NOMINAL prediction match (+ ACHILLES + autodiff), NOT BFP-match.

## ALL-GAUSSIAN VALIDATION (apples-to-apples, the new reference)
Built the Gaussian ACHILLES C reference + Gaussian ADoNIS and compared like-for-like:
- ACHILLES: `_oracle_out/run_T2K_C_fate_gauss.yml` (Probability: Gaussian). Single 2M runs hit a
  DETERMINISTIC mid-run SIGSEGV (fixed Seed 12345678 always crashes at the same event ~60k), so
  `scripts/gen_ach_gauss_batched.py` batches over distinct seeds (override Options.Initialize.Seed),
  keeps partials, and `scripts/combine_ach_gauss.py` concats with the POOLED normalization
  (per-event nb = w_i·σ̄_b/Σw_all, weight_to_nb=1; reduces to single-file for B=1). 11 batches ->
  2,024,263 events, σ_b spread 2.4%. (ACHILLES native arm64 `achilles:fullcascade` ~4.6 ms/event.)
- ADoNIS: 5-seed Gaussian banks (`configs/gen_c_{res,qe}_5seedG.yaml`, cylinder=false, pool, 600). RES
  413,908 ev, QE 150k. 1-seed Gaussian wall time ~840 s (cascade ~10 ms/ev).
- Matrix `gen_cc_matrix.py CG` (new "CG" material) -> `paper_figures/cc_matrix_CG.pdf`.

RESULT (ACH/ADO, main channels ~1%): CC0π QE incl 0.996, CC0π RES incl 1.012, CC0π both 0.998,
CC1π RES incl 0.994, CC1π both 0.985; QE/RES proton splits ~0.99-1.03. **The earlier "RES 9% low"
(1.092) was a LOW-STATS artifact (1 ADoNIS seed vs 50k ACHILLES) — gone at 5-seed/2M.** Off cells are
genuinely low-stats: CC1π·QE (cascade-created pions only, N~35-126) + a few 2p tails; CC0π·RES 0p
(0.773, N=30) is the known soft-knockout tail. Pipeline: `scripts/_pipeline_5seed_gauss.sh`.

PERF note (`scripts/_pool_timing.py`): pool ~9x the legacy SEGMENT cascade per replica (intrinsic
per-step compaction, NOT my records which add ~0; max_steps 600 vs 260 is a real 2x but 600 is the
production value). cylinder==gaussian speed. The legacy segment is fast only because it is the
approximate proton-only path (the thing lead #2 fixed).

## STAGE 3 COMPLETE: pool-backed differentiable tune wired + proven end-to-end
`cc0pi_tune_adonis.py main()` now defaults to the POOL engine (env `CC0PI_POOL=1`, `CC0PI_N`,
`CC0PI_NREP`; legacy via `CC0PI_POOL=0`) — `_build`/`_hist` = `build_replica_pool`/`model_hist_pool`.
Proof run (NQE=NRES=10000, NREP=2, T2K dpt): replica bank built, 400-iter Adam fit converged via
autodiff through `pool_fsi_reweight`, BFP + Hessian produced (s_abs railed at clip 3.0, s_scat 1.00,
χ²/ndf 1.40). So the pool IS the differentiable blueprint end-to-end. (Railed BFP + Hessian-at-rail are
tiny-stats artifacts of the proof, NOT meaningful values; a real tune needs more stats — but the pool
is ~9x the legacy segment per replica, so full-stats banks are expensive: the open perf question.)
The fit itself succeeded; only the final plot line had a leftover legacy `model_hist` ref (fixed).

## Log
- Baseline: `tests/test_cascade_reweight_records.py` + `tests/test_cascade_pool.py` → 8 passed (pre-work).
- **Stage 1 done** (`scripts/_pool_rec_check.py`, C RES 15814 ev, caps (Kp,Kn)=(32,256)):
  - forward output byte-identical with/without record accumulation (walk untouched);
  - `pool_fsi_reweight(record,1,1)` ≡ 1 bit-exact (maxdev 0.0);
  - zero record overflow (pion nh max 7, nucleon ns max 43 — caps generous);
  - off-nominal reweight responsive: mean reweight (1.3,1.0)=0.9995, (1.0,0.7)=1.41, (1.3,0.7)=1.41.
  Machinery: `_pion_step`/`_nucleon_step` stats surfaced through `make_pool_stepper` → per-step `rec`;
  `run_cascade_pool(rec_caps=(Kp,Kn))` accumulates via `_rec_scatter` (running write-index, scratch-slot
  drop); `pool_fsi_reweight` = legacy `pion_branch_reweight`×`nucleon_scat_reweight` on the joint record.
  Engine-API threading (cascade_carbon_v2 returning the record) deferred to Stage 3.
- **Stage 2 done**: autodiff==FD on the real pool record (`scripts/_pool_rec_check.py` @θ=(1.15,0.85)):
  grad_ad=[151.21,-20280.20] vs grad_fd same, maxreldiff 2.7e-8 → PASS. Synthetic-record pytest
  `tests/test_pool_fsi_reweight.py` 3/3 (nominal identity, factorization==legacy, autodiff==FD). No
  θ-walk threaded (would violate kind-1; walk already θ-independent).
- **Stage 3a done**: `cascade_carbon_v2(rec_caps=(Kp,Kn))` returns the joint per-event FSI record (4-tuple
  default unchanged; 5-tuple with record). Gate `scripts/_engine_rec_check.py`: per-event aligned
  (n=events), nominal≡1 bit-exact, forward identical 4-tuple vs 5-tuple — PASS (also after the rec opt-in
  refactor). `build_replica_pool`/`model_hist_pool` (pool-backed blueprint, Gaussian) added to
  cc0pi_tune_adonis. Pool-vs-legacy nominal comparison: see `scripts/_pool_blueprint_compare.py`.

## FULL CLEANUP (this session): pool is now the SOLE engine, Gaussian-only, de-versioned
Executed the approved full-cleanup plan in 5 gated, committed phases. The pool is the single cascade
engine; BFS-forward, the legacy `DiscreteCascadeFSI`/`DiscreteNucleonFSI` segment classes, the
`_propagate_*`/`propagate_*` walks, `RealCascadeFSI`, and the Cylinder interaction-probability are all
removed. ~1700 lines deleted net.

- **Phase 1** (`2949e84`): `adonis/fsi/pool_fsi.py` shared helper (`run_fsi`/`lead_proton`); `cc0pi_tune_adonis`
  is pool-only (legacy `build_replica`/`model_hist`/`hist_nb`/`_mk_ev` removed); `cc0pi_tune_closure`→pool.
  Gate: tune exit 0; closure recovers θ*=(1.3,0.8,1.0) within errors (pulls +0.45/+1.34/+0.50σ, χ²/ndf
  0.19→0.53 at reduced stats); `test_pool_fsi_reweight` 3/3.
- **Phase 2** (`687c4b4`,`d8de47b`,`af303fd`): migrated CC0π gens (h_cc0pi, cc0pi_disaggregated,
  gen_t2k_cc0pi_combined) to the pool; **rm** 5 superseded bespoke gens (gen_t2k_cc0pi/cc1pi_adonis,
  gen_minerva_cc0pi_adonis, gen_pion_fate, cc1pi_pion_survival — the config-driven engine_rich+matrix
  pipeline supersedes them). **cc1pi_fig_tki unified on the validated pool extraction** (user decision):
  `res_C` emits the workflow-schema rich bank (`prot[]`/`prot_origin`/`prot_gen` + `cr_p4`/`cr_pid` + `pi_nsc`,
  via `pool_fsi.proton_candidates` = verbatim `workflow/generate.run_one_seed`); `cc1pi_signal.ado_select`
  consumes `prot[]` (mirrors `workflow/signal`); `KNOCKOUT_MODE` re-expressed as an origin/gen filter; Npid
  threaded from the generator. NOTE: `t2k_cc1pi_rich_adonis.npz` must be regenerated (gen_cc1pi_rich) for the
  new schema before its run-at-import consumers (cc1pi_sweep) work.
- **Phase 3** (`7ef9941`): cascade_fate_dump→pool (dropped the BFS branch); retired BFS-only/legacy-vs-pool
  dev tools (_res_diag, _pool_blueprint_compare, _pool_timing, cascade_viz, later _pool_cyl_vs_gauss).
  **CAPABILITY LOSS (accepted):** per-step cascade trajectory visualization — it was a BFS-only feature
  (pion_segment `track_steps`); the pool `while_loop` has no StepTrace. The MC-truth track tree
  (`tracking.finalize`/`print_tree`) remains for cascade introspection.
- **Phase 4** (`2d2a21d`): deleted BFS-forward (pion_segment, nucleon_segment, make_kernel, run_cascade,
  _nucleon_kernel, old cascade_carbon) + cascade_carbon_v2's BFS else-branch; deleted _propagate_discrete/
  propagate_discrete/DiscreteCascadeFSI + _propagate_nucleon_discrete/propagate_nucleon_discrete/
  DiscreteNucleonFSI + RealCascadeFSI; forced Gaussian in `_pion_step`/`_nucleon_step`; removed the
  cylinder/prob config fields + CascadeHyperparams.cylinder + the generate.py pass; stripped `cylinder:`
  from 50 configs/*.yaml; deleted 5 obsolete tests (test_cascade_{real,discrete,early_exit,reweight_records},
  test_flux_and_real_fsi) + the BFS bit-exactness cross-checks in test_cascade_pool.
- **Phase 4b** (`b113cf1`): stripped the now-removed `cylinder=` kwarg from 22 scripts (frozen dataclass
  rejects it); de-versioned identifiers cascade_carbon_v2→cascade_carbon, _cascade_pool_v2→_cascade_pool,
  build_replica_pool→build_replica, model_hist_pool→model_hist. `currents_opt_v1` LEFT AS-IS (it is the
  upstream ACHILLES Fortran source filename in a provenance docstring, not a versioned identifier).
- **Phase 5**: full suite + forward pool-gen smoke. The **forward smoke caught a real regression the unit
  tests missed**: deleting the legacy blocks also removed two MID-FILE module-level imports
  (`nn_elastic_sigma` from nucleon_cascade, `nn_inelastic as nni`) that the KEPT `_nucleon_step` depends on
  → `NameError` at cascade runtime (unit tests use a toy stepper, so they passed). Re-added both at the top
  import block (no circular import). Lesson: deleting a contiguous line range can take out mid-file imports
  used by code OUTSIDE that range — a forward run, not just unit tests, is the real gate.
