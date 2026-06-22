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
2. Thread sabs/sscat into the pool walk (default nominal=no-op); gate reweight(nominal,θ)==in-walk(θ);
   autodiff==FD.
3. Single pool-backed FSI engine + migrate blueprint (cc0pi_tune*, cc0pi_disaggregated, cc1pi_fig_tki,
   gen_t2k_cc0pi/cc1pi_adonis) + reweight tests. Gate: cc0pi_tune via pool ≈ legacy within stats; ADoNIS
   vs ACHILLES forward unchanged. COMMIT (pre-cleanup checkpoint).
4. Cleanup: remove BFS forward + legacy classes + propagate_discrete; engine switch; extract cascade_real
   utilities, drop RealCascadeFSI/cascade_exact/dead tests. Gate: full suite green.

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
