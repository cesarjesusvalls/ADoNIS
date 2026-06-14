# Cascade MC-truth tracker (Geant4-style, differentiable)

Goal: a modular, channel-agnostic tracker over the differentiable BFS cascade for (1) diagnostics
(termination step, fates) and (2) visualization (show genuine sampling in a fully-differentiable system).
Two independent toggles, both default OFF -> production path bit-exact.

## Data model (adonis/fsi/tracking.py)
- TrackBank (summary, cheap): per track (n,MAX_TRACKS): track_id, parent_id, pdg, end_process (PROC_* enum),
  gen, origin, p4_birth/p4_death, pos_birth.  daughters derived ONCE at finalize() from parent_id (numpy).
- StepTrace (viz): (n,MAX_TRACKS,MAX_STEPS,7) float32 = per-step pos(3)+p4(4) from the march scan.
- track_id = GEN_STRIDE*gen + flat_slot (unique per event); parent_id set at spawn; ride through compact()
  exactly like the proven origin/gen provenance fields (never enter physics -> bit-exact).

## Memory/event (float32 steps): summary ~MAX_TRACKS*0.18 KB (T=64 -> 11.3 KB; 30k ev -> 340 MB).
## Steps viz-only: T*MAX_STEPS*28 B (T=64,260 -> 466 KB/ev -> N~120 ev ~ 56 MB).

## Stage 1 DONE: summary over the nucleon BFS, wired into cascade_full (empty_batch/compact/_nucleon_kernel/
   run_cascade/cascade_carbon_v2).  Smoke (931 RES ev, P6/g3): parent linkage 1347 ok / 0 dangling; gens
   {0,1} correct; pdg {2212:1031,2112:316}; all escape (nucleons don't absorb).  overflow 0.  Production
   physics unchanged by construction (track_id/parent_id never read by any kinematics).
## TODO: include the pion tracks (primary + created, with absorb/convert fates) + link pion-knockouts to
   the pion; the per-step StepTrace from the march; a viz demo + autodiff(exit)!=0 check.

## Stage 2 DONE: per-step StepTrace + viz + differentiability
- Added cfg.track_steps (DEFAULT OFF -> production bit-exact): the pion AND nucleon marches reuse `body`
  verbatim and additionally stack (pos, p4, alive) per step from the reference scan (forced when tracking).
  Threaded via term["traj"] = (pos(nsteps,n,3), p4(nsteps,n,4), alive(nsteps,n)) through pion_segment /
  nucleon_segment.  All march callers (apply x2, cc1pi_fig_tki, engine) absorb the appended traj.
- Regression: track_steps OFF -> cascade_carbon_v2 + DiscreteNucleonFSI.apply run identically (traj None,
  pterm pid {-211,-1,0,111,211}, overflow 0).  New fields/outputs never enter kinematics -> bit-exact.
- tracking.termination_step / step_summary: physical termination = first step |pos|>R_nucleus.
- scripts/cascade_viz.py (N=600): radius 6.55 fm, step 0.04 -> 10.4 fm max path.  median exit 157 steps,
  11.4% still geometrically INSIDE at the 260 cap.  NOTE: this is the FULL-traversal scan metric; the
  scattered-pion tail (multiple scatters -> path >10.4 fm) hasn't reached |pos|>R yet.  Production uses
  early_exit (marches until NO particle can still interact; inert = outside+outward excluded) so those are
  inert, not truncated -- n_trunc (truly able-to-interact at cap) is far smaller.  If we ever need the full
  geometric traversal in the scan, bump max_steps; for physics (early_exit) 260 is fine.
- DIFFERENTIABILITY (the headline): the walk is SAMPLED (frozen trajectories, shown), the observable is
  differentiable via the per-trajectory kind-1 reweight w(theta).  d(sum w_FSI)/d(sabs) = -1.19, NaN-free.
  Grad through RAW input-momentum rethrow is NaN by construction (discrete scatter acos/normalize 0/0 VJPs)
  -> exactly why kind-1 weights exist.  This is the correct "sampled yet fully differentiable" statement.
- Memory (float32 steps, persisted): summary T*~180 B (T=64 -> 11.3 KB/ev); steps T*MAX_STEPS*28 B
  (T=64,260 -> 466 KB/ev -> viz N~120 ~ 56 MB).  In-flight march traj is float64; cast to float32 on save.
