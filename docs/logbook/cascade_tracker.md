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
