# Logbook: tuning the real ADoNIS CC0π chain to T2K δp_T / δα_T

Goal: gradient-fit the FSI knobs (σ_abs, σ_scatter) of the REAL differentiable chain to
the T2K CC0π-Np STV data (and use the working CC0π/T2K sample for speed-up +
differentiability work).

## #1 — status reconstruction + diagnostics (2026-06-12)

State found (all evidence, no new fits):

- Forward, no fit (paper_figures/cc0pi_dat_dpt_data.png, regenerated 2026-06-12 09:50):
  ADoNIS vs ACHILLES δα_T χ²/ndf 0.36, δp_T 0.43 (ratio panels within ±2%); T2K data
  (arXiv:1802.05078) overlaid. Forward agreement is done.
- scripts/cc0pi_tune_t2k.py (TOY structural model + free norm A, δp_T, full 8×8 cov,
  two-replica χ², Hessian errors) — COMPLETED 2026-06-11: BFP χ²/ndf 6.3;
  corr(σ_abs,σ_scatter) = −0.83, σ_abs essentially unconstrained. Toy can't reach the
  data peak.
- scripts/cc0pi_tune_nominal.py — COMPLETED: REAL ADoNIS nominal χ²/ndf = 1.44 vs the
  toy fit plateau 4.5 → the toy structure is the limitation, motivating the real-chain tune.
- scripts/cc0pi_tune_adonis.py (REAL chain: frozen QE+RES proposal, DiscreteCascadeFSI /
  DiscreteNucleonFSI kind-1 reweights, no free norm; N=120k+120k, 150 iters + 12 Hessian
  evals) — NEVER COMPLETED: no cc0pi_tune_adonis.png exists, no run log found.

Diagnostics run today (smoke + scaling, logs /tmp/tune_adonis_smoke.log, _scale.log):

- N=8k+8k: pipeline runs end-to-end; loss(nominal)=32.79, grad=(−9.13, −22.0) — the
  machinery IS differentiable and finite.
- N=20k+20k: 16.5 s per jitted value_and_grad eval, peak RSS 5.0 GB.
  Extrapolated to the script's 120k+120k: ~minutes/eval and O(30 GB) RSS → the 150-iter
  loop is hours-to-infeasible on this machine. This is consistent with the run never
  finishing.
- N=60k (after a 20k trace in the same process): jax.errors.UnexpectedTracerError —
  REAL BUG: adonis/fsi/cascade_discrete.py:66 `_load_qmc_configs` caches a value created
  inside a jit trace (float64[36000]) in global state; any re-trace picks up the stale
  tracer. Same family as the df7d6c9 fix (cache numpy, jnp.asarray per call).

Open items / candidate next steps (not started):
1. Fix the `_load_qmc_configs` tracer-leak cache (numpy in cache, asarray at use).
2. Speed: the walk is theta-INDEPENDENT (kind-1: trajectories at nominal, weights carry
   theta — cascade_discrete.py:393/551). Candidate big win: split apply() into
   walk(key) [precompute once per replica key, non-diff] + weight(theta, walk)
   [cheap product per iter]; also a fixed bank of replica keys to avoid re-walking.
   Memory: chunk events through the cascade to cap RSS.
3. Differentiability gate for the REAL chain: autodiff-vs-FD closure in (σ_abs, σ_scatter)
   at moderate N (the toy has closures via 98d487d; the real chain does not yet).
4. δα_T: not fit anywhere yet (δp_T only). datResults.root has its own 8×8 covariance →
   standalone δα_T fit is possible with the same machinery; a JOINT (δp_T, δα_T) fit
   needs the cross-covariance, which the release does not provide.
