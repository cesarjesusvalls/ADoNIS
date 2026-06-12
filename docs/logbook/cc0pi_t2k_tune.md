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

## #2 — tracer fix + walk/weight split (2026-06-12)

- Tracer leak FIXED (commit a9dd7bf): numpy caches + per-call asarray in _load_qmc_configs,
  _load_density, _jax_grids, _build_angular (df7d6c9 pattern). Gate: two-size retrace passes.
- WALK/WEIGHT SPLIT implemented in cascade_discrete.py: the walk is theta-independent
  (sabs/sscat only enter the post-scan kind-1 reweight, lines ~390/~520), so the propagators
  now also return COMPRESSED walk records (pion: <=16 branch slots, hits measured <=8;
  nucleon: <=48 in-slab slots, measured <=20), and pion_branch_reweight /
  nucleon_scat_reweight are pure (theta, records) functions.
- Gates (all pass):
  * records reweight == in-propagation weight: bit-exact at 3 thetas (max dev 7e-16 = 1 ulp
    XLA fusion); nominal -> exactly 1.
  * model_hist(theta, replica) == hist_nb(theta, key) (same key): max rel dev 4.4e-16 over
    3 thetas; GRADIENTS IDENTICAL (rel 0.0).
  * autodiff == central FD through the records (tests/test_cascade_reweight_records.py).
- Speed: value_and_grad eval 16.5 s -> 13.4 ms (N=8k; ~1200x). The fit cost is now the
  one-time replica-bank walk (NREP=8).
- Driver rewritten (cc0pi_tune_adonis.py): replica bank, two-replica chi2 over distinct
  replica pairs, 400 iters, Hessian over the bank; observable generalized (argv[1]="dat"
  for delta_alphaT, datResults.root verified: 8 bins, own 8x8 cov).
- Production dpt run + full test suite launched (logs /tmp/tune_adonis_production.log,
  /tmp/adonis_split_suite.log).

## #3 — REAL-chain dpt tune COMPLETED (2026-06-12, paper_figures/cc0pi_tune_adonis.png)

- Timing: total 860 s = proposal+jit ~6.5 min + 8 replicas (38 s each after warm-up) +
  400 fit iters at 393 ms/it + Hessian over the bank. (Pre-split forecast: hours + ~30 GB.)
- NOMINAL (theta=1,1): chi2/ndf = 2.05 with profiled norm A=0.881 (8x8 full covariance).
  [The older cc0pi_tune_nominal.py quoted 1.44 -- different normalization handling/stats;
  not directly comparable.]
- BFP: sigma_abs = 3.0000 +/- 2.37 (RAILED at the [0.3,3] clip; error spans the whole range
  -> dpt does NOT constrain sigma_abs), sigma_scatter = 1.0753 +/- 0.248 (consistent with
  nominal), corr = +0.10, chi2/ndf(BFP) = 1.78.
- Reading: the best-fit curve is visually indistinguishable from nominal; ADoNIS nominal is
  statistically compatible with the T2K dpt data, FSI knobs only weakly constrained by dpt
  alone. The sigma_abs rail is a flat direction (noise-driven), not a physics pull.
- Correction to the split commit message (7877be0): full suite count was 90 passed, not 92.

## #4 — dat tune + the ACHILLES-vs-data reference (2026-06-12)

dat tune (paper_figures/cc0pi_tune_adonis_dat.png): NOMINAL chi2/ndf = 2.83 (A=0.891);
BFP s_abs = 0.30 (railed at the LOWER clip; opposite rail from dpt), s_scat = 0.868,
chi2/ndf 2.86 (NO improvement). Hessian errors = nan at the rail (H not pos-def there;
expected). Both observables railing sigma_abs in opposite directions with no chi2 gain =
sigma_abs is a flat noise-driven direction; the FSI scales cannot fix the dat shape
(model rises less steeply than data in bins 2-3 and the last bin).

ACHILLES vs T2K data, SAME convention (full cov, floated norm, chi2/7), computed from
t2k_cc0pi_tki_achilles.npz + the regenerated cc0pi_disaggregated.npz:

| prediction              | dpt  | dat  | A           |
|-------------------------|------|------|-------------|
| ACHILLES                | 1.84 | 2.12 | 0.875/0.911 |
| ADoNIS Cylinder (npz)   | 1.63 | n/a  | 0.892       |
| ADoNIS Gaussian nominal | 1.76 | 2.43 | 0.881/0.891 |

- ADoNIS tracks ACHILLES vs data (the project goal); both carry a common ~11% norm excess
  over T2K (A~0.88 for BOTH) -- shared model feature, not an ADoNIS artifact.
- The earlier "nominal 1.44" (cc0pi_tune_nominal.py, Jun 11) is NOT reproducible from
  current files: cc0pi_disaggregated.npz (gitignored) was REGENERATED on Jun 12 morning
  with the cascade fixes (escape plane, charge-resolved scatter); the identical
  computation on today's npz gives 1.63. Bin edges verified identical to the data release.

Status: plan items complete -- tracer fix (a9dd7bf), walk/weight split + gates (7877be0),
real-chain differentiability test (test_cascade_reweight_records.py), dpt fit + figure,
dat fit + figure, ACHILLES reference chi2. Possible follow-ups (not started): joint
(dpt, dat) fit blocked on missing cross-covariance; physics-side question of the common
A~0.88 norm excess; Hessian guard at clip rails in the tune script.

## #5 — closure + M_A third knob (2026-06-12, commit bc0c793)

- M_A knob (exact): amps2 is QUADRATIC in the axial scale -> 3 evals per frozen proposal
  give per-event (a,b,c); fit-time weight elementwise via axial_reweight_dipole at the
  event's amplitude Q2 (QE: leptonic Q2/dirac convention; RES: de-Forest-shifted Q2 via
  return_q2). Gates: nominal bit-identical, quadratic identity 5e-15, AD==FD 3.5e-8,
  =1 exactly at MA=1 (tests/test_ma_reweight.py). Effect: MA=1.15 -> +14..19%/bin;
  FSI (1.6,0.7) -> +8..52%/bin. NaN fix: zero-weight rejected QE draws (negative Q2)
  sanitized to identity records.
- Closure machinery (scripts/cc0pi_tune_closure.py): pseudo-data = ADoNIS @ theta* from
  4 INDEPENDENT walks; exact tune machinery recovers from nominal. Optimisation history
  persisted to /tmp/adonis_tune_runs/*.npz; --plot-only re-renders figures w/o recompute
  (also added to the data-fit driver).
- Closure result (dpt, theta*=(1.6, 0.7), MA free at truth 1.0, norm profiled):
  sigma_abs 1.65+/-1.30 (pull +0.04), sigma_scatter 0.90+/-0.39 (+0.51), M_A 0.96+/-0.17
  (-0.25); chi2/ndf 0.07->0.26. UNBIASED recovery; errors honest and large because the
  profiled norm absorbs most of the theta* effect (A_nom=1.143; residual shape 0.07) --
  quantifies the weak knob sensitivity of norm-profiled single-STV fits.
- 3-knob closures with theta*=(1.6, 0.7, MA*=1.15), figures cc0pi_tune_closure_dpt_ma{,_abs}.png,
  histories /tmp/adonis_tune_runs/closure_dpt_ma{,_abs}.npz:
  * PROFILED norm: pulls +0.21 / +0.64 / -1.19 sig; A_nom=1.306 absorbed the combined effect
    (residual shape 0.15); M_A central value moved the WRONG way (0.96 vs 1.15) -- the
    norm-profiled single-STV fit cannot separate M_A from the normalization.
  * ABSOLUTE norm: pre-tune chi2/ndf 2.18 -> post-tune 0.35; pulls -0.11 / +0.46 / +0.34 sig
    (M_A = 1.226 +/- 0.223 vs truth 1.15). The tune VISIBLY moves the prediction onto the
    pseudo-data and recovers all three injected parameters. The model is absolutely
    normalized, so the absolute fit is legitimate; for real data it would expose the common
    ~12% ADoNIS/ACHILLES norm excess instead of hiding it in A.
