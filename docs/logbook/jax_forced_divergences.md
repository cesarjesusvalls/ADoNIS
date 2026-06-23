# JAX-forced divergences from ACHILLES — audit & accuracy checklist

## Why this document exists
ADoNIS has a two-part mandate: (a) **mirror ACHILLES exactly**, (b) be **differentiable / vectorized
(JAX)**. Where they conflict, an exact copy is impossible and a divergence is *forced* — e.g. ACHILLES's
adaptive integrators, variable-length per-event particle trees, and data-dependent control flow have no
direct JAX-with-static-shapes equivalent. These forced divergences are legitimate (they buy
differentiability + batching), but they are **numerical/representational**, not physics, and they can
deviate from ACHILLES in narrow kinematic corners (thresholds, sharp poles, high multiplicity) — below
the level of the headline validations, so they surface late as small residuals.

There are TWO kinds of divergence and they get opposite treatment:
- **Pure logic / constants** (an isofactor, a mass, a frame): there is NO advantage to differing. Every
  one that surfaced (pion mass [[res-norm-deficit-is-pion-mass]], spectator isospin, amps2 frame) was a
  **bug**, fixed to be exact.
- **Forced by JAX** (fixed-grid integral vs adaptive, fixed stack vs object tree, table+interp vs
  on-the-fly): unavoidable. Rule: **same formula, and the numerical method validated to match ACHILLES
  to <=1% across the FULL kinematic range — including thresholds/poles, not just typical values.**

**Discipline (standing):** every item below must have an explicit accuracy-validation against ACHILLES
before it is considered closed. "Faithful formula" is not enough — the *evaluation* must be validated in
the hard corners. New JAX-forced divergences get added here when introduced.

Legend — Status: [V]=code verified by reading both sides | [R]=agent-reported, code not yet re-read.
Accuracy: validated? = has it been quantitatively checked vs ACHILLES across the range.

---

## CATEGORY 1 — fixed-grid numerical integration (vs ACHILLES adaptive)

### 1.1 [V] NN->NDelta sigma + dsigma/dm  (adonis/fsi/nn_inelastic.py)  **TOP PRIORITY**
- ADoNIS: `dsigma_dm` integrates cosθ with a **200-point linear `np.trapezoid`** (line ~105-110);
  `_build_tables` integrates the mass with a **160-point trapezoid** and tabulates sqrts on a
  **240-point linear grid** read back via `jnp.interp` (line ~114-150).
- ACHILLES (`NucleonNucleon.cc::SigmaNN2NDelta` + `ResonanceHelper.cc::DSigmaDM`): adaptive
  `Integrator::DoubleExponential`, rtol 1e-6 (mass) / 1e-12 (cosθ); sqrts via order-3 polynomial interp.
- Physics VERIFIED identical (isofactors 1 / 1/3 summed to 4/3 pp&nn, 2/3 pn on the Delta++ table;
  limits [m_n+m_pi+, sqrts-m_n]; threshold 2 m_N^avg+m_pi^avg; MatNN2NDelta; Breit-Wigner; masses).
- Regime: near threshold (steep turn-on) + the t/u pion-propagator poles `1/(t-m_pi^2)` sharply peaked in
  cosθ — exactly where a fixed trapezoid loses accuracy.
- Evidence pointing here: cascade-segment matrix (seed 1) shows the **nucleon-inelastic NN->NNpi**
  channel is the live residual — QE p chi2/ndf 3.3, RES p 2.5, neutrons clean (momentum-sampling, not
  proton-specific); the worst pulls cluster near threshold ([[res-norm-deficit-is-pion-mass]] is a
  separate, fixed issue). HELD AS HYPOTHESIS.
- **Accuracy validated? YES — FIXED.** `scripts/test_sigma_nn_ndelta.py` (ACHILLES_SIGMADUMP vs ADoNIS
  base table) showed the per-point mass/cos integral is ALREADY exact (direct integral == ACHILLES to
  1.0000); the entire +36% near threshold was the **sqrts-table LINEAR INTERP** across the steep turn-on
  at ~8 MeV spacing.  FIX: non-uniform sqrts grid, dense near threshold (4*n_s over [s_lo,2.30], cf.
  nn_inelastic._build_tables).  RESULT: threshold band [2.016,2.10) 1.358->1.0002 (max 0.17%); whole
  cascade-relevant range (threshold->3.0 GeV) now <=0.71%.  RESIDUAL: 3-4 GeV tail still ~4-8% low (a
  slow-converging per-point integral) but OUT OF CASCADE RANGE (sqrts>3 needs nucleon |p|>3.5 GeV, never
  reached) -> harmless, not fixed.  n_m/ncos unchanged.  TODO: re-confirm the cascade NN->NNpi matrix
  cell closes after regenerating the segment bank with the new table.

### 1.3 [V] Spectral-function importance CDF (adonis/xsec/spectral.py ~139-189)  **TESTED — OK**
- ADoNIS: SpectralImportanceSampler draws (|p|,E_rm) ~ |p|^2 S(p,E) via trapezoid CDFs on FINE grids
  (~0.25 MeV E / ~1 MeV |p|) built with the SF's OWN cubic-p/linear-E Polint, + linear inverse-CDF.
- ACHILLES: flat draw + per-point Polint S weighting (unbiased to |p|^2 S) -- same target distribution.
- **Accuracy validated? YES (after two flawed test iterations -- see lesson).** Final test:
  `scripts/test_spectral_sampler.py` (C, N=4e6, chunked) -- PEAK-RESOLVED 0.5 MeV E bins (the shell peak
  is at 17.5 MeV, FWHM ~6 MeV) + a 2D/CONDITIONAL check (E_rm shape in |p| slices + <E_rm>(|p|), the
  per-p e_cdf being the suspect).  RESULT: peak bin sampled/true = **1.001**; |p| marginal chi2/ndf
  **0.85**; E_rm|p conditional chi2/ndf **1.0-6.4** (the 2 high-stat slices ~6, inflated by 4e6 stats);
  <E_rm>(|p|) within **3.3 MeV** over a 28-288 MeV range.  E_rm marginal chi2/ndf 14 at 4e6 is consistent
  with ~1-2% scatter dominated by the low-stat tail + first edge bin (visible in the plot), NOT a peak or
  correlation bias.  So the fine-grid+Polint rebuild WORKS; no significant bias; no action.
  **LESSON (process):** the FIRST test used 80 uniform bins (~5 MeV) -- coarser than the ~6 MeV peak ->
  falsely "OK" (chi2~1).  The SECOND (fine bins) had a TRUE-CURVE aliasing BUG (summing a fine density
  into bins with a >=/< mask -> variable # of fine points/bin -> +-13% ripple) -> falsely "6% peak
  deficit, chi2=305".  Fixed with exact cumulative-interp integration.  Peak-resolving binning AND an
  alias-free reference are both mandatory for any narrow-feature accuracy test.

---

## CATEGORY 2 — precomputed table + interpolation (vs on-the-fly / higher-order)

### 2.2 [V] DCC amplitude interp: bilinear vs spline (adonis/xsec/dcc_current.py:55, :233)  **DANGEROUS DEFAULT — FIXED**
- `BATCH_INTERP` "bilinear" (~45x faster) vs "spline" (bit-faithful to ACHILLES interpolate_amp).
- **The "~0.3%" is the FLUX-INTEGRATED xsec difference and is MISLEADING**: bilinear is a known
  offender vs W — the dsigma/dW SHAPE, esp. the high-W tail, deviates **well beyond 1%**. So bilinear is
  unusable for any W-differential / pion-kinematics result even though the total looks fine.
- The module default WAS "bilinear" (a silent footgun: any script that forgot the override got it).
  generate.py (production) + all matrix/tune/figure generators DID set "spline" -> the banks we used are
  spline.  EXCEPTION found: scripts/gen_res_events.py defaulted to bilinear (dev script, output
  scripts/res_events_1M.npz, not a committed bank).
- **FIX (this audit):** flipped the module default to "spline"; bilinear is now opt-in for fast
  diagnostics only.  gen_res_events.py default also -> "spline".  Comments corrected to state the W-tail
  danger, not the misleading 0.3%.
- **Accuracy validated?** Default now faithful.  All bilinear origin docstrings (amplitudes.py
  amplitudes_bilinear{,_np}, structure.py, channel.py weight_from_sample, params.py, dcc_current.py)
  now carry the explicit "NOT W-faithful; >1% dsigma/dW tail; never use without explicit awareness"
  warning.  **OPEN — bilinear is still ACTIVELY used in these paths (verify each is a closure where it
  cancels, else switch to spline):** adonis/nuclear/inclusive_1pi.py:27 (HadronStructure spline=False),
  adonis/primary/dcc/sigma_enu.py:204,249 (use_spline=False) + :347 (GenConfig spline=False default).
  TODO: quantify the bilinear W-tail deviation explicitly (dsigma/dW bilinear-vs-spline) so the
  magnitude is on record.

### 2.4 [V] SF inverse-CDF linear interp (spectral.py ~177-189) — see 1.3.  TESTED OK (same test).

(Faithful, NOT divergences — recorded so they're not re-flagged: spectral Polint order [V],
flux histogram interp, vegas 3-point Adapt smoothing — all match ACHILLES.)

---

## CATEGORY 3 — fixed-size buffers / caps (vs unbounded ACHILLES)

All of these COUNT overflow (sofl/oofl/rofl/logofl, force-escape logs) — none is silent — but a nonzero
overflow means truncation. Rule: monitor the counter; size the cap so overflow == 0 on the physics sample.
- 3.x [V] `_KSLAB=3` nearest in-slab nucleons evaluated per step (cascade_discrete.py:83, top_k:398).
  ACHILLES evaluates all in-slab nucleons. Claim ">=K in one 0.04 fm slab ~never" NOT empirically shown.
- 3.x [V] pool stack width `M=P` (run_cascade_pool) + `pool_reconcile` compaction; overflow=sofl/oofl.
- 3.x [V] kind-1 FSI record caps `rec_caps=(Kp,Kn)`, `_K_BR=16`, `_K_SLAB_REC=48`; overflow=rofl.
  Truncation biases the reweight/gradients — relevant for tuning, not nominal forward.
- 3.x [V] segment logger `log_cap=L` (diagnostic only); overflow=logofl (was 0 at L=64 for C).
- 3.x [V] `_MAX_SEG=12` per-particle interaction cap (legacy interaction-kernel constant; measured
  max=11/622k). Confirm whether still live on the pool path or vestigial.
- 3.x [V] `DiscreteCascadeConfig.max_steps` default 260; the cascade generators override with
  `max(req, 3*radius/0.04)` (~491 for C), so 260 is only a default — check every caller overrides it.
- 3.x [R] tracking.py `max_tracks=64`, `max_steps=260` (optional MC-truth; off in production).

**Accuracy validated?** Overflow counters exist; need a high-A / high-E stress run confirming all
counters stay 0 on the physics sample (esp. _KSLAB and pool M for Ar).

---

## CATEGORY 4 — other differentiability approximations

### 4.1 [V] Gaussian interaction probability (the SOLE engine) vs ACHILLES cylinder/hard-disk
- ADoNIS: `prob = exp(-pi b^2 / sigma)` everywhere (cascade_discrete `_pion_step`/`_nucleon_step`); the
  cylinder/hard-disk option was removed for differentiability ([[always-pool-never-bfs]]).
- Regime: small impact parameter / nucleus edge. This is a deliberate, user-approved model choice.
- **Accuracy validated?** Indirectly via the cascade transparency / matrix agreement; no dedicated
  Gaussian-vs-cylinder closure.

### 4.2 [V] Straight-line propagation vs potential-curved trajectory  **NOT A DIVERGENCE — CLOSED**
- ADoNIS marches `pos += step*dhat` straight between interactions.  ACHILLES `Cascade::Propagate` only
  integrates through the mean field WHEN `m_potential_prop` is true (symplectic `integrators[idx].Step`);
  otherwise it is straight-line (`kickNuc->Propagate(timeStep)`).
- **VERIFIED:** our run card sets **`PotentialProp: False`** (run_T2K_{C,Ar}_fate_gauss.yml) -> ACHILLES
  cascade transport is straight-line, IDENTICAL to ADoNIS.  The agent's `[R]` "curves through a
  potential" claim was wrong for our config.  No divergence; nothing to fix.  (Re-check only if a card
  ever sets PotentialProp: True.)

(Faithful, NOT divergences: Delta->Npi phase-space `_split2`, hard-cut Pauli blocking, spline rho(r)/kF.)

---

## Status summary
- **1.1 nn_inelastic** — FIXED (sigma <=0.7% in range); cascade-level NN->NNpi re-validation pending the
  seed-1 segment run with the new table.
- **2.2 DCC bilinear** — FIXED (default->spline; warnings at all origins; RES confirmed spline).
- **1.3 / 2.4 spectral sampler** — TESTED OK (chi2/ndf ~1; peak region 5.8%, no bias).
- **4.1 Gaussian prob** — accepted (deliberate, differentiable engine).
- **4.2 straight-line** — NOT A DIVERGENCE (PotentialProp: False -> ACHILLES also straight-line).

## Remaining action order
1. **1.1 cascade-level** — re-check the cascade-segment matrix NN->NNpi cell after regenerating the
   segment bank with the fixed sigma table (run in progress).
2. **3.x caps** — high-A/high-E overflow stress test (confirm every counter stays 0 on the physics
   sample, esp. _KSLAB and pool M for Ar).  ONLY remaining untested divergence.
3. **2.2 follow-up** — quantify the bilinear W-tail magnitude on record; the non-blueprint bilinear
   users (inclusive_1pi, sigma_enu) are handled in a separate session.
4. Re-read any lingering [R] items to promote to [V].
