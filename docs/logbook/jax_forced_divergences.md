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
- **Accuracy validated?** Test built: `scripts/test_sigma_nn_ndelta.py` (ACHILLES_SIGMADUMP dump of
  `SigmaNN2NDelta(sqrts,1GeV,Delta++)` vs ADoNIS base table, ratio vs sqrts). RESULT: PENDING (rebuild).

### 1.3 [R] Spectral-function importance CDF (adonis/xsec/spectral.py ~139-173)
- ADoNIS: trapezoid CDF on fixed ~0.25 MeV (E) / ~1 MeV (|p|) grids + linear inverse-CDF sampling.
- ACHILLES: flat draw over the coarse grid weighted by cubic Polint of S(p,E) — no precomputed CDF.
- Regime: sharp removal-energy peak (~15 MeV) + low-|p|; comment already flags a peak under-sampling bias.
- **Accuracy validated?** NO — needs a (p,E) marginal comparison vs ACHILLES.

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
- **Accuracy validated?** Default now faithful.  TODO: quantify the bilinear W-tail deviation explicitly
  (dsigma/dW bilinear-vs-spline) so the magnitude is on record.

### 2.4 [R] SF inverse-CDF linear interp (spectral.py ~177-189) — see 1.3 (same mechanism).

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

### 4.2 [R] Straight-line propagation vs ACHILLES potential-curved trajectory (cascade_discrete ~330)
- ADoNIS marches `pos += step*dhat` straight between interactions; ACHILLES integrates through the mean
  field. Regime: low-energy hadrons deep in the nucleus. Severity likely low (most cascades high-E).
- **Accuracy validated?** NO.

(Faithful, NOT divergences: Delta->Npi phase-space `_split2`, hard-cut Pauli blocking, spline rho(r)/kF.)

---

## Action order
1. **1.1** — run `scripts/test_sigma_nn_ndelta.py`; if ADoNIS deviates near threshold, refine the
   integration (more cos/mass points or adaptive; finer sqrts table) until <=1% across the range, then
   re-check the cascade-segment matrix NN->NNpi cell. (Pure accuracy; brings ADoNIS *closer* to ACHILLES.)
2. **3.x caps** — high-A/high-E overflow stress test (all counters 0).
3. **2.2** — confirm production forward gen uses `BATCH_INTERP="spline"`.
4. **1.3 / 2.4** — SF (p,E) marginal vs ACHILLES.
5. Re-read the [R] items to promote them to [V] (the agent inventory is a starting point, not verified).
