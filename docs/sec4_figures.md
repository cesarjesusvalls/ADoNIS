# Section 4 — figure plan

## The claim

Not "we can fit" — everyone can fit. The claim only a **differentiable** generator supports is:

> Gradients make it cheap to **validate** the quoted uncertainty, so you stop assuming the quadratic
> (Hessian) error is right and start measuring whether it is.

Reference point: the T2K ND280 constraint paper (arXiv:2606.14015, *Constraining Neutrino Interaction
Uncertainties for Neutrino Oscillation Measurements at the T2K Experiment*), 75 xsec parameters, 22
samples, ~5000 bins. What they do:

- **hybrid-frequentist** — Minuit2 MIGRAD + HESSE, i.e. the inverse numerical Hessian, explicitly
  "assuming the likelihood has a Gaussian profile";
- **Bayesian** — MCMC (MaCh3, Metropolis–Hastings), marginalised 1-D posteriors;
- non-Gaussianity is **acknowledged** for the CCQE **removal energy** (our `Eb_shift`) and the Pauli
  blocking dials, and reported as violin plots of the posterior with the HPD point;
- goodness of fit from 2000 pseudo-experiments (p-value 57.5%);
- **no Feldman–Cousins** in the ND fit, and **coverage is never measured** — robustness rests on the two
  frameworks agreeing;
- boundary handling: parameters *whose prior central value sits on a physical boundary* are known to
  break the minimiser, and their response functions are **mirrored** across the boundary.

Every step they skip or approximate is skipped because it is expensive without gradients. That is the
gap §4 fills.

---

## Fig A — closure, and that the error bar is real

Existing: `analysis/paper/sec4_closure/fig_closure_summary.py` (+ `fig_dists.py` / `fig_dists_all.py`).

Panels:

- **(a)** recovery of all 16 dials on the injected truth, quadratic σ beside the profile interval;
- **(b)** the same truth refit 500× (statistics only) — the histogram must have the width the profile
  predicted;
- **(c)** χ²_data across those fits vs χ²(ndf), ndf counting only bins that constrain;
- **(d)** the boundary case, `Eb_shift` ~1.2σ from its wall;
- **plus pre/post-fit distributions** — `fig_dists.py` for four representative observables (T2K and
  MINERvA CC0π δp_T, an (e,e′) QE ω, a π⁺–C reaction σ) and `fig_dists_all.py` for the full sample set.
  These make the fit concrete: nominal misses, best fit lands on the closure data.

Truth for the reference fit: **all 16 dials displaced** (0.4–1.25 σ_prior, mixed signs), with `Eb_shift`
deliberately left near its wall at 0.50 — `scratchpad/truth16.txt`, label `sec4_all16` /
ensemble `sec4_ens16`.

## Fig B — non-Gaussianity triage

Profiled Δχ² at ±3σ against the parabola's 9.0, for **all 16 dials**. Measured on `sec4_all16`:

| dial | Δχ²(−3σ, +3σ) |
|---|---|
| `delta_strength` | 58.2, 1.0 |
| `M_A_res` | 1.8, 19.9 |
| `s_NN_elastic[2]` | 15.2, 8.0 |
| `Eb_shift` | 1.6, 8.1 (scan bounded at the wall) |
| other 12 | 8.0 – 10.5 |

The point is not that four dials are non-Gaussian — it is that **we checked all 16 and it cost ~4 min per
dial** (17 nodes × 13.4 s, one SLURM task per dial). T2K cannot afford this for 75 parameters, which is
precisely why they need a second MCMC framework to discover the same fact.

## Fig C — the boundary case done properly

The wall scan: coverage of **Wilks / flat-prior Bayes / Feldman–Cousins** 68% intervals as the truth
approaches the `Eb_shift` wall. 1-D (other 15 dials fixed at truth, exhaustive grid scan, no optimiser),
20k toys per point, interval built **per toy**:

| truth | d/σ | Wilks | Bayes flat | FC |
|---|---|---|---|---|
| 0.01 | 0.00 | 83.8 | **0.0** | 67.7 |
| 0.10 | 0.26 | 81.6 | 51.0 | 68.0 |
| 0.20 | 0.56 | 71.9 | 72.8 | 68.8 |
| 0.50 | 1.37 | 66.8 | 74.6 | 68.2 |

Wilks over-covers to 84% near the wall; a flat-prior credible interval reaches **0%** coverage; FC holds
68% everywhere. Supporting panels: the FC test statistic showing the boundary peak as an atom at
Δχ² = 0 (43% at the wall, 0.7% at the truth), and the belt running from Chernoff's on-boundary value
(0.226) to Wilks' 1.00.

This is the most novel result in the section and a direct, quantitative caveat on how removal-energy-type
parameters are reported today.

## Fig D — the cost argument, measured (the keystone)

The same Asimov fit, three ways, same start / bounds / data:

- **A** MIGRAD, numerical derivatives — what T2K runs;
- **B** MIGRAD, analytic autodiff gradient — isolates the value of *having* a gradient;
- **C** Gauss-Newton / trust-region-reflective — adds the value of exploiting the least-squares
  structure (JᵀWJ).

Reported in wall time **and** model evaluations (the expensive object), plus accuracy against truth and a
dial-by-dial comparison of HESSE's error against the Gauss-Newton one. A vs B vs C separates iteration
count from per-iteration cost — the naive "16 extra evaluations per Jacobian" understates it, because a
variable-metric method also needs many more iterations.

Fig D is the keystone: it converts B and C from "nice extra studies" into "things that were previously
infeasible".

---

## Status

| | |
|---|---|
| Fig A closure + profile (`sec4_all16`) | done |
| Fig A ensemble (`sec4_ens16`, 500 toys) | running |
| Fig A pre/post-fit dists | scripts exist, need a run on `sec4_all16` |
| Fig B | data in hand (profile), figure not written |
| Fig C 1-D wall scan | done |
| Fig C 16-dial FC | not run — needs the nuisance-throw fix below |
| Fig D | benchmark running (job 34523089) |

## Open issues

1. ~~`multisample_fc.py` throws nuisances at the reference truth~~ **FIXED** — it now uses the standard
   *profile construction*: nuisances are thrown at their conditional best fit ν̂̂(t) given θ = t on the
   observed data, obtained with one extra fit per grid point. Not yet exercised on a real belt.
2. **`M_A_res` / `S_Δ` multi-modality in the 16-dial fit is not explained.** Measured: identical toy data
   from 6 dispersed starts converges to χ² differing by up to 2.9 on 3/8 toys, and the spread lives almost
   entirely in that −0.995-correlated pair (3.9σ and 2.9σ, vs <0.42σ for 13 of the other dials).
3. **Test T2K's mirroring** against our multi-start for the boundary. `Eb_shift`'s nominal is 0.010 —
   exactly its physical floor — so every fit starts *on* the wall, which is the documented failure mode.

   Mechanism: a response function is mirrored by defining ω(−x) := ω(+x), i.e. even about the boundary.
   The contrast with us is the whole point — `sf_reweight` **clamps** E_b ≤ 0, so the prediction is
   *constant* below zero, χ² is exactly flat, the gradient vanishes and the region is an absorbing trap;
   a *mirrored* response instead has a restoring gradient that pushes the minimiser back. For us this is
   a reweight-level change: evaluate at `|Eb_shift|` rather than `max(Eb_shift, 0)`, drop the bound, and
   report the magnitude.

   Two caveats: the mirrored likelihood is even, so it carries twin minima at ±x̂, and the boundary
   becomes a stationary point by symmetry. Mirroring fixes the **minimisation**, not the statistics —
   the interval still needs FC. Same structure as the `M_A` mirror minimum (M_A enters as M_A²), which
   we currently handle with a bound instead.

## Cost reference (measured, turing RTX 2080 Ti, 250k events / 1.9M resident)

| | |
|---|---|
| one TRF fit, 15–16 dials | 13.4 s (nfev 22, njev 13) |
| profile, one dial (17 nodes) | ~4 min → whole profile ~9 min sharded one task per dial |
| one FC toy (pinned + global fit) | ~27 s |
| FC belt, one dial (15 grid pts × 300 toys) | ~2.2 h wall, one task per grid point |
| 1-D FC belt (nuisances fixed) | minutes — every toy is arithmetic on a precomputed grid |
