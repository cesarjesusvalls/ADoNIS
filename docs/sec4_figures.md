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

## Fig A — the whole closure argument

`fig_closure_summary.py` + `fig_dists.py` / `fig_dists_all.py`. Per dial, the three interval recipes
against the thing they all claim to describe:

- **(a)** quadratic sigma / profile (Wilks) / **FC**, side by side, truth marked
- **(b)** the **toy distribution** per dial -- the empirical sampling distribution
- **(c)** chi2_data across the toys vs chi2(ndf), ndf counting only bins that constrain
- **(d)** E_b zoom, the dial where the three separate
- plus pre/post-fit distributions (4 representative observables, and the full sample set)

For the ~12 well-behaved dials all three coincide with the toys -- that IS the result, and it is why the
expensive machinery is only needed on a few. **Non-Gaussianity needs no separate figure**: it is exactly
where the symmetric quadratic bar and the asymmetric profile bar disagree in (a). Quote the magnitudes in
the text (Delta chi2 at +-3 sigma against the parabola's 9.0): S_Delta 58.2/1.0, M_A_res 1.8/19.9,
s_NN_elastic[2] 15.2/8.0, E_b 1.6/8.1, the other 12 between 8.0 and 10.5.

FC costs ~2.2 h/dial, so run it on the four flagged dials only and report "= profile" for the rest.

May split on panel count (A1 intervals vs toys, A2 pre/post-fit + chi2) -- decide at render time.

## Fig B — coverage vs distance from the boundary  (supporting)

1-D E_b, other 15 dials fixed, exhaustive grid scan (no optimiser), 20k toys per point, interval built
PER TOY:

| truth | d/sigma | Wilks | Bayes flat | FC |
|---|---|---|---|---|
| 0.01 | 0.00 | 83.8 | **0.0** | 67.7 |
| 0.10 | 0.26 | 81.6 | 51.0 | 68.0 |
| 0.20 | 0.56 | 71.9 | 72.8 | 68.8 |
| 0.50 | 1.37 | 66.8 | 74.6 | 68.2 |

Generalises beyond the single truth in Fig A, and is the only place the flat-prior Bayesian failure is
quantified. Supporting panels: the FC test statistic showing the boundary peak as an atom at Delta chi2 = 0
(43% at the wall, 0.7% at the truth), and the belt running from Chernoff's 0.226 to Wilks' 1.00.

## Fig C — the cost argument (keystone)

Same Asimov fit three ways, same start / bounds / data, warm-up before timing and counters at the engine:

| method | wall s | model evals | Jacobians | chi2 | max abs err | valid |
|---|---|---|---|---|---|---|
| MIGRAD numerical | 183.6 | 1329 | 0 | 1.601 | 4.9e-01 | True |
| MIGRAD autodiff | 98.2 | 571 | 26 | 1.601 | 4.9e-01 | True |
| Gauss-Newton/TRF | **10.1** | **11** | **12** | 1.2e-24 | 8.0e-14 | True |

**1329 -> 11 model evaluations, 18.1x wall clock**, and both MIGRAD variants return valid=True while
sitting 0.49 from truth with E_b pinned on its floor -- and HESSE reports its error 3.1x too large there.
Caveat: the autodiff row still builds all 16 forward JVPs and contracts them; MIGRAD needs only the scalar
gradient, which one reverse-mode VJP would give for ~1-2 model evals. Now possible (BinSpec unification)
but not yet done, so that row understates the gradient-only benefit.

## Status

| | |
|---|---|
| closure + profile (`sec4_all16`, 16/16 dials) | done |
| Fig C cost benchmark | done |
| Fig B 1-D wall scan | done |
| Fig A ensembles, single start | running (see the arms below) |
| Fig A pre/post-fit dists | scripts exist, need a run on `sec4_all16` |
| **FC belts for the 4 flagged dials** | **not run -- the long pole for Fig A** |

## What we tested and dropped

Three candidate fixes for the E_b boundary were tried. Recording the negative results because they are
what justifies the simple production configuration:

1. **Mirroring (T2K's remedy) -- tested, not adopted.** Single-start fits, clamp vs mirror, identical
   otherwise: the atom goes 7.2% -> 0.0% on the floor, but the fraction with |E_b| < 0.1 is **16.0% vs
   15.8%** and the medians agree. It moves the point mass into a continuous spread near zero; the
   inference is unchanged. 49.8% land on the negative branch, confirming the mechanism works -- it just
   buys nothing. Kept in the code, env-gated (`S4_EB_MIRROR=1`), off by default.

2. **TRF vs the original LM -- controlled A/B running** (`sec4_ens_lm1` vs `sec4_ens_c1`, single start,
   clamp, same truth, only the minimiser differs). The 21.6% -> 7.2% atom drop previously attributed to
   TRF was NOT controlled (the truth changed between those runs). What is controlled: on the 12 wall
   toys TRF reached lower chi2 on 12/12, but by a mean of 0.50 and **10/12 landed in the same place**,
   and LM never satisfied its own tolerance (0/12) while TRF certifies at ~1e-6.

3. **Multi-start -- dropped from the production configuration.** It targets the `M_A_res`/`S_Delta` two
   basins (the RES weight is g^T M g with g = (1,d) x (1,r,r*pp), hence QUADRATIC in delta_strength, so
   d -> prediction is two-to-one about that parabola's vertex). It costs xN per toy (8 starts = 230-500
   s/toy vs ~84 s single) and does nothing for E_b, which mirroring/TRF already showed is not an
   optimiser problem.

The residual -- ~16% of fits placing E_b near zero when the truth is 0.50 -- is **not** an optimiser
artefact at all. It is genuine near-boundary statistics, and it is what Fig A's FC column and Fig B are
for.

## Open issues

1. ~~`multisample_fc.py` throws nuisances at the reference truth~~ **FIXED** -- now the standard *profile
   construction*: nuisances thrown at their conditional best fit given theta = t on the observed data.
   Not yet exercised on a real belt.
2. **Reverse-mode VJP gradient** for the Fig C autodiff row. Now possible (BinSpec unification made the
   model differentiable end to end) but not implemented, so that row understates the gradient-only
   benefit -- 26 gradients cost 26 x 1.05 s as full Jacobians instead of ~26 x 0.3 s as VJPs.
3. `BeamSample.jac_blocks` still bins on the host in the `S4_JAX_BIN=1` path (model evaluation is on
   device for all 8 samples, Jacobians for 5 of 8). Irrelevant while the flag is off.

## Cost reference (measured, turing RTX 2080 Ti, 250k events / 1.9M resident)

| | |
|---|---|
| one TRF fit, 15–16 dials | 13.4 s (nfev 22, njev 13) |
| profile, one dial (17 nodes) | ~4 min → whole profile ~9 min sharded one task per dial |
| one FC toy (pinned + global fit) | ~27 s |
| FC belt, one dial (15 grid pts × 300 toys) | ~2.2 h wall, one task per grid point |
| 1-D FC belt (nuisances fixed) | minutes — every toy is arithmetic on a precomputed grid |
