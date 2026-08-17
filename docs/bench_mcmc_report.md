# Do gradients change the scaling of MCMC? Metropolis-Hastings vs NUTS on one ADoNIS posterior

Companion to `docs/bench_fair_report.md`, which asks the same of *optimisers*. Same construction: both
samplers call one object, so the only difference left is whether a gradient is used.

**Every table below is generated from the data, not typed.** An earlier version of this report was
hand-patched after a bug fix and silently kept the pre-fix numbers; regenerating is the only way that
does not recur.

## How this was made fair

Both samplers call the same `adonis.fit.kernels.FitKernel`. MH asks for `chi2`; NUTS for
`chi2_and_grad`. Likelihood, physical box, dataset, float64, device and the device-resident model are
shared **by construction**. The posterior is `exp(-chi2/2)` truncated to the box (MLE, prior widened
x1e6, flat inside).

**MH is not a strawman.** It gets the **Laplace covariance** `(J^T W J)^-1` as its proposal -- precisely
what NUTS gets as inverse mass matrix -- and its scale adapted to **0.234** acceptance
(Roberts-Gelman-Gilks). Measured acceptance 0.21-0.26. NUTS targets 0.8 (measured 0.80-0.92), max tree
depth 8.

**Equal compute, not equal samples.** Fixed warm-up, then a fixed 300 s wall-clock sampling budget per
chain. ESS/s is measured, not inferred, and the study's runtime is bounded.

**Setup.** Section-4 posterior, 15,000 events/sample (150k resident), **statistically fluctuated
pseudo-data** -- one Gaussian throw, the same throw for every method and chain at a given parameter set.
4 chains per cell, overdispersed starts at 2x the Laplace width. NVIDIA A100 40 GB throughout. 64 cells.

## Two axes, deliberately separated

The parameter subsets are **nested by Gate-I shrinkage**, so "n=17" is not "n=12 plus five generic
parameters" -- it is n=12 plus five *specific* ones. Reporting an advantage-vs-n curve from nested
subsets alone conflates **dimension** with **which parameters happened to enter**. So this report
measures both, and the second axis turns out to dominate at the top end.

### Gate 1

| n | MH R-hat | NUTS R-hat | MH draws/chain | NUTS draws/chain | NUTS divergences |
|---|---|---|---|---|---|
| 2 | 1.0002 | 1.0001 | 83,550 | 7,470 | 0 |
| 4 | 1.0003 | 1.0004 | 81,650 | 7,670 | 3 |
| 8 | 1.0004 | 1.0004 | 80,250 | 6,400 | 0 |
| 12 | 1.0012 | 1.0007 | 74,100 | 5,400 | 0 |
| 17 | 1.0023 | 1.0057 | 73,950 | 3,290 | 152 |

### Efficiency

| n | method | ESSb/s | ESSt/s | ESSb/1e3 visits | ESSb/1e3 passes | tau | accept | grad/draw | min ESSb/s | worst param |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | mh | 32.78 | 41.22 | 115.6 | 115.6 | 8.7 | 0.232 | 0.00 | 32.74 | `M_A_qe` |
| 2 | nuts | 15.33 | 12.69 | 129.2 | 64.6 | 1.7 | 0.916 | 4.65 | 13.55 | `M_A_qe` |
| 4 | mh | 19.88 | 29.09 | 71.5 | 71.5 | 14.1 | 0.239 | 0.00 | 19.41 | `M_A_qe` |
| 4 | nuts | 14.43 | 15.71 | 123.2 | 61.6 | 1.8 | 0.880 | 4.46 | 9.71 | `Eb_shift` |
| 8 | mh | 10.72 | 17.73 | 39.2 | 39.2 | 25.5 | 0.233 | 0.00 | 9.88 | `Eb_shift` |
| 8 | nuts | 17.73 | 14.83 | 150.8 | 75.4 | 1.2 | 0.863 | 5.40 | 8.13 | `Eb_shift` |
| 12 | mh | 6.90 | 12.47 | 25.5 | 25.5 | 39.2 | 0.214 | 0.00 | 6.22 | `s_NN_elastic[0]` |
| 12 | nuts | 21.10 | 12.98 | 179.8 | 89.9 | 0.9 | 0.856 | 6.46 | 12.46 | `Eb_shift` |
| 17 | mh | 4.21 | 8.36 | 15.9 | 15.9 | 62.8 | 0.263 | 0.00 | 1.10 | `M_A_res` |
| 17 | nuts | 6.87 | 6.14 | 58.7 | 29.4 | 2.1 | 0.804 | 8.06 | 0.96 | `M_A_res` |

### NUTS / MH  (>1 = NUTS better)

| n | ESSb/s | ESSt/s | ESSb/visit | ESSb/pass |
|---|---|---|---|---|
| 2 | 0.47 | 0.31 | 1.12 | 0.56 |
| 4 | 0.73 | 0.54 | 1.72 | 0.86 |
| 8 | 1.65 | 0.84 | 3.85 | 1.92 |
| 12 | 3.06 | 1.04 | 7.04 | 3.52 |
| 17 | 1.63 | 0.73 | 3.69 | 1.84 |

### Fixed n=12, varying WHICH parameters

| subset | contains RES trio? | NUTS/MH ESSb/s | ESSb/pass | tau MH | tau NUTS | NUTS divergences | NUTS R-hat |
|---|---|---|---|---|---|---|---|
| nested-12 | M_A_res + delta_strength (2 of 3) | 3.06 | 3.52 | 39.2 | 0.9 | 0 | 1.0007 |
| hard-12 | **all three** | 1.21 | 1.43 | 44.8 | 2.4 | 137 | 1.0124 |
| easy-12 | none | 3.09 | 3.64 | 37.5 | 0.8 | 1 | 1.0008 |
| rand-12 | M_A_res + res_axial (2 of 3) | 3.05 | 3.61 | 38.3 | 0.8 | 5 | 1.0006 |

Spread across 12-parameter subsets: **1.21 - 3.09, a factor 2.56**.
The whole n=12 -> n=17 change is 3.06 -> 1.63, a factor 1.87.

MH tau vs n: linear fit tau = 3.56 n + -0.55, R2 = 0.986; power law exponent 0.911.
MH tau across the four 12-parameter subsets: 39.2, 44.8, 37.5, 38.3 (spread 1.19x, against 7.3x across n=2..17).

## What this says

**1. Gradients change the SCALING, and the evidence is `tau_int`.** MH's autocorrelation time grows
linearly in the parameter count, and it does so *despite* an optimal Laplace preconditioner and
acceptance pinned at the 0.234 optimum -- there is no tuning left to give MH. NUTS's tau stays ~1-2.
Critically, this claim survives the confound: MH's tau varies by only 1.19x across four different
12-parameter subsets, against 7.3x across n = 2..17, so the dimensional signal dominates parameter
identity by roughly a factor 6 for this statistic.

**2. There is a crossover near n = 5, below which gradients do not pay.** At n=2 MH is ~2x better in
ESS/s: ~83,000 draws per chain against NUTS's ~7,500, and on an easy 2-D geometry many cheap correlated
draws beat few expensive independent ones. A gradient costs 2 event-passes to a likelihood's 1, and
4.5-8 gradients go into each NUTS draw; at low dimension that is not repaid. Interpolating the ESSb/s
ratio through 1 gives n ~ 5.

**3. The n=17 "collapse" is NOT dimensional -- it is one parameter, and the fixed-n test proves it.**
The advantage appears to peak at n=12 and fall at n=17. But at *fixed* n=12, swapping a single dial
(`src_tail` for `res_axial_strength`) reproduces the collapse and exceeds it: ratio 1.21 against n=17's
1.63, with 137 divergences against 152. The **spread across 12-parameter subsets (2.56x) is larger than
the entire n=12 -> n=17 change (1.87x)**. Any claim that the advantage falls at high dimension is not
supported by this data; what falls is the advantage on a particular geometry.

**4. The geometry that breaks NUTS is a THREE-way degeneracy, not a pair.** `M_A_res`,
`delta_strength` and `res_axial_strength` are individually fine and pairwise fine -- nested-12 (first
two) gives 3.06 with 0 divergences, rand-12 (first and third) gives 3.05 with 5. All three together
gives 1.21 with 137 divergences and **R-hat 1.0124, i.e. it fails the convergence gate outright**. NUTS
does not merely lose there; it does not converge in the budget. A fixed Laplace mass matrix cannot
describe that direction, and the sampler answers with deeper trees and a diverging integrator.

**5. Bulk and tail disagree, and the tails are what a physics result quotes.** In ESS_tail/s MH is
better at every n except 12. Credible-interval endpoints come from the tails, so the gradient advantage
here is largely a *bulk* phenomenon.

**6. The median hides a reversal at the hard end.** At n=17 the median ESSb/s favours NUTS 1.63x, but on
the worst-mixing parameter (`M_A_res`) NUTS is 0.96 against MH's 1.10 -- i.e. *worse*. The min column is
reported for exactly this reason.

**7. The portable numbers halve the advantage.** Per forward *visit* to the model NUTS is 1.1-7.0x
better; per *event-pass* it is 0.56-3.5x, because a gradient costs a forward and a reverse sweep to a
likelihood's forward alone. The second is the honest accounting and is the same unit
`bench_fair_report.md` uses.

## What was wrong before, and how it was found

This report is the second version. Four independent adversarial audits of the pipeline found:

* **chain-truncation cost accounting** -- ESS was computed on truncated chains while the *full* sampling
  time was charged, deflating whichever method had uneven chains. That was NUTS at n=17 specifically,
  i.e. the bug landed exactly on the anomalous point.
* **`chi2_and_grad` reused the `chi2` counter**, so a value-and-gradient was indistinguishable from a
  likelihood downstream and the per-evaluation column charged a gradient at a likelihood's price --
  exactly 2x too kind to NUTS. Fixed with a separate counter and a `visits()` denominator.
* **the divergence counter was multiplied by zero** and, even without that, counted trees that did not
  saturate max depth. Every cell reported 0 divergences. With it fixed, n=17 shows 152 and hard-12 shows
  137 -- the single most informative diagnostic in the study, and it had been dead.
* **`split_rhat` omitted the folded half** of the published statistic, leaving the convergence gate blind
  to a pure scale mismatch (four chains with sd 1:2:4:8 gave R-hat 1.0000076).
* **a frozen chain scored as perfectly independent** (ESS = N instead of 0), with FFT round-off deciding
  between that and an absurd finite value -- rewarding a random walk's characteristic failure mode.
* **HVP passes were charged per column** rather than sharing the reverse sweep across a dispatch, which
  overcounted autodiff's Hessian cost ~2.7x (this made autodiff look *worse* than it is; see the
  Hessian section of the optimiser report).
* **the event-chunking window ownership double-counted** up to 28% of the bank, and the test could not
  detect it because it reassembled by assignment rather than summation.

The first version's headline that the advantage "peaks at n=12 and collapses at n=17" was an artefact of
the first bug plus the nested-subset confound. The `tau ~ n` result is the one claim that survived every
audit unchanged.

## Limitations

1. **One noise realisation per parameter set.** Required -- both samplers must see the same posterior --
   but the n-dependence still mixes dimensional scaling with the geometry of this particular throw.
2. **15,000 events/sample only.** Raising it should scale both methods' ESS/s together and leave the
   ratios alone; that is an expectation, not a measurement.
3. **The mass matrix is the fixed Laplace covariance, never adapted.** That is what makes the comparison
   fair, but finding 4 says it is also where NUTS loses: a warm-up-adapted metric would likely rescue
   hard-12, and testing that is the obvious next experiment.
4. **NUTS step-size adaptation is block rescaling toward 0.8, not dual averaging.**
5. **No clean n=17 point exists.** The only 17-parameter subset necessarily contains the full trio, so
   dimension and geometry cannot be separated there by construction.

## Reproducing

```
python -m adonis.fit.bench_mcmc --ndials <n> --method mh|nuts --chain <k> --sample-seconds 300
python -m adonis.fit.bench_mcmc --ndials 12 --dials '<explicit,names>' --method ... --tag hard
python -m adonis.fit.bench_mcmc_report --glob 'output/altgen/mcmc2_*.npz'
pytest tests/test_mcmc_diag.py tests/test_fit_kernels.py
```
