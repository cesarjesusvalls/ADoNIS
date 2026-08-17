# Do gradients change the scaling of MCMC? Metropolis-Hastings vs NUTS on one ADoNIS posterior

Companion to `docs/bench_fair_report.md`, which asks the same question of *optimisers*. Same principle:
both samplers call **one object**, so the only difference left is whether a gradient is used.

## How this was made fair

Both samplers call the same `adonis.fit.kernels.FitKernel`. MH asks it for `chi2`; NUTS asks for
`chi2_and_grad`. Likelihood, physical box, dataset, float64, device and the device-resident model are
therefore shared **by construction**, not by agreement. The posterior is `exp(-chi2/2)` truncated to the
physical box, MLE (prior widened x1e6, so flat inside).

**MH is not a strawman, and nothing else about this study matters if it is.** A random walk with an
isotropic proposal on a posterior whose worst pair correlates at -0.995 would lose by orders of magnitude
and say nothing about gradients. So MH gets:

* a proposal preconditioned by the **Laplace covariance** `(J^T W J)^-1` at the best fit -- *precisely*
  the information NUTS receives as its inverse mass matrix, so neither method is handed geometry the
  other lacks;
* a scale adapted during warm-up to **0.234** acceptance, the Roberts-Gelman-Gilks optimum for
  random-walk Metropolis. Measured acceptance across the grid: 0.214-0.268.

NUTS adapts its step size toward 0.8 (measured 0.81-0.92) with max tree depth 8.

**Equal compute, not equal samples.** Each chain warms up, then samples for a fixed **wall-clock budget**
(300 s). Neither method is advantaged by a draw count chosen to suit the other, ESS/s is measured rather
than inferred, and the study's runtime is bounded by construction.

**Setup.** Section-4 posterior at 15,000 events/sample (150k resident), 282-bin frozen mask, **statistically
fluctuated pseudo-data** (one Gaussian throw, *the same throw for every method and chain at a given n* --
otherwise the two samplers would be exploring different posteriors). Nested dial subsets n = 2, 4, 8, 12,
17 ordered by Gate-I shrinkage, the same ordering the optimiser benchmark uses. 4 chains per cell, started
overdispersed at 2x the Laplace width from a shared start. NVIDIA **turing** (not A100 -- see Limitations).
40 cells (4 chains x 2 methods x 5 dial counts), ~1.6M likelihood evaluations.

**Diagnostics are hand-rolled** (`adonis/fit/mcmc_diag.py`; this environment has no arviz/numpyro) and
unit-tested against cases with known answers in `tests/test_mcmc_diag.py`: iid draws (ESS = N), AR(1) with
correlation rho (ESS = N(1-rho)/(1+rho), tau = (1+rho)/(1-rho)), offset chains and within-chain drift
(Rhat >> 1), and rejection-induced ties. That test caught a real bug: the Geyer pairing started at lag 1
instead of lag 0, dropping the leading 1 from `tau = 1 + 2 sum rho_t`. It inflated ESS **3.3x at rho=0.5
but only 1.02x at rho=0.95** -- nearly harmless for badly-mixing chains, badly wrong for well-mixing ones,
i.e. it would have systematically flattered whichever sampler mixes better. That is the quantity this
report exists to measure.

## Gate 1: convergence

Rank-normalised split-Rhat, maximum over parameters. **All cells pass** (< 1.01).

| n | MH max Rhat | NUTS max Rhat | MH draws/chain | NUTS draws/chain |
|---|---|---|---|---|
| 2 | 1.0001 | 1.0000 | 51,000 | 4,820 |
| 4 | 1.0004 | 1.0002 | 49,450 | 5,010 |
| 8 | 1.0006 | 1.0005 | 50,050 | 4,260 |
| 12 | 1.0026 | 1.0003 | 51,600 | 3,390 |
| 17 | 1.0053 | 1.0072 | 48,250 | 2,100 |

## Gate 2: the same stationary distribution

Efficiency is meaningless if the two samplers are not sampling the same thing, and that failure is
**silent**: a sampler stuck in one mode reports a *small* ESS, which reads as an honest efficiency number
rather than a broken one. So the marginals are compared before any ratio is quoted -- worst dial per n,
difference of means in units of the combined MCSE:

| n | worst dial | \|dmean\|/MCSE | sd ratio | verdict |
|---|---|---|---|---|
| 2 | M_A_qe | 0.89 | 0.989 | PASS |
| 4 | sabs | 1.42 | 0.992 | PASS |
| 8 | s_piN_elastic | 1.10 | 0.973 | PASS |
| 12 | sabs | 1.49 | 1.008 | PASS |
| 17 | s_NN_inelastic[2] | 1.31 | 1.017 | PASS |

## Efficiency

| n | method | ESS_bulk/s | ESS_tail/s | ESS_bulk / 1e3 logp | ESS_bulk / 1e3 pass | tau_int | accept | grad/draw |
|---|---|---|---|---|---|---|---|---|
| 2 | mh | **19.02** | **22.49** | 112.17 | 112.17 | 8.6 | 0.233 | 0 |
| 2 | nuts | 9.66 | 8.06 | 126.98 | 63.49 | 1.7 | 0.917 | 4.74 |
| 4 | mh | **11.61** | **17.35** | 68.37 | 68.37 | 14.2 | 0.240 | 0 |
| 4 | nuts | 9.40 | 9.99 | 122.81 | 61.40 | 1.8 | 0.880 | 4.58 |
| 8 | mh | 6.44 | **10.76** | 37.88 | 37.88 | 25.9 | 0.233 | 0 |
| 8 | nuts | **11.82** | 9.56 | 150.10 | 75.05 | 1.2 | 0.863 | 5.55 |
| 12 | mh | 3.76 | 6.81 | 22.42 | 22.42 | 40.0 | 0.214 | 0 |
| 12 | nuts | **13.13** | **8.04** | 177.79 | 88.90 | 0.9 | 0.857 | 6.55 |
| 17 | mh | 2.61 | **5.06** | 15.65 | 15.65 | 62.0 | 0.262 | 0 |
| 17 | nuts | **3.32** | 3.13 | 43.32 | 21.66 | 2.1 | 0.812 | 10.98 |

NUTS / MH, so >1 means NUTS wins:

| n | ESS_bulk/s | ESS_tail/s | ESS/logp | ESS/pass |
|---|---|---|---|---|
| 2 | 0.51 | 0.36 | 1.13 | 0.57 |
| 4 | 0.81 | 0.58 | 1.80 | 0.90 |
| 8 | 1.84 | 0.89 | 3.96 | 1.98 |
| 12 | **3.49** | 1.18 | **7.93** | **3.96** |
| 17 | 1.27 | 0.62 | 2.77 | 1.38 |

## What this says

**1. Gradients change the SCALING, and the evidence is `tau_int`.** MH's autocorrelation time grows
essentially linearly in the parameter count -- **8.6, 14.2, 25.9, 40.0, 62.0** at n = 2, 4, 8, 12, 17 --
which is the textbook random-walk result, and it holds *despite* an optimal Laplace preconditioner and
acceptance pinned at the 0.234 optimum. There is no tuning left to give MH. NUTS's tau stays at **0.9-2.1
and is flat in n**. That is the answer to the question as posed: the advantage is asymptotic in n, not a
constant factor.

**2. There is a crossover near n = 6, and below it gradients do not pay.** At n = 2 MH is 2x better in
ESS/s: it takes ~50,000 draws per chain to NUTS's ~4,800, and on an easy 2-D geometry many cheap
correlated draws beat few expensive independent ones. A gradient costs ~2 event-passes against 1 for a
likelihood, and ~5-11 gradients are spent per NUTS draw; at low dimension that overhead is not repaid.

**3. The advantage is NOT monotone: it peaks at n = 12 (3.5x) and falls to 1.27x at n = 17.** NUTS's
gradients per draw jump 6.6 -> 11.0 and its tau rises 0.9 -> 2.1 there. The dial ordering explains it:
n = 15-17 adds `res_axial_strength`, `s_NN_elastic[2]`, `axial_strength`, completing the RES axial block --
the degenerate geometry that fails Gate I and whose worst pair correlates at -0.995. Both samplers
degrade; NUTS degrades faster, because a *fixed* Laplace mass matrix is a poorer description of that
posterior and the sampler answers by building deeper trees. A gradient method is not immune to bad
geometry; it is differently sensitive to it.

**4. Bulk and tail disagree, and the tails are what a physics result quotes.** In ESS_tail/s, **MH is
better at n = 2, 4, 8 and 17**; NUTS wins only at n = 12. Credible-interval endpoints come from the tails,
so on this posterior at these dimensions the gradient advantage is largely a *bulk* phenomenon. Anyone
quoting the 3.5x without this row is quoting the half of the result that flatters autodiff.

**5. The portable numbers halve the advantage.** In ESS per 10^3 *likelihood evaluations* NUTS looks
1.1-7.9x better; in ESS per 10^3 *event-passes* it is 0.57-3.96x, because a gradient costs two passes
(forward + reverse) to a likelihood's one. The second is the honest accounting of what autodiff buys, and
it is the same unit `bench_fair_report.md` uses, so the two studies can be read on one footing.

## Limitations

1. **Hardware is turing, not the A100 used in the optimiser report.** ESS/s therefore cannot be compared
   across the two documents; the ratios and the portable per-pass columns can. The grid was moved to
   turing mid-study for queue reasons and the earlier A100 cells were **discarded rather than merged**:
   mixing GPUs across the n-axis would have meant the scaling curve partly measured which card each cell
   landed on.
2. **Event count is fixed at 15,000/sample.** Event scaling was declared secondary; a larger N raises the
   cost of every likelihood and gradient roughly equally, so it should shift ESS/s uniformly without
   moving the ratios -- but that is an expectation, not a measurement.
3. **One noise realisation per n.** The optimiser study found seed-to-seed spread to be its dominant
   uncertainty (GN iteration counts varied 13-56 on the same problem). Here the throw is fixed so both
   samplers see one posterior, which is required for the comparison, but it means the *n*-dependence
   mixes genuine dimensional scaling with the particular geometry of this throw.
4. **NUTS step-size adaptation is the production scheme** (block-wise rescaling toward 0.8), not dual
   averaging. It reaches the target, but a dual-averaging warm-up might do better at n = 17 where the
   geometry is worst.
5. **The mass matrix is fixed at the Laplace covariance and never adapted.** That is what makes the
   comparison fair -- both methods get the same geometry and nothing else -- but a production NUTS would
   adapt it during warm-up, and finding 3 suggests that is exactly where n = 17 loses.

## Reproducing

```
python -m adonis.fit.bench_mcmc --ndials <n> --method mh|nuts --chain <k> --sample-seconds 300
python -m adonis.fit.bench_mcmc_report --glob 'output/altgen/mcmc_*.npz'
pytest tests/test_mcmc_diag.py          # the diagnostics, against known answers
```
