# What a fit costs: Gauss-Newton vs MIGRAD on one shared kernel

**Supersedes `docs/bench_minimizers_report.md` (commit `93d8fa7`), which is withdrawn.** That report
compared two *implementations*, not two algorithms: Gauss-Newton reached the model through a host round
trip (per-event weights, and for the Jacobian per-event **derivatives**, shipped back and binned with
`np.bincount`) while MIGRAD's objective was fully device-resident. 106 ms against 14.9 ms for identical
physics. The whole comparison moved by 3.6x when one environment variable was flipped, which is proof
enough that it was not measuring an algorithm. Its **counts** survive; every wall-clock number in it does
not. See `docs/bench_fair_plan.md` for the diagnosis and the plan this report executes.

## How this was measured

Both minimisers call **one object**, `adonis.fit.kernels.FitKernel`, built from a single per-sample
function `f_s(x) -> binned model blocks`. Residuals, chi2, the gradient (one reverse VJP) and the full
residual Jacobian (vmapped forward JVP, **binning inside the jit**) are all derived from that one
function. The only thing that differs between the arms is which of those four objects the algorithm asks
for. Objective, prior block, whitening, dead-bin mask, box, start point, precision and device are shared
by construction, so they cannot silently diverge.

The kernel was accepted against the host path before any timing was taken
(`adonis.fit.verify_kernels`). Rather than assert a tolerance, that test **decomposes** the disagreement:

| stage changed | difference |
|---|---|
| binning: `np.bincount` -> device `segment_sum` | 1.07e-17 |
| vmapped jvp -> looped jvp | 2.10e-16 |
| re-jitting the bank reweight alone, no binning | 1.06e-07 per event |
| -> after binning (thousands of events per bin) | 2.63e-11 per bin |
| chi2 floor at the closure truth | **1.62e-11** |

So the rewrite itself is exact; the entire residual difference is XLA re-optimising the reweight
arithmetic when `_wf` is inlined into a larger program. It is shared identically by both arms, and it
bounds how deep a convergence target may be set (nothing below ~1e-8).

**Setup.** Section-4 closure, MLE (prior widened x1e6), 10 samples, 282 live bins. sigma and the live-bin
mask are **frozen from the production 250k run** at every event count, so the event axis is not
confounded by a moving bin count (unfrozen, it moves 136/175/219/282). Dials are nested subsets of the
Gate-I 17 ordered by shrinkage. Truth is injected on the fitted dials only, so every n is a reachable
closure. Data is that truth plus per-bin Gaussian noise; the throw is generated once per realisation and
every method fits the same data. `Eb_shift` starts off its lower bound (starting on a wall measures
time-to-fail; that behaviour is a separate, completed result). One process per (N, n) cell, NVIDIA A100
40 GB, float64, one Jacobian dispatch. 342 fits.

## The primary result: noise, n=17, 125,000 events/sample (1.25M resident)

| method | wall-clock | event-passes | objective evals |
|---|---|---|---|
| Gauss-Newton | **1.58 s** [0.63-1.90] | 573 [247-704] | 48 [13-56] |
| MIGRAD + autodiff gradient | 4.75 s [4.60-6.17] | 484 [474-713] | 416 |
| MIGRAD, no gradient | 12.26 s [12.03-14.22] | 1656 [1625-1918] | 1656 |

Ratios to Gauss-Newton: **3.0x** and **7.8x** in wall-clock; **0.84x** and **2.89x** in event-passes.

**Read those two rows against each other, because they disagree, and the disagreement is the finding.**
Gradient-driven MIGRAD does *less* total event work than Gauss-Newton here — 484 passes against 573 —
and still takes three times as long. Gauss-Newton's advantage in this regime is not arithmetic saved, it
is arithmetic **batched**: its 573 passes arrive in ~50 dispatches, because one Jacobian carries all 17
tangents in a single launch, while MIGRAD's 484 arrive in ~450 separate calls each paying its own launch
and host sync. Anyone quoting "3x faster" without that sentence is quoting a property of the dispatch
structure and calling it a property of the mathematics.

## The contrast: Asimov (no noise), n=17

| method | wall-clock | event-passes | objective evals |
|---|---|---|---|
| Gauss-Newton | 0.46 s | **171** | **9** |
| MIGRAD + gradient | 4.10 s | 438 | 388 |
| MIGRAD, no gradient | 8.89 s | 1203 | 1203 |

Ratios: 8.9x / 19.3x in wall-clock, 2.6x / 7.0x in event-passes. Asimov is *deterministic* — the seed
spread is exactly zero, which is a check on the harness — and Gauss-Newton's cost here is **exactly
independent of the event count**: 171 event-passes and 9 iterations at 15k, 30k, 60k and 125k alike.
9 iterations reproduces the historically recorded closure count.

Asimov flatters Gauss-Newton and must not be quoted alone. GN approximates the Hessian by `J^T J`,
dropping a term proportional to the residual; on a perfect closure that term vanishes, GN *becomes*
Newton and converges quadratically. Real data leaves a residual of O(sqrt(ndf)), the dropped term stays
finite, and the cost rises from 9 iterations to a median of 48 with a spread of 13-56.

## Scaling

Fitted on the medians across four event counts (15k -> 125k, an 8.3x span) at n=17, noise cells:

```
wall-clock vs events,  log t = alpha log N
    gn        alpha = 0.88      near compute-bound
    migrad+g  alpha = 0.58      partly overhead-bound
    migrad    alpha = 0.56

event-passes vs dials, log p = beta log n     (at 125k)
    gn        beta  = 1.26
    migrad+g  beta  = 1.04
    migrad    beta  = 1.55
```

**Gauss-Newton's wall-clock advantage narrows as statistics grow.** Its work is batched and therefore
compute-bound (alpha ~ 0.9); MIGRAD's is spread over hundreds of small calls whose cost is partly fixed
overhead, so its wall-clock grows more slowly with N. Extrapolating the 3.0x at 125k to the production
2.4M-event configuration gives roughly **2x**, not 3x. That extrapolation is the weakest number in this
report and is flagged as such — see Limitations.

In the dial count, the intrinsic Gauss-Newton Jacobian is 1 primal + n tangents, i.e. linear; the
measured beta = 1.26 exceeds that because the iteration count also grows with n. Gradient-free MIGRAD's
beta = 1.55 is the compounding of a per-gradient cost of ~2n objective calls with an iteration count
that itself grows.

## What this supports, and what it does not

**Supported.** The structural argument about derivative *counts*, which is implementation-independent
and N-independent: 9 objective evaluations to converge the closure, against 388 for MIGRAD with a
gradient and 1203 without. One reverse VJP delivers the gradient regardless of dial count; a finite
difference costs ~2n objective calls, and HESSE ~2n^2. The covariance is a by-product of a Jacobian the
Gauss-Newton fit has already built, whereas MINUIT must evaluate for it.

**Not supported by this report.** Any claim that Gauss-Newton does less arithmetic than a
gradient-driven quasi-Newton method on realistic (noisy) data — measured, it does slightly more. The
wall-clock advantage is real on this hardware and comes from batching.

## Limitations

1. **No production-statistics point.** 250k events/sample (2.4M resident) is absent: the fused Jacobian
   fails to load its compiled CUBIN there even on a 40 GB card and at dial batch 1, so dial-batching
   cannot rescue it. The 2x extrapolation above rests on a 4-point fit over 8.3x and should not be
   quoted as a measurement. Recovering it needs the Jacobian split into two device-resident stages, or
   chunking over **events** rather than dials — the latter is exact and costs no extra passes, since
   binning is a sum over events.
2. **One missing cell**, N=125k / n=12, for the same reason; it does not affect any number quoted here.
3. **Seed spread is the dominant uncertainty**, not timing jitter. Gauss-Newton's iteration count varies
   13-56 across noise realisations on the same problem, because the Hessian term it drops scales with
   the residual and the throw is what sets the residual. Five realisations per cell; medians quoted with
   min-max throughout. Any single-realisation number is close to meaningless.
4. **One GPU model.** All wall-clock is A100 40 GB. Event-passes are the portable quantity; the
   convention is primal = 1 pass, each JVP tangent = 1, one VJP = 2 (forward + reverse sweep), and the
   raw counts are stored so a reader can re-weight them.
5. **Stopping rules were not compared.** Times are to a *common* accuracy target, measured against the
   lowest chi2 any method reached in that cell; a method that never reaches a target is recorded as not
   reaching it and never scored as a fast time.

## Reproducing

```
python -m adonis.fit.verify_kernels --sig-cap 20000          # acceptance gate, must pass first
python -m adonis.fit.bench_fair --sig-cap <N> --ndials <n> --noise-seeds 1,2,3,4,5
python -m adonis.fit.bench_report --glob "output/altgen/cell_amp_N*_n*.npz"
```

One process per cell: each `FitKernel` loads ~50 CUBIN modules onto the device and `jax.clear_caches()`
does not unload them, so building several kernels in one process exhausts the CUDA context regardless of
how much memory is free. That failure is why several earlier runs died on a *smaller* kernel than one
that had just succeeded.
