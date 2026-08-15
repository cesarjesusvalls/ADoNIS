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

**Do not read those alphas as a trend in the advantage.** They are fitted straight across a REGIME
CHANGE, and averaging over it is misleading. Taken directly, the wall-clock ratio to Gauss-Newton at
n=17 runs

    migrad+g (noise)    5.70 -> 3.46 -> 2.92 -> 3.01      (15k, 30k, 60k, 125k)
    migrad   (noise)   15.66 -> 8.70 -> 7.18 -> 7.78
    migrad+g (asimov)  11.93 -> 10.50 -> 8.60 -> 8.92
    migrad   (asimov)  28.52 -> 23.31 -> 18.83 -> 19.32

falling from 15k to 60k and then FLAT. At small event counts MIGRAD is overhead-dominated -- hundreds of
tiny calls, each paying a fixed launch and host sync -- while Gauss-Newton's batched work is already
compute-bound. By ~60k both are compute-bound and scale together, so the ratio stabilises. The honest
extrapolation to production statistics is therefore that the advantage **stays at ~3x and ~8x**, not
that it decays.

(An earlier draft of this report inferred "~2x at production" from the single-power-law fit. That was
wrong, and it is precisely the error the alphas invite: a power law fitted through two regimes describes
neither.)

**The parameter axis is where the advantage genuinely grows.** Fitted at 125k, cost ~ n^gamma:

| | wall-clock | event-passes |
|---|---|---|
| gn (asimov) | n^0.42 | n^0.92 |
| gn (noise) | n^0.81 | n^1.26 |
| migrad + gradient | n^0.89-0.96 | n^0.98-1.04 |
| migrad, no gradient | n^1.39-1.55 | n^1.39-1.55 |

Gauss-Newton's wall-clock is **sub-linear in the dial count**: a new dial is one more tangent column in
a dispatch that was already being launched, so it is close to free. Gradient-free MIGRAD grows as
~n^1.5, because each gradient costs ~2n objective calls *and* the iteration count rises with n.
Gradient-driven MIGRAD sits between at ~n^0.9, since the VJP itself is O(1) in n and only the call count
grows. The wall-clock ratio at 125k accordingly runs 2.7x (n=2) -> 4.4x -> 11.3x -> 19.3x (n=17) on
Asimov. Seventeen dials is the small end of what a real analysis wants, so this is the axis the argument
should lean on.

In event-passes, the intrinsic Gauss-Newton Jacobian is 1 primal + n tangents, i.e. linear; the measured
beta = 1.26 under noise exceeds that because the iteration count also grows with n (it is 0.92 on
Asimov, where the iteration count is flat at 9).

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
   cannot rescue it. The ratios are flat over the last doubling (60k -> 125k), so carrying them to 250k
   is a mild extrapolation rather than a leap, but it is still an extrapolation and not a measurement. Recovering it needs the Jacobian split into two device-resident stages, or
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

## The Hessian: what the Laplace/Occam factor costs, and whether the free route is usable

The Occam correction needs `log det V_nuis` at every node of a profile scan -- 221 nodes in 1-D, 7938 in
the 2-D corner. There are three routes, and they differ in accuracy as well as price. Measured at four
real nodes (one dial pinned 1 sigma off the best fit, the other 16 re-minimised), 600k resident events:

| route | wall/node | event-passes/node | objective calls | over the 7938-node corner |
|---|---|---|---|---|
| A `J^T W J`, by-product of the fit | **0.39 ms** | **0** | 0 | ~3 s, 0 passes |
| B exact Hessian, autodiff HVPs | 0.18 s | 48 | 0 | 0.40 h, 381k passes |
| C MINUIT HESSE, finite differences | 1.35 s | ~285 | ~285 | **2.98 h, 2.26M passes** |

Route A is free because the inner fit already built J. Route B is ~n HVPs (forward-over-reverse), i.e.
**O(n)** for the exact matrix. Route C is O(n^2) objective evaluations, each a full pass over the
resident events. B beats C by ~7x in wall-clock and ~6x in event-passes. (MINUIT's *other* covariance,
the one `migrad()` accumulates, is a quasi-Newton approximation built along the path taken; feeding that
into a log-det puts a path-dependent error into the very factor being computed, so HESSE is the honest
MINUIT route and it is the expensive one.)

### Is the free route accurate enough?

`J^T W J` is not the Hessian: it drops `sum_b r_b d2m_b`, exact only at zero residual. Differences in
`log det V` from the exact Hessian, in units where a difference `d` shifts the Occam-corrected chi2 by
`d` (so compare with the Delta chi2 = 1 scale of a 1-sigma interval):

| pinned dial | GN - exact (noise) | GN - exact (Asimov) | HESSE - exact (noise) |
|---|---|---|---|
| node 0 | +0.222 | +0.251 | 0.010 |
| node 1 | +0.040 | +0.059 | 0.003 |
| node 2 | +0.101 | +0.123 | 0.010 |
| node 3 | +0.018 | +0.050 | 0.607 |

HESSE agrees with the autodiff Hessian to ~0.01 on three of four nodes, which validates the HVP
implementation against an independent computation. (The 0.607 outlier is HESSE's own finite-difference
step, not the reference.)

**The null control.** At an UNPINNED Asimov best fit the residual is identically zero, so `J^T W J` must
BE the exact Hessian:

    max|H/2 - J^T J| / max|J^T J| = 4.9e-10      log det V:  GN - exact = -1.2e-06

It is. Note what this control is *not*: a profile NODE is not a null however noiseless the data, because
pinning a dial off the minimum leaves a residual by construction. An earlier reading of this benchmark
treated the Asimov nodes as a null and would have accepted a broken Hessian.

**Why the error is what it is.** At the noise best fit the dropped term is a *tiny* elementwise
perturbation -- 2.4e-04 relative -- yet it moves `log det V` by 0.145. To first order

    d(log det V) = -tr(A^-1 dA),     A = J^T W J,   dA = sum_b r_b d2m_b

A perturbation PROPORTIONAL to A would give n x 2.4e-04 ~ 4e-03, i.e. negligible. The observed 0.145 is
~36x that, because dA is not proportional to A: it carries weight in the directions where A^-1 is large.
The bound n x (relative size) x cond ~ 17 x 2.4e-04 x 260 ~ 1.06 contains it comfortably. So the
Gauss-Newton error in the Occam factor is not driven by the residual being large -- it is driven by dA
landing on the worst-constrained directions of an ill-conditioned nuisance block (cond ~260), which is
exactly the regime an Occam factor is introduced to handle.

(An earlier draft called this "an amplification of ~600x", which was simply 0.145 / 2.4e-04 -- the ratio
of a dimensionless log-det shift to a max-norm relative matrix perturbation. Those are not the same kind
of quantity and the ratio is not an amplification factor.)

### Recommendation

**Use route B, the exact autodiff Hessian.** The free route carries an uncontrolled error of order
0.02-0.25 in the quantity being computed -- up to a quarter of a Delta chi2 = 1 unit -- in a term whose
whole purpose is to be a correction. Route B removes that for 48 event-passes per node, and is still ~7x
cheaper than the MINUIT route that would be needed to get the same accuracy without autodiff. The
existing corner scans store `logdet_Vnuis` computed the Gauss-Newton way and should be revisited.

**What is NOT established here.** These four nodes are four *different pinned dials*, each at 1 sigma. A
profile scan varies ONE dial across many positions, and what enters the profile is the *variation* of the
Occam term along that scan, not its absolute value -- a constant offset cancels. Whether the
Gauss-Newton error is roughly constant along a scan (and so largely cancels) or varies with position is
not measured here. The 0.02-0.25 spread bounds it from above; establishing the actual effect on a quoted
interval needs one dial scanned at several positions, which is the obvious follow-up.
