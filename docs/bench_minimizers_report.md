# Minimiser and derivative benchmarks — measured numbers for the paper

Everything here was measured on the ADoNIS section-4 multisample closure between 2026-08-14 and
2026-08-15. It is intended to ground the computational-cost claims in the text with numbers rather than
call-counting arithmetic. **Read the "What these numbers are not" section before quoting any of them.**

Scripts: `adonis/fit/bench_minimizers.py`, `bench_scaling.py`, `bench_tol_boundary.py`,
`bench_laplace_node.py`. Outputs in `output/altgen/bench_*.npz`.

---

## 0. Setup, and the one caveat that applies everywhere

| | |
|---|---|
| problem | section-4 closure, `configs/fits/sec4_P1.yaml` |
| dials | 17 (Gate-I subset of the 28-knob basis) |
| samples | 10 (T2K ×2, MINERvA ×4, (e,e′), π⁺/p/n beams) |
| bins | 325 total |
| hardware | one NVIDIA RTX 2080 Ti (11 GB), jax 0.10.1, float64 |
| estimator | MLE (prior widened ×1e6), Asimov unless noise is stated |

**Statistics caveat.** Production is `sig_cap = 250 000` events/sample (**2 409 416 resident**). The
benchmarks run at **60 000/sample (600 000 resident)** or less, because the differentiable objective
needs device memory *on top of* the resident banks and 11 GB cannot hold both at production scale.

**`sig_cap` is not a pure statistics knob.** Fewer events raises the per-bin MC error, which masks more
bins, so ndf changes too:

| events/sample | total resident | live bins (of 325) |
|---|---|---|
| 15 000 | 150 000 | 136 |
| 30 000 | 300 000 | 175 |
| 60 000 | 600 000 | 219 |
| 250 000 (production) | 2 409 416 | 282 |

So each event point is its own fit, not the same fit with less data. **Method ratios transfer; absolute
times do not.**

---

## 1. Cost of the derivative objects  ← the O(n) / O(n²) claims

n = 17, 600 000 events, identical objective for every row. HESSE is the median of 3 runs, each on a
fresh `Minuit` instance (`hesse()` caches, so repeating on one instance would time a no-op).

| object | seconds | objective calls | vs autodiff |
|---|---|---|---|
| MINUIT gradient (2n finite differences) | 0.380 | 34 | **11.5×** |
| **ADoNIS gradient (1 reverse-mode VJP)** | **0.033** | ~2 | 1.0× |
| MINUIT Hessian (`HESSE`, ~2n²) | 3.530 | ~578 | **13.5×** |
| **ADoNIS Jacobian (n forward JVPs)** | **0.261** | ~17 | 1.0× |

The Jacobian is not merely a cheaper gradient: it is the **full 325 × 17 residual Jacobian**, and `JᵀJ`
is the Gauss-Newton Hessian at no extra cost. So the honest comparison for it is the HESSE row, which it
beats by 13.5× while returning strictly more information.

**Phrasing note.** "Finite differences inflate by a factor of n" is right in *call count* (2n = 34) but
**11.5× in time**, because one VJP costs 2.2 function calls rather than 1. State which.

### Why each method needs what it needs

* MIGRAD is quasi-Newton on the **scalar** χ². It needs ∇χ², a length-n vector; reverse mode gives that
  in **one** VJP at cost independent of n.
* Gauss-Newton needs the **residual Jacobian** J = ∂rᵢ/∂θⱼ to form `(JᵀWJ)δ = −JᵀWr`. The curvature
  `JᵀWJ` cannot be reconstructed from ∇χ² — note ∇χ² = 2JᵀWr is already J *contracted* with the
  residual, i.e. exactly the one VJP MIGRAD gets.
* J here is tall (325 × 17), so **forward** mode is correct: one JVP per column = 17 passes, against 325
  VJPs for reverse. Reverse would win only if dials outnumbered bins.

---

## 2. Scaling in parameters and events

Truth injected **on the fitted dials only**, so χ²_min = 0 at every n and each point is a well-specified
closure. (An earlier version injected the full 17-dial truth and froze the unfitted dials, which made
every n < 17 point misspecified — χ²_min of 610/516/434 at n = 4/8/12 — and produced meaningless
iteration counts. Those numbers are withdrawn.)

### Per-call costs — the scaling laws, measured

| | n=4 | n=8 | n=12 | n=17 |
|---|---|---|---|---|
| χ² per call, 150k events | 4.6 | 4.6 | 4.6 | 4.6 ms |
| χ² per call, 300k | 8.3 | 8.3 | 8.4 | 8.3 ms |
| χ² per call, 600k | 15.0 | 14.9 | 14.9 | 14.9 ms |
| Jacobian per call, 150k | 25.4 | 39.1 | 54.5 | 76.7 ms |
| Jacobian per call, 300k | 44.7 | 68.4 | 97.7 | 138.7 ms |
| Jacobian per call, 600k | 80.4 | 123.9 | 190.7 | 264.5 ms |

* **χ² is flat in n at every event count → O(1).**
* **Jacobian is linear in n, and its slope doubles as events double** (3.9 → 7.2 → 14.2 ms/dial) → O(n·N).
* Reverse-mode gradient costs 2.2 × a χ² call, independent of n.

### Full fit times (Asimov, medians of 2)

| events | n | GN | MIGRAD+grad | MIGRAD (no grad) |
|---|---|---|---|---|
| 150k | 4 | 1.20 s | 0.39 | 0.48 |
| 150k | 8 | 1.51 | 1.06 | 1.48 |
| 150k | 12 | 1.68 | 0.60 | 2.09 |
| 150k | 17 | 2.18 | 2.15 | 5.02 |
| 300k | 4 | 1.32 | 0.64 | 0.82 |
| 300k | 8 | 1.91 | 1.17 | 2.06 |
| 300k | 12 | 2.23 | 1.94 | 4.34 |
| 300k | 17 | 2.62 | 4.88 | 9.52 |
| 600k | 4 | 1.55 | 1.09 | 1.44 |
| 600k | 8 | 2.20 | 2.09 | 3.63 |
| 600k | 12 | 3.12 | 3.62 | 8.53 |
| 600k | 17 | **3.98** | **5.20** | **13.11** |

**GN converges in 7–10 iterations at every n and every event count** on a clean closure. Derivative-free
MIGRAD's call count grows roughly ∝ n (134 → 422 → 591 → 1427 at 150k), as expected from 2n evaluations
per gradient.

**The ranking depends on the operating point** — MIGRAD+grad wins at few dials and few events, GN wins as
both grow, because GN's iteration count is fixed while its per-iteration cost is O(n).

---

## 3. Headline single-point comparison (n = 17, 600k events, Asimov)

| method | median s | value calls | derivative calls | max \|Δθ\|/σ_prior |
|---|---|---|---|---|
| **GN (TRF)** | **3.4** | 9 | 9 | 8.6e-13 |
| MIGRAD + gradient | 5.2 | 388 | 24 | 0.134 |
| MIGRAD, no gradient | 13.1 | 1154 | 0 | 0.134 |

Ready-made sentence, with correct units: *a derivative-free minimiser needs ~1150 function evaluations
where Gauss-Newton converges in 9 iterations, and roughly a third of that (386) when handed the exact
gradient.* **386/1154 = 0.334.**

**Do not turn "9 vs 1150" into a speed claim.** Each GN iteration costs a full Jacobian, 17.4 × a
function call. In wall-clock the advantage is **3.9×** over derivative-free MIGRAD and **1.5×** over
gradient-fed MIGRAD, not ~100×.

---

## 4. Behaviour under statistical noise — the ranking reverses

50 toys, closure truth fixed, only the statistical throw varying, **identical data for every method in a
given toy** (the throw is generated once per toy, before any minimiser runs).

| method | median s | value calls | derivative calls | median χ² |
|---|---|---|---|---|
| GN (TRF) | 15.31 | 57 | 31 | 205.22 |
| **MIGRAD + gradient** | **5.53** | 403 | 33 | 205.95 |
| MIGRAD, no gradient | 16.97 | 1527 | 0 | 205.18 |

**Gradients are worth 3.1×** to MIGRAD (16.97 → 5.53 s). Derivative-free MIGRAD and GN cost about the
same (16.97 vs 15.31 s).

### Why MIGRAD is faster here, when it was slower on Asimov

With the measured per-call costs, the time decomposes exactly:

| | function/residual | derivative | total |
|---|---|---|---|
| GN | 57 × 106 ms = 6.0 s | 31 × 260 ms = **8.1 s** | 14.1 s (measured 15.3) |
| MIGRAD+grad | 403 × 14.9 ms = 6.0 s | 33 × 33.5 ms = **1.1 s** | 7.1 s (measured 5.5) |

Both spend ~6 s on function evaluations; **the whole difference is the derivative term.** Two causes:

1. **GN needs 31 Jacobians on noisy data, not 9.** Gauss-Newton approximates the Hessian by `JᵀJ`,
   dropping a term proportional to the residual. On an Asimov closure the residual → 0, the
   approximation becomes exact, and GN converges quadratically — the 9 iterations. With noise the
   residual is O(√ndf) and never vanishes, so convergence degrades and it takes 31 steps.
   **The Asimov benchmark flatters Gauss-Newton.**
2. Its derivative object costs 7.8× more per call (260 vs 33.5 ms), and that gap is structural — O(n)
   against O(1).

**The correct framing is that the ranking is a property of the residual at the solution**, not of the
methods: near-zero residual favours Gauss-Newton, finite residual favours a cheap-gradient quasi-Newton.

---

## 5. Boundary behaviour — E_b on its wall

`Eb_shift` has a hard lower bound and its **nominal value is that bound** (`_EB_EPS = 0.01 = phys_lo`),
so with the production `start: nominal` every fit begins *on* the wall.

**50 toys, truth E_b = 0.5 (off the wall), statistical noise, start at nominal.** Expected boundary
occupancy is the censoring rate Φ((0.01 − 0.5)/σ_post) = **10.8%**, with σ_post = 0.397 measured from
GN's own scatter.

| method | at wall | measured | z vs null | binomial p |
|---|---|---|---|---|
| **GN (TRF)** | 2/50 | 4.0% | −1.56 | 0.17 — **consistent** |
| **MIGRAD + gradient** | 32/50 | **64.0%** | **+12.1** | 3e-19 |
| **MIGRAD, no gradient** | 26/50 | **52.0%** | **+9.4** | 7e-13 |

**MIGRAD over-produces boundary solutions by a factor of 5–6.** GN reproduces the expected rate.

### The tolerance is not the knob

Separate ensemble, 24 Asimov toys with E_b thrown uniformly off the wall (expectation exactly 0, since
the Asimov MLE *is* the truth):

| tol | at wall |
|---|---|
| 0.1 | 9/24 (37.5%) |
| 0.01 | 9/24 (37.5%) |
| 1e-3 | 9/24 (37.5%) |

Three orders of magnitude in `tol` change nothing — and not even the same 9 toys each time. GN: 0/24.

**Cause: the start point.** Re-running the identical 24 toys with E_b started at 4.0 instead of nominal
gives **0/24** at the wall, and toys with \|ΔE_b\| > 0.1σ fall from 7/24 to 2/24. MINUIT maps bounded
parameters through a transform whose derivative vanishes at the limit, so a parameter starting on the
bound sees zero internal gradient — a stationary point no convergence tolerance can escape. Escape
requires the χ² pull, which is why failures cluster at truths near the wall (all failures had
E_b* < 3.6 MeV; all 8 toys above it escaped).

**Practical recommendation: start E_b off its floor for any MINUIT-based fit.** It removes a 37.5%
failure mode for free. It does not close the gap to GN, which needs no such help because TRF handles
bounds by reflection rather than by a degenerating transform.

**Tightening tol also buys nothing else:** at tol = 1e-8 versus 0.1, χ² improves by 1e-4 and the boundary
fraction is unchanged within errors, for **8× the time** (63.8 s vs 8.2 s) and 9× the calls (5488 vs 586).

---

## 6. Cost of the Laplace (Occam) correction — measured per node

The correction needs `log det V_nuis` at every profile node, where `V_nuis` is the covariance of the
nuisance dials with the scanned dial held **fixed** (so the dimension is n−1 = 16, not 17).

* **Autodiff:** the inner fit is Gauss-Newton, so it has already built J at the solution and
  `V = (JᵀWJ + P)⁻¹` falls out of it. The marginal cost is one decomposition of a 16 × 16 matrix.
  Nothing extra is evaluated.
* **MINUIT:** `m.covariance` after `migrad()` is an *accumulated* BFGS-style approximation, not an exact
  Hessian — putting it into a log-det injects an uncontrolled error into the very factor the correction
  is about. The defensible route is `m.hesse()`, ~2(n−1)² evaluations.

Measured at four real nodes (one dial pinned 1σ from the BFP, the other 16 re-minimised), 600k events.
HESSE is the median of 3 fresh `Minuit` instances per node.

| node (pinned dial) | inner fit | autodiff log-det | MINUIT `hesse()` |
|---|---|---|---|
| `M_A_qe` | 6.78 s | 73.6 µs | 3.14 s |
| `M_A_res` | 9.16 s | 102.0 µs | 3.23 s |
| `axial_strength` | 7.49 s | 124.1 µs | 3.29 s |
| `res_axial_strength` | 7.51 s | 67.6 µs | 3.16 s |
| **median** | **7.50 s** | **87.8 µs** | **3.20 s** |

* The autodiff log-det is **0.0012% of the inner fit** — genuinely a by-product.
* `hesse()` adds **0.43 × the cost of the inner fit itself** to every node.
* Ratio between the two routes: **~36 000×**.

Node counts in this analysis: **221** for the 1-D profile (17 dials × 13 nodes), **7 938** for the 2-D
corner (441 nodes × 18 pair shards). Extra cost of the MINUIT route:

| | nodes | MINUIT extra | autodiff extra |
|---|---|---|---|
| 1-D profile | 221 | **0.20 h** (12 min) | 0.019 s |
| 2-D corner | 7 938 | **7.05 h** | 0.70 s |

At production statistics (2.4M events, ~4×) that becomes ~0.8 h and ~28 h.

**Do not write that this is impossible without autodiff.** 12 minutes for the 1-D scan is entirely
practical. The defensible claim is the corner: *hours of dedicated computation instead of a 0.7-second
by-product*, growing as O(n²) per node — at all 28 knobs rather than 17 it is another ~2.7×.

---

## 7. The claim that *does* survive "could not be attempted at all"

The NUTS sampler, measured from the production run: **24 000 samples over 16 chains, 281 710 gradient
evaluations** (median tree depth 4).

| route | cost |
|---|---|
| reverse-mode AD | 281 710 × 33.5 ms ≈ **2.6 h** |
| finite differences | 281 710 × 2n = **9.6 M** model evaluations × 14.9 ms ≈ **40 h** |

and the finite-difference version additionally inherits the simulator's own stochastic noise in a
quantity the symplectic integrator amplifies along every trajectory — which no amount of CPU time fixes.

---

## What these numbers are not

1. **Not production statistics.** 600k resident events, not 2.4M. Ratios transfer, absolute times do not.
2. **Not a pure event scan.** Live bins move 136 → 175 → 219 → 282 across the event axis; statistics and
   ndf change together.
3. **Not an ensemble, except where stated.** The timing benchmarks are single-dataset with repeats
   (2 per fit, 5 per per-call cost, 3 for HESSE). Only §4 and §5 are toy ensembles.
4. **The dial axis takes a prefix of the Gate-I subset**, so "n dials" is a different physics problem at
   each n, not one problem rescaled. Per-call scaling laws are robust to this; iteration counts less so.
5. **One GPU model throughout** (RTX 2080 Ti); every job's node and GPU were recorded and checked before
   combining.

## Corrections made during this work, worth knowing about

* An earlier n-scan injected the full truth and froze unfitted dials, making n < 17 misspecified
  (χ²_min 434–610). Withdrawn and re-run.
* An earlier significance column divided by the *measured* fraction's error rather than the null's.
  Corrected; GN's apparent −2.8σ boundary deficit became a consistent p = 0.17.
* Timings that included XLA compilation inside the clock (65 of 72 s in one case) were discarded. The
  failure was invisible because it hit every repeat equally and so produced a *tight, convincing* spread.
* A wrapper that served MIGRAD's value and gradient from one cached `value_and_grad` made
  "MIGRAD with gradients" appear *slower* than without. Value and gradient now come from separate
  compiled objectives.
