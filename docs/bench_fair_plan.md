# Plan: a defensible GN-vs-MINUIT workload comparison

**Status: plan only, nothing implemented.** Written 2026-08-15 after the first round of benchmarks was
found to compare implementations rather than algorithms. Everything in
`docs/bench_minimizers_report.md` (commit `93d8fa7`) is **withdrawn** pending this work; see §6 for
exactly which numbers survive.

---

## 0. What went wrong the first time

Flipping one environment variable (`S4_JAX_BIN`) moved GN from 3.4 s to 12.3 s at a fixed problem while
leaving MIGRAD untouched. A comparison that moves by 3.6× under a switch that changes no mathematics is
not measuring the algorithms. The cause is that the two arms reach the model through different code:

* **GN** (`eng.model`, `eng.jac`) computes per-event weights on device, **ships them to the host**, and
  bins with `np.bincount`.
* **MIGRAD's objective** (`chi2_fn` → `model_blocks_jax` → `bin_w_dev`) stays on device and returns a
  scalar.

Measured consequence: 106 ms vs 14.9 ms for the *same physics*. So most of what I attributed to
"Gauss-Newton's derivative is expensive" was a host round-trip.

---

## 1. The questions this design has to answer

**Q1. Algorithms or implementations?**
Algorithms. Both minimisers must sit on one shared, best-effort primitive layer, so the only difference
left is which derivative object the algorithm consumes.

**Q2. What must be identical; what may differ?**
Identical: objective (including the prior block), bin mask, bounds, start point, precision, data,
hardware, and the primitives that evaluate model/derivatives. May differ: the algorithm, and the
derivative object it asks for.

**Q3. How do you compare minimisers with different stopping rules?**
You do not compare single wall-clock numbers. GN stops on the projected gradient (`gtol = 1e-8`,
absolute) and reaches chi2 ~ 1e-24; MIGRAD stops on EDM and reaches 1.9. Those are different amounts of
work. Instrument both to record `(t, chi2, max|dtheta|/sigma)` per iteration and report **time to reach
common accuracy targets**. Neither native stopping rule enters the comparison.

**Q4. What if a method cannot reach a target?**
Report non-convergence. Never score a failure as a fast time.

**Q5. What unit makes the result portable?**
**Event-passes** — full passes over the resident events — alongside wall-clock. Counts are
implementation- and hardware-independent; wall-clock is their realisation on one machine and must always
be tagged with its configuration.

**Q6. What is the deliverable?**
Computing workload as a function of **(N_events, n_params)**, not a headline number.

---

## 2. Oddities found while reading the fitting code

| # | Oddity | Location | Consequence |
|---|---|---|---|
| 1 | `JAC_BATCH = 16` against 17 dials → dispatches of 16+1. The comment says "Default = all dials in one call"; the value does not achieve that. | `multisample.py:57` | one wasted primal pass, and it lands specifically on the n=17 point of any scan |
| 2 | The Jacobian ships **per-event derivatives** to the host: `G = np.asarray(call(ks))` is `(B, n_events)` ≈ 77 MB per sample per iteration at B=16, N=600k | `multisample.py:76` | dominates GN's cost; the binning that follows is `np.bincount` on the host |
| 3 | The device binning path loops **per dial × per dataset** in Python (~170 tiny dispatches, each returning a few floats) | `multisample.py:193` | device path measured *slower* than host (944 ms vs 260 ms), contradicting its own docstring |
| 4 | GN's residual path is host-bound; MIGRAD's objective is one fused device kernel | `multisample.py:170` vs `:173` | 106 ms vs 14.9 ms for identical physics — the root of the unfair comparison |
| 5 | `eng.jac(th, subset)` is called **twice at the same theta** in the post-fit block | `fitters.py:157` and `:166` | one entirely redundant Jacobian per fit |
| 6 | `chi2_fn` has **no prior term**; `trf_fit` includes one | `multisample.py:359` vs `fitters.py:93` | structurally different objectives (numerically negligible under MLE, but that must be verified, not assumed) |
| 7 | `_batched_jac` halves the batch on device OOM **silently** | `multisample.py:79-84` | a timing can degrade with no signal in the output |
| 8 | The live-bin mask depends on `sig_cap` (via `mcerr`) | `multisample.py:481` | an event scan changes ndf as well as statistics: live bins 136/175/219/282 at 150k/300k/600k/2.4M |
| 9 | `chi2_fn()` returns a **fresh `jax.jit`** on every call | `multisample.py:401` | invites recompilation inside a timed region — this actually happened, and cost 65 of 72 measured seconds |
| 10 | `_JAX_BIN` and `JAC_BATCH` are read from the environment at **import** time | `multisample.py:57,85` | the configuration that produced a number is invisible in the output |

Items 2–4 are one defect: **there is no fused "model → bins" primitive**, so both existing paths are
compromises and GN happens to sit on the worse one for its access pattern.

---

## 3. Work plan

### Phase 1 — Fused device-resident primitive layer  *(the only substantial code)*

New `adonis/fit/kernels.py`. For a given engine + dial subset, build from **one** source:

| primitive | returns | how |
|---|---|---|
| `residuals(x)` | `(nbin + n,)` | whitened data residuals stacked with the prior block |
| `chi2(x)` | scalar | sum of squares of the above |
| `grad(x)` | `(n,)` | one reverse-mode VJP |
| `jac(x)` | `(nbin + n, n)` | vmapped JVP with the **binning inside the jit** |

Non-negotiables:

* the Jacobian returns `nbin x n` floats and **never** per-event arrays to the host;
* dial-batching is allowed only where device memory requires it, and batching changes wall-clock but not
  event-passes, so it is recorded and reported;
* one jitted object per (engine, subset), built once and reused — no `jax.jit` inside a timed region;
* data and weights are arguments, not captured constants, so toys retarget without recompiling.

**Acceptance test, before any timing is taken:** residuals, chi2, gradient and Jacobian agree with the
existing host path to ~1e-12 at several theta, including the truth and the start. Correctness first.

### Phase 2 — Put both minimisers on it

`trf_fit` consumes `residuals` + `jac`; MIGRAD consumes `chi2` + `grad`. The prior block is present in
both and asserted numerically equal. Remove the duplicated post-fit Jacobian (oddity 5).

### Phase 3 — Time-to-accuracy instrumentation

Both record `(wall time, chi2, max|dtheta|/sigma_prior)` per iteration — GN through the existing
`record=` hook, MIGRAD through its callback. Report time and event-passes to reach:

* `chi2 - chi2_min < {1, 1e-2, 1e-4, 1e-8}`
* `max|dtheta|/sigma < {1e-1, 1e-3, 1e-6}`

This is what settles whether GN's 31 iterations under noise are real or an artefact of chasing an
absolute `gtol` far below any meaningful accuracy.

### Phase 4 — Fatal audit before timing

Assert, print, and refuse to run on failure: binning path identical; one Jacobian dispatch (or the batch
count recorded); `jax_enable_x64`; device is GPU; host-vs-device objective agreement < 1e-10; prior block
fraction of chi2 < 1e-6; live-bin counts equal on both sides; start points equal; no OOM fallback fired;
`S4_*` configuration echoed into the output file.

### Phase 5 — Run the grid

* **N (events/sample):** 15k, 30k, 60k, 125k, 250k — production included
* **n (dials):** 2, 4, 8, 12, 17 — **nested subsets of the Gate-I 17**, ordered by shrinkage
  (best-constrained first), so n=17 is exactly the sec4 fit and every smaller n is a sub-problem of it.
  **Estimator stays MLE (prior widened x1e6) at every n**, matching sec4.
  *Rejected alternative, and why:* extending the axis to the full 28-knob basis would have brought in
  dials that fail Gate I (shrinkage > 0.5), and to keep the objective well-posed there I proposed
  switching to MAP. That was wrong twice over. It changes WHICH FIT is being benchmarked — the same
  class of move as the environment-variable flip that invalidated the first report — and the premise
  did not survive checking: under MLE the prior is widened, not removed, so the Hessian is
  `J^T W J + 1e-12 diag(1/prior^2)` and never actually singular; shrinkage > 0.5 measures the
  posterior-to-prior width RATIO, not vanishing data curvature; and the sec2 28-knob Jacobian has
  nonzero gradient for all 28 knobs, so no column of J is empty. Restricting the axis to the 17
  removes the question instead of legislating an answer to it.
* **Conditioning is measured, not assumed.** cond(`J^T W J` + prior) and its smallest singular value are
  recorded at every grid point. If ill-conditioning degrades one minimiser more than the other at
  larger n, that IS the result the scan is for; it is reported, never designed away.
* **Mask frozen** from one reference configuration so the N axis is pure statistics (fixes oddity 8)
* **Data:** closure truth **plus statistical noise**, 3 realisations per point, throw shared across
  methods, workload reported as the median; **plus one Asimov fit per point** to quantify the
  small-residual effect
* **Start:** nominal, with **E_b off its wall** (settled: on-wall starts measure time-to-fail, and the
  wall behaviour is already a completed separate result)
* One engine build per N, run as parallel jobs, one GPU each

### Phase 6 — New report

Written from scratch. Workload in event-passes and wall-clock, every number tagged with its
configuration.

---

## 4. Why noise is the primary regime

Measured at 17 dials / 600k events:

| | GN iterations | GN time | MIGRAD+grad |
|---|---|---|---|
| Asimov | 9 | 3.4 s | 5.2 s |
| + noise | 31 | 15.3 s | 5.5 s |

Gauss-Newton approximates the Hessian by `J^T J`, dropping a term proportional to the residual. On a
perfect closure the residual vanishes, GN *becomes* Newton and converges quadratically. Real data leaves
a residual of O(sqrt(ndf)), so the dropped term stays finite. **An Asimov-only surface would overstate GN
everywhere**, which is why noise is primary and Asimov is kept only as a contrast.

Caveat carried into Phase 3: part of the 9 -> 31 may be the absolute `gtol`, not convergence behaviour.
The trajectories will separate the two.

---

## 5. Open risks

1. **Production statistics may not fit on an 11 GB card.** Untested under a correct configuration. The
   fused Jacobian removes the large host transfer but the vmapped JVP's device intermediates still scale
   as n x n_events; dial-batching is the mitigation, and it is recorded when it fires.
2. **Noise reintroduces E_b censoring.** The closure truth sits ~1.2 sigma from the bound, so a
   fluctuation can legitimately push the MLE onto it. Detected and reported separately, never folded
   into a median.
3. **Fixing the Jacobian touches production fit code**, not just the benchmark. It needs its own
   correctness test before any physics result is regenerated with it. Expected upside: it removes a
   ~77 MB/sample/iteration host transfer from the real fits.

---

## 6. What survives from the withdrawn report

**Survives** (counts and structure, not wall-clock):

* iteration and call counts: GN 9 (Asimov) / 31 (noise); MIGRAD+grad 386 value + 24 gradient; MIGRAD
  1154 value. Ratio 386/1154 = 0.334.
* derivative-object call counts: 2n for a finite-difference gradient, ~2n^2 for HESSE, 1 VJP, n JVPs.
* NUTS: 24,000 samples, **281,710 gradient evaluations**.
* profile node counts: 221 (1-D), 7,938 (2-D corner).
* E_b boundary results: 64% vs 10.8% expected at +12 sigma; tol-independent across 0.1/0.01/1e-3;
  fixed by moving the start off the wall (9/24 -> 0/24).
* live-bin counts vs statistics: 136/175/219/282.

**Withdrawn** (every wall-clock number, all configuration-dependent): per-call costs, all fit times, the
scaling grid, the derivative-timing table, and the per-node Laplace timings. The Laplace *ratio* argument
survives qualitatively (the covariance is a by-product of a Jacobian the fit already has, versus ~2(n-1)^2
evaluations) but its seconds must be re-measured.
