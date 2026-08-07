# Sec 4 binning decisions — DECIDED, PARKED (2026-08-07)

Decisions are final; **activation is deferred** until we're ready to regenerate the Gate-I chain.
The enabling machinery is already on `main` and is **inert** (every new field defaults to off).
The YAML activations live on branch **`sec4-binning`**.

## Why these changes exist

`adonis/analysis/sample.py:_binidx` **clips values into range before digitizing** — "overflow folded
into the edge bins ... the binning convention every driver uses". So any observable whose histogram
range is narrower than its kinematic reach gets an edge bin that is an *integral over everything
beyond*, drawn as if it were one narrow bin. Measured on the banks:

| observable | reach vs range | overflow |
|---|---|---|
| `ee_omega:omega` | ω to ~2222 MeV, range ends 950 | large visible spike |
| `t2k_cc0pi:pmu` | p_μ to 14 GeV/c, range ends 2 GeV/c | 1.35% of events = **5.77% of the cross section** |

This is **not wrong for the fit** (model and data are binned identically, χ² is valid) but it is wrong
for the figures, and it dilutes the dial sensitivity of those bins.

## The decisions

0. **THE FIX: drop overflow instead of folding it. Do not cut the sample.** (user, 2026-08-07)
   `_binidx` no longer clips; an out-of-range value simply enters **no bin**. The event stays
   selected, weighted and available to every *other* observable — only that one histogram omits it.
   This is strictly better than the alternatives considered:
   - no `mu_win`/`omega_win` surgery, so `dpt`/`dalphat` keep the exact NUISANCE selection
     (a shared `mu_win` cut would have silently truncated them — 5.77% of the cross section);
   - one change fixes every observable, present and future, with no per-observable config field.

   A histogram then integrates to the cross section **inside its range**, not the sample total.
   That is the honest reading of a binned differential distribution. Model and data go through the
   same function, so χ² is unaffected.

1. **ω — 45 → 24 bins**, `linspace: [50.0, 950.0, 25]` (24 × 37.5 MeV).
   QE peak and Δ shoulder both stay resolved at 24 bins (verified on 2.1M events).

2. **p_μ — uniform `[0, 3000]` × 100 MeV (30 bins).**
   3.53% of the cross section sits above 3 GeV; with no-clip it is simply outside the measurement.
   Verified on 1.3M events: the tail now falls monotonically through the last bin
   (last/previous = 0.914; it was a ~4.5× step).

3. **`cos_mu` → `th_mu_rad`**, 22 bins on a round 0.1 rad grid, top edge = `arccos(-0.6) = 2.2143`
   (= the `cos_mu` signal cut, so the histogram boundary IS the selection boundary):
   `[0.0, 0.1, ..., 2.1, 2.2143]`

   **Why θ not cos:** `dσ/dcosθ = (dσ/dθ)/sinθ`. As θ→0 the Jacobian diverges, so `dσ/dcos` is
   enhanced toward cosθ→1 and the last cos bin (= θ ∈ [0, 0.40]) shows a big step that reads as a
   binning bug. In θ the same distribution is smooth. Not a bug — a coordinate artifact.

   **The shoulder at θ ≈ 0.15–0.35 rad is PHYSICAL** and deliberately left unresolved (it is a
   property of the selection, not of any fitted dial). It is the proton requirement
   (`p_win: [450,1000]`, `cth: 0.4`): forward muon → low Q² → recoil proton below threshold.
   Measured proton-acceptance efficiency vs θ_μ: flat at **~11-13% below 0.35 rad**, rising to
   **>50% by 1.2 rad**; overall the proton cut keeps 33% (867k of 2.62M).
   Confirmed not a channel effect (QE 813k vs RES 54k, same shape).

   Do **not** bin finer than ~0.1 rad: below 0.05 rad the MC is noise-dominated (44 events in a
   0.01 rad slice with mean weight 8.1e-9, ~8× the local average).

## What blocks activation

`analysis/paper/physfit/multisample.py:349`

```python
assert got == want, f"sample composition != multisample_carbon: ..."
```

The engine asserts its observable list against `output/altgen/multisample_carbon.npz`, which records
`t2k_cc0pi:cos_mu` and the current bin counts. **sec2 and sec3 read the same npz.** So activating any
of the three changes the dskeys and aborts every sec4 driver until the Gate-I object is rebuilt.

## Checklist when we ARE ready

1. `git merge sec4-binning` (or cherry-pick `3f0f412`) — brings the no-clip `_binidx` **and** both YAMLs.
   Note `EleBeamSignalDef.omega_win` on main is then redundant; it stays as an inert capability
   (genuinely cutting the sample) but nothing uses it — consider removing to avoid two mechanisms.
2. Rebuild the Gate-I object: `analysis.paper.sec3_gradients.build_multisample` → `multisample_carbon.npz`
3. Re-run **sec2 + sec3** (they read that npz — their figures WILL move)
4. Re-run sec4: closure → profile → ebwall → coverage → corner
5. Update `figure_draft.py` captions — several are already stale independently of this
   (Fig 4.1 says "(MLE)", Fig 4.1b quotes an old injection, Fig 4.3 says 80 toys)

## Related open items (independent of the binning)

- **4.3 filename mismatch**: coverage writes `sec4_coverage.png`, `figure_draft.py` points at
  `sec4_fig43a_coverage_mle.png` (5 days stale) — the draft would silently use the old file.
- **4.4 corner** is still built from the 50k bank, not 250k like 4.1/4.1b/4.2.
- `t2k_cc0pi:cos_mu` is the only cos observable affected; no other sample config uses one.
