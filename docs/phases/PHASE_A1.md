# Phase A1 — EM single-pion current (electron probe)

Plan ref: `docs/REPRODUCTION_PLAN.md` Phase A1. Build the **electromagnetic** 1π
production current (e⁻ probe), gate it with a closure (grad wrt vector-FF knobs) and an
oracle (vs ACHILLES EM 1π). Prerequisite for **Fig 1** (inclusive (e,e′), Phase B2) and
the **e4ν** comparisons (Phase I1).

## What's already there (big head start)
- **`build_zmtx` already implements the EM mode** (`adonis/primary/dcc/assembly.py`,
  `mode=10`): it drops the axial current and the CC pion-pole, and applies the **EM
  isospin** rule (proton → `vec` block; neutron I=1/2 → isoscalar `isv` block; I=3/2 → `vec`).
  So the hadron-current assembly for EM is essentially done — A1 is mostly the *leptonic*
  side + the right *channel set*.
- **`lepton_tensor_em`** (`adonis/primary/dcc/lepton.py`) — written: one-photon-exchange
  QED tensor `L^{μν}=2(k^μk'^ν+k'^μk^ν−g^{μν}k·k')`, **symmetric only** (no V-A
  antisymmetric term → contracts only the symmetric/real part of W^{μν}).
- **Oracle mode runs** (Phase 0): `run_electron_*` configs; electron RES single-pion on a
  free nucleon works (same 1H/1N stationary-target trick as A3).

## The two physics pieces A1 must get right
1. **The 1/Q⁴ photon propagator → the total σ is forward-divergent.** Unlike the ~constant
   CC W-propagator (Q²≪M_W²), the EM cross section carries an explicit `e⁴/Q⁴` that
   strongly weights low Q². On a free nucleon with no angular cut this makes the **total σ
   formally divergent** as Q²→0 (forward scattering) — which is why ACHILLES's electron
   example configs (`run_electron_*`) all use `HardCuts: true` with
   `AngleTheta PIDs:11 range:[10,90]`. **Consequence for A1:** there is no clean "total
   σ(E_e)" to compare (an un-cut EM run never converges — confirmed: a free-proton EM run
   with `HardCuts:false` ran for minutes with no result). The EM comparison must be at a
   **fixed scattering angle / within an angular acceptance** — i.e. dσ/dΩdE′ — which is
   exactly what (e,e′) experiments and the paper's **Fig 1** (JLab, θ_e′=15.541°) measure.
   So A1's natural gate is dσ/dΩdE′ (or a σ-within-acceptance) at a fixed angle, not an
   inclusive total. The `e⁴/Q⁴` factor (Q²-dependent) must be in the EM weight regardless;
   it does NOT cancel into the calibration constant.
2. **EM isospin coupling.** `angular_kernel` currently hardcodes an **isovector** current
   `cbg(1.0, tcrz, …)`. The photon is isoscalar **+** isovector; `build_zmtx` routes the
   isoscalar piece via the `isv` block, but whether the existing `angular_kernel`
   (tcrz=0) reproduces the EM coupling for *both* parts must be **validated against the
   oracle** (the A3 method: candidate channels → match ACHILLES channel ratios). If the
   isoscalar CG needs `cbg(0,0,…)`, add an EM variant of `angular_kernel`.

## EM channels (4, vs 3 for CC)
e⁻ on a free nucleon, photon (Δq_charge=0):
- `e⁻ p → e⁻ p π⁰`   (proton target)
- `e⁻ p → e⁻ n π⁺`   (proton target)
- `e⁻ n → e⁻ n π⁰`   (neutron target)
- `e⁻ n → e⁻ p π⁻`   (neutron target)
(STATUS.md's Phase-2 "EM oracle, 4 channels, 122.8 nb" used the older diagonal `dcc_xsec`;
A1 builds the EM path in the full-tensor framework instead.)

## RESOLVED: the EM current works — the proxy was the problem, not the physics
Building the **real EM weight** (`lepton_tensor_em` × 1/Q⁴ × angle cut, `current="EM"`)
reproduces the EM physics: the proton channel ratio σ(e p→e p π⁰)/σ(e p→e n π⁺) =
**0.908** vs ACHILLES **0.914** (<1%). So `build_zmtx` mode=10's EM isospin and the
`EM_CHANNELS` were correct all along — the misleading thing was the crude `sum|W_T+W_L|`
proxy, which ignored the 1/Q⁴ weighting (it shifts each channel's Q² distribution
differently). EM closure (dσ/d pw_norm) is exact (autodiff==FD, 2e-14). The remaining
work is just the multi-energy oracle bridge test + figure (oracle scan generating).

## Build plan (oracle-anchored, like A3)
- ☑ **A1.0** `lepton_tensor_em` written.
- ☑ **A1.1** `EM_CHANNELS` (mode=10, tcrz=0, 4 channels) in `structure.py` — validated:
  proton π⁰/π⁺ = 0.908 vs ACHILLES 0.914. No `angular_kernel` change needed.
  - **(superseded) earlier dead-end with the structure-function proxy:**
  - **Finding (the crux): naive EM channel defs do NOT reproduce the EM proton ratio.**
    With `Channel(itiz=+1, tiz=+0.5, tpinz=+0.5, tpiz={0,1}, tcrz=0, mode=10)` for the two
    proton channels, the model's per-channel structure-function proxy gives
    **π⁰/π⁺ ≈ 1.21**, but ACHILLES has **325.3/355.9 ≈ 0.91** (π⁺ > π⁰). A ~30% flip → the
    EM isospin coupling is wrong as-is. `build_zmtx` mode=10 routes the isoscalar via the
    `isv` block, but `angular_kernel` hardcodes the **isovector** Clebsch `cbg(1.0,tcrz,…)`
    and never adds the isoscalar `cbg(0,0,…)` piece — so the photon's I=0+I=1 interference
    is missing. **Next:** port the EM isospin from the Fortran `interpolate_amp` EM branch
    (the isoscalar/isovector current decomposition) into an EM-specific `angular_kernel`
    (or channel coupling), and re-check against 0.91 (and the `1N` neutron channels).
    *(Caveat: the proxy ignores the 1/Q⁴ weighting + angle cut, which could shift the ratio
    somewhat; the full EM weight (A1.2) is the definitive test — but a 1.21-vs-0.91 flip is
    almost certainly a real isospin issue, not a proxy artifact.)*
  - **Tested one fix hypothesis — WRONG direction (reverted).** Fortran detail: the EM branch
    uses `zampv` (isovector) for I=3/2 and `zampv_is` (isoscalar) for I=1/2 waves
    (`amp_dcc_sl_module.f` 925/962). Hypothesis: route *all* EM I=1/2 (proton too) to the
    isoscalar `isv` block. Result: proxy ratio → **2.40** (worse; target 0.91). So it is NOT
    a simple block-swap; reverted (CC path verified unchanged). **Two lessons:** (1) EM I=1/2
    likely needs the isovector AND isoscalar pieces *each with their isospin Clebsch*
    (cbg(1,0,…) + the isoscalar coupling), combined — port the full Fortran EM current, don't
    guess a single block; (2) **drop the crude `sum|W_T+W_L|` proxy** — it ignores 1/Q⁴ and
    the angle cut and may mislead. Build the real EM weight first (A1.2), generate the EM
    oracle (`1H`+`1N`, angle-cut, ~60 s/run), and judge channel ratios from the actual σ.
- ☑ **A1.2** EM weight path: `current="EM"` in `sample_final_state` (lepton_tensor_em +
  `1/Q⁴` + angle range); `em_sigma_channels_at` (per-channel σ-in-acceptance, chunked).
- ☑ **A1.3** Closure: `em_dsigma_dpw_closure` (grad of EM σ wrt the P33 `pw_norm` vector-FF
  knob), autodiff==FD to 2e-14. Gate: `test_em_closure_dpw`.
- ☑ **A1.4** Oracle: `em_sigma_oracle` vs ACHILLES (electron on 1H+1N, `[10,90]` cut,
  5 energies 0.7–2.2 GeV, `data/oracle/freenucleon_em_sigma.csv`). **Validated set: the two
  proton channels + the neutron total**, single-constant bridge → **max rel 2.6%, mean 1.3%,
  c-spread 1.6%** (gate `test_em_sigma_oracle`; figure `scripts/make_a1_em_figure.py` →
  `figures/a1_em_sigma.png`). The proton π⁰/π⁺ ratio matches at every energy
  (0.865/0.866 … 0.948/0.947). **→ building block for Fig 1 / e4ν.**
- ☑ **A1.5 — the neutron π⁰/π⁻ split. FIXED.** The missing piece was the **`isign=-1`
  neutron-amplitude phase** (`amp_dcc_sl_module.f:644`, *"correct phase for neutron amp"*),
  applied to the `zampv_is` block for EM only — which our loader stores without it. Using
  `-isv` for the EM neutron I=1/2 block (`build_zmtx_batched` + `build_zmtx`) makes the
  neutron split match ACHILLES at every energy (nπ⁰/pπ⁻: 0.753/0.754 … 1.035/1.037), and the
  **full 4-channel bridge passes: max rel 2.1%, c-spread 1.4%**. CC is untouched (the change
  is in the `mode>=10, itiz==-1` branch only). Two earlier hypotheses had been ruled out
  (isoscalar CG `cbg(0,0,…)` — Fortran uses the isovector CG for all EM waves; and routing
  EM I=1/2→isv for both nucleons — breaks the proton); the real fix was the neutron phase.

## Status: A1 COMPLETE
The differentiable EM single-pion current reproduces ACHILLES for **all four channels** to
**≤3%** (single-constant bridge, σ-in-[10,90]°-acceptance, E_e 0.7–2.2 GeV), with an exact
closure (dσ/d pw_norm, autodiff==FD to 2e-14). Building block for **Fig 1** (inclusive
(e,e′), Phase B2) and **e4ν** (Phase I1) is ready. Gates: `test_em_closure_dpw`,
`test_em_proton_channel_ratio`, `test_em_sigma_oracle` (4 channels). Figure:
`figures/a1_em_sigma.png`.

## Validation result (A1 essentially complete)
The EM single-pion current reproduces ACHILLES to **≤3%** for the proton channels and the
neutron total, fully differentiable (EM closure autodiff==FD to 2e-14). The one open item
is the neutron π⁰/π⁻ split (A1.5) — a localized isospin refinement that does not affect the
proton channels (the e4ν / Fig-1-relevant ones) or the neutron inclusive rate.

## Anchor data (ACHILLES, e⁻ on 1H, E_e=1.5 GeV, `AngleTheta[10,90]`, nb-in-acceptance)
Confirmed the 1/Q⁴ story: **un-cut diverges (never converges); WITH the angle cut it
converges in ~60 s** at `Accuracy 5e-3`. Per-process σ (ACHILLES process order):
- `e⁻ p → e⁻ p π⁰` : **325.3 nb**
- `e⁻ p → e⁻ n π⁺` : **355.9 nb**
- `e⁻ n → e⁻ n π⁰` , `e⁻ n → e⁻ p π⁻` : 0 (hydrogen has no neutron — use `1N` for these)
- Total (in acceptance): **681.2 nb** (±0.2%)

Note the EM isospin structure differs from CC: here the π⁺ and π⁰ proton channels are
**comparable** (355.9 vs 325.3), whereas CC was strongly `p→pπ⁺`-dominated. This per-channel
split is the key thing A1.1's EM channel/coupling definitions must reproduce — validate the
candidate `EM_CHANNELS` against these ratios (and the `1N` neutron-channel run) the same way
A3 validated the CC channels. EM oracle generation is tractable: ~60 s/run with the angle cut.
