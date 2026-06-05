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
1. **The 1/Q⁴ photon propagator.** Unlike the ~constant CC W-propagator (Q²≪M_W²), the EM
   cross section carries an explicit `e⁴/Q⁴` that strongly weights low Q². This must be a
   factor in the EM weight (Q²-dependent — it does NOT cancel into the calibration
   constant). It also makes the σ integral sharply low-Q²-peaked → **ACHILLES EM runs are
   much slower to converge than CC** (observed: minutes vs seconds at the default
   `Accuracy 5e-3`; use a looser `Accuracy` (e.g. 5e-2) for oracle generation, or a Q²
   floor).
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

## Build plan (oracle-anchored, like A3)
- ☑ **A1.0** `lepton_tensor_em` written.
- ☐ **A1.1** Define `EM_CHANNELS` (mode=10, tcrz=0, the 4 channels) in `structure.py`;
  decide whether `angular_kernel` needs an EM isoscalar variant — **verify against the
  oracle channel ratios**.
- ☐ **A1.2** EM weight path: a `current="EM"` switch (GenConfig) selecting `lepton_tensor_em`
  + the `e⁴/Q⁴` factor + EM channels; a `sigma_enu`-style σ(E_e) scan.
- ☐ **A1.3** Closure: grad of σ wrt a **vector-FF / pw_norm** knob (the EM analog of A3's
  dσ/dM_A), autodiff==FD.
- ☐ **A1.4** Oracle: ACHILLES electron on 1H/1N (loose `Accuracy`), σ(E_e) per channel;
  the same single-constant bridge test as A3. **→ building block for Fig 1 / e4ν.**

## Open data point (anchor)
EM 1H (e⁻ p → e⁻ p π⁰ / e⁻ n π⁺) σ at E_e=1.5 GeV: _to be filled from the loose-Accuracy
probe_ (`_oracle_out/em_1H_fast.yml`). The CC-vs-EM σ ratio and the proton channel split
will anchor A1.1/A1.2.
