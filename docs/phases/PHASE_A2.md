# Phase A2 — NC single-pion current (neutral-current neutrino)

Plan ref: `docs/REPRODUCTION_PLAN.md` Phase A2. The neutral-current 1π current
(ν → ν via Z), gated by a closure (grad wrt M_A) and an oracle (vs ACHILLES NC 1π).
Prerequisite for **MicroBooNE NC π⁰** (Phase I4).

## What it reuses / what's new
- **Leptonic side = CC**: the NC lepton is a massless neutrino with the same V−A
  structure, so `lepton_tensor_cc` is used unchanged (no 1/Q⁴, no angle cut — the Z
  propagator is ~constant, like the W). NC runs converge as fast as CC.
- **New: the NC hadronic current** carries the **sin²θ_W weak mixing**
  (`amp_dcc_sl_module.f:288–294, 1004–1050`):
  - isovector vector × `vfac = 1 − 2 sw2`  (sw2 = 0.2312) for all waves;
  - isoscalar vector × `vvfac(itiz)` added for I=1/2  (`vvfac(+1) = −2sw2` proton,
    `vvfac(−1) = +2sw2` neutron);
  - the axial current kept (no pion pole — that is CC-only, `mode>0`).
  Implemented as a `mode <= -1` branch in `build_zmtx_batched` / `build_zmtx`:
  `src = i32·(vfac·vec) + (1−i32)·(vfac·½(vec−isv) + vvfac·½(vec+isv))`.

## Channels (4, same final states as EM, mode = −1, tcrz = 0)
`NC_CHANNELS` in `structure.py`: ν p→ν p π⁰, ν p→ν n π⁺, ν n→ν n π⁰, ν n→ν p π⁻.

## Status
- ☑ **Mode verified**: NC config runs (Phase-0 open item closed) — `Leptons:[12,[12]]`
  on a free nucleon; per-process σ printed; 4 channels (neutron channels vanish on 1H).
- ☑ **Couplings validated**: the NC proton ratio σ(ν p→ν p π⁰)/σ(ν p→ν n π⁺) =
  **1.678** vs ACHILLES **1.670** (<0.5%) — the sin²θ_W vector couplings are correct.
  (Note the NC proton split is π⁰-dominated, unlike CC's p→pπ⁺ dominance or EM's ~even.)
- ☑ **Closure**: d(total NC σ)/dM_A autodiff==FD (NC keeps the axial). Gate
  `test_nc_closure_dMA`.
- ☑ **Oracle**: `nc_sigma_oracle` — single-constant **4-channel** bridge vs ACHILLES (ν on
  1H+1N, 5 energies 0.7–2.2 GeV, `data/oracle/freenucleon_nc_sigma.csv`): **max rel 1.3%,
  mean 0.6%, c-spread 0.7%** — passed on the first try. Gate `test_nc_sigma_oracle`; figure
  `scripts/make_a2_nc_figure.py` → `figures/a2_nc_sigma.png`.

## Status: A2 COMPLETE
The differentiable NC single-pion current reproduces ACHILLES for **all four channels** to
**≤1.3%** (even tighter than EM's 2.1%), closure dσ/dM_A exact. Notably the neutron π⁰/π⁻
split is correct **without** any extra phase (NC keeps `isign=+1`; only EM needed the
neutron-amplitude phase flip). The NC axial taken as the CC axial block (isospin CG with
tcrz=0) is evidently correct — no axial-driven residual at the ≤1.3% level. Prereq for
MicroBooNE NC π⁰ (Phase I4) ready.
