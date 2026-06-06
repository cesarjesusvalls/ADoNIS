# Phase C — Observables, signal definitions & TKI (+ the FSI signature)

Plan ref: `docs/REPRODUCTION_PLAN.md` Phase C. The analysis layer on top of the EventRecord:
leptonic/hadronic observables, single-transverse kinematic imbalance (TKI), and signal
(topology) selections — all differentiable, so any distribution can be fit to data and any
FSI/production knob recovered from it.

## Done
**Observables** (`adonis/observables/kinematics.py`) — pure `EventRecord -> (N,)` functions,
all built from the lab 4-momenta so the model histograms identically to the oracle:
- leptonic: `enu`, `Q2`, `W`, `lepton_energy`, `lepton_costheta`;
- hadronic: `ppi_mag`, `ppi_costheta_lab`, `nucleon_mom`, `cos_theta_star`, `phi_star`;
- **TKI**: `delta_pT` (|p_T^lep + p_T^had|), `delta_phiT`, `delta_alphaT` — the
  single-transverse imbalances that isolate nuclear effects (Fermi motion + FSI).
A registry `OBSERVABLES` exposes them by name for batch histogramming.

**Signals** (`adonis/signal/base.py`) — boolean `select(event) -> (N,) mask`; `weight()`
multiplies the (differentiable) event weight by the mask, so predictions are signal-restricted
without breaking gradients. `AllEvents`, `HasPion(pid)`, `KinematicCut(obs, lo, hi)`, and the
FSI-topology pair **`CC1Pi`** (surviving pion, `pid_pi != 0`) / **`CC0Pi`** (absorbed,
`pid_pi == 0`). After the cascade, absorption marks pions `pid_pi=0`, so these partition the
sample into the CC1π and (FSI-fed) CC0π topologies.

## The FSI signature in TKI (ties C ↔ D)
`scripts/make_tki_figure.py` → `figures/tki_fsi_dpt_c12.png`: CC1π `delta_pT` on real
DCC-produced ¹²C events, **before vs after** the cascade `ToyCascadeFSI`. With no FSI the
imbalance reflects only Fermi motion (a peak at ~110 MeV, small tail); the cascade
(scatter + absorption) drags strength into a **high-δp_T tail** and converts CC1π → CC0π by
absorption — exactly the FSI signature T2K/MINERvA use:
- δp_T tail (>300 MeV): **1.7% (pre-FSI) → 21% (post-FSI)**;
- absorption-driven **CC0π fraction ~49%** at `(σ_sc, σ_abs) = (0.35, 0.22) fm⁻¹`.

## Gates (`tests/test_tki_fsi.py`)
- **topology partition** — `CC1Pi`/`CC0Pi` are mutually exclusive + exhaustive; FSI creates a
  sizable CC0π rate (0.2–0.8).
- **FSI broadens δp_T** — the post-FSI CC1π tail (>300 MeV) is >3× the pre-FSI tail.
- **differentiable post-FSI observable** — the CC1π mean δp_T is differentiable in σ_scatter,
  autodiff == FD (frozen proposal, rel < 1e-3).

(TKI/observable closures vs M_A also run inside `tests/test_ma_closure.py` /
`tests/test_grad_fs.py` on the bare production.)

## Remaining (optional)
- ☐ A concrete experiment `SignalDef` (e.g. lepton+pion-only TKI, dropping `p_N` from the
  hadronic system) to mirror a specific T2K/MINERvA selection for a data overlay.
- ☐ `FluxModel` beyond the monochromatic source (a real flux histogram) for flux-averaged
  distributions — the `Monochromatic` flux is in place; a histogram flux is a drop-in.
