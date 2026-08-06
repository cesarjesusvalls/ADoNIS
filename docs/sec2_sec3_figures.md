# sec2 & sec3 figures — what they are, and how they connect to the sec1 banks

The paper keeps **two** §2/§3 figures (all others were pruned 2026-08-06):

| figure | script | one-line |
|---|---|---|
| `sec2_shrinkage_subsets` | `analysis/paper/sec2_fisher/make.py` (+ `configs/paper/sec2_subsets.yaml`) | Which knobs each **probe type** can constrain (Fisher / Gate I) |
| `sec3_gradients_shape` | `analysis/paper/sec3_gradients/make.py` | **Where** (which bins) each fittable knob's gradient information comes from |

**Both are two views of ONE object:** `output/altgen/multisample_carbon.npz` — a single Jacobian
`J[i,k] = ∂(dσ/dx)_i / ∂θ_k` over **28 knobs** (`physical_fit.PNAMES`/SPEC) and every analysis bin `i`.
sec2 gates on it (Fisher); sec3 shows its per-bin shape. Neither does a bank pass — they only read this npz.

## The chain: banks → per-sample Jacobians → multisample_carbon.npz → the two figures

**1. Per-sample bank→Jacobian npzs** (`output/altgen/*.npz`), each an *exact* autodiff Jacobian: one
`jax.jvp` per knob through `bank_reweight.weight_jit` (differentiating the **frozen** cascade — not finite
differences, not a surrogate). Same 28-knob SPEC each.

| npz | script | bank | measurement (NUISANCE signal) |
|---|---|---|---|
| `physfit_gate1` | `physical_fit.py` | `nu_T2K_C/merged` (`$ADONIS_EVENT_BANK`) | T2K CC0π-Np + CC1π⁺-Np STV |
| `physfit_minerva` | `minerva_fit.py` | `nu_MINERvA_C/merged` | MINERvA CC0π-Np STV |
| `physfit_minerva_ptpz` | `minerva_ptpz_fit.py` | `nu_MINERvA_C/merged` | MINERvA qelike μ pT/p∥ |
| `physfit_electron` | `electron_fit.py` | `beam_e_C_hv/merged` | (e,e') EM QE/RES ω |

**2. FSI-only beam Jacobians** — computed on the fly by `beams/beam_fisher.beam_jacobian(beam)` for
`pip`/`prot`/`neut`, from the π⁺/p/n–C beam banks (reaction, absorption / π-production cross sections).

**3. Stack** — `analysis/paper/sec3_gradients/build_multisample.py` (`build()`): loads each per-sample npz,
**row-slices** it to the observables that experiment actually reported (`NPZ_SAMPLES`, e.g. T2K kept to its
7 STV+μ vars, dropping ppi/cospi/multiplicities), appends the beam Jacobians, and writes
`multisample_carbon.npz` = `J, sigma, prior, pnames, dskeys, row0, F, V, shrink`. Fisher is additive
(`F = SᵀS`, `S_ik = σ_prior_k·J_ik/σ_i`), so any subset/stack of samples is just a row-slice — that is why
one npz answers both "which observable class measures a knob" and "which sample does".

## What each figure computes

- **`sec2_shrinkage_subsets`** — Gate I: with a prior on every knob, does the DATA determine it? Asimov
  posterior `V = (JᵀC⁻¹J + Π⁻¹)⁻¹`; a knob is **FIT** when marginalized `σ_post/σ_prior < 0.5`. The single
  panel groups samples into **columns** (config `by_probe`): `ν` (T2K+MINERvA), `e beam` (e,e'),
  `hadron beam` (π⁺,p,n), and `ALL` combined. Dark = tighter constraint; orange outline = FIT; rows grouped
  by physics block. (The pruned figures were per-sample / cumulative-ladder variants + a failure-mode
  scatter + a degeneracy eigen-spectrum — all on the same npz, none are paper figures.)
- **`sec3_gradients_shape`** — the per-bin pull `S_ik` normalized to each knob's own peak, for the Gate-I
  **fittable** subset (marginalized shrinkage < 0.5 on the combined fit). Signed diverging map (a knob
  raises/lowers a bin); rows = fittable knobs by physics block, columns = bins grouped by observable. Reads
  as a pair with sec2: *which* knobs are constrainable (sec2) and *where* their information lives (sec3).

## Connection to the sec1 samples/banks

- **Same weak banks + same NUISANCE signal defs/observables as sec1:** `nu_T2K_C` (sec1 fig07/08 vars:
  dpt,dat,pmu,cosmu / pn,dptt,daT), `nu_MINERvA_C` (fig09 vars: mnv_dat/pn/dpt + qelike ptmu/pzmu), and the
  π⁺/p/n–C beam banks (same family as fig03).
- **One deliberate difference — the electron sample:** sec2/sec3 use `beam_e_C_hv` (monochromatic hard-vertex
  e⁻ beam, E=2222 MeV, θ∈[14,17]°) for a clean EM QE/RES ω Jacobian, **not** sec1 fig0456's e4ν bank
  `beam_e_C_1159` (1.159 GeV). Same physics probe, different kinematic point chosen for the gradient.
- **The relationship:** sec1 = "does ADoNIS reproduce ACHILLES" (forward histograms). sec2/sec3 = the
  differentiable-MC payoff on those *same* samples — *differentiate* the bank reweight to get J, then read
  off which knobs the data constrains and where the gradient comes from.

## ⚠️ Regeneration (staleness)

`multisample_carbon.npz` and the per-sample `physfit_*.npz` were built **before** the 2026-08 high-stat bank
swap + the joint-amps2 (exact cross-term) fix, so the two kept figures are stale. To refresh correctly:
rebuild the per-sample Jacobians on the current banks (`physical_fit.py`, `minerva_fit.py`,
`minerva_ptpz_fit.py`, `electron_fit.py` — each a bank `jvp`), then
`python -m analysis.paper.sec3_gradients.build_multisample`, then
`python -m analysis.paper.sec2_fisher.make` and `python -m analysis.paper.sec3_gradients.make`.
