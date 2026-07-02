# Information content of a differentiable generator — per-bin/per-dataset Fisher decomposition

Branch `callum_a_infocontent`. Script: `scripts/info_content.py`; persisted run arrays:
`/tmp/adonis_tune_runs/info_content.npz` (figures re-render via `--plot-only`).
Bank: the main checkout's 1M event bank (`ADONIS_EVENT_BANK`), 1 874 303 events (1 000 000 QE + 874 303 RES).

## Setup

- 4 params, all nominal = 1.0, entering only via the exact `bank_reweight.bank_weight` reweight:
  `M_A` (QE+RES, cross-channel), `axial_strength` (QE-only), `res_axial_strength` (RES-only),
  `pion_pole` (RES-only).
- 5 histograms, 2 channels, all with the released covariances:
  T2K CC0π-Np STV `dpt`,`dat` (8+8 bins, ROOT release) and T2K CC1π⁺Np STV `pN`,`dpTT`,`daT`
  (4+5+3 bins, PRD 103 112009 text release, ×(1e33·13) → nb/unit per CH).
- CC1π free-H: generated once at nominal (`generate_H`, 4×50k, σ_acc = 5.59e-7 nb after cuts) and added as a
  **frozen** per-bin offset → the CC1π Jacobian is carbon-only. Limitation of this demonstrator, not the method.
- Per-bin Jacobian: `jax.jacfwd` of the per-event weight (one (N,4) pass, reused by every dataset — the bank
  is threaded as a function argument, never a jit constant).

## Bug found and fixed on the way: CC1π STV acceptance was missing the forward cuts

`bank_plot.signal_cc1pi_stv` applied only the momentum windows; the T2K tight signal (and the validated
`make_plots.block_cc1pi_stv` / `workflow.signal.select_signal` path, `SignalDef(cth=COS70)`) also requires
cos θ > cos 70° on μ, π⁺ **and** the leading proton, with the leading proton picked among window+angle
accepted protons. Evidence (pN, nb/MeV per CH):

| | bin1 | bin2 | bin3 | bin4 | nominal χ²/ndf |
|---|---|---|---|---|---|
| windows-only (bug, C-only)   | 3.60e-9 | 8.68e-9 | 2.77e-9 | 2.92e-10 | 24.6 |
| +cos70 (fixed), C + free-H   | 6.59e-9 | 4.57e-9 | 1.05e-9 | 0.93e-10 | 1.18 |
| T2K data                     | 4.65e-9 | 4.28e-9 | 1.26e-9 | 0.79e-10 | — |

Fixed at the source: `bank_plot.leading_proton_window(..., cth)` + `signal_cc1pi_stv` (now `_CTH = cos 70°`);
`scripts/info_content.py` threads those constants (no literals). Signal counts 36 181 → 14 637 (daT).
`bank_arrows.py` (only other consumer) inherits the fix.

## Validation ladder (all gates passed before trusting anything)

| gate | result |
|---|---|
| nominal identity `bank_weight(θ_nom)` vs production `weight_jit(nominal)` | max abs diff **1.49e-13** |
| autodiff vs central FD (ε=1e-3), per dataset | max rel err **3.5e-7** (all 5 datasets 2.4–3.5e-7) |
| pseudo-data closure: inject θ*=(1.15, 0.90, 1.20, 0.80), joint-fit from nominal | recovered (1.15001, 0.89999, 1.20002, 0.80266); max bias **2.7e-3** (in pion_pole, the near-flat direction; others ≤2e-5) |

## Nominal (fixed acceptance), absolute, per dataset

| dataset | nbin | n_sig | χ²/ndf |
|---|---|---|---|
| CC0π dpt  | 8 | 339 069 | 1.80 |
| CC0π dat  | 8 | 340 046 | 1.86 |
| CC1π pN   | 4 |  14 530 | 1.18 |
| CC1π dpTT | 5 |  14 422 | 1.08 |
| CC1π daT  | 3 |  14 637 | 0.28 |

(CC0π values match the tune.py-era χ²/ndf ≈ 1.9.)

## Fisher decomposition (at nominal)

- Fisher-content share `F^(d)_kk / Σ_d F^(d)_kk` (fig `info_content_fisher_breakdown.png`):
  - `axial_strength`: **100 % CC0π** (55 % dpt + 45 % dat; CC1π shares 0 %).
  - `res_axial_strength`: **87 % CC1π** (33+31+23 %) + 12 % CC0π (RES-in-CC0π via π absorption).
  - `M_A`: split across **all five** — 75 % CC0π + 24 % CC1π (the only cross-channel param).
  - `pion_pole`: total F_kk ~**4000× smaller** than M_A's; Cramér-Rao σ = **13.4** (vs 0.28/0.28/0.40 for
    M_A/g_A^QE/g_A^RES) — a near-flat direction of this dataset combination.
- Per-bin info-weighted gradients J_ik/σ_i (fig `info_content_gradient_heatmaps.png`): every param has
  nonzero gradient in every bin; peak bins differ (M_A/g_A^QE peak in dpt/dat bin 1; g_A^RES in CC1π pN
  bin 2 / dpTT bin 3, and in the CC0π dpt tail bins 7–8).
- Fisher correlations: (M_A, g_A^QE) = −0.97, (M_A, g_A^RES) = −0.87, (g_A^QE, g_A^RES) = +0.85.

## Joint 4-param fit to real T2K data (full covariances)

Absolute (primary, per the normalization stance; ndf = 28−4 = 24):

| | χ²/ndf |
|---|---|
| nominal | 1.68 |
| best fit | **1.47** |

| param | BFP ± Hessian σ |
|---|---|
| M_A | 1.269 ± 0.443 |
| axial_strength | 0.671 ± 0.262 |
| res_axial_strength | 0.609 ± 0.425 |
| pion_pole | 0.300 **(rail — Hessian error invalid at the [0.3,3] clip)** |

Per-dataset profiled-norm variant (shape-only contrast; ndf = 19): nominal 1.71 → BFP 1.55;
M_A = 0.966 ± 0.336, axial_strength = 1.221 ± 0.674, res_axial_strength = 1.72 (rail-adjacent, σ invalid),
pion_pole = 0.300 (rail). Profiled A_d at BFP: 0.71/0.72 (CC0π), 0.55/0.53/0.64 (CC1π).

pion_pole rails in **both** stances — consistent with its Fisher content (σ_CR = 13.4): the datasets carry
essentially no information on it, so the fit parks it wherever χ² is marginally lower.

## Figures

- `output/figures/info_content_gradient_heatmaps.png` — per-bin ∂(dσ/dx)_i/∂θ_k / σ_i, 5 panels, peak-bin stars.
- `output/figures/info_content_fisher_breakdown.png` — dataset×param Fisher share + stacked total F_kk (log).
- `output/figures/info_content_joint_fit.png` — data/nominal/absolute-BFP/profiled-BFP overlays (5 obs) +
  Fisher correlation matrix.

## Timing

Full run ≈ 14 min on the 1M bank (CPU): 76 s load+datasets, 2 s per-event Jacobian (N×4), ~3 min per
400-iteration Adam fit (three fits: closure, absolute, profiled), figures <1 s from the npz.
