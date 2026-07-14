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

---

# Session 2: M_A channel split + SF-knob validation + 17-param fit

## M_A split into M_A_qe / M_A_res

`M_A` was ONE shared knob (same dipole ratio applied to `qe_ma` and `res_ma` records). Split into two
independent knobs, each reweighting only its own channel's amps2 record — still exact, no bank change:
`full_knobs.nominal_knobs` (M_A_qe, M_A_res), `_hv_qe`/`_hv_res`, `bank_reweight.bank_weight`; consumers
updated (`closure_full` FIT registry now injects both; `bank_taylor` default). At nominal both = 1.0 →
identical weights (identity gate re-verified in the full17 run below). The original 4-param demo (`--set
axial4`) now uses M_A_qe.

## SF-knob validation (`scripts/sf_knob_checks.py`, fig `sf_knob_checks.png`)

The SF reweight is `w = sf_norm · tail(|p|) · S(p/kF, E−Eb)/S(p,E)` on a cubic-B-spline(p)×linear(E)
interpolant; upstream sampler uses 4-pt Lagrange(p)×linear(E) on the same table. Carbon SF table: 40×80
nodes, p∈[10,790], E∈[2.5,397.5] MeV.

| # | check | result |
|---|---|---|
| 1 | grid uniformity (B-spline assumption) | exactly uniform: Δp=20, ΔE=5 MeV (rel spread 0) |
| 2 | node exactness of prefiltered spline | max rel err 3.3e-11 at S>0 nodes; ≤1.6e-30 at S=0 nodes |
| 3a | S(p,E) spline vs upstream @ 1.87M event points | rms 2.6e-2, pointwise; outliers at the S≈0 edge |
| 3b | reweight RATIO spline vs upstream @ kF=1.1 | median 8e-4; mean w 1.31570 vs 1.31577; **weighted impact 3.0e-3**; max 209 confined to S₀<1e-4·max (2.6% of events) |
| 4 | nominal identity | Eb=0: w≡1 exactly. Eb=ε anchor: mean bias −5.7e-5, 30/1.87M events at w=0 |
| 5 | sf_norm | linear exactly: per-event \|dw/dnorm − w/norm\| ≤ 1.1e-16; AD/FD = 1.0000000000 |
| 6 | src_tail | matches 1+(s−1)σ((p−300)/80) to 6e-5; per-event grad = σ·w_nom to 3e-5; AD/FD = 1.0000000000 |
| 7 | kF_sf gradient | AD correct: per-event AD-vs-FD disagrees only for **24/1.87M events** at the S≈0 support edge (per-event \|J\| up to 1032; FD non-convergent there: sum AD/FD 0.9975→0.865 non-monotone in eps). Excluding them: AD/FD = 0.9986 @eps=1e-5. Physics sign: d⟨\|p\|⟩_w/dkF = +166 MeV/unit (stretch, as expected) |
| 8 | Eb_shift | one-sided branch AD/FD = 0.9955–0.9995 @ Eb = ε/2/5/15 MeV; S-clamped (w=0) fractions 0%/3.6%/22%/60%; w(Eb<0) ≡ w(0) exactly (clamp) |

Notes: (i) the spline-vs-Lagrange interpolant difference is the *documented* differentiability divergence
(sf_reweight.py docstring) — its physical impact is 3e-3 on the reweighted cross section at a 10% kF
deformation and 0 at nominal (ratio ≡ 1 both ways); (ii) the 22%/60% zero-weight fractions at Eb = 5/15 MeV
mean large Eb shifts KILL a large part of the sample (events pushed off the SF support) — the Eb fit
direction is one-sided and increasingly non-linear.

## full17 gradient stage (17 params: M_A_qe, M_A_res, 4 SF, 11 FSI; 28 bins)

Checkpoint: `/tmp/adonis_tune_runs/info_content_full17_grad.npz` (gradient/Fisher persisted before the fits).

- Nominal identity after the M_A split: **1.29e-14** (bit-exact preserved).
- Per-knob per-bin AD-vs-FD (eps=1e-3): **16 of 17 knobs pass at ≤ 6e-6**
  (M_A_qe 3.4e-7, M_A_res 3.5e-7, kF_sf 5.9e-6, sf_norm 3e-9, src_tail 2e-9, all 11 FSI 2e-8–1.2e-6).
  The single failure is **Eb_shift = 3.3e-2** — eps-scan diagnostic below.
- **Fisher eigen-spectrum** (normalized F̃ = D^-1/2 F D^-1/2; condition number 4.9e19):
  - λ = −9.2e-17 (**exact zero**): `+0.77·sabs +0.47·s_piN_cex +0.44·s_piN_elastic` — an exactly flat
    pion-FSI combination (per-event structural test below).
  - λ = 3.2e-4: `+0.64·sf_norm −0.50·M_A_qe −0.41·src_tail −0.25·M_A_res` — normalization vs form-factor.
  - λ = 1.1e-3: `−0.51·s_NN_el[pp] −0.44·sf_norm −0.41·s_NN_el[pn] +0.35·s_piN_cex`.
  - Stiffest: λ = 8.2 (`s_NN_el[pn] + f_NN_cex − src_tail + s_piN_cex` mix), λ = 3.5, λ = 2.9.
  - 28 bins vs 17 params: the spectrum spans 20 decades — most of the parameter space is measured only
    through a handful of stiff combinations. Cramér-Rao uses pinv (measurable subspace only).
- Ops note: the 17-tangent `jacfwd`/`jax.hessian` OOM'd this 16 GB machine (tangents on every (N,64)
  FSI-record intermediate ≈ 16 GB) — replaced by per-knob `jvp` / per-row Hessian (peak ≈ one forward pass).
  Machine shared with the altgen session's fits (7.4 GB peaks) during these runs.

## full17 fits (LM on the jvp Jacobian; npz `/tmp/adonis_tune_runs/info_content_full17.npz`)

Fit engine: reverse-mode (`value_and_grad`) is ~7 s/eval on this graph vs ~0.3 s forward (gather-heavy
FSI/SF VJPs) — Adam@600 iters ≈ 1.5–2 h/fit. Replaced by box-clipped Levenberg–Marquardt on the
forward-mode per-bin Jacobian (17 jvps + 17×17 solve per step): closure converged in 8 steps; all
3 fits + GN Hessians in ~2 h wall total. Errors are Gauss–Newton (`V = pinv(JᵀC⁻¹J)`); pinv because the
Fisher is singular; rails flagged.

- **Pseudo-data closure (17 params)**: injected θ* recovered to ~1e-7 in 14/17 params with final χ²=1.7e-8;
  residual error is confined to the pion-FSI knobs (sabs −0.28, s_piN_cex −0.26, s_conv −0.19,
  s_piN_elastic −0.17) and is **98.1% aligned with the Fisher null eigenvector** — in knob space
  ≈ (+0.5,+0.5,+0.5,+0.5) on (sabs, s_piN_elastic, s_piN_cex, s_conv): a COMMON rescaling of all pion
  interaction cross sections is unmeasurable (to first order) by these five distributions; χ² returns to
  its minimum anywhere on that manifold.
- **Absolute joint fit** (ndf=11): χ²/ndf 3.66 → **2.50**. Constrained: kF_sf = 0.929 ± 0.144,
  f_NN_cex = 0.40 ± 1.12, M_A_qe = 0.94 ± 1.87, M_A_res = 0.90 ± 1.48 (only their stiff combinations are
  measured). 7/17 params at clip rails with GN σ 13–235 (pion knobs) — the degenerate subspace parks at
  rails without moving χ².
- **Profiled-norm variant** (ndf=6): χ²/ndf 5.41 → 3.15, A_d = 1.98/2.01 (CC0π), 0.63/0.64/0.72 (CC1π);
  shape params run to extremes (M_A_qe → 0.49, Eb → 27 MeV) compensated by A_d ≈ 2 — with 17 free shape
  knobs the profiled norm absorbs so much that the shape/norm decomposition is no longer meaningful
  (evidence for the blueprint's absolute-first stance).
- M_A split content: **M_A_qe = 100 % CC0π** (53 dpt + 47 dat); **M_A_res = 92 % CC1π** (35 pN + 32 dpTT +
  25 daT) — the per-channel axial masses are informed by disjoint datasets. kF_sf carries the largest
  single-param information (F_kk ≈ 3.5e3, 63 % CC0π dpt); σ_conv and σ_NNinel[pp/nn] carry essentially none.
- Figures: `info_content_{gradient_heatmaps,fisher_breakdown,joint_fit}_full17.png`.

## Post-fit diagnostics (`scripts/fd_gate_diagnose.py`)

1. **Eb_shift gate failure was an FD artifact — AD is correct.** Per-bin worst AD-vs-FD rel err vs the FD
   bracket: 3.33e-2 @ eps=1e-3 → 2.95e-4 @ 1e-4 → **8.7e-6 @ 1e-5** (events with per-event AD≠FD:
   600 → 58 → 16). With the faithful LINEAR-in-E interpolant, w(Eb) is piecewise-linear in Eb; central FD
   across a kink (E−Eb crossing an E-node) averages two slopes while AD returns the exact local one. FD
   converges to AD as eps shrinks below the kink spacing → gate passes at 8.7e-6.
2. **The pion-FSI flat direction is a PER-EVENT structural identity, not a data limitation.**
   |J·v| ≤ 1.3e-21 per event (per-bin ≤ 8e-17 vs |J| scales up to 1.65) for
   v ≈ (+0.5,+0.5,+0.5,+0.5)/2 on (sabs, s_piN_elastic, s_piN_cex, s_conv). Code cause
   (`adonis/fsi/cascade_discrete.py:fsi_pion_reweight`): each pion HIT is reweighted by s_real·D0/D with
   D = s_abs·sa + s_el·ss_el + s_cex·ss_cex + s_conv·si — under a common rescale s of all four, D = s·D0
   and the factor is exactly 1 at ANY s. The pion record reweights the branch split conditional on a hit,
   but NOT the hit/no-hit probability (no survival factor). The nucleon reweight DOES carry the total-σ
   response (exp(−a/g)/exp(−a) hit factor + (1−pk)/(1−p0) no-hit factor).

   > **ANSWERED (2026-07-14, main session): COVERAGE GAP, not walk design.** In-walk gate — scale ALL
   > FOUR pion σ by a common s = 1.2 *inside the walk* (patched `oset_xsec.abs_cross_section`,
   > `cascade_mb.jax_channel_sigmas_resolved`, `jax_conversion_sigma`; same generator, seeds, config),
   > and compare with what the reweight predicts for that same s. 5275 live RES events on C:
   >
   > | | mean π interactions/event | absorbed hits | branch frac |
   > |---|---|---|---|
   > | nominal walk | 0.6673 | 1161 | 0.3298 |
   > | **reweight** @ s=1.2 | **0.6673** (w ≡ 1, max\|w−1\| = 8.9e−16) | 1161 | — |
   > | **in-walk** σ×1.2 | **0.7448 (+11.6%)** | 1285 (+10.7%) | 0.3271 |
   >
   > The walk's pion interaction probability is `prob = exp(−π·perp²/σ_tot)`, `σ_tot = sa+ss+si`
   > (`cascade_discrete._pion_step`) — it DOES respond to a σ scale; the reweight cannot express that.
   > The branch fraction is unchanged (0.3298 → 0.3271), confirming the mechanism: the common factor
   > cancels in `sa/σ_tot`, so 100% of the missing response is the hit/no-hit probability.
   >
   > ⇒ **`sabs`/`s_piN_elastic`/`s_piN_cex`/`s_conv` are BRANCH-ONLY knobs today**: they redistribute
   > channels among interactions that already happened and cannot change how many happen. Their Gate-I
   > sensitivity is *understated*, and the flat direction is a record artifact, not physics. A π⁺–C
   > transparency sample cannot constrain the σ_π magnitudes until this is fixed.
   > **Fix** = mirror the nucleon record: log every pion CANDIDATE (not just realized hits) with its
   > `a = π·perp²/σ_tot`; reweight gains `g = D/D0`, hit factor `(pk/p0)·(s_real·D0/D)`, no-hit factor
   > `(1−pk)/(1−p0)`. Needs a record-schema change + bank regen (~21 h).

   Until then, only 3 of the
   4 pion knobs are independent in any fit, exactly.

## Code consolidation (2026-07-04)

All session code moved into `analysis/t2k/differentiability/` (see its `README.md` for usage):
`scripts/info_content.py` → `info_content.py`; `scripts/{sf_knob_checks,fd_gate_diagnose}.py` →
`validate.py` (`--sf` / `--fisher`). Module cleanup in the same commit: `bank_arrows_{incl,obs}` merged
into `bank_arrows` (modes `cc0pi|cc1pi [stv]|incl`); `grad_{2d,all}` merged into `grad_arrows`
(`--2d`/`--hi-stats`); `closure_full` merged into `closure --full`; knob enumeration moved to
`full_knobs.knob_specs` (single source; `grad_arrows._specs` kept as alias); stale Taylor-era consumers
deleted (`bank_taylor`, `bank_showcase`, `bank_validate`, `bank_plot.hist_gradient/hist_diag_curv` — all
read `D1/D2/D3` fields the full-record bank no longer stores). Earlier `scripts/...` paths in this logbook
refer to the pre-consolidation locations (git history preserves them).
