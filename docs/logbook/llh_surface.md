# LLH surface visualization — exact multi-parameter χ² vs the FD-Hessian-at-BFP

**Session:** Callum_B (`callum_b_llhsurface`). **Goal:** map the *exact* T2K CC0π log-likelihood (χ² with the
full covariance) over dense 2D/3D parameter grids using the fast differentiable event-bank reweight, and
contrast the true Δχ² contours against the Gaussian ellipse from `jax.hessian` at the best-fit point (BFP):
non-Gaussian tails, curved degeneracy valleys ("bananas"), rail effects. Blueprint = the CC0π T2K tune.

Evidence, not conclusions. Numbers are per-run; figures under `output/figures/llh_*`.

## Machinery

- **Model.** χ²(θ) is re-summed from the frozen 1M event bank. `bank_reweight.weight_jit(JB, θ, grids)` gives
  the EXACT per-event weight w(θ) (26 knobs, no Taylor). The CC0π-signal mask (`bank_plot.signal_cc0pi`:
  primary-pion-absorbed + T2K acceptance) and the per-event bin index are θ-independent (frozen kinematics),
  so a bin content is `segment_sum(w(θ)·mask, bin_idx)` → χ²(θ) is jit/grad/hessian-able in every knob.
- **Speed.** Slice the bank to the 340,046 CC0π-signal events (of 1,874,303; 318,692 QE + 21,354 RES) →
  one exact reweight is ~53 ms warm. A 61×61 grid ≈ 3.3 min/pair; a 25³ cube ≈ 14 min. No separability
  approximation — the exact `weight_jit` is evaluated at every grid point.
- **χ².** Absolute (primary; CLAUDE.md absolute stance) `χ² = rᵀ C⁻¹ r`, r = model−data, over the full T2K
  STV covariance; shape-profiled (closed-form global A) as a secondary view. **All of it lives in one module:
  `analysis/t2k/differentiability/llh_surface.py`** (CLI modes `--gates`/`--2d`/`--3d`/`--combo`/`--plot-only`);
  T2K data + CC1π/free-H assembly reused from `info_content` (single source of truth). (2026-07 consolidation
  of the six scratch `scripts/cc0pi_llh*.py` demonstrators into this module.)
- **Robust covariance.** `robust_cinv` = eigen-floored inverse (floor eigenvalues at `rtol·max(eig)`, an
  SVD/pinv with a relative cutoff), condition number reported — the same robust handling the 2D pcos
  covariance needed. STV blocks are well-conditioned: **cond(dpt)=2.98e3, cond(dat)=14.0, 0 eigs floored.**

## Constant-folding NaN (fixed)

`bank_weight` returns **NaN when JB is baked into a jit as a closed-over constant** — XLA constant-folds the
21.8M-row FSI gather (340k events × 64 nucleon-step cap) and produces value-dependent NaN. With JB/grids as
jit **operands** (`weight_jit(JB, θ, grids)`) the result is finite. Fix: the χ² helpers call `weight_jit`
(JB operand) and are **never** wrapped in an outer `jax.jit` that would recapture JB; speed comes from
`weight_jit`, and `jax.grad`/`jax.hessian` trace *through* it with JB still an operand.

## Validation gates (`llh_surface.py --gates`, 1M bank)

| gate | result |
|---|---|
| A. slicing bit-exactness: `Σ weight_jit(full)[mask]` vs `Σ weight_jit(sliced)` | rel **0.0e0** — PASS |
| B. nominal footprint: model_vec(nominal) vs bare-w0 forward (dpt / dat) | max per-bin 1.7e-2 / 2.3e-4; **aggregate 1.6e-5 / 1.2e-4** (the nominal SF Eb=1e-2 + FSI f_NN_cex=0.5 reweights the w0-forward omits) — informational |
| C. AD-Hessian vs central FD (M_A_qe × axial_strength @ (1.05,0.95)) | relmax **2.5e-7** — PASS |

Nominal joint χ²/ndf (dpt+dat, ndf=16): **abs 1.81, shape 1.61**. (Consistent with the ~1.89 CC0π tune baseline.)

## Note on the merged main (2026-07-04)

Fast-forwarded onto `origin/main` (info-content + M_A-split + consolidation commits). Consequences used here:
- **`M_A` split into `M_A_qe` + `M_A_res`** (per-channel axial dipole masses) — the hard-vertex pair is
  `M_A_qe × axial_strength` (both act on the QE amps² record → strongly anti-correlated).
- The other session flagged an **exact pion-FSI flat direction**: equal rescale of
  {sabs, s_piN_elastic, s_piN_cex, s_conv} is a per-event invariance of the pion kind-1 reweight. My FSI
  pair `sabs × s_piN_elastic` and the `fsi_trio` 3D cube sit inside this degenerate group.
- The CC0π+CC1π combination reuses info_content's dataset assembly (CC1π `signal_cc1pi_stv`, nb/CH units,
  **frozen nominal free-H offset** via `generate_H`).

## Results — 2D surfaces (CC0π dpt+dat, ng=61, absolute χ²)

`llh_surface.py --2d` → `output/figures/llh_surface_<pair>_dptdat.png`, npz in `/tmp/adonis_tune_runs/llh_*`.
Each figure: (0) true Δχ² heat + true contours (white, Δχ²=1/4/9) + Hessian ellipse (red dashed) at the BFP;
(1) profiled (min over partner) vs conditional (slice at BFP) vs Gaussian (σ from V₀₀) along p₀; (2) the
non-Gaussianity residual Δχ²_true − Δχ²_Gauss.

| pair | BFP | χ²/ndf | rail | non-Gaussianity: max\|dev\| (frac of 9) · RMS · area(true/Gauss)@1σ |
|---|---|---|---|---|
| `M_A_qe × axial_strength` | (1.236, 0.684) | 1.56 | no | **23.5 (2.61) · 5.6 · 1.23** |
| `sabs × s_piN_elastic`    | (1.254, 0.512) | 1.78 | no | **9.9 (1.10) · 2.8 · 1.67** |
| `qe_norm × res_norm`      | (0.856, 1.428) | 1.51 | no | **0.00 (0.00) · 0.00 · 1.00** |

Evidence:
- **`qe_norm × res_norm` is EXACTLY Gaussian** (residual at the 1e-12 numerical-noise floor; true contours and
  the Hessian ellipse are perfectly concentric; profiled≡conditional≡Gaussian). This is the expected control:
  a per-channel normalization is a *linear* scale on the model, so r=model−data is linear in (qe_norm,res_norm)
  → χ² is *exactly quadratic* → the Hessian is exact. Confirms the machinery reproduces the Gaussian limit when
  the model truly is linear. (QE norm tightly constrained ~0.82–0.90; RES norm loose — QE dominates CC0π rate.)
- **`M_A_qe × axial_strength` is a strong curved banana** (both act on the QE amps² record → corr ≈ −0.97,
  V=[[0.186,−0.115],[−0.115,0.076]]). The Hessian ellipse hugs the true valley only locally at the BFP; along
  the valley the true Δχ² stays low where the Gaussian ellipse predicts ≫9 (max residual 23.5 inside the true
  3σ region). Profiled Δχ²(M_A_qe) is nearly flat while the conditional is a tight parabola — the classic
  degeneracy signature.
- **`sabs × s_piN_elastic` is a broad soft valley** (whole grid spans only Δχ²≲4.5; true 1σ region is 1.67× the
  Gaussian-ellipse area). Consistent with the flagged exact pion-FSI flat direction — with s_piN_cex/s_conv held
  nominal the projection onto (sabs, s_piN_elastic) is soft but not exactly flat; sabs is nearly unconstrained
  when s_piN_elastic floats (profiled Δχ²(sabs) flat across 0.75–2.3).

## Results — 3D slices (CC0π dpt+dat, ng=25 cube, absolute χ²)

`llh_surface.py --3d` → `output/figures/llh_3d_<key>.png`. Each figure: the three profiled 2D projections
(min over the 3rd param) with the projected Hessian ellipse, plus the normalized 3×3-Hessian eigen-spectrum
(stiff vs flat directions) — this is the "beyond FD-Hessian-at-BFP" read: a *single* number (a determinant or
a marginal σ) hides that one eigen-direction is nearly flat.

**`M_A_qe × axial_strength × res_norm`** — BFP (0.911, 0.939, 1.475), interior (no rails), χ²/ndf 1.51,
cond(H)=194. Normalized-Hessian eigenvalues (small = flat):
- λ=0.0126 → −0.71·M_A_qe +0.70·axial (the QE-axial banana direction — the soft valley),
- λ=0.889 → +0.95·res_norm (res_norm nearly its own eigen-direction, moderately stiff),
- λ=2.099 → the coherent QE-rate direction (all same sign).

The BFP here differs from the 2D `maqe_axial` BFP (1.24, 0.68) because res_norm floats to 1.48 and absorbs
rate — a concrete example of a BFP moving when a dimension is added.

**`sabs × s_piN_elastic × s_piN_cex`** — BFP (2.40, 0.976, 1.920) with **sabs at the upper rail (2.40)**;
χ²/ndf 1.78; cond(H)=**8.7e3** (near-singular). Normalized-Hessian eigenvalues:
- λ=**3.4e-4** → **+0.80·sabs +0.49·s_piN_cex +0.34·s_piN_elastic** — a *same-sign* near-flat direction, i.e.
  the flagged EXACT pion-FSI invariance (equal rescale of {sabs, s_piN_elastic, s_piN_cex, s_conv}) restricted
  to these 3 of the 4 knobs. The eigenvalue is ~8000× smaller than the stiff one → an essentially flat plane;
  the fit slides along it until sabs rails. A Gaussian error from `det(H)`/`inv(H)` is meaningless here.
- λ=0.154 → −0.75·s_piN_elastic +0.65·s_piN_cex (the elastic↔cex trade), λ=2.846 → the coherent-rate direction.

Rail effect: at the sabs rail the fitted Hessian is not a valid curvature (the minimum is a boundary point);
the eigen-*direction* is still the robust readout — it is a property of the surface, not of the rail.



## Results — CC0π+CC1π combination (5 datasets, absolute χ²)

`llh_surface.py --combo` builds a combined absolute χ² over the FIVE T2K STV datasets — CC0π dpt, dat
(per-nucleon 1e-38) + CC1π+Np pN, dpTT, daT (nb/CH + **frozen nominal free-H offset**) — all re-summed from
the ONE bank, dataset assembly REUSED from `info_content.build_datasets` (single source of truth: acceptance,
units, free-H). The bank is sliced to the UNION of CC0π ∪ CC1π signal events (353,653 = 340,046 CC0π +
14,637 CC1π) so the exact reweight stays ~57 ms. Covariance conditions: CC1π pN 8.5e2, dpTT 39, daT 6.5.
Nominal combined χ²/ndf = 1.44 (ndf=28). It exposes the same `make_chi2` interface, so the 2D machinery is
reused unchanged (`--ng 51`, obs_tag `cc0cc1`).

**Does CC1π break a CC0π degeneracy?** (`output/figures/llh_cc0cc1_norm_degeneracy.png`, from the saved BFP/V):

| pair | dataset | BFP | σ (QE norm, RES norm) | non-Gauss |
|---|---|---|---|---|
| `qe_norm × res_norm` | CC0π only | (0.856, 1.428) | (0.067, **0.426**) | 0.00 |
| `qe_norm × res_norm` | CC0π+CC1π | (0.874, 0.903) | (0.066, **0.135**) | 0.00 |
| `M_A_qe × axial`     | CC0π only | (1.236, 0.684) | banana (corr −0.97) | 23.5 |
| `M_A_qe × axial`     | CC0π+CC1π | (1.247, 0.676) | banana (unchanged)  | 37.1 |

- **RES norm: σ 0.43 → 0.13 (3.2× tighter)** and the BFP moves 1.43 → 0.90 — CC1π+Np directly measures the
  RES channel and collapses the res_norm direction CC0π alone leaves loose (CC0π has only ~6% RES). QE norm is
  unchanged (0.067 → 0.066): CC0π already pins it, CC1π adds no QE-norm information. Both surfaces stay
  **exactly Gaussian** (linear norms). This is the headline degeneracy-breaking result.
- **`M_A_qe × axial` is unchanged by adding CC1π** (BFP 1.236,0.684 → 1.247,0.676; same banana). Expected:
  M_A_qe/axial are QE-only knobs, so the CC1π (RES) datasets are ≈constant in them — an orthogonal dataset
  cannot break a degeneracy it has no sensitivity to. (The non-Gauss number rises 23.5 → 37.1 only because the
  combined χ² has a deeper floor; the banana *shape* is identical.) `qe_res_norm` combined grid is zoomed to
  the tightened constraint (`COMB_RANGES`); the norm shape-χ² uses a single global A across mixed units and is
  a documented secondary only.

## Reproduce

```
export ADONIS_EVENT_BANK=/Users/homelab/Lab/Playground/projects/DIFFGEN/ADoNIS/output/event_bank
M=analysis/t2k/differentiability/llh_surface.py
python -u $M --gates                 # validation gates
python -u $M --2d    --ng 61         # 3 pairs, 2D (dpt+dat)
python -u $M --3d    --ng 25         # 2 cubes (ma_axial_resnorm, fsi_trio)
python -u $M --combo --ng 51         # combination (qe_res_norm, maqe_axial) + degeneracy overlay
# figures -> output/figures/llh_*   (gitignored, per-worktree);  npz -> /tmp/adonis_tune_runs/llh_*
```

## TL;DR

The exact bank reweight lets the *full* Δχ²(θ) surface be mapped, not just its curvature at one point. The
Gaussian/Hessian ellipse is **exact iff the model is linear in θ** (`qe_norm × res_norm`: residual 1e-12), and
**fails predictably when it is not**: hard-vertex knobs (`M_A_qe × axial`) make a curved banana the ellipse
only fits locally, and pion-FSI knobs (`sabs × s_piN_elastic`, `fsi_trio`) form a near-flat valley/plane
(3×3-Hessian eigenvalue 3.4e-4 along the flagged equal-rescale invariance) where a single Gaussian error is
meaningless. Adding the CC1π+Np channel tightens RES norm 3.2× but leaves the QE-axial banana untouched.

