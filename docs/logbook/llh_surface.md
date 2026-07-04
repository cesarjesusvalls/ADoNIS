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
  STV covariance; shape-profiled (closed-form global A) as a secondary view. Files: `scripts/cc0pi_llh_lib.py`
  (χ² + robust C⁻¹), `scripts/cc0pi_llh_surface.py` (2D + Hessian), `scripts/cc0pi_llh_3d.py` (3D cube),
  `scripts/cc0pi_llh_gates.py` (validation).
- **Robust covariance.** `robust_cinv` = eigen-floored inverse (floor eigenvalues at `rtol·max(eig)`, an
  SVD/pinv with a relative cutoff), condition number reported — the same robust handling the 2D pcos
  covariance needed. STV blocks are well-conditioned: **cond(dpt)=2.98e3, cond(dat)=14.0, 0 eigs floored.**

## Constant-folding NaN (fixed)

`bank_weight` returns **NaN when JB is baked into a jit as a closed-over constant** — XLA constant-folds the
21.8M-row FSI gather (340k events × 64 nucleon-step cap) and produces value-dependent NaN. With JB/grids as
jit **operands** (`weight_jit(JB, θ, grids)`) the result is finite. Fix: the χ² helpers call `weight_jit`
(JB operand) and are **never** wrapped in an outer `jax.jit` that would recapture JB; speed comes from
`weight_jit`, and `jax.grad`/`jax.hessian` trace *through* it with JB still an operand.

## Validation gates (`scripts/cc0pi_llh_gates.py`, 1M bank)

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

`scripts/cc0pi_llh_surface.py` → `output/figures/llh_surface_<pair>_dptdat.png`, npz in `/tmp/adonis_llh/`.
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

## Results — 3D slices  (TODO)

## Results — CC0π+CC1π combination  (TODO)
