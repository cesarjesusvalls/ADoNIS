# Sec 4 & Sec 5 — figure review catalogue

Draft figures for **Sec 4 (fitting & statistical interpretability)** and **Sec 5 (computational performance
& novel methodologies)**, all on branch `sec4-fitting`. PNGs live in `output/paper/` (gitignored); every one
regenerates from committed code. Engine: `analysis/paper/physfit/multisample.py` (nonlinear closure on the
16 Gate-I dials across T2K + MINERvA + (e,e′) + π/p/n beams). MLE = no prior; MAP = with the Gate-I prior.

---

## Sec 4 — fitting & statistical interpretability

| fig | file | code | shows | key numbers | status |
|---|---|---|---|---|---|
| **4.1 recovery + 1-D uncertainty** | `sec4_fig41_recovery.png` | `sec4_closure/fig_recovery.py` | MLE best-fit on every injected-truth star; Gaussian σ vs profile (Δχ²=1) interval per dial | unbiased; Gaussian≈profile except the weak cross-section dials | ✅ |
| **Eb-wall** (non-Gaussian headline) | `sec4_ebwall.png` | `sec4_closure/fig_ebwall.py` | E_b injected at 0.15 MeV near its wall: profile one-sided (walled), Gaussian σ leaks into unphysical E_b<0 | Gaussian [−0.10, 0.40]; profile [wall, 0.42] | ✅ **strong** |
| **4.2 pre/post distributions** | `sec4_fig42_dists.png` | `sec4_closure/fig_dists.py` | nominal (pre) vs best-fit (post) vs data for T2K + MINERvA δp_T (ν), (e,e′) ω_QE, π⁺–C σ_reac | fit tracks data across all probes | ✅ |
| **4.3a bulk coverage** | `sec4_fig43a_coverage_mle.png` | `sec4_closure/coverage_fig.py` | 80-toy MLE ensemble: pooled pull vs N(0,1), per-dial calibration, χ² goodness | pull **+0.01 ± 1.07**, KS p=0.50; χ²~χ²(354) | ✅ |
| **4.3b Eb-wall coverage** | `sec4_fig43b_ebcov.png` | `sec4_closure/fig_ebcov.py` | per-toy Gaussian vs profile intervals at the wall + coverage % | Gaussian 75% / profile 70% vs 68% | ⚠️ **PARKED / opt-out** — weak, stat-limited (40 toys), *over*-covers; panel (a) visual OK, panel (b) not trustworthy |
| **4.4 corner** | `sec4_fig44_corner.png` | `sec4_closure/corner_fig.py` | 16-dial lower-triangle: exact 68/95% contours (blue) vs Gaussian ellipse (orange) | degeneracies M_A_res×{kF, S_Δ, N_SRC}, s_NN_pn×f_NN_cex | ✅ |

**Supporting / diagnostic Sec-4 figures** (not the canonical set, but built along the way):
- `sec4_mle_closure.png` — the all-16 MLE recovery (predecessor of 4.1), `sec4_closure/mle_fig.py`.
- `sec4_closure_random16_asym.png` — asymmetric (MINOS) errors: profile min + Δχ²=1 vs GN symmetric σ, `asym_fig.py`.
- `sec4_prior_pull.png` — MLE vs MAP: the fit is unbiased; BFP≠truth is the prior pull (∝ shrinkage), `prior_pull_fig.py`.
- `sec4_traj_s10_convergence.png` — distance-to-truth vs LM/GN iteration, `convergence.py`.
- `sec4_closure_random16_profile.png` — per-dial exact profile vs GN parabola (1-D non-Gaussianity), `profile_fig.py`.

---

## Sec 5 — computational performance & novel methodologies

| fig | file | code | shows | key numbers | status |
|---|---|---|---|---|---|
| **5.1/5.3 scaling & advantage** | `sec5_fig_scaling.png` | `sec5_methods/fig_scaling.py` | (a) 120-pair corner Taylor vs grid vs profile, (b) fit Jacobian autodiff vs finite-diff, (c) derivative-tensor cost vs order m | corner **10 s / 1.6 hr / 45 hr**; jac **935 ms vs 4.0 s**; sweeps 16→3876 | ✅ |
| **5.2 autodiff non-Gaussian corner** | `sec4_closure_r16_noprior_corner_validate.png` | `sec4_closure/corner_taylor.py` | exact grid (black) vs 2nd-order-model Taylor from autodiff (orange) vs Gaussian (blue), 4 pairs | Taylor tracks exact to ~1–2σ | ✅ |
| **5.4 hybrid grid-bad × Taylor-good** | `sec5_fig_hybrid.png` | `sec5_methods/fig_hybrid.py` | (E_b, X) contour: exact vs pure-Taylor (misses wall) vs hybrid (grid E_b × Taylor) | hybrid nails kF; captures wall + residual Taylor error for degenerate M_A_res | ✅ |
| **5.5 differentiability-advantage table** | `sec5_fig_table.png` | `sec5_methods/fig_table.py` | per-task diff-engine vs finite-difference cost (gradient, Hessian, error prop, corner, m-th deriv, hybrid) | measured | ✅ |

---

## Statistics framing (Sec 4)

Two objects both called χ²: **data-space χ²** (goodness of fit) is χ²(ndf) even for a nonlinear model
(data errors are Gaussian) → robust; the **parameter posterior** is where non-Gaussianity lives → the
symmetric-σ pull assumes it's Gaussian. **Coverage = interval coverage**, and the correct interval for a
non-Gaussian likelihood is the **profile (Δχ²=1)** one, which equals the asymmetric-error object. So
coverage and non-Gaussian errors are one story: Gaussian σ covers only in the Gaussian regime; the profile
covers always. The Eb-wall figure is the clean boundary illustration (Gaussian interval goes unphysical).

## Open items
- **4.3b** parked (needs ~300 toys + Eb*≈0.05 to resolve, if pursued at all).
- Optional: a **threshold** non-Gaussian companion (`s_NN_inelastic` at low beam momentum) and a **curved
  degeneracy** pair to headline 4.4.
- Sec 5.4 hybrid: 3rd-order Taylor would tighten the degenerate-dial (M_A_res) residual.
- Regenerate any figure: `python -m analysis.paper.<module>` (see the code column).
