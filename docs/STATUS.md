# diffpi status

Differentiable ACHILLES single-pion production for gradient-based tuning of the
underlying physics parameters to data. See `docs/STRATEGY.md` for the plan and
`docs/CONTAINER.md` for the ACHILLES oracle image / CI.

## Phase 1 — toy differentiable components (COMPLETE)
Validated score-function / reweighting machinery on toy analogues, then a full
integrated chain. Components A–E + B→C→G + A→(C+B+D)→G all close. Three hard-won
gotchas baked in: detach sampled continuous values before the shape score weight;
use the unbiased two-replica loss; validate gradients with the expected-histogram
Jacobian. (See git history; modules `kernel/chain/harness`, `component_*`, `integrate_*`.)

## Phase 2 — real DCC physics, differentiable (COMPLETE)
The single-pion **production** forward model on the real ACHILLES inputs, validated
against ACHILLES oracles, with exact/clean gradients.

| step | what | gate |
|---|---|---|
| DCC amplitudes | parse `dcc_EW.dat`, JAX bilinear interp + knobs (`dcc_loader`,`dcc`) | grads verified |
| EM oracle | ACHILLES e⁻ on ¹²C → clean single-pion reference (`oracle/`) | 122.8 nb, 4 channels |
| oracle parse | events → dσ/dQ², dσ/dW, pion spectra (`parse_hepmc`) | Δ(1232) peak |
| **xsec assembly** | angle-integrated diagonal bilinear σ(W,Q²;knobs) (`dcc_xsec`) | 7/7; Δ peak, **grad==FD exact** |
| spectral fold | σ folded over ¹²C spectral fn → nuclear dσ/dW (`spectral`,`dcc_fold`) | 4/4; FWHM 85→107=oracle |
| form factors | Kelly + axial z-exp/dipole; **M_A** as Q²-reweight knob (`form_factors`) | 11/11 |
| **CC oracle** | ACHILLES νe CC → weak single-pion; validates vec+**axial** path | 5/5; axial=59% of σ |
| det. closure | recover (pw_norm[p33], M_A) from pseudo-data, exact | 0.20→0.20, 1.20→1.20 |
| stoch. closure | recover same from **noisy MC fold** (two-replica loss) | 0.158/0.15, M_A 1.233/1.25 |

**Headline result:** the axial mass M_A is recovered by gradient descent through the
real DCC amplitude assembly — both noise-free and under realistic Monte-Carlo noise.

## Phase 2.5 — full hadron-tensor port (FAITHFUL; reproduces the oracle)
Replaced the diagonal bilinear with the faithful port of `amp_dcc_sl.f::amplitude()` +
`interpolate_amp` (Wigner-d / helicity / isospin / Legendre), contracted with the REAL
CC lepton tensor `L_{mu,nu} W^{mu,nu}` (V-A interference) and the πN 2-body phase space.

| step | what | gate |
|---|---|---|
| zmtx assembly | `interpolate_amp` port: 8-comp current, parity fill, current-cons, pion-pole (`hadron_assembly`) | 6c |
| angular kernel | isospin CG × spin-orbit CG × Legendre × azimuth on a quadrature grid; helicity→Cartesian zj_mu; W^{mu,nu} | Hermitian 1e-18; Δ peak; σ_L present; grad clean |
| structure fns | W_T, W_L on (Q2,W) grid, summed over CC isospin channels (`hadron_xsec`) | no NaN; Δ at 1210 |
| lepton tensor | boost to πN-CM (q∥z), `L_{mu,nu}` with V-A, contract (`lepton_tensor`) | q·L=0; L·W>0 (physical events) |
| **fold vs oracle** | L·W × k_pi(W)/W, spectral fold → dσ/dW vs νe-CC oracle | **relL2 4e-4, peak 1218=oracle** |

**Headline:** the full port reproduces the ACHILLES neutrino dσ/dW to ~0.04% (relative
L2), peak position exact. The decisive physics was the **πN 2-body phase space k_pi(W)/W**
(= `fnuc*k_pi/(16 pi^3 W)` in `amp_dcc_sl.f`): it rises from threshold and had been
dropped when building the tensor directly; with the real lepton tensor carrying the
flux, the hadronic factor is the BARE k_pi/W (not k_pi/(E_gamma W) — that double-counts
the EM flux). This — not any binding/W-scale shift — was the ~10 MeV peak offset.

### ACHILLES-exact kinematics (DONE) + the EW-isospin fix
`dcc_fold_full.py` matches ACHILLES (verified vs `currents_pi_dcc.f90` /
`res_spectral_model.f90`): off-shell E_struck=mqe−E_removal (mqe=938.919); amplitude at
the on-shell-rebalanced **Q2_adj**=|q⃗|²−(ω−E_removal−T_N)², observable binned in the true
leptonic Q2; hard cuts W∈[1076.957,2000], Q2∈[0,5 GeV²]; full angular coverage (θ→180°,
needed for the dσ/dQ² reach); m_π=138.04 (`achilles_const`). FResV/FResA are Q²-CONSTANT.

**EW-isospin fix** (the Δ-vs-second-resonance balance): `interpolate_amp` rotates the
weak I=1/2 vector current to the isovector **(V−IS)/2** (I=3/2 stays raw); `build_zmtx`
now does this. Channel fractions then match the oracle to <1% (p→pπ⁺ .664/.661, n→pπ⁰
.192/.193, n→nπ⁺ .144/.146). Result (`validate_fold_full.py`, vs 200k event-level oracle):
**dσ/dW relL2 6.5e-4, dσ/dQ² relL2 6.8e-4, peak 1218 exact, Q² reach 1.47 exact, M_A
gradient autodiff==finite-diff.** Residual: ~3% at the lowest-Q² bin (open).

## Phase 3 — fit REAL data (next)
The faithful, differentiable forward model is ready. Remaining for a NUISANCE fit:
neutrino flux fold, FSI cascade, lab-frame STV observable + cuts, χ² vs data+covariance.

## Phase 3 — fit REAL data (NOT STARTED)
To fit a NUISANCE release (MINERvA CC0π STV / T2K CC1π STV) we need the full event
chain the production model doesn't yet have:
1. **Neutrino flux fold** — replace the monochromatic beam with a real flux spectrum
   (`Achilles/flux/`, or the flux bundled in the T2K STV txt).
2. **FSI cascade** — propagate the produced pion (absorb/scatter): reuse the Phase-1
   toy cascade, or port the real Oset/MB amplitudes. CC0π *requires* pion absorption.
3. **Lab-frame final state** — full lepton+pion+nucleon kinematics, then the signal
   observable (single-transverse variables) + selection cuts.
4. **χ² vs data + covariance** — extract data+cov (uproot for MINERvA ROOT; plain-text
   for T2K), transcribe the NUISANCE signal definition/binning.

All inputs are local (ACHILLES + NUISANCE clones); zero external dependencies.
