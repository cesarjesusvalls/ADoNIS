# diffpi status

Differentiable ACHILLES single-pion production for gradient-based tuning of the
underlying physics parameters to data. See `STRATEGY.md` (parent dir) for the plan.

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

### Known, bounded approximations (deferred to the full hadron-tensor port)
The xsec assembly is the **angle-integrated diagonal bilinear** (Option B), not the
full `amp_dcc_sl.f` hadron tensor (Option A: Wigner-d / helicity / boosts / isospin).
Consequences, all noted in-code:
- per-partial-wave weights are (2J+1)+single-K, not the exact multipole normalization;
- transverse only (no σ_L); factorized leptonic flux (Hand for EM, flat-propagator for CC);
- no V-A interference in the weak current.
These cause the residual "blue vs orange" shape gap and the missing CC low-Q² turnover.
The diagonal assembly is the **regression oracle** for the full port (must collapse to
it when angle-integrated). Prerequisite for the port: reconcile the file's `idx` →
`zmtx(ixi1=1..8 = photon-pol × nucleon-helicity)` mapping in `read_amp`.

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
