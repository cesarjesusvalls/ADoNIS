# Phase B — QE channel + inclusive (e,e′) → Fig 1

Plan ref: `docs/REPRODUCTION_PLAN.md` Phase B. B1 QE 1-nucleon vertex + SF fold;
B2 EM inclusive dσ/dω = QE + 1π (Fig 1, ¹²C); B3 ⁴⁰Ar SF swap (Fig 1, ⁴⁰Ar).

## B1 — free-nucleon CCQE σ(E_ν) — DONE
`adonis/primary/qe/llewellyn_smith.py`: the analytic **Llewellyn-Smith** CCQE
dσ/dQ² for ν_μ n → μ⁻ p, with the **Kelly vector form factors** (F1, F2; new in
`form_factors.py`) and a dipole axial FF (F_A; F_P from PCAC). Integrated over the
kinematic Q² range → σ(E_ν).

- **Physical absolute units from first principles** (prefactor G_F²cos²θ_c — *no*
  calibration): σ rises from threshold and plateaus at **~1.05 ×10⁻³⁸ cm²** above 1 GeV,
  the known free-nucleon CCQE value.
- **Oracle (absolute)**: vs ACHILLES `QE_Spectral_Func` on a stationary neutron (ν_μ on
  `1N`), `data/oracle/freenucleon_ccqe_sigma.csv`. Since both sides are absolute nb, the
  gate checks **model/ACHILLES ≈ 1 directly** (no bridging constant). With ACHILLES's
  default **z-expansion** axial FF (`ff="zexp"`): **0.997–1.002 across 0.5–3 GeV, max 0.3%**.
  (With the **dipole** FF the model is 1.01–1.03, max 3.2% — the dipole is kept as the
  M_A-tunable closure handle; the z-exp matches ACHILLES's FF choice.)
- **Closure**: dσ/dM_A autodiff==FD (analytic, exact; dipole FF). Gates: `tests/test_ccqe.py`.

## Remaining
- ☐ **B1b** Optional: z-expansion axial FF in the LS σ (close the ~3%); EM/NC elastic
  variants of the 1-nucleon current (reuse the Kelly FFs) for the inclusive fold.
- ☑ **B2 — EM inclusive dσ/dω = QE + 1π** — DONE (Fig-1 structure) on ¹²C, vs the JLab-config ACHILLES (E=2.222 GeV,
  θ=15.541°). Build the QE 1-nucleon hadron tensor W^{μν}_1N folded over the ¹²C spectral
  function (reuse `SpectralFunction`) + the EM 1π piece (A1), summed → dσ/dω. **→ Fig 1 (¹²C).**
- ☐ **B3 — ⁴⁰Ar spectral function** swap: point the fold at the `pke40{p,n}` tables (present
  in the image) → Ar inclusive. **→ Fig 1 (⁴⁰Ar).**

## Status
B1 (the QE vertex, free-nucleon CCQE) is done and validated absolutely vs ACHILLES. The
inclusive (e,e′) fold (B2/B3, Fig 1) is the remaining build — it composes the QE 1-nucleon
tensor (here) with the spectral fold and the A1 EM-1π piece.

## B2 update — QE inclusive response (PWIA) DONE
`adonis/nuclear/qe_inclusive.py`: the plane-wave impulse-approximation QE (e,e') response, folding the single-nucleon elastic response (Kelly FFs) over the 12C spectral function S(p,E). At the JLab kinematics (E=2222 MeV, theta=15.541 deg) it gives a QE peak at omega~227 MeV (the relativistic quasi-free peak sqrt(q^2+M^2)-M+E_b ~207, + response skew) with a **Fermi-motion width FWHM~159 MeV** (expected k_F q/M ~147). Gate: tests/test_qe_inclusive.py. Remaining: add the 1pi inclusive piece (from A1, integrated over the pion) for the full QE+1pi dsigma/domega, and overlay the JLab-config ACHILLES run -> Fig 1.

## B2 complete — QE + 1pi inclusive (Fig 1 structure)
`adonis/nuclear/inclusive_1pi.py`: the 1pi/Delta contribution to dsigma/domega, folding the EM transverse/longitudinal structure functions W_T,W_L (EM_CHANNELS hadron tensor) over S(p,E) at the lepton (omega,q). The Delta bump sits at omega~459 MeV (expected (m_D^2-M^2+Q^2)/2M ~491), well above the QE peak (~223). The total QE+1pi (figures/inclusive_ee_c12.png) reproduces the two-peak structure of the paper's Fig 1. test_inclusive_two_peak_structure.
