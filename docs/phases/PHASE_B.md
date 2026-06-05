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
  gate checks **model/ACHILLES ≈ 1 directly** (no bridging constant):
  **1.01–1.03 across 0.5–3 GeV, max 3.2%**.
- **Closure**: dσ/dM_A autodiff==FD (analytic, exact). Gates: `tests/test_ccqe.py`.
- *Note:* the consistent ~2–3% (model slightly high) is the **dipole** axial FF vs
  ACHILLES's default **z-expansion** (`axial_zexpansion` already in `form_factors.py`) and/or
  the spectral S(0,E) weight; swapping in the z-expansion FF is the obvious refinement.

## Remaining
- ☐ **B1b** Optional: z-expansion axial FF in the LS σ (close the ~3%); EM/NC elastic
  variants of the 1-nucleon current (reuse the Kelly FFs) for the inclusive fold.
- ☐ **B2 — EM inclusive dσ/dω = QE + 1π** on ¹²C, vs the JLab-config ACHILLES (E=2.222 GeV,
  θ=15.541°). Build the QE 1-nucleon hadron tensor W^{μν}_1N folded over the ¹²C spectral
  function (reuse `SpectralFunction`) + the EM 1π piece (A1), summed → dσ/dω. **→ Fig 1 (¹²C).**
- ☐ **B3 — ⁴⁰Ar spectral function** swap: point the fold at the `pke40{p,n}` tables (present
  in the image) → Ar inclusive. **→ Fig 1 (⁴⁰Ar).**

## Status
B1 (the QE vertex, free-nucleon CCQE) is done and validated absolutely vs ACHILLES. The
inclusive (e,e′) fold (B2/B3, Fig 1) is the remaining build — it composes the QE 1-nucleon
tensor (here) with the spectral fold and the A1 EM-1π piece.
