# Phase F — Real Oset absorption

Plan ref: `docs/REPRODUCTION_PLAN.md` Phase F1. The Oset pion self-energy in nuclear
matter (the absorption + quasi-elastic in-medium widths) — the FSI absorption physics the
cascade's `PionAbsorptionOneStep` uses, with the **C_Q/C_A2/C_A3 coefficients as the
paper's tunable absorption knobs**.

## Done (F1)
`adonis/fsi/mb/oset.py`: faithful transcription of ACHILLES `OsetCrossSections.cc/.hh` —
the imaginary parts of the Δ self-energy as quadratics in x = T_π/m_π times a density power:

    abs_NN (x,ρ) = q(x;C_A2)·(ρ/ρ0)^β(x)
    abs_NNN(x,ρ) = max(q(x;C_A3),0)·(ρ/ρ0)^{2β(x)}      (clamped >=0 for T_π < ~50 MeV)
    qe    (x,ρ) = q(x;C_Q)·(ρ/ρ0)^α(x)
    q(x;a)=a0 x²+a1 x+a2 ,  α=q(x;C_ALPHA), β=q(x;C_BETA)

Coefficients copied verbatim from `OsetCrossSections.hh` (C_Q={-5.19,15.35,2.06},
C_A2={1.06,-6.64,22.66}, C_A3={-13.46,46.17,-20.34}, C_ALPHA, C_BETA, ImB0=0.035).

**Validated** (`tests/test_oset.py`, the plan's "oracle = the Oset parameterization values"):
- the total absorption self-energy **peaks at T_π ~200 MeV** (the Δ region, where pion
  absorption is known to peak); abs_NN dominates at low T_π (s-wave-like), abs_NNN turns on
  above ~85 MeV;
- the 3N piece is correctly **clamped to 0** at low T_π;
- **closure**: Im Σ_abs is exactly differentiable in the C_A2 (and C_Q) strength knobs
  (autodiff==FD to 1e-13) — the F-phase tunable handle.

## F2 — real Oset absorption cross section — DONE (un-blocked)
The earlier "soft-block" was wrong: the full ACHILLES **source** is local (`../Achilles`), so
`OsetCrossSections.cc::AbsCrossSection` was directly portable. `adonis/fsi/oset_xsec.py` is a
**line-by-line JAX port** of the exact absorption (s+p wave, the Δ-propagator + `pAbsorptionFactor`
/`sAbsorptionFactor` + the self-energy density powers) AND the 9-channel QE cross sections, in
absolute mb, with the verbatim constants from `OsetCrossSections.hh`. Differentiable in the
kinematics and the Oset coefficients. Gate `tests/test_oset_xsec.py`: σ_abs peaks ~42 mb at the
Δ, falls off, charge-exchange structure correct. This is the **real, untuned** absorption the
cascade (`cascade_real.py`) uses — no optical-relation approximation.
- The self-energy itself (the tunable FSI knobs C_Q/C_A2/C_A3) is done + exactly
  differentiable; it is what the cascade and the D-phase FSI consume. A future
  momentum-dependent FSI σ_abs(T_π) can read its Δ-peaked shape directly (cf. PHASE_D
  "Remaining": resonant FSI rates).
