# Phase logs

One markdown file per phase of `docs/REPRODUCTION_PLAN.md`, written so work is
**resumable**: each records what was verified, what was built, the gates that pass,
the exact commands, and the open issues / next steps. Read the relevant phase doc
before continuing that phase.

| File | Phase | State |
|---|---|---|
| [PHASE0.md](PHASE0.md) | 0 — inventory & oracle-mode verification | ☑ done (inventory + mode matrix) |
| [PHASE_A3.md](PHASE_A3.md) | A3 — free-nucleon σ(E_ν) → Fig 2 | ◐ vertex+oracle+muon+units done (ν_e & ν_μ ≤3% vs ACHILLES, c_μ/c_e=0.998, physical nb); only the external ANL/BNL data overlay remains |
| [PHASE_A1.md](PHASE_A1.md) | A1 — EM 1π current (electron probe) | ☑ done — EM current validated ≤3% vs ACHILLES (all 4 channels; neutron split fixed via isign=-1), closure exact |
| [PHASE_A2.md](PHASE_A2.md) | A2 — NC 1π current | ☑ done — NC current (sin²θ_W) validated ≤1.3% vs ACHILLES (all 4 channels), closure exact |
| [PHASE_B.md](PHASE_B.md) | B — QE vertex + inclusive (e,e′) | ☑ B1 CCQE <0.5% abs; B2/B3 inclusive (e,e′) QE+1π (Fig 1, ¹²C+⁴⁰Ar); validated vs ACHILLES (e,e′) oracles w/ ratio+χ²: 1π χ²/ndf~15 (binding fix), QE χ²/ndf~9 (v_L/v_T Rosenbluth fix, peak 208 vs 192) |
| [PHASE_C.md](PHASE_C.md) | C — observables, signals & TKI | ☑ leptonic/hadronic/TKI observables + CC1π/CC0π topology signals; FSI distorts δp_T (tail 2%→21%, CC0π 49%), post-FSI observable differentiable |
| [PHASE_D.md](PHASE_D.md) | D — FSI scaffold & differentiability proof (toy) | ☑ `ToyCascadeFSI` ports the cascade to `FSIModel.apply(EventRecord)`; closure ~9e-7; joint M_A+σ_sc+σ_abs recovery (1.150,0.349,0.218 vs 1.15,0.35,0.22); D2 Oset-shaped momentum-dependent σ_abs (peaks at Δ) |
| [PHASE_E.md](PHASE_E.md) | E — meson-baryon scattering (DCC PWA) | ☑ πN sector: σ(W) all 3 charge channels (9.3:2.2:1 Δ ratio), angular dσ/dΩ (1+3cos²θ at Δ); ηN/KΛ/KΣ soft-blocked on missing tables |
| [PHASE_G.md](PHASE_G.md) | G/H — cascade engine + both modes | ☑ cascade image built + RUNS; Fig c12_ar40 oracle generated (π-¹²C Virtual+Propagating, π-⁴⁰Ar), all Δ-peaked; model-side FSIModel follows |
| [PHASE_F.md](PHASE_F.md) | F — Oset pion absorption | ☑ F1 done — Oset self-energy transcribed (C_Q/C_A2/C_A3 knobs), peaks at the Δ, exactly differentiable; abs cross-section fold remains |
| [PHASE_I.md](PHASE_I.md) | I — exclusive e/ν comparison | ☑ matched-kinematics e vs ν_e RES oracles; axial signature (ν more Δ-peaked in W, less forward in cos θ*); ADoNIS reproduces both, χ²/ndf W 1.5–2.4, cos θ* 4.5–7.3 |

Legend: ☐ todo · ◐ partial · ☑ done.

## Showstoppers (autonomous, this environment)
- **Standalone π–nucleus cascade σ oracle** (Fig c12_ar40, G2/H2): the published image lacked
  `achilles-cascade` and the host has no `cmake`/`gfortran`. **FULLY RESOLVED** — built a
  cascade-enabled image inside Docker (`docker/Dockerfile.cascade`), debugged the
  interaction-registry segfault (force-link `AchillesCascadeInteractions --no-as-needed` +
  `Plugin::Manager`), and generated the Fig-c12_ar40 oracle (π-¹²C both modes + π-⁴⁰Ar, all
  Δ-peaked). No longer a showstopper. See `PHASE_G.md`.
- **A3 ANL/BNL data overlay** (#6): the paper's σ(E_ν) points are the external Wilkinson-2014
  reanalysis (not in the `../nuisance` clone, which has dσ/dQ²); `uproot` isn't installed.
  Soft-blocked — and not the model's job (the model↔ACHILLES validation is done).
