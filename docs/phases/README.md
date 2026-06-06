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
| [PHASE_B.md](PHASE_B.md) | B — QE vertex + inclusive (e,e′) | ◐ B1 done — free-nucleon CCQE (Llewellyn-Smith) reproduces ACHILLES QE to ≤3% (absolute, no constant); inclusive fold (Fig 1) remains |
| [PHASE_D.md](PHASE_D.md) | D — FSI scaffold & differentiability proof (toy) | ◐ toy cascade recovered + verified (forward-unbiased, 5-param differentiable); FSIModel port + full joint closure remain |
| [PHASE_E.md](PHASE_E.md) | E — meson-baryon scattering (DCC PWA) | ◐ E1 πN σ(W): π⁺p Δ(1232) peak reproduced (208 mb @ 1220 MeV); π⁻p/ηN/KΛ channels + angular dσ/dΩ remain |
| [PHASE_G.md](PHASE_G.md) | G/H — cascade engine + both modes | ☑ cascade image built + RUNS; Fig c12_ar40 oracle generated (π-¹²C Virtual+Propagating, π-⁴⁰Ar), all Δ-peaked; model-side FSIModel follows |

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
