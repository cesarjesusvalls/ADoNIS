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

Legend: ☐ todo · ◐ partial · ☑ done.

## Confirmed showstoppers (autonomous, this environment)
- **Standalone π–nucleus cascade σ oracle** (Fig c12_ar40, Phases G2/H2): the image lacks
  the `achilles-cascade` binary (`ACHILLES_ENABLE_CASCADE_TEST` OFF) **and** this environment
  has no `cmake`/`gfortran` to build it. So the standalone π-A reaction/absorption cross
  sections can't be oracle-gated here. *Mitigation:* cascade-as-FSI inside event generation
  (`Cascade: Run: True`) still works → Phase-I FSI-on observables are NOT blocked; only the
  standalone π-A σ figures are. Needs a cascade-enabled image rebuild (packages:write) or a
  machine with the C++/Fortran toolchain.
- **A3 ANL/BNL data overlay** (#6): the paper's σ(E_ν) points are the external Wilkinson-2014
  reanalysis (not in the `../nuisance` clone, which has dσ/dQ²); `uproot` isn't installed.
  Soft-blocked — and not the model's job (the model↔ACHILLES validation is done).
