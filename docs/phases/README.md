# Phase logs

One markdown file per phase of `docs/REPRODUCTION_PLAN.md`, written so work is
**resumable**: each records what was verified, what was built, the gates that pass,
the exact commands, and the open issues / next steps. Read the relevant phase doc
before continuing that phase.

| File | Phase | State |
|---|---|---|
| [PHASE0.md](PHASE0.md) | 0 â inventory & oracle-mode verification | â done (inventory + mode matrix) |
| [PHASE_A3.md](PHASE_A3.md) | A3 â free-nucleon Ï(E_Î½) â Fig 2 | â vertex+oracle+muon+units done (Î½_e & Î½_Î¼ â¤3% vs ACHILLES, c_Î¼/c_e=0.998, physical nb); only the external ANL/BNL data overlay remains |
| [PHASE_A1.md](PHASE_A1.md) | A1 â EM 1Ï current (electron probe) | â done â EM current validated â¤3% vs ACHILLES (all 4 channels; neutron split fixed via isign=-1), closure exact |
| [PHASE_A2.md](PHASE_A2.md) | A2 â NC 1Ï current | â done â NC current (sinÂ²Î¸_W) validated â¤1.3% vs ACHILLES (all 4 channels), closure exact |
| [PHASE_B.md](PHASE_B.md) | B â QE vertex + inclusive (e,eâ²) | â B1 done â free-nucleon CCQE (Llewellyn-Smith) reproduces ACHILLES QE to â¤3% (absolute, no constant); inclusive fold (Fig 1) remains |
| [PHASE_D.md](PHASE_D.md) | D â FSI scaffold & differentiability proof (toy) | â toy cascade recovered + verified (forward-unbiased, 5-param differentiable); FSIModel port + full joint closure remain |
| [PHASE_E.md](PHASE_E.md) | E â meson-baryon scattering (DCC PWA) | â E1 ÏN Ï(W): Ïâºp Î(1232) peak reproduced (208 mb @ 1220 MeV); Ïâ»p/Î·N/KÎ channels + angular dÏ/dÎ© remain |
| [PHASE_G.md](PHASE_G.md) | G/H â cascade engine + both modes | ☑ cascade image built + RUNS; Fig c12_ar40 oracle generated (pi-12C Virtual+Propagating, pi-40Ar), all Delta-peaked; model-side FSIModel follows |

Legend: â todo Â· â partial Â· â done.

## Showstoppers (autonomous, this environment)
- **Standalone Ïânucleus cascade Ï oracle** (Fig c12_ar40, Phases G2/H2): the published image
  lacks the `achilles-cascade` binary and the host has no `cmake`/`gfortran`. **RESOLVED by
  building a cascade-enabled image inside Docker** (`docker/Dockerfile.cascade`, toolchain in
  the container) â `achilles:cascade` with the `achilles-cascade` binary. See `PHASE_G.md`.
  (Remaining: a segfault in the cascade interaction setup on the first build â the
  factory-registry/visibility issue, addressed by adding `-fno-visibility-inlines-hidden`.)
- **A3 ANL/BNL data overlay** (#6): the paper's Ï(E_Î½) points are the external Wilkinson-2014
  reanalysis (not in the `../nuisance` clone, which has dÏ/dQÂ²); `uproot` isn't installed.
  Soft-blocked â and not the model's job (the modelâACHILLES validation is done).
