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

Legend: ☐ todo · ◐ partial · ☑ done.
