# Argon CC1π ~7–9% high — pion absorption in the Ar cascade (vs ACHILLES)

## Evidence (overnight C+Ar 6h run, 2026-06-20; resonance RES + Pauli-fix + pool)
ADoNIS vs ACHILLES, proc-matched references (C `t2k_cc1pi_rich_ach_FSI_proc.npz`,
Ar `t2k_cc1pi_rich_ach_FSI_Ar_proc.npz`; QE=proc 200, RES=proc 401/402). ACH/ADO σ:

| channel | Carbon | Argon |
|---|---|---|
| CC0π QE-only (proc 200)      | 1.001 | 0.999 |
| CC0π QE+RES (all proc)       | 1.003 | 1.004 |
| CC1π RES-only (proc 401/402) | 0.980 | **0.932** |
| CC1π full (RES + QE-π⁺)      | 0.983 | **0.915** |

Seeds: C QE 10, C RES 13, Ar QE 10, Ar RES 7 (n_per_seed 30000). Shapes are FLAT (Ar CC1π per-obs
χ²/ndf ≤ 2.0, ratios scatter uniformly around 0.93) → a pure **normalization** offset, not a shape bug.

## Decomposition (absolute σ, Ar)
- QE→CC0π: ADoNIS 4.4154e-5 vs ACH 4.4125e-5 (1.001) — bang on.
- RES→CC0π (π absorbed) = combined−QE: ADoNIS 0.550e-5 vs ACH 0.575e-5 → **0.957 (ADoNIS ~4% low)**.
- RES→CC1π (π survives): ADoNIS 2.576e-6 vs ACH 2.401e-6 → **0.932 (ADoNIS ~7% high)**.

Consistent single cause: **ADoNIS absorbs slightly fewer pions in the Ar cascade than ACHILLES** →
strength moves from CC0π(absorbed) into CC1π(surviving). The CC0π total barely moves because it is
QE-dominated (absorbed is ~11% of CC0π); the CC1π channel is small so the same Δσ is a large fractional
effect.

## Full 3x2 matrix (ACH/ADO, all six channel x topology combinations)
|        | QE-only | RES-only | QE+RES |
|--------|---------|----------|--------|
| C  CC0π | 1.001 | 1.019 | 1.003 |
| C  CC1π | 1.077 | 0.980 | 0.983 |
| Ar CC0π | 0.999 | 1.045 | 1.004 |
| Ar CC1π | **0.581** | 0.932 | 0.915 |

Two separable pion-cascade effects, both growing C->Ar:
- **RES-absorbed** (CC0π RES-only): C 1.019, Ar 1.045 -> ADoNIS absorbs slightly LESS (low in absorbed).
- **Cascade-CREATED π⁺** (CC1π QE-only; NN-inelastic / Δ on the QE primary path): C 1.077, **Ar 0.581**
  -> ADoNIS OVER-creates FSI pions, dramatically on Ar.  Small channel (σ 1.3e-7 vs RES CC1π ~2.7e-6,
  ~5% of CC1π), so the Ar full CC1π 0.915 is mostly the RES-only 0.932 deficit, worsened by this excess.
Both push CC1π up.  Unified: ADoNIS's Ar pion cascade absorbs less AND creates more than ACHILLES.

## A-dependence
Carbon CC1π RES-only is 0.980 (~2% high); Argon is 0.932 (~7% high). The deficit in pion absorption
**grows with nucleus size** (C12 → Ar40). Points at the cascade pion-absorption rate not scaling
correctly with density/radius/path-length on the larger nucleus — NOT a normalization or selection
issue (CC0π and QE are correct on both).

## Status: PARKED (evidence only, not yet diagnosed)
The carbon pion-absorption residuals were closed earlier (docs/logbook/cascade_transport_residual.md:
abs isospin partition + charge-resolved scatter σ). Argon reveals a residual the carbon tuning did not
cover. Next step when picked up: read the cascade pion-absorption path (PionAbsorption / mean-free-path
vs local density) and compare ADoNIS vs ACHILLES on Ar specifically — likely the absorption σ or the
density/⟨path⟩ on the larger nucleus. Do NOT hot-fix; fix at the source if a real difference is found.

## Artifacts
- Banks: `t2k_cc{0,1}pi_engine_rich_{c6h,ar6h}.npz` (gitignored).
- Configs: `gen_{c6h,ar6h}_{qe,res}.yaml`, `ana_cc{0,1}pi_*_{c6h,ar6h}.yaml`.
- Figures: `paper_figures/cc{0,1}pi_*_{C6H,AR6H}.png`. Runner: `scripts/_run_overnight_c_ar.sh`.
