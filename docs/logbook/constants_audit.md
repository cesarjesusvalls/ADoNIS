# Physical-constants / mass-convention audit — ADoNIS vs ACHILLES (2026-07-20)

Exhaustive audit of **every** physical-constant use site in `adonis/`, triggered by the recurrence of
avg-vs-physical mass bugs across many prior sessions (commits `fd852af`, `2abde14`, `2b3aaa1`,
`1e52129`, `119d039`, `fd14c46`, `8848583`, `862ac1e`, `dba93e4`). Root cause of the recurrence:
prior audits were **symbol-scoped** (grep the mass symbol in formulas) and missed masses hidden inside
`jnp.clip(...)` floors / thresholds / on-shell energies, which read as numerical-safety bounds.

**Guardrail:** `tests/test_constants_vs_achilles.py` now (1) asserts every constant value == ACHILLES,
and (2) fails if any NEW `(2*M_N)**2` / `s/4 - M_N**2` avg-mass-at-threshold appears in the FSI cascade
outside the verified-correct whitelist. Run it in CI.

## Scope
- **513** constant-use sites across 98 files, over **43** distinct constants (ledger: 5-agent audit +
  mechanical grep). **94** high-risk "mass at clip/threshold/on-shell" sites individually adjudicated.
- Method: 5 parallel Sonnet auditors (nucleon cascade / NN xsec / meson cascade / primary xsec+EW /
  constants source), each diffing value AND use-context against the specific ACHILLES mirror function.

## Constant VALUES — verdict: essentially all correct
All of `mp, mn, mN, mpip, mpi0, mdelta, HBARC, HBARC2, alpha, GF(+unit), sin2w, cos2w, MZ, MW, GAMZ,
GAMW, Vud, Vus, ee, cw, sw, MB_TO_FM2, TO_NB, W_THR, MASS_PDG_*` match ACHILLES bit-for-bit or by
identical formula. Only sub-ppm truncations found (`HBARC` literal 3e-10; `HBARC_GEVFM` 3.5e-5 — fixed).
"Missing" constants (`mrho, mlambda, msigmam, msigma0, AMU, NAVOGADRO, GAMMA_E`) all correspond to
ACHILLES channels ADoNIS does not implement (Interference Model, hyperon production, optical potential,
electron-PDF) — not oversights.

## Use-context DIVERGENCES — FIXED 2026-07-20

| Site | Bug | Fix | Channel |
|---|---|---|---|
| `cascade_discrete.py:412` | elastic `sqrts` floor `(2*M_N)²` avg-mass (default-on) | physical per-pair floor `(m1+m2)²`, now the only path | pp elastic |
| `cascade_discrete.py:416` | NΔ `pcm` = `s/4 - M_N²` (avg, equal-mass approx) | exact Källén with physical per-pair masses | NΔ |
| `cascade_discrete.py:477` | NΔ `sqrts` floor `(2*M_N)²` avg-mass | physical per-pair floor | NΔ |
| `cascade_discrete.py:479` | Δ-mass clip bounds avg-mass + π⁰ | floor = neutron+π⁺, ceiling = √s − physical recoil mass | NΔ |
| `cascade_discrete.py:716-718` | pion absorption symmetric CM split, avg `M_N` both nucleons | asymmetric split, physical `mA/mB` (ACHILLES PionAbsorption.cc:175-181) | π absorption |
| `cascade_discrete.py:789-792` | πN↔ηN conversion recoil split avg `M_N` | physical recoil-baryon mass per `q_bary` | eta conversion |
| `anl_xsec.py:59 (_pcm2)` | piN elastic/CEX σ normalization uses **physical** π/p masses | ANL-code masses (138.5, 938.5), matching ACHILLES MesonBaryonAmplitudes.hh:110 | piN elastic — **was ~38% off at threshold** |
| `cascade_real.py:246` | legacy continuum scatter recoil avg-`M_N` default | physical charge-exchange-aware recoil mass | (legacy) |
| `nucleon_cascade.py:98,103,104,111` | dead scan path: avg mass at Fermi-sample/floor/σ/scatter | physical per-pair throughout | (dead) |
| `nn_inelastic.py:36` | `HBARC_GEVFM=0.19732` truncated (3.5e-5) | `197.3269804/1000` | NΔ Blatt-Weisskopf |

## VERIFIED-CORRECT — avg `M_N` here is RIGHT (do NOT "fix"; whitelisted in the guard)
- `cascade_discrete.py:55` formation zone `M_N² − dot4` — ACHILLES `Particle.cc:10` uses `Constant::mN`.
- `cascade_discrete.py:379-380` recap threshold `_e_phys − M_N` — ACHILLES `Cascade.cc:638` uses `Constant::mN`.
- `oset_xsec.py` uses `Constant::mN` throughout by design (OsetCrossSections.cc).
- Elastic `sig_el` per-pair mass (`:405-414`), elastic-CEX & NΔ-decay recoil masses, `nn_inelastic`
  per-charge Δ dict + MN_HEAVY/MPI_HEAVY convention, `k_F=cbrt(3π²ρ)·HBARC`, `MB_TO_FM2=0.1`.
- `thr_safe=clip(thr,1e-6)` in `nn_elastic_sigma`: differs from ACHILLES only within ~0.27 keV of
  threshold (zero-measure vs Fermi-motion spread); needed for autodiff. Not a bug.

## KNOWN-REMAINING (documented, not production bugs)
- `nn_elastic_sigma` (`nucleon_cascade.py:43-54`) lacks ACHILLES's `plab≥6 ∧ √s<10 GeV` intermediate
  branch → ~15-18% low at plab=6 (irrelevant to nuclear FSI, needs GeV-scale relative NN momentum).
- `nn_inelastic.py:114` NΔ table `s_hi=4.0` vs ACHILLES 6.0 → channel zeroed for 4<√s<6 GeV (high-E tail).
- `mb/nn_delta.py` (TEST-ONLY, not in production; production NΔ is `nn_inelastic.py`) uses a single Δ
  pole mass/width for all charges + avg-from-precise `M_N`. Gives false confidence if used to validate.
- Naming trap: `constants.py:59 M_N = mn` (neutron, a DCC alias) collides with the cascade's
  `ox.M_N` (average). Inert today; do not "clean up" the imports.

## Physics impact on Deviation 2 (nucleon reaction deficit)
The elastic floor fix (`:412`) raises near-threshold pp σ but the extra hits are low-momentum-transfer
and Pauli-blocked, so the **reacted** rate moves only 0.913→0.918 (identical-config replay, n=8000).
**The mass bugs are NOT the driver of the ~9% reacted deficit** — that remains open.
