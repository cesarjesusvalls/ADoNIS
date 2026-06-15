# Argon (and general-nucleus) support

**Goal.** Generalize ADoNIS from carbon-only to any registered nucleus, with Argon-40 as the proof.
User-requested explicitly ("make any material available"), then reproduce the T2K CC0π/CC1π figures
on Ar — same flux/selection, different nucleus (acknowledged unphysical for T2K; the point is to
exercise the machinery and compare ADoNIS-vs-ACHILLES on Ar).

**Architecture.** Every per-nucleus input is the single source of truth in
`adonis/workflow/materials.py::REGISTRY` (NuclearTarget), threaded → config → generators + cascade.
Nothing nucleus-specific is hardcoded downstream.

## Inputs present (verified)
| nucleus | density (p / n) | configs | spectral (n / p) | status |
|---|---|---|---|---|
| C-12 | c12_density.txt (both) | QMC_configs.out.gz (A=12) | pke12{n,p}_tot.data | runnable (production) |
| Ar-40 | rho_Ar_p.txt / rho_Ar_n.txt | AR40_configs_RMF_achilles.out.gz (A=40) | pke40{n,p}_tot.data | wiring |
| O-16 | rho_O_{p,n}_*.txt | 16O_*_configs.out.gz | **none (no pke16)** | NOT registered (no spectral) |

Density files copied into `ADoNIS/data/nuclear/` (rho_Ar_p.txt, rho_Ar_n.txt). Spectral + configs
resolve relative to `Achilles/` (already present there).

## ACHILLES reference behavior (Explore over Achilles/src, cited)
- Nucleus.cc:36-80 — ALWAYS reads SEPARATE proton & neutron densities (for C both files identical).
- Nucleus.cc:49-51 — radius = first r where rho_proton < 1e-6 fm^-3.
- Nucleus.cc:212-238 — PER-SPECIES local Fermi momentum k_F^s = cbrt(3π²ρ_s)·ℏc (ρ_s = p or n by PID).
- Configuration.cc:19-65 — config header [A Nconfigs maxWgt minWgt]; A read from header; per config
  A×"isospin x y z" + weight line + blank line. QMC and RMF share this format byte-for-byte.
- SpectralFunction.cc:8-38 — header [ne np] then np momentum blocks; pke40 == pke12 format, larger grid.
- Run card: data/default/40Ar.yml (split densities + RMF configs); NuclearModel SpectralP/N → pke40{p,n}.

## Staged plan (carbon bit-exact gate after S1–S4)
- S1 ✓ registry: NuclearTarget carries density_p/density_n/spectral_n/p/configs/A/Z; added Ar, kept C/H.
- S2 ✓ density loader per-species: `_load_density(name_p, name_n)` → (rgrid, rho_p, rho_n, radius);
  cascade k_F per nucleon isospin; total density = rho_p+rho_n (was 2·rho_p). N=Z → identical.
- S3 ✓ config loader reads A from header (dropped hardcoded A=12); QMC/RMF share the parser.
- S4 — spectral threading: qe/res accept SpectralFunction paths; rebuild RES proc table + importance
  sampler from chosen SFs (today they are import-time carbon globals).
- S5 — generate.py wiring: resolve_targets(material) feeds spectral/density/config/A through.
- S6 — ACHILLES Ar reference via docker (T2K flux + 40Ar.yml).
- S7 — Ar gen/ana configs → the cc0π/cc1π figures, ADoNIS-vs-ACHILLES.

## Noted approximation to validate (CLAUDE.md: discuss divergences)
- `cascade_discrete.kf_pi` (pion-absorption product-A Fermi momentum at the pion vertex): outgoing-
  nucleon species is channel-dependent; kept on proton density (carbon-exact). For Ar this is a
  per-species approximation in the absorption Pauli block — flag for validation vs ACHILLES Ar.

## Bit-exact gate
`scripts/_gate_carbon_bitexact.py` diffs a regenerated 2-seed carbon bank (gen_basecarb.yaml) vs the
golden snapshot (`*_basegold.npz`). Must be BIT-EXACT after each of S1–S4.
- S2+S3 gate: <result pending>
