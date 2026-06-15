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

## S6 — ACHILLES Ar reference (BLOCKED: local emulation crash)
- Ran `_oracle_out/run_T2K_Ar_virt.yml` (T2K flux + 40Ar.yml, NEvents 500k) via
  `docker run --platform linux/amd64 -v _oracle_out:/out <image> /out/...yml` (documented invocation).
- Result: SIGSEGV during Nucleus setup (right after "parsing nuclear name '40Ar'" / DecayHandler
  rescaling), before any event generation. exit 255.
- ISOLATION: the SAME image segfaults at the SAME setup point for CARBON too (run_T2K_virt.yml,
  NEvents=80) -> NOT Ar-specific.  All Ar inputs verified present in the image (40Ar.yml, rho_Ar_p/n,
  RMF configs, pke40, flux).  Image unchanged (created 2026-06-05, digest 0c36e0d8...).
- But local hepmc from 2026-06-06 exist (T2K_CH_virt.hepmc 3.2 GB, ar40_*.hepmc) -> ACHILLES ran
  locally then.  => host-side Docker/Rosetta emulation regression in the last ~9 days, not the image.
- seccomp=unconfined + -m 8g + --shm-size=2g did NOT help.
- Image: ENTRYPOINT ./bin/achilles, WORKDIR /achilles, CMD run.yml (bundled 50k 12C verified config).

### CORRECTION — the "host/Rosetta regression" conclusion was WRONG (premature).
- The image's BUNDLED run.yml (documented `docker run <image>`: e- @1108 MeV, QESpectral+RES, 12C,
  no cascade, NEvents=50000) runs CLEANLY end-to-end on this host (RES xsec ~120 nb, QE ~95 nb,
  "Finished optimization").  => container + emulation are FINE.
- My earlier carbon "crash" was a BAD TEST: NEvents=80 is far too small for the Vegas optimization.
- Minimal-Ar test (bundled config, ONLY 12C->40Ar + pke12->pke40): runs CLEANLY -- RES on 40Ar
  xsec ~400 nb, QESpectral QE on 40Ar ~303 nb, "Finished optimization".  => ACHILLES v0.3.0 DOES
  neutrino/lepton-on-Ar; the Ar nucleus + RMF configs + split density + pke40 all load fine.
- So the T2K-Ar SIGSEGV is NOT Ar and NOT the host -- it is a config element in my full T2K card.
  Suspects (my card vs the working bundled one): QE via FortranModel(QE_Spectral_Func) [bundled uses
  C++ QESpectral], Cascade.Run=True, nu_mu+T2K-flux beam.  RES FortranModel already works on Ar.
- Single-variable test running: run_T2K_Ar_virt2.yml = T2K-Ar with ONLY QE FortranModel->QESpectral
  (nu beam + cascade kept).  <result pending>
- LESSON: read the container docs / run the verified bundled config FIRST; don't conclude from a
  hand-tweaked card.  [[dont-declare-conclusions-prematurely]]

### ROOT CAUSE & FIX (the cascade crash was MY invocation, not the host/image/Ar)
- I was running the PUBLISHED amd64 image `ghcr.io/cesarjesusvalls/achilles:oracle` WITH
  `--platform linux/amd64` -> Rosetta EMULATION.  The cascade interaction factory setup SIGSEGVs
  under emulation (no-cascade paths survive, which masked it).
- The WORKING method = LOCALLY-BUILT NATIVE arm64 images, run WITHOUT `--platform`.  NOT all of them
  run the full neutrino+cascade generator without crashing:
  - `achilles:fullcascade` -> CONFIRMED works end-to-end for neutrino + FortranModel QE+RES + cascade
    (carbon smoke: "Event Run Concluded - Success!", 4000 ev hepmc; 12C QE xsec 4.29e-5 == target).
  - `achilles:scatrec` / `achilles:cascade*` -> CRASH at setup on the full neutrino+cascade card (both
    C and Ar); they are for the cascade-only `achilles-cascade` (pion+nucleus) binary.
- FSI run command (native, the recipe -> now in docs/CONTAINER.md):
  `docker run --rm -v "$PWD/_oracle_out":/out --entrypoint /achilles/bin/achilles achilles:fullcascade /out/<card>.yml`
  (NO --platform).  Sporadic SIGSEGV-on-exit (139) possible; partial hepmc stays valid -> batch seeds.
- Documented: docs/CONTAINER.md (canonical), memory achilles-fsi-native-image.md.
- The no-FSI Ar reference (amd64 emulated, no cascade) also works; native is preferred.

## Bit-exact gate
`scripts/_gate_carbon_bitexact.py` diffs a regenerated 2-seed carbon bank (gen_basecarb.yaml) vs the
golden snapshot (`*_basegold.npz`). Must be BIT-EXACT after each of S1–S4.
- S2+S3 gate: <result pending>
