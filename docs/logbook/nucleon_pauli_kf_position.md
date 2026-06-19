# Nucleon-elastic Pauli k_F evaluated at the wrong position (slow-proton over-interaction)

## Finding (single-pass primary-fate loop, carbon, vs ACHILLES `achilles:fatedump`)
Tracking each PRIMARY nucleon to its first interaction (matched to ACHILLES's "consume at first scatter"
definition) and comparing the **INTERACTED fraction vs initial |p|**, p/n disaggregated, high stats
(ADoNIS 115k live protons; ACHILLES 50k, 48k protons), with stat errors:

ADoNIS **over-interacted low-momentum protons** vs ACHILLES (Δ/σ): [75,125) 3.5σ, **[125,175) 5.3σ,
[175,250) 5.1σ**, [250,400) 1.6σ, [400,700) 0.2σ.  (Capture itself agreed <1σ; the 2σ seen at 20k was
under-powered — 50k made it a clean 5σ.)

## Root cause (confirmed)
The NN-elastic Pauli block requires BOTH outgoing nucleons' |p| to exceed the LOCAL k_F.  ADoNIS
evaluated the **leading** outgoing nucleon's k_F at the **struck background nucleon's position**
(`kf_lead` from `_rho_species(|npos[ar,j]|)`), not the leading's own position.  ACHILLES
`PauliBlocking(paOut)` uses k_F at the **outgoing's own position** (`paOut.Position()` = the leading's
position).

For large-sigma scatters the cylinder impact parameter is large (`sqrt(sigma/pi)`; np sigma ~170 mb
near threshold -> reach ~2.3 fm), so the struck nucleon sits up to ~the impact parameter (~2 fm)
transverse from the leading, at a different (often lower) density -> lower k_F -> the leading's block is
too lenient -> **sub-k_F leading outgoing leaked through** (measured: 12.5% of non-blocked NN scatters
had an outgoing below local k_F; leading 8.2%).  This is momentum-dependent: fast protons -> small
sigma -> struck nucleon adjacent -> correct k_F -> no leak (agrees >250 MeV); slow protons -> huge
sigma -> far struck nucleon -> leak -> over-interaction.  Exactly the observed |p|-dependence.

Everything else was verified identical ADoNIS<->ACHILLES: NN-elastic sigma parametrization (byte-for-byte),
sqrts (invariant mass), carbon density file (`c12_density.txt` == ACHILLES `c12.prova.txt`, identical),
local k_F formula, Fermi-momentum sampling (`kf*cbrt(U)`), QMC config positions, cylinder probability
(`b^2<sigma/pi`), step (0.04 fm), capture rule (KE<10 MeV at boundary).

## Fix
`adonis/fsi/cascade_discrete.py`: evaluate `kf_lead` at the LEADING's own position `|pos|` (per-species),
in BOTH `_nucleon_step` (pool) and `_propagate_nucleon_discrete.body` (bfs).  The recoil's k_F (`kf_j`,
at the struck vertex = the recoil's position) was already correct.

## Confirmation
Post-fix INTERACTED vs |p| (Δ/σ): [75,125) 0.7, **[125,175) 0.3, [175,250) 0.6**, [250,400) 0.9,
[400,700) 0.2 -- all <1.5σ.  The 5σ over-interaction is gone.  bit-exact nucleon-step test
(`_nucleon_step` == bfs body) still passes (both fixed identically).

## Tooling (repeatable)
- ADoNIS single-pass fate: `scripts/cascade_fate_dump.py [C|Ar] [proton|neutron] N out pauli max_steps [bfs|pool]`.
- ACHILLES reference: `achilles:fatedump` (native arm64; instrumented `Cascade.cc`: `FATE ... posr=`,
  `NSCATR`), run card `_oracle_out/run_T2K_C_fate*.yml`, env `ACHILLES_FATEDUMP=1`.  NB: the NSCATR dump
  MUST sit BEFORE the `particles.push_back` loop (a post-push reference dangles -> SIGSEGV).  Build with
  `--provenance=false`.
- Matched per-primary comparison: classify primaries as INTERACTED (nsc>0|made_pi), CAPTURED
  (~interacted & p_fin<1), ESCAPED; compare to ACHILLES status 29/26/1 per |p| bracket WITH stat errors.

## Downstream (NOT yet done) -- this changes the cascade
Reduces slow-nucleon scattering -> shifts proton multiplicity / topology / the carbon 0p tension and ALL
banks.  Needs: re-baseline the carbon cascade bit-exact tests (they freeze the old buggy output);
regenerate QE+RES banks; re-validate CC0pi/CC1pi (incl/0p/1p/2p) vs ACHILLES, C and Ar.
