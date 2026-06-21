# Primary-pion cascade FATE study (ADoNIS ↔ ACHILLES), the p/n method applied to pions

## Motivation
The CC0π/CC1π × {QE,RES,QE+RES} × {0p,1p,2p} matrix (`scripts/gen_cc_matrix.py`) showed the headline
channels agree ~1%, but **CC0π·RES** (RES pion absorbed → 0π) is off (C: 0p 0.42, 1p 0.68, 2p 0.88;
−4 to −7σ, worst at low proton multiplicity) and **CC1π·QE** (cascade-created π⁺) is erratic but
ACHILLES-stat-limited.  Goal: track PRIMARY pions through the cascade and find where ADoNIS diverges
from ACHILLES in bins of momentum — the same diagnostic used for p/n (`cascade_fate_dump.py` +
`cascade_fate_compare.py` + ACHILLES `achilles:fatedump`).

## Tools (new, mirror the nucleon study)
- `scripts/cascade_pion_fate_dump.py` — ADoNIS: primary RES pions (from `res_xsec.generate`) seeded by
  `setup_carbon`, propagated single-pass by iterating the validated `_pion_step`; fate per initial-|p|
  bin (TRANSMITTED/QUASI_ELASTIC/CHARGE_EX/ABSORBED/CONVERTED) → npz.  Also records the πNN→NN
  absorption-product nucleon momenta+charges.
- `scripts/cascade_pion_fate_compare.py` — transmitted-vs-reacted (transparency) per |p| bin.
- `scripts/cascade_pion_fate_split.py` — absorbed/scattered/CEX split via per-event pairing of the
  ACHILLES FATE lines with the FATE_FS final-state pion counts (single-primary-pion events).

## ACHILLES instrumentation (image `achilles:fatepion2`)
All local images had lost the FATE instrumentation (the `:fatedump` tag now emits only RESDUMP).  The
source survives in `Achilles/src/Achilles/Cascade.cc`; two faithful edits were added (same FATEDUMP
design, gated by env `ACHILLES_FATEDUMP=1`):
1. snapshot filter (line ~271): `IsNucleon()` → `IsNucleon() || IsPion()` (snapshot primary pions too).
2. `FATE_FS` line: added final-state pion counts `npip/npi0/npim` (separate absorbed from scattered).
Build: `docker build -f docker/Dockerfile.fullcascade -t achilles:fatepion2 <Achilles source>`.
Run: `docker run --rm -e ACHILLES_FATEDUMP=1 -v "$PWD/_oracle_out":/out --entrypoint
/achilles/bin/achilles achilles:fatepion2 /out/run_T2K_C_fate.yml 2>&1 1>/dev/null | grep "^FATE"`.
NB: RESDUMP floods stderr during integration — keep only `^FATE` lines; FATE fires at event-gen time.

## Result (carbon, π⁺)
Every category agrees with ACHILLES bin-by-bin (no ≥3σ flags), 50k-event ACHILLES (11185 primary π⁺):
- transmitted (transparency): overall ADoNIS 0.479 vs ACHILLES 0.491; all |p| bins <1.5σ.
- absorbed: peaks ~0.34 at 250–300 MeV (Δ region) on BOTH; all bins ≤1.5σ.
- scattered / charge-exchange: all bins ≤2.5σ, no flags.
- πNN→NN absorption-product nucleons are HARD (~450–590 MeV; only ~1–3% of absorptions give 0 protons
  above the 250 MeV ejection threshold) — and the residual-doc appendix already verified the absorption
  3-body kinematics bit-identical to ADoNIS `abs_one`.

Figures: `paper_figures/pion_fate_vs_p_C.png`, `pion_fate_stacked_C.png`, `pion_fate_pull_C.png`,
`pion_absprod_spectrum_C.png`.

## Conclusion
**The pion FSI is reproduced bin-by-bin** — pion reaction rate, absorbed/scattered/CEX split, and
absorption-vertex kinematics all match ACHILLES.  The CC0π·RES "0-proton excess" is therefore **NOT**
in the pion cascade or the absorption vertex; it is downstream on the **nucleon-transport** side
(re-cascade of the hard absorption-product nucleons + the RES recoil proton: energy loss / Pauli /
secondary scatters degrading them below threshold differently than ACHILLES).  Next: final proton
spectrum / multiplicity of RES-absorbed CC0π events in the full cascade, ADoNIS vs ACHILLES.
