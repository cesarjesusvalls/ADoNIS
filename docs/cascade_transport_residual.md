# Pion-cascade ADoNIS↔ACHILLES residual — investigation log

Status: **absorption SOLVED** (commit `9124afb`); a separate **~4% scatter-RATE deficit** remains
OPEN. Last updated 2026-06 (this session).

## RESOLUTION of the +5% absorption (commit `9124afb`)

**Root cause:** the pion-absorption cross section that competes in the cascade is isospin-decomposed
by partner channel (Nucl.Phys.A568 Table 1; `PionAbsorption.cc:85-136`, see analysis below). For
**LIKE-CHARGE pairs (π⁺p, π⁻n)** charge conservation forces both outgoing nucleons identical (p+p /
n+n), so only the opposite-isospin partner channel survives and the absorption seen by the cascade is
**(5/6)·oset_abs**, not the full `oset_abs`. Every other (π,N) pair keeps the full `oset_abs` (its 3
modes sum back to it). ADoNIS used the full `oset_abs` for *all* pairs → over-absorbed π⁺ on protons
(the dominant Δ⁺⁺ channel) → the documented ~5%.

This enters BOTH the interaction probability and the branching: `Interaction::TotalCrossSection`
(`Interactions.cc:38`) sums all channel xsecs, including the reduced absorption, and that total drives
`prob = exp(−πb²/σ_tot·0.1)` (`Cascade.cc:644/671`) and `SelectChannel`.

**Fix** (`cascade_discrete.py`, after the σ eval): `sa *= where((π⁺&p)|(π⁻&n), 5/6, 1)`. No tuned
constants. Effect on absolute ABSORPTION (mb), 1M ADoNIS vs 1.2M-att ACHILLES:

| p (MeV) | ABS ach / ado before | ABS ach / ado AFTER |
|---|---|---|
| 245 | 173.9 / 181.5 (+4.4%) | 173.9 / **173.2** (−0.4%) |
| 305 | 166.2 / 175.6 (+5.7%) | 166.2 / **164.9** (−0.8%) |
| 335 | 143.9 / 151.7 (+5.4%) | 143.9 / **141.9** (−1.4%) |

**How it was localized — per-pion CASCSEQ instrumentation.** Built `achilles:cascade-seq` (dumps, per
event, `CASCSEQ nscat=N nabs=N` = pion scatters / absorptions; counting only pion interactions, scatter
vs abs by presence of an outgoing pion — see *Build & usage* below). Over 354k matched-beam events vs
ADoNIS's `nsc`:

| metric (among reacted π) | ACHILLES | ADoNIS (pre-fix) | ADO/ACH |
|---|---|---|---|
| mean scatters / reacted π | 1.055 | 1.051 | **0.996** |
| abs fraction / reacted π | 0.309 | 0.331 | **1.070** |

The scatter *multiplicity matched* — the gap was entirely the **vertex abs/scatter branching** (+7%),
NOT the transport walk (which the earlier hypothesis blamed). That pointed straight at σ_abs entering
the branching, and reading `PionAbsorption::CrossSection` exposed the isospin partition ADoNIS lacked.

## OPEN: residual ~4% scatter-RATE deficit

With absorption fixed, REACTION (=scatter+abs) is now uniformly low: ach/ado **0.959 / 0.954 / 0.961**
at p=245/305/335. Absorption matches (above), so this is a **scatter-rate** deficit (~4%, energy-
INDEPENDENT → geometric/normalization, not vertex physics). Cause **currently UNIDENTIFIED**; ~2% of
the 4% is the correct consequence of the absorption fix (lower σ_tot for π⁺p lowers prob — yet ACHILLES
has the same lower σ_tot and still reacts more, so a real residual remains).

**Ruled OUT this session (read both sources / config):**
- *Nucleon positions* — ADoNIS already samples the SAME QMC configs ACHILLES uses
  (`cascade_discrete.sample_nucleons` → `_load_qmc_configs`, weighted `jax.random.choice`;
  `cascade.yml Configs: QMC_configs.out.gz`). NOT smooth-ρ. (An earlier draft wrongly blamed smooth-vs-QMC.)
- *SRC / Fermi gas* — oracle is `FermiGas: Type Local, Params: []` (no correlated SRC tail); ADoNIS
  uses the same local FG `kf·∛U`. Configs are exactly CM-centered (centroid 0.0000 fm).
- *Pauli blocking* — ACHILLES `PauliBlocking` uses the LOCAL per-species kf (`∛(ρ_species·3π²)·ℏc`)
  at the outgoing-nucleon position; matches ADoNIS `_kf_local` (211 vs 211 MeV at vertices).
- *Selection* — ACHILLES `Interacted`+`Project`+`BetweenPlanes` = smallest-impact-parameter passer with
  independent per-nucleon Gaussian rolls = ADoNIS step pick. Consume-on-interaction, escape (sphere vs
  external_test plane), σ_scat magnitude (0.3%), step 0.04 — all match.
- *Reaction definition* — ACHILLES accepts (writes) iff `event.History().size()>0`; a Pauli-blocked
  attempt records no node (`if(hit)` guard, `Cascade.cc:739`), same as ADoNIS (`interacted=is_abs|is_scat`).

**Open candidate under test:** the only un-matched step is ACHILLES's per-event **rigid random rotation**
of each config (`Configuration.cc:77-89`) — ADoNIS fires all pions along +z through UNROTATED configs.
Argued negligible (centered, isotropic, 36000 samples) but being verified directly (`/tmp/ado_rot.py`,
rotated vs unrotated reaction).

---

# (historical) Original investigation: the +5% absorption

## The observable

π⁺ + ¹²C transparency (ACHILLES `CrossSection` mode: beam uniform in p∈[80,500] MeV over a disk
of radius R=10 fm, fired through ¹²C). Cross sections via `sigma = πR² · P(.|p)`, πR²=3141.6 mb.

- ADoNIS side: `adonis.fsi.cascade_discrete` step mode, fired from z=−12, b uniform in the R=10 disk
  (harness `/tmp/transparency*.py`, `/tmp/hi_reac_abs.py`).
- ACHILLES side: re-ran `_oracle_out/cascade_virt_c12.yml` to **1.2M attempts** (32× the committed
  72k oracle) to beat the error down from ±6–10 mb to **±1.5–2.5 mb** — without this the residual is
  invisible. Batch: `/tmp/ach_batch.sh` → `_oracle_out/absrun/abs_*.hepmc`; extract with
  `python scripts/cascade_abs_from_hepmc.py _oracle_out/absrun/abs_*.hepmc` (drop `--nbins`, it has a
  bug that treats the value as a filename).

### High-stat result (1M-event ADoNIS step vs 1.2M-att ACHILLES)

| p (MeV) | REACTION ach / ado | ABSORPTION ach / ado | abs/reaction ach / ado |
|---|---|---|---|
| 245 | 578.5±4.6 / 566.4±1.2 | 173.9±2.5 / 181.5±0.7 | 0.301 / 0.320 |
| 305 | 606.0±4.7 / 583.8±1.2 | 166.2±2.5 / 175.6±0.7 | 0.274 / 0.301 |
| 335 | 525.0±4.4 / 508.8±1.2 | 143.9±2.3 / 151.7±0.7 | 0.274 / 0.299 |

**ADoNIS: reaction −3%, absorption +5%, scatter −7%** (scatter = reaction − abs). The abs/scatter
branching is tipped ~+8–10% toward absorption. All ~3–4σ at this stat.

This is the SAME residual documented in commit `9189390`: "RES abs_frac 0.270 → 0.228 (ACH 0.218)"
— i.e. ADoNIS pion absorption fraction ~+4.6% high. The CC0π *ratios* (QE-C 1.000, RES-C 0.987) are
fine because the residual washes down through production/QE/signal-def; the raw absorption keeps the
~5%. It is within the validated test band (`tests/test_cascade_vs_achilles_oracle.py` locks 0.85–1.20)
and the level commit `449a106` claimed ("~5%"), but it is a coherent one-sided offset, not noise.

## What is PROVEN to match (read both sources line-by-line + instrumented dump)

The per-nucleon physics is bit-identical — the residual is NOT in any single interaction component:

| component | ACHILLES location | verdict |
|---|---|---|
| interaction probability `exp(−π b²/σ)`, σ=xsec/10 | `Cascade.cc:41` | bit-identical to ADoNIS `exp(−π perp²/(sig·0.1))` |
| σ_abs (Oset p+s wave), σ_scat (DCC) | `OsetCrossSections.cc`, `MesonBaryonInteractions.cc` | bit-exact: `scratch/compare_oset_abs.py`/`compare_mb_scat.py` on `/tmp/instr.log` give 1.000 / 1.003 over all 31k samples |
| Oset kinematics: effective `0.6·kf²` for s, actual `vrel` | `OsetCrossSections.cc:32-53` | faithfully ported (`oset_xsec._kinematics`) |
| Fermi sampling: local FG `cbrt(ρ_species·3π²)·ℏc`, `kf·cbrt(U)` | `Nucleus.cc:165,212`,`GenerateConfig:144` | identical to `_kf_local`/`sample_nucleons`; `225` is Global-FG only (unused) |
| nuclear density | `c12.prova.txt` | identical file to ADoNIS `c12_density.txt` (∫=6=Z, per-species) |
| Pauli blocking `|p|<kf(pos)`, pion never blocked, block→continue | `Cascade.cc:774,695-737` | identical condition + outgoing-nucleon positions (pion@pos, struck@pos) |
| absorption 3-body kinematics (E*=√s/2, p*, isotropic, boost) | `PionAbsorption.cc:138-197` | algebraically identical to `abs_one` |
| scatter 2-body kinematics (E, p_f, **angle axis = incoming-pion CM dir**, masses) | `MesonBaryonInteractions.cc:65-192` | identical to `_two_body_cm_scatter`; `_CH_MASS=[139.57,134.98,139.57]` = `ParticleInfo().Mass()` |

## Fixes made this session (both correct; committed; neither closes the gap)

1. **`d109552`** channel-specific scatter angle. ADoNIS sampled `cos_cm` from one (π⁺p, pure I=3/2)
   dσ/dΩ for ALL channels; ACHILLES uses the channel/W-specific partial-wave dσ/dΩ (`Get_CSpoly_W`).
   Built per-(π_in,nuc,π_out) angular table from `_CHANNELS` isospin `cg`, threaded `chan_idx`.
   π⁺p (Δ⁺⁺, dominant) unchanged. **Effect: 305 MeV unchanged; 245 MeV abs went UP 181.5→185.7**
   — the single-channel approx was *partially compensating* the residual (right number, wrong reason).
2. **`34a29b2`** absorption partner must conserve charge (match `FindClosest`): π⁺p forces a neutron
   partner, π⁻n forces a proton. Fixes a CC0π charge-violating final state. **Transparency-neutral**
   (3-body s dominated by pion E + 2 mN; partner Fermi momentum barely shifts the Pauli block).

## Ruled out

- **Scatter angle is the lever?** No. Forcing isotropic (a fresh-process bracket — note the jit-cache
  trap: monkeypatching a function after the first trace is silently ignored) makes abs *worse*
  (305: 175.6→179.6), and the correct channel-specific fix doesn't close it either.
- **Escape (sphere vs plane).** ACHILLES external_test escapes at the z=radius plane, internal pions
  (and ADoNIS) at the |pos|>radius sphere (`Cascade.cc:530`). Negligible: QMC nucleons only reach
  ~5 fm, so the off-axis sphere-cut (z=√(R²−b²)) passes no extra nucleons, and post-scatter pions use
  the sphere in both. Radius value: ACHILLES `minDensity=1e-6` (`Nucleus.cc:49`) vs ADoNIS 4e-6 → ADoNIS
  R≈6.05 fm slightly smaller, but no nucleons out there.
- **Gaussian-prob normalization, pion masses, partner charge** — all match / transparency-neutral.

## Current hypothesis & next step

Every per-interaction component is identical, so the residual is in the **multi-nucleon transport
accumulation** (how the discrete-Glauber walk combines the matched per-step physics across scatters).

**Measurement (1) DONE — inputs match (`/tmp/eval_dist.py`):** ADoNIS's per-eval input distributions
over the matched beam are nearly identical to ACHILLES `/tmp/instr.log` (OSETABS/MBSCAT):
dens mean 0.0987 vs 0.0986 (percentiles match), vrel 0.866 vs 0.859, W 1233 vs 1229, fermi 211 vs 211,
~12 nucleons passed/pion. So the geometry, which-nucleons, and per-eval kinematics are CORRECT. The
+5% is therefore **definitively** in the transport walk, not the inputs. (The vrel +0.8% is even the
wrong sign: σ_abs∝1/vrel → would make ADoNIS absorb *less*.)

**Next — measurement (2):** rebuild an instrumented `achilles:cascade-instr` that dumps the **per-pion
interaction sequence** (n_scatter before absorb/escape, and the interaction positions) and compare the
scatter-multiplicity distribution to ADoNIS (`propagate_discrete` returns `nseg`). The hepmc records
only the final state (no history), so this needs the rebuild. The signature to explain: ADoNIS
**scatter −7%, absorption +5%** with identical per-step physics — i.e. ADoNIS pions undergo fewer
net scatters and tip to absorption, despite matched cross sections, geometry, Pauli, and kinematics.
Candidate walk-level effects to test once ACHILLES's per-pion sequence is in hand: (a) order/timing of
the consume + re-evaluation across a scatter, (b) whether ACHILLES re-rolls already-passed nucleons
after a direction change differently, (c) multi-nucleon shadowing along the path.

## Build & usage: `achilles:cascade-seq` (per-pion interaction-sequence dump)

Instrumentation (in the ACHILLES source tree, `git diff` shows it; ~18 lines):
- `include/Achilles/Cascade.hh`: `mutable size_t m_nscat_seq{}, m_nabs_seq{};` member counters.
- `src/Achilles/Cascade.cc` `Evolve`: reset both to 0 at start; before `Reset()` at the end,
  `fprintf(stderr, "CASCSEQ nscat=%zu nabs=%zu\n", m_nscat_seq, m_nabs_seq);`.
- `Cascade.cc` `FinalizeMomentum`, inside `if(hit)`: if either particle is a pion, scan
  `particles_out` — `has_pi_out ? m_nscat_seq++ : m_nabs_seq++` (scatter vs absorption).
  (`OsetCrossSections.cc` / `MesonBaryonInteractions.cc` also carry the older OSETABS/MBSCAT dumps.)

Build the image (local; do NOT pass `--platform`):
```
cd Achilles && docker build -t achilles:cascade-seq .
```
Run one config (stderr carries the CASCSEQ lines):
```
docker run --rm -v "$PWD/_oracle_out:/out" --entrypoint /achilles/bin/achilles-cascade \
    achilles:cascade-seq /out/absrun/<cfg>.yml 2> cascseq.log
```
Batch over seeds to accumulate ~100k+ events (`/tmp/ach_seq_batch.sh`): for each seed it `sed`s
`cascade_virt_c12.yml` (seed, `NEvents: 30000`, output hepmc name) and runs the image, appending
stderr to one log; ~30k events in ~25s, sporadic SIGSEGV (exit 139) but the CASCSEQ lines already
flushed are valid. Parse: `grep CASCSEQ <log> | sort | uniq -c` → the joint (nscat,nabs) histogram;
"reacted" = any line with nscat≥1 or nabs≥1, "absorbed" = nabs≥1. Compare to ADoNIS `propagate_discrete`
`nsc`/`absorbed` over the matched beam (`/tmp/ado_seq.py`).

## Key artifacts / reproduction

- ACHILLES image `achilles:cascade` (LOCAL build; do NOT pass `--platform` — forces a failing pull,
  unlike the ghcr `:oracle` image). Instrumented: `achilles:cascade-instr` (dumps OSETABS/MBSCAT).
- Run: `docker run --rm -v "$PWD/_oracle_out:/out" --entrypoint /achilles/bin/achilles-cascade
  achilles:cascade /out/absrun/<cfg>.yml`. `NEvents`≈attempts; ~83k att in ~15s; sporadic SIGSEGV
  (exit 139) but partial hepmc keeps valid acc/att — batch over seeds.
- ADoNIS step-mode transparency: `/tmp/hi_reac_abs.py` (1M/energy, ±1 mb).
- See also: docs `cc0pi_fsi_investigation.md`; commits `3cfbeea` (Fig-3 cascade validation, "~5%
  residual is transport"), `449a106` (discrete-Glauber, oracle norm), `9189390` (struck-isospin).

## Unrelated WIP in `cascade_discrete.py` (dormant, `cfg.algo` default "step")

`cfg.algo="interaction"` is an experimental jump-to-next-interaction kernel (~15× faster) that is
statistically off vs step at high pion momentum (+35–60% absorption) — a SEPARATE, larger bug, not
this ~5% residual. `fast_xsec` (slab-restricted σ eval, bit-exact ~1.15×) is on by default in step.
