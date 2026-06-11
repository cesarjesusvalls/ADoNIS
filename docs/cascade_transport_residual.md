# Pion-cascade ADoNIS↔ACHILLES residual (~5% absorption) — investigation log

Status: **OPEN**. Last updated 2026-06 (this session).

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
