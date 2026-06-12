# Pion-cascade ADoNIS↔ACHILLES residual — investigation log

**Status.** The two large π⁺¹²C transparency residuals are SOLVED, each by a real ACHILLES physics
difference (no tuned constants):

| residual | root cause | commit |
|---|---|---|
| absorption **+5%** | ACHILLES PionAbsorption **isospin partition** (π⁺p, π⁻n use (5/6)·oset_abs) | `9124afb` |
| reaction **−4%** | DCC scatter σ was **isospin-averaged**; ACHILLES is charge-resolved per nucleon | `1ad6cf0` |

After both, π⁺¹²C transparency matches ACHILLES to ~1% in BOTH reaction and absorption at 245/305/335.
A small **~1–1.3% reaction residual at 305/335 MeV** remains (entirely in *scatter-survived*; absorption
matched) — OPEN, under investigation (see §4). It is the only visible ACH/ADO difference left in the
T2K CC0π figures (it shifts the Δ region of dσ/dW). Last updated 2026-06.

The observable throughout: π⁺+¹²C transparency, ACHILLES `CrossSection` mode — beam uniform in
p∈[80,500] MeV over a disk R=10 fm; σ = πR²·P(reaction|p), πR²=3141.6 mb. ADoNIS:
`adonis.fsi.cascade_discrete` step mode, fired from z=−12 through the same R=10 disk.

---

## 1. Absorption +5% → PionAbsorption isospin partition (`9124afb`)

**Root cause.** The pion-absorption cross section that competes in the cascade is isospin-decomposed by
partner channel (Nucl.Phys.A568 Table 1; `PionAbsorption.cc:85-136`). For **like-charge pairs (π⁺p,
π⁻n)** charge conservation forces both outgoing nucleons identical (p+p / n+n), so only the
opposite-isospin partner channel survives → the absorption seen by the cascade is **(5/6)·oset_abs**,
not the full `oset_abs`. Every other (π,N) pair keeps the full `oset_abs` (its 3 modes sum back to it).
ADoNIS used the full `oset_abs` for *all* pairs → over-absorbed π⁺ on protons (the dominant Δ⁺⁺
channel). It enters BOTH the interaction probability and the branching: `Interaction::TotalCrossSection`
(`Interactions.cc:38`) sums all channel xsecs incl. the reduced absorption.

**Fix** (`cascade_discrete.py`, after the σ eval): `sa *= where((π⁺&p)|(π⁻&n), 5/6, 1)`. Absolute
ABSORPTION mb (1M ADoNIS vs 1.2M-att ACHILLES):

| p (MeV) | ABS ach/ado before | ABS ach/ado AFTER |
|---|---|---|
| 245 | 173.9 / 181.5 (+4.4%) | 173.9 / **173.2** (−0.4%) |
| 305 | 166.2 / 175.6 (+5.7%) | 166.2 / **164.9** (−0.8%) |
| 335 | 143.9 / 151.7 (+5.4%) | 143.9 / **141.9** (−1.4%) |

**How found.** Per-pion `CASCSEQ` instrumentation of ACHILLES (see §5) showed mean scatters/reacted-π
MATCHED (ADO/ACH 0.996) but abs/reacted +7% → the gap was the **vertex abs/scatter branching**, not the
transport walk (the earlier hypothesis). Reading `PionAbsorption::CrossSection` then exposed the
isospin partition ADoNIS lacked.

---

## 2. Reaction −4% → charge-resolved scatter σ (`1ad6cf0`)

**Root cause.** The DCC scatter cross section was isospin-AVERAGED over a p/n target
(`cascade_mb.jax_channel_sigmas`): every nucleon saw `(σ_πp + σ_πn)/2` regardless of its charge.
ACHILLES `MesonBaryonInteraction` uses `GetCchannel(pion, baryon)` — the σ for the SPECIFIC struck
nucleon. At the Δ the asymmetry is large: σ(π⁺p)=198 mb (pure-I=3/2 Δ⁺⁺) vs σ(π⁺n)=68 mb (ratio ~2.9).
Because reaction is NONLINEAR in σ (`1−Π(1−exp(−πb²/σ))`), averaging the dominant Δ⁺⁺ proton channel
down to 133 mb suppressed proton scattering and the neutron over-estimate did not compensate → a ~4%
first-pass reaction deficit. (This bites only the *discrete per-nucleon* cascade where the nonlinear
`exp(−πb²/σ)` is evaluated for a specific nucleon. The mean-field `cascade_real` uses λ=ρσ which is
LINEAR, so p/n averaging is exact there for N=Z — `cascade_real` was not affected and not changed.)

**Fix.** `jax_channel_sigmas_resolved(W, pion_in, nuc_idx)` from the per-(pin,nuc,pout) grid (no p/n
average); thread the struck nucleon charge (`nisp`) into the cascade σ eval. Now consistent with the
already-charge-resolved scatter ANGLE (`d109552`). REACTION mb (ach/ado):

| p (MeV) | REACTION before | REACTION AFTER |
|---|---|---|
| 245 | 0.959 | **0.995** |
| 305 | 0.954 | **0.986** |
| 335 | 0.961 | **0.988** |

**How found.** Localized the deficit to the first-pass reaction (Pauli-INDEPENDENT via the
`ACHILLES_NO_PAULI` toggle); a host-side first-pass calculator (`/tmp/ado_chargeres.py`) then compared
averaged vs charge-resolved σ_scat directly: 0.1242 → 0.1290, matching ACHILLES 0.1286. The old "σ
matches (`compare_mb_scat` 1.003)" check had only validated the AVERAGE — correct on average but wrong
per-nucleon, and the nonlinearity does not commute with the p/n average.

---

## 3. Current transparency match (tightened oracle)

Combined ACHILLES oracle ~6.5M attempts (±~1.8 mb reaction / ±~1.5 mb abs), 1M-event ADoNIS,
charge-resolved + isospin-partition + all-36k-config build. ADO/ACH:

| p (MeV) | REACTION | ABSORPTION |
|---|---|---|
| 245 | 575.4 / 577.0 = **0.997** (0.7σ) | 169.2 / 170.1 = 0.994 |
| 305 | 597.8 / 606.5 = **0.986** (~4σ) | 166.1 / 166.1 = 1.000 |
| 335 | 518.9 / 525.6 = **0.987** (~3σ) | 144.1 / 143.6 = 1.003 |

Absorption fully matched (<0.6%). Reaction matched at 245; a real ~1.3% residual at 305/335 → §4.

---

## 4. OPEN: ~1–1.3% reaction residual at 305/335 MeV

**It is entirely in scatter-survived; absorption is matched.** Clean fixed-energy comparison, BOTH
codes with Pauli ON, charge-resolved σ (ACHILLES `achilles:scatrec` CASCSEQ at KickMomentum [pe−3,pe+3];
ADoNIS `propagate_discrete`):

| | reacted | abs | scatter-surv (=reac−abs) | mean nscat/pion |
|---|---|---|---|---|
| **305** ACH | 0.1942 ±0.0007 | 0.0527 | 0.1416 | 0.2120 |
| **305** ADO | 0.1892 (−2.6%) | 0.0532 (**matched**) | 0.1360 (−4.0%) | *(measuring)* |
| **335** ACH | 0.1669 ±0.0010 | 0.0460 | 0.1209 | 0.1942 |
| **335** ADO | 0.1645 (−1.4%) | 0.0464 (**matched**) | 0.1181 (−2.3%) | *(measuring)* |

So with σ_scat, σ_abs and the abs/scatter branching all matched, ADoNIS produces fewer
**scattered-and-survived** pions at high energy. **Root cause (found via the event-by-event replay, see
Logbook #8/#10): ADoNIS Pauli-blocks ~2% more scatters than ACHILLES at high pion energy.** On identical
nuclei the no-Pauli first-pass MATCHES (1.009) but the with-Pauli reaction is −1.6% — so it is neither a
σ/geometry nor a multiplicity effect; it is the Pauli scatter-blocking step. (The earlier
"multiplicity matched / it's in the accumulation" framing was wrong.)

**Ruled OUT for this residual (all measured / matched):**
- **σ_scat(W)** — charge-resolved σ matches ACHILLES `MBSCAT` dump to **<0.5% in every W bin** for both
  π⁺p and π⁺n (1.005/1.002/1.001 across W 1150–1600).
- **σ_abs** — ADoNIS `abs_cross_section` vs ACHILLES `OSETABS` at 305 (feeding ACHILLES's exact
  Tpi/dens/fermi/vrel): ratio **0.9998**, median 1.0000.
- **Nucleon positions** — ADoNIS samples the SAME QMC configs (`_load_qmc_configs`, weighted choice);
  loading all 36000 vs the old 20000 cap is transparency-neutral (`bc1ee24`).
- **Config rotation** — ACHILLES rotates each config; ADoNIS doesn't. VERIFIED null (`/tmp/ado_rot.py`,
  rotated vs unrotated reaction identical, incl. with charge-resolved σ).
- **SRC / Fermi gas** (oracle FG Local, no SRC; configs CM-centered), **max_steps/truncation** (500 vs
  2000 identical), **fast_xsec K=3** (True==False bit-exact), **escape sphere-vs-plane** (no nucleons
  out there), **W distribution** (Δ-peak frac [1200,1260] identical 0.207, med within 4 MeV), **Pauli**
  (faithful; block fractions bracket ACHILLES 0.263). See §6 for the full ledger.

### Logbook — chasing the ~1.3% (do NOT re-run these; sequential)

Convention: "reacted" = P(≥1 interaction). Fixed-energy = KickMomentum [pe−3,pe+3]. Numbers at 305 MeV
unless noted.

1. **charge-resolved σ fix** → reaction 0.959/0.954/0.961 ⇒ 0.997/0.986/0.988. Closed 4% → ~1.3%.
2. **tightened oracle** (6.5M att): abs fully matched (≤0.6%); reaction 245 matched, **305/335 ~1.3% low
   (3–4σ)**. This is the residual.
3. **σ_scat(W)** vs ACHILLES MBSCAT dump: <0.5% in every W bin (π⁺p & π⁺n). → NOT σ_scat.
4. **σ_abs** vs ACHILLES OSETABS at 305 (ACHILLES's exact Tpi/dens/fermi/vrel): ratio 0.9998. → NOT σ_abs.
5. **config rotation** (added to ADoNIS, charge-res σ): null (≤0.2%). → NOT rotation.
6. **config count** 20000→all 36000: transparency-neutral. → NOT config count.
7. **nucleon momentum sampling**: ADoNIS local-FG `kf·∛U` vs ACHILLES *dumped* |p|: mean 158.2 vs 158.3,
   rms 169.1 vs 169.2, |p|/kf=0.751, frac|p|>kf=0.0007 (no SRC tail). → NOT momentum sampling.
8. **EVENT-BY-EVENT REPLAY, no-Pauli** (instr `ACHILLES_DUMPCFG` dumps beam+12 nucleons+outcome; replay
   ADoNIS analytic P(react) on the *identical* dumped nuclei): ADoNIS **0.2142** vs ACHILLES empirical
   **0.2123 ±0.0058** = **1.009 (+0.3σ) MATCHED**. → ADoNIS's per-config first-pass physics is CORRECT
   (not a b↔σ correlation / roll-accumulation bug). Decisive.
9. **ADoNIS analytic on OWN 36k configs @305** (same calc as #8): 0.2100 ±0.0009 vs ACHILLES no-Pauli
   0.2123 ±0.0058 → 0.4σ, consistent (ACHILLES no-Pauli stat too loose to resolve 1%).
10. **WITH-Pauli replay** (`ACHILLES_DUMPCFG` Pauli ON; 103302 configs @305; `propagate_discrete`
    Pauli-ON on identical nuclei): ADoNIS 0.1908 vs ACHILLES 0.1952 = −2.3%. *Provisional read "it's the
    Pauli step" — WRONG, see #12.*
11. **scatter block fraction on identical configs**: ADoNIS 0.2362 vs ACHILLES `SCATREC` 0.2324; recoil
    |p| 320.9 vs 320.7; kf 203.5 vs 201.9 — **nearly equal**. A 1.6%-rel block diff cannot make −2.3%
    reaction. → the Pauli *block fraction* is NOT the cause. (Killed the #10 read.)
12. **cascade vs its OWN analytic, identical configs** (the tell): no-Pauli CASCADE 0.2070 vs no-Pauli
    ANALYTIC `1−Π(1−p)` 0.2120 — **disagree +2.4%**, yet the analytic matched ACHILLES (#8). So the bug
    is INSIDE the discrete cascade, not σ. **Disabling the escape** → cascade no-Pauli 0.2121 (= analytic
    0.2120 ✓) AND cascade with-Pauli 0.1953 (= ACHILLES 0.1952 ✓✓). **ROOT CAUSE = the escape geometry.**

**ROOT CAUSE (FIXED).** ADoNIS escaped the *beam* pion on a **sphere** `|pos|>radius`; ACHILLES's
un-scattered beam pion is `external_test` and escapes at the **z≥radius PLANE** (`Cascade.cc:532-553`,
`if(status==external_test){ if(Z<radius) continue; }`). For an off-axis pion the sphere cuts the track
at z=√(R²−b²) < R, dropping the exit-side nucleons → ~1–2% fewer reactions, worst at high energy/large b.
Fix (`cascade_discrete.py` step escape): beam pion (`nsc==0`) → plane `pos_z≥radius`; once it scatters
(`nsc>0`, internal) → sphere. (This was wrongly "ruled out as negligible" early on — it is THE residual.)
Validated on identical configs (#12) and on the high-stat transparency (§3 will be updated).

13. **post-fix transparency** (1M ADoNIS own configs, escape fix) vs tightened oracle:
    REACTION 245 **+1.5%**, 305 **+0.6%**, 335 **0.0%**; abs all matched. So the fix closes 305/335 but
    245 now OVERSHOOTS. But the **replay on identical configs matches at BOTH** (245 1.007/0.6σ,
    305 1.0005) — so the escape fix itself is correct; the +1.5% at 245 is the **own-config vs
    ACHILLES-dumped-config ensemble difference** (~0.7%), on top of the fix being within stats on
    identical configs. Candidate was config ROTATION (plane escape breaks z-symmetry) — but TESTED and
    REJECTED: with the escape fix, rotate=True (1.016/1.013/1.009) vs rotate=False (1.012/1.009/1.006)
    is ~null/slightly worse (N=300k noise). So the residual ~0.7% own-vs-dumped-config difference is NOT
    rotation; cause unidentified (sub-1%, near the oracle/stat floor). NET: escape fix closes 305/335
    (the main residual, −1.4%→matched) but 245 reads +1.5% high; on identical configs ADoNIS matches
    ACHILLES to <0.7% at both.

---

## 5. Instrumentation & reproduction

ACHILLES images (LOCAL build; do NOT pass `--platform` — forces a failing pull):
- `achilles:cascade` — clean oracle. Run: `docker run --rm -v "$PWD/_oracle_out:/out" --entrypoint
  /achilles/bin/achilles-cascade achilles:cascade /out/absrun/<cfg>.yml`. `NEvents`≈attempts; ~83k
  att/15s; sporadic SIGSEGV (exit 139) but partial hepmc keeps valid acc/att — batch over seeds.
- `achilles:scatrec` — all env/guarded dumps to stderr (rebuild `docker build -f Dockerfile.cascade
  -t achilles:scatrec .`):
  - `CASCSEQ nscat=N nabs=N` (per event, always on) — pion scatters/absorptions; in `Evolve`
    (reset) + `FinalizeMomentum` inside `if(hit)` (`has_pi_out ? nscat++ : nabs++`).
  - `SCATREC pmag= kf= blocked=` (per scatter) — outgoing-nucleon |p|, local kf, Pauli flag.
  - `GETXSEC b2= xsec= prob= rad= pid=` (per roll, env `ACHILLES_GETXSEC=1`) — the actual per-roll
    (impact², σ_total, prob, nucleon radius/charge) in `Interacted` (un-short-circuited).
  - `ACHILLES_NO_PAULI=1` — `PauliBlocking` returns false (first-pass / ablation studies).
  - `OSETABS …` / `MBSCAT pidm= pidb= W= tot=` — per-eval Oset-abs / MB-scatter σ + inputs.

Extraction / harnesses (scratch, in `/tmp`):
- `scripts/cascade_abs_from_hepmc.py <hepmc...>` — reaction+absorption σ(p) from a CrossSection-mode
  hepmc (σ = πR²·acc/att; abs = no-final-pion). Drop `--nbins` (bug: treats value as filename).
- `/tmp/ado_seq.py` (ADoNIS reacted/abs/nsc over the matched beam), `/tmp/hi_reac_abs.py` (per-energy
  transparency), `/tmp/ado_chargeres.py` (averaged-vs-resolved σ first-pass), `/tmp/ado_firstpass.py`
  (host-side first-pass), `/tmp/ado_rot.py` (rotation test), `/tmp/ach_bigbatch.sh` (oracle batch).
- ADoNIS ablation knobs: `DiscreteCascadeConfig.pauli` (Pauli on/off), `.prob` (gaussian/cylinder/pion),
  `.fast_xsec`, `.cylinder`.

---

## 6. Appendix — full ruled-out ledger & per-nucleon match

Per-nucleon physics proven bit-identical (read both sources + instrumented dump):

| component | ACHILLES location | verdict |
|---|---|---|
| interaction probability `exp(−πb²/σ)`, σ=xsec/10 | `Cascade.cc:41` | bit-identical to ADoNIS |
| σ_abs (Oset p+s), σ_scat (DCC) | `OsetCrossSections.cc`, `MesonBaryonInteractions.cc` | `compare_oset_abs`/`compare_mb_scat` 1.000 / 1.003 (averaged); charge-resolved σ_scat <0.5%/W |
| Oset kinematics: effective `0.6·kf²` for s, actual `vrel` | `OsetCrossSections.cc:32-53` | ported (`oset_xsec._kinematics`) |
| Fermi sampling: local FG `∛(ρ_species·3π²)·ℏc`, `kf·∛U` | `Nucleus.cc:165,212` | identical (`_kf_local`/`sample_nucleons`); `225` Global-FG unused |
| density | `c12.prova.txt` | identical file to ADoNIS `c12_density.txt` |
| Pauli `|p|<kf(pos)`, pion never blocked, block→continue | `Cascade.cc:803-808,697-711` | identical |
| absorption 3-body kinematics | `PionAbsorption.cc:138-205` | identical to `abs_one`; partner charge-conserving (`34a29b2`) |
| scatter 2-body kinematics + angle (channel-specific dσ/dΩ) | `MesonBaryonInteractions.cc:65-192` | identical to `_two_body_cm_scatter`; channel angle `d109552` |
| selection (smallest-impact passer, indep rolls) | `Interacted`+`Project`+`BetweenPlanes` | identical to ADoNIS step pick |
| reaction definition (`History().size()>0`, no node on Pauli-block) | `RunCascade.cc:199`, `Cascade.cc:739` | identical to `interacted=is_abs|is_scat` |

Wrong leads worth remembering: the **smooth-ρ-vs-QMC** guess (ADoNIS already uses QMC configs); the
**pauli on/off "over-blocking"** read (it compared ADoNIS-no-block to ACHILLES-WITH-block — apples to
oranges); the **σ median** comparison from `GETXSEC` (ACHILLES `Interacted` short-circuits on the first
passer, so its dumped per-roll set is a biased subset — not directly comparable to ADoNIS's full set).
jit-cache trap: monkeypatching a module fn or global after the first trace is silently ignored — route
toggles through a static `cfg` field or a fresh process.

## 7. Unrelated WIP (dormant)

`cfg.algo="interaction"` — experimental jump-to-next-interaction kernel (~15× faster), statistically off
vs step at high pion momentum (+35–60% absorption); a SEPARATE, larger bug. `fast_xsec` (slab-restricted
σ eval, bit-exact ~1.15×) is on by default in step mode.
