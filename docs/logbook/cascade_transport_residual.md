# Pion-cascade ADoNIS↔ACHILLES residual — investigation log

**Status.** The two large π⁺¹²C transparency residuals are SOLVED, each by a real ACHILLES physics
difference (no tuned constants):

| residual | root cause | commit |
|---|---|---|
| absorption **+5%** | ACHILLES PionAbsorption **isospin partition** (π⁺p, π⁻n use (5/6)·oset_abs) | `9124afb` |
| reaction **−4%** | DCC scatter σ was **isospin-averaged**; ACHILLES is charge-resolved per nucleon | `1ad6cf0` |
| reaction **−1.3% @305/335** | beam-pion **escape** was a sphere; ACHILLES uses the z≥radius **plane** | `1b938a2` |

After all three, π⁺¹²C transparency matches ACHILLES at 305/335 to <0.6% (pulls +1.7σ / −0.0σ reaction;
abs |pull|≤0.4σ). The ONLY remaining item is a **+1.5% (+3.9σ) reaction overshoot at 245 MeV** — the
escape fix adds a ~uniform +1.5–1.8%, which over-corrects the lowest bin (pre-fix deficit there was only
−0.3%). Split (logbook #14): on identical configs the escape-fixed cascade is +0.8% (+1.4σ) at 245, and
~+0.7% more comes from ADoNIS's own-config ensemble vs ACHILLES's. Both small/OPEN (see §4). Last
updated 2026-06.

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

## 3. Current transparency match (tightened oracle, AFTER the escape fix `1b938a2`)

Combined ACHILLES oracle ~6.5M attempts (±~1.8 mb reaction / ±~1.0 mb abs), 1M-event ADoNIS, all three
fixes + all-36k configs. ADO/ACH (pull):

| p (MeV) | REACTION (pull) | ABSORPTION (pull) |
|---|---|---|
| 245 | 585.4±1.2 / 577.0±1.8 = **1.015 (+3.9σ)** | 169.6 / 170.1 = 0.997 (−0.4σ) |
| 305 | 610.3±1.2 / 606.5±1.8 = **1.006 (+1.7σ)** | 165.8 / 166.1 = 0.998 (−0.2σ) |
| 335 | 525.6±1.2 / 525.6±1.7 = **1.000 (−0.0σ)** | 143.4 / 143.6 = 0.998 (−0.2σ) |

Absorption fully matched (|pull|≤0.4σ). Reaction 335 perfect, 305 +1.7σ, **245 overshoots +3.9σ (+1.5%)**
→ §4. (Pre-escape-fix this table was 0.997/0.986/0.987 — the −1.3% at 305/335 was the escape; the fix
over-corrects only 245.)

---

## 4. OPEN: +1.5% reaction overshoot at 245 MeV (residual, post-escape-fix)

The two big residuals and the 305/335 −1.3% are fixed (§1–3). What remains is a **+3.9σ (+1.5%) reaction
overshoot at 245 MeV** introduced by the escape fix. The escape fix is itself correct (it matches
ACHILLES's `external_test` plane escape and was proven exactly at 305 by the replay, logbook #12). The
245 overshoot decomposes (logbook #13–14) into two small pieces:
- **+0.8% (+1.4σ) in the cascade itself**: the escape-fixed cascade on IDENTICAL ACHILLES configs gives
  1.008 at 245 (vs 1.0005 at 305). So the plane escape slightly over-adds at low energy — likely the
  ADoNIS radius (≈6.05 fm, minDensity 4e-6) vs ACHILLES (≈6.55 fm) interacting with the plane cut, or a
  low-energy post-scatter-escape detail. Not yet pinned.
- **~+0.7% from the config ensemble**: ADoNIS's own sampled nuclei react slightly more than ACHILLES's
  dumped nuclei at 245 (own-config transparency +1.5% vs identical-config replay +0.8%).

Both are sub-percent and only at the lowest bin; absorption is matched everywhere. Next: test the radius
value (set ADoNIS escape radius = ACHILLES's 6.55) and re-replay at 245.

### (historical) how the 305/335 −1.3% was hunted — see Logbook below
The fixed-energy table that drove the hunt (BOTH codes Pauli ON, charge-resolved σ): at 305 ADoNIS
reacted 0.1892 vs ACHILLES 0.1942 (−2.6%) with absorption matched — i.e. the deficit sat in
scatter-survived. The chain σ_scat→σ_abs→sampling→Pauli→**escape** is the Logbook. (An earlier draft
mis-attributed this to Pauli over-blocking; logbook #11–12 corrected it to the escape geometry.)

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

13. **post-fix transparency** (1M ADoNIS own configs, escape fix) vs tightened oracle, REACTION pulls:
    245 **+3.9σ (1.015)**, 305 **+1.7σ (1.006)**, 335 **−0.0σ (1.000)**; abs all matched (|pull|≤0.4σ).
    The fix closes 305/335 but 245 overshoots.
14. **245 replay** (escape-fixed `propagate_discrete` Pauli-ON on 120000 IDENTICAL ACHILLES dumped
    configs, 6 keys): ADoNIS 0.1843 vs ACHILLES 0.1827 = **1.008 (+0.8%, +1.4σ)** — cf. 305 replay
    1.0005. So on identical nuclei the escape-fixed cascade is +0.8% at 245 (vs exact at 305): the plane
    escape slightly over-adds at low energy. The remaining +0.7% (own-config transparency +1.5% −
    identical-config replay +0.8%) is an own-vs-ACHILLES-config ensemble difference. Config ROTATION
    tested + REJECTED as its cause (rotate on/off ~null at N=300k). Both pieces sub-1%; OPEN.
    NEXT: set ADoNIS escape radius = ACHILLES Radius() (≈6.55 vs 6.05 fm) and re-replay at 245.

---

## 4b. Absorbed-RES (CC0π·RES) low-proton over-production → dropped abs neutron (lead #2)

Distinct from the transparency residual above: in the **CC0π·RES channel** (RES event whose primary π is
absorbed → 0π final state), ADoNIS over-produced **low ejected-proton multiplicity** on ¹²C vs ACHILLES-C
(FSI, proc 401/402). Old banks (both dev-cap cv5 *and* high-cap hicap — so NOT a buffer effect):

| ejected p | OLD cv5 | OLD hicap | ACH/ADO target |
|---|---|---|---|
| 0p | 0.476 (−4.7σ) | 0.426 (−5.2σ) | 1.0 |
| 1p | 0.736 (−5.3σ) | 0.726 (−5.2σ) | 1.0 |
| 2p | 0.884 (−5.7σ) | 0.879 (−5.3σ) | 1.0 |

(ACH/ADO; <1 = ADoNIS over-produces that multiplicity.)

**Root cause.** ADoNIS pion absorption emitted **proton-only slots**: `abs_one` returned
`protA = where(A_is_p, pa, 0)` (neutron momenta zeroed) and the spawn charges were hardcoded
`s1_q/s2_q = 1`. The neutron product was dropped (and the 2-neutron mode emitted *nothing*). ACHILLES
re-cascades **both** absorption nucleons (`Cascade.cc` `particles_out[0],[1]`); a neutron product can
knock out a proton downstream. Dropping it lost those secondary proton knockouts → events stuck at low
multiplicity.

**Fix** (`adonis/fsi/cascade_discrete.py` `_pion_step`): `abs_one` returns both full 4-vecs `pa,pb` and
their true charges `abs_qa,abs_qb`; both nucleons spawn into the pool with their real charge and
re-cascade. The pool `_nucleon_step` already handles neutron projectiles (`same_iso=isp==nisp`,
recoil charge `bg_proton=nisp[ar,j]`), so an absorbed neutron correctly knocks out a proton.

**Result** (NEW high-cap `lead2`, 2-seed, 40k primaries):

| ejected p | NEW lead2 | was | verdict |
|---|---|---|---|
| 0p | 0.543 (−3.1σ) | 0.43 | improved, residual (N=31) |
| 1p | 0.912 (−0.9σ) | 0.73 | **closed** |
| 2p | 1.045 (+1.1σ) | 0.88 | **closed** |
| incl (≥1p) | 1.049 (+2.2σ) | 1.045 | unchanged |

The bulk 1p/2p −5σ over-production is RESOLVED — multiplicity redistributes 0p→1p→2p exactly as
predicted by re-cascading the abs neutron. BFS path (`_propagate_discrete`) still proton-only —
non-production (always-pool), no re-cascade pool there.

**Two residuals remain after lead #2 — both MAPPED (`scripts/_lead2_map.py`):**

1. **Total absorbed-RES rate ADoNIS ~4.5% low** (CC0π meson-veto, muon window, no proton/pion cut:
   ADO 2.4024e-6 vs ACH 2.5104e-6, ACH/ADO=1.045). Decomposed via the **FSI-invariant muon-only RES
   rate** (cascade doesn't touch the muon, so this isolates σ_RES in the acceptance):
   - **pre-FSI σ_RES (muon acceptance): ACH/ADO=1.013** (1.3% — MINOR; the muon-only RES rate ADO
     1.1732e-5 vs ACH 1.1887e-5).
   - **absorption fraction (cascade): ACH/ADO=1.031** (absorbed/muon-only = ADO 0.2048 vs ACH 0.2112).
     **DOMINANT.** ADoNIS under-absorbs RES pions by ~3% in the T2K acceptance.
   - 1.013 × 1.031 = 1.044 ✓.
   - (An earlier draft wrongly attributed this to pre-FSI σ_RES; that used CC0π incl+0p which
     *undercounts* absorbed events — incl needs a proton in [450,1000] and 0p needs 0 protons >250, so
     events with a proton in [250,450] fall in neither. The veto-only count above is the correct total.)

   → **Next lead:** the ~3% RES-pion under-absorption. Candidates: π⁰ channel (RES is a π⁺/π⁰ mix; the
   fate study that matched ACHILLES was π⁺-only); production-point path length (RES π born at the vertex
   inside the nucleus vs the transparency beam fired from outside); RES π momentum spectrum vs the
   transparency beam range.

2. **0p residual VERDICT (4× stats, `lead2hi` 8-seed, N_ado 90 vs 47; `scripts/_lead2_0p_hi.py`).**
   - NOT global softness: absorbed-RES leading-proton mean 682.9 (ADO) vs 681.4 (ACH); bulk spectrum
     matches 1–2% in every bracket; multiplicity 1p/2p/3p/4p all within 2–4% (1.04/1.02/0.98/1.03).
   - REAL & localized (persists at 4× stats, 0p ACH/ADO=0.623): ADoNIS over-produces absorbed-RES events
     whose LEADING proton lands just below 250 MeV — [0,175)/[175,225)/[225,250) ratios 0.39/0.67/0.70;
     0p-subset leading-|p| mean 189 (ADO) vs 211 (ACH).
   - Mechanism: lead #2's re-cascaded absorption-neutron knockouts are SOFT secondary protons; in the
     rare 2-neutron-absorption channel the only protons are these soft knockouts → leading <250 → 0p.
     ADoNIS's soft secondary-knockout tail is slightly more abundant/softer than ACHILLES.
   - Magnitude: 0p is 0.1% of absorbed-RES → negligible for any paper observable. Understood, not a
     stats artifact, parked as a sub-percent soft-knockout-tail difference in the 2-neutron channel.

**Conclusion:** lead #2 closes the cascade-side absorbed-RES multiplicity SHAPE (1p/2p). The remaining
absorbed-RES rate deficit (~4.5%) is dominated by the **cascade under-absorbing RES pions by ~3%**
(not pre-FSI σ_RES, which is only ~1.3% here). That ~3% π-absorption gap is the next cascade lead.

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

## 8. RES pi0 charge-exchange residual (10-seed C matrix, 2026-06-24)

Overnight 10-seed C cascade-vertex/segment matrix (refill engine, 6.65M ADoNIS primaries vs ACHILLES
/tmp/cascade_segments_C_ach.txt, 530k segs).  Bulk agreement is good: nucleon channels (incident p/n,
transmit/elastic) chi2/ndf 0.76-0.99; RES p/n all <1.2.  The one persistent residual that GROWS with
stats (4-seed 2.03 -> 6: 2.46 -> 8: 2.57 -> 10: 2.54) is **RES incident pi0, channel charge-exchange**:
chi2/ndf = 33.1/13 = 2.54.  Localized to MID-MOMENTUM: ADoNIS UNDER-produces the pi0 charge-ex fraction
by ~27-36% in 400-800 MeV (400-500 ratio 1.359 pull +3.5; 500-600 1.270; 600-700 1.314; 700-800 1.355),
while 0-400 MeV agrees (ratios ~1.0).  pi+ charge-ex is milder (1.33), pi- fine (0.50).  The other pi0
channels are good (transmit 1.74, elastic 0.62, abs 0.37, conv 1.68).
Pre-existing (the earlier-session "QE/RES secondary-pion charge-ex ~2.1-2.3sigma" note); NOT a Stage-3
effect -- the refill engine is bit-exact to the pre-refill distributions (per-event RNG + refill gates).
Lower-priority secondary cells: QE pi+ elastic 2.97 / pi- elastic 3.63 (very low N, not significant),
QE p inelastic 2.47, QE n elastic 2.14, RES n inelastic 1.84.
TODO (separate session): localize the pi0 charge-ex deficit to the DCC charge-exchange cross section
(jax_channel_sigmas_resolved out_ch distribution) vs ACHILLES at W ~ 400-800 MeV pi0 + nucleon.

## 9. fig3 matched-geometry transparency (pool engine, R=10 both sides, 2026-06-27)

Rebuilt the π⁺¹²C transparency as paper Fig. 3 on the SOLE pool engine (`analysis/paper/fig3_cascade_
absorption/make.py`): the ADoNIS side drives `cascade_full.run_cascade_pool` + `_pion_step` in ACHILLES's
EXACT `InitCrossSection` beam geometry (beam_r uniform in a disk R_DISK=10 fm, z₀=−1.05·R_nuc, status
external_test → no formation zone).  ACHILLES side = `run_cascade_pip_C.yml` (VirtRes interactions;
uniform-momentum range beam [80,500], σ(p)=πR²·n_react/(n_tried·Δbin/Δrange), n_tried from the
GenCrossSection counter; 8 seed batches, 41.6k reactions).

**Geometry fix vs the old oracle (commit 3cfbeea).**  The old CSVs used MISMATCHED radii — ACHILLES
πR² with R=6.5, ADoNIS "R from ρ tail" — so the old "~1.33×" was partly a normalization mismatch.  With
R=10 on BOTH sides ACHILLES gives **641 mb at the Δ (290 MeV)**, matching the physical DUET π⁺¹²C
reaction σ (~600 mb); the old R=6.5 gave only 270 mb (truncated the ρ tail).  R=10 is the correct disk.

**Reaction-counting bug found + fixed (the real cause of the apparent residual).**  A first pass defined
the ADoNIS reaction as the kind-1 record `nh>0`.  `nh` records the *sampled* pion hit BEFORE Pauli
blocking (`_pion_step` returns `has_hit` at cascade_discrete.py:618; the Pauli check sets `is_scat=False`
but the record still fires).  ACHILLES counts a reaction ONLY when the interaction is NOT Pauli-blocked —
`Cascade::FinalizeMomentum` adds the history vertex inside `if(hit)` (Cascade.cc:903,996), and
`reaction = History().size()>0`.  So `nh>0` over-counted blocked scatters; the over-count is largest at
low pion momentum where the recoil nucleon is soft and Pauli blocking is strongest.  This produced a
SPURIOUS reaction excess: ADO/ACH = 1.10 at the Δ growing to ~1.5 at 90–130 MeV, χ²/ndf = 51.6 — entirely
in the scattering channel (absorption was untouched, since a blocked scatter leaves the pion alive so the
"no surviving pion" absorption tag was already correct).
Fix: define reaction from the primary pion's ACTUAL fate — `prim_fate∈{ABSORB,CONVERT}` OR an actual
scatter `out["nsc"]>0` (`nsc += is_scat` counts only non-blocked scatters, cascade_discrete.py:608) —
NOT `nh`.

**Result after the fix (80k ADoNIS vs 8-seed ACHILLES, matched R=10):**
- REACTION: mean ADO/ACH = **1.001** over the Δ (230–350 MeV); per-bin within a few % (statistics);
  χ²/ndf = **2.9**.  Low-p 90 MeV now 1.01 (was 1.5).
- ABSORPTION: abs fraction ADoNIS **0.308** vs ACHILLES 0.305; χ²/ndf = **4.9** (a few high-p bins
  scatter at low absolute σ / lower stats).
- Both σ(p) peak at the Δ; shapes and magnitudes agree.

CONCLUSION: the π⁺¹²C cascade transport is faithful to ACHILLES at the ~1% level (reaction at the Δ
<1%, absorption fraction <1%) with the SAME untuned Oset+DCC and no fitted constants — the earlier
"~1.10–1.5× reaction residual" was a fig3-side counting artifact, not cascade physics.  This is
consistent with §1–3 (identical-config replay <1%).  The §8 pi0 charge-exchange channel residual is
SEPARATE (a DCC out-channel split, not the reaction total) and still open.

## 10. pi0 CEX residual: the DCC cross section is REFUTED as the cause (2026-06-27)

Followed up §8's TODO (localize the π⁰ charge-exchange deficit to the DCC cross section).  The
cross section is BIT-FAITHFUL to ACHILLES — it is NOT the cause:
- **Amplitude file identical**: `data/achilles/.../ANL_0-0.dat` is byte-for-byte the ACHILLES repo file
  (96095 B, `diff` clean).  W grid [1080, 2200] MeV (pion lab |p| up to 2096) — covers the whole
  400–800 MeV (W 1300–1557) region; both sides clamp to 0 only beyond the grid.
- **Wave column order identical**: ADoNIS `WAVES` ≡ ACHILLES `PWnames`/`L_vec`/`twoJ_vec`/`twoI_vec`
  (L 0,0,1,1,1,1,2…; 2I 1,3,1,1,3,3…; 2J 1,1,1,3,1,3…).  No I=1/2 scrambling.
- **Formula identical**: ADoNIS `_channel_sigma` == ACHILLES `CalcCrossSectionW_grid` — same CG products
  (π⁰p→π⁺n CEX = √2/3·(A₃ᐟ₂−A₁ᐟ₂)), same Σ_LJ (2J+1)|Σ_I CG_I·A_LJI|², same pref 197.32²·10·2π·4W²/PF.
- **Resulting CEX fraction is textbook-correct** (JAX cascade path `jax_channel_sigmas_resolved`):
  σ_cex/(σ_el+σ_cex) for π⁰ = 0.35 at the Δ (W1225; the 2:1 elastic:cex ratio), rising to ~0.50 in the
  I=1/2 second-resonance region (W~1475), falling back to ~0.38 by W1600.  Charge-resolved per struck
  nucleon (π⁰p→π⁺n needs p; π⁰n→π⁻p needs n).  out_ch sampling = inverse-CDF on these σ, == ACHILLES
  SelectChannel.

So the per-scatter π⁰ CEX cross section AND fraction match ACHILLES at fixed W.  The §8 segment-matrix
residual (CEX low by 27–36% at 400–800 MeV lab |p|, growing with stats) therefore lives DOWNSTREAM of
the cross section.  Leading remaining candidate: the CEX fraction is steeply W-dependent above the Δ
(0.35→0.50 over W 1225→1475 while σ_el falls 90→10 mb), so a small difference in the W DISTRIBUTION at
fixed lab |p| — i.e. the struck-nucleon Fermi-momentum/relative-angle sampling at the scatter vertex —
shifts the lab-|p|-binned CEX fraction much more than the (steeply-falling, less sensitive) elastic
absolute.  NEXT: compare the π⁰-scatter CEX fraction per **W** bin (not lab |p|) on identical inputs
(ACHILLES VERTEXDUMP/achilles:vertex replayed through the ADoNIS segment logger), to confirm it is the
W-distribution/binning rather than a per-scatter difference.

## 11. pi0 CEX per-W test: per-scatter physics is FAITHFUL; sec-8 residual is not in the CEX (2026-06-27)

Decisive per-W test (`scripts/pi0_cex_perW.py`, fig `output/figures/pi0_cex_perW.png`).  Fired a pi0 BEAM
at uniform lab |p| [80,900] into 12C on BOTH sides and compared the scatter CEX fraction cex/(el+cex):
- ADoNIS: drove `_pion_step` directly (500k pi0, R_DISK=8), recording per first-scatter (lab |p|, W,
  el/cex).  ACHILLES: `achilles:vertex` VERTEXDUMP, 9 seed batches (~18.4k pi0 scatters).
- Logic: f_cex(W) is proven bit-identical (sec 10), and CEX(lab|p|)=∫f_cex(W)·P(W|lab|p|)dW, so this
  tests the W DISTRIBUTION at fixed lab |p| (= the struck-nucleon Fermi/relative-angle sampling = transport).

RESULTS:
- **Cascade runtime CEX-vs-W lies exactly on the cross-section curve** (right panel): no runtime W/sigma
  bug.  (0.34 at the Delta dip W~1270, rising to ~0.49 at W~1475, falling to ~0.38 by 1600.)
- **CEX-vs-lab|p| agrees ADoNIS/ACHILLES to <5% (statistics)** across every bin (left panel):
  ratio A/C = 1.01, 0.95, 0.99, 0.98, 1.03, 0.98 for [200,400]..[800,900].  18.4k ACH / 49k ADO scatters.

So the per-scatter pi0 CEX -- cross section AND transport (the W-distribution at fixed lab |p|) -- is
FAITHFUL to ACHILLES.  A controlled pi0 beam does NOT reproduce the sec-8 RES residual (27-36% deficit).
CONCLUSION: the sec-8 RES pi0-CEX residual is NOT in the cascade CEX physics.  It is upstream/RES-specific
-- either the RES-PRODUCED pi0 momentum/angular population (primary RES generation, feeding the segment
matrix) or a segment-matrix stats/binning artifact -- NOT the DCC charge-exchange cross section or the
pion transport.  (The cross-section hypothesis of sec 8/10 is now fully closed: refuted analytically in
sec 10 and confirmed in-cascade here.)  If the RES residual matters for CC1pi it should be chased in the
RES pi0 spectrum, not the cascade.

## 12. All-channel per-W test: pi+/pi0/pi-/p/n faithful; Pauli effect on CEX (2026-06-27)

Generalized the per-W test to all 5 beam hadrons (`scripts/cascade_channel_perW.py` + `_plot.py`, fig
`output/figures/cascade_channel_perW.png`).  Pions: CEX fraction cex/(el+cex); nucleons: inelastic
fraction inel/(el+inel) (NN->NN pi).  ADoNIS drives `_pion_step`/`_nucleon_step` directly; ACHILLES
`achilles:vertex` VERTEXDUMP (8 seed batches each).  ADoNIS/ACHILLES ratio vs lab |p| (binomial errors):
- pi+ CEX: 0.89-1.01 ; pi0 CEX: 0.95-1.01 ; pi- CEX: 0.91-1.02 (low-p [200,400] ~0.9 for pi+/pi-, ~3sigma)
- p inel: 1.00-1.10 ; n inel: 0.99-1.16 (high-p <2%; low-p/near-threshold bins noisy).
Nucleon inelastic turns on correctly at the Delta-production threshold (W~2017, lab|p|~800) -> ~0.4,
tracking ACHILLES and the pre-Pauli NN->NDelta cross section.

DIFFERENTIAL-PAULI effect (pi0, confirmed): the post-Pauli runtime CEX-vs-W sits BELOW the bare
cross-section ratio (sum-pull -4.6 sigma over W 1240-1520); with cfg.pauli=False it returns to the curve
(+0.6 sigma).  Above the Delta the CEX recoil nucleon is softer than the elastic recoil (I=1/2 angular
content) -> CEX is Pauli-blocked slightly more -> post-Pauli CEX fraction suppressed below the bare sigma
ratio.  ACHILLES applies the same Pauli blocking, so ADoNIS-vs-ACHILLES (both post-Pauli) still agree.

NOTE (diagnostic-only bug, not ADoNIS): `_nucleon_step` returns `do` = ELASTIC-only; inelastic is
signaled by the created-pion spawn pio[4]=pi_alive with do=False.  A first nucleon driver gated scatters
on `do` -> showed inel=0; fixed to (do | pio_alive).  ADoNIS produces NN inelastic correctly.

Net: every per-scatter cascade channel (pion el/cex/abs, nucleon el/inel) is faithful to ACHILLES; the
sec-8 RES pi0-CEX residual remains attributable to the RES pi0 POPULATION (upstream), not the cascade.

## 13. pi+/pi- low-p CEX deficit = nucleon-SELECTION geometry (pre-Pauli), not Pauli/xsec (2026-06-27)

The all-channel per-W plot left panels show pi+/pi- CEX ~10% LOW vs ACHILLES at low lab |p| [200,400]
(W~Delta): pi+ 0.89+/-0.03 (-3.7s), pi- 0.91+/-0.02 (-4.5s); pi0 is fine (1.01).  Decisive test
(ACHILLES_NO_PAULI on both sides):
  pi+ [200,400]: post-Pauli A/C 0.89+/-0.03 ; NO-Pauli A/C 0.90+/-0.05  -> deficit PERSISTS w/o Pauli.
So it is NOT a Pauli-implementation difference (that cancels here), and NOT the cross section (bit-exact,
sec 10).  Remaining cause: the per-step NUCLEON-SELECTION geometry -- ADoNIS's Gaussian discrete-Glauber
(closest nucleon passing exp(-pi b^2/sigma)) vs ACHILLES's exact impact-parameter walk -- picks the p/n
hit ratio slightly differently.  It only shows in pi+/pi- because their CEX is CHARGE-ASYMMETRIC (pi+ CEX
= pi+ n -> pi0 p needs a NEUTRON) while at the Delta sigma(pi+ p)=198 >> sigma(pi+ n)=68, so a small p/n
selection bias swings the small (diluted) CEX fraction ~10%.  pi0 is immune (both nucleons give CEX
symmetrically -> selection-independent).  This is the documented continuum/discrete transport residual
surfacing in the charge-asymmetric channel; small absolute (diluted channel), sub-% expected on CC1pi.

CAVEAT on the right-panel curve: sigma_cex/sigma_scat (nucleon-SUMMED) is the bare-sigma ratio assuming
P(hit nucleon) ∝ sigma_total (dilute limit).  It is EXACT for pi0 (symmetric) -- no-Pauli runtime lands
on it (+0.6s) -- but only APPROXIMATE for pi+/pi- (no-Pauli runtime sits ~ -10s below it) because the
geometric selection deviates from sigma-weighting when sigma is large/asymmetric (Delta).  So for pi+/pi-
judge agreement by the LEFT panel (ADoNIS vs ACHILLES), not the right-panel curve.

## 14. pi+/pi- CEX deficit LOCALIZED: nucleon selection over-weights protons ~5% (2026-06-27)

Chased the sec-13 pi+/pi- low-p CEX deficit to its mechanism via a pi+ beam, recording per scatter the
recoil-nucleon charge (proton/neutron HIT fraction among scatters).  Measured ADoNIS vs ACHILLES vs the
sigma-weighted prediction (sigma_scat(pi+p)/(sigma_scat(pi+p)+sigma_scat(pi+n)) at each scatter's W):

| lab|p| | ADoNIS runtime | sigma-weighted pred | ACHILLES |
|---|---|---|---|
| [200,400] | 0.773 | 0.738 | 0.744 |
| [400,600] | 0.631 | 0.615 | 0.595 |
| [600,900] | 0.292 | 0.305 | 0.318 |

**ACHILLES follows the sigma-weighted prediction; ADoNIS does NOT** -- it over-selects protons by ~3-5%
at/below the Delta (and slightly under at high p).  Since pi+ CEX needs a NEUTRON (pi+ n -> pi0 p) and at
the Delta sigma(pi+p)=198 >> sigma(pi+n)=68, a ~5% proton-over-selection -> ~11% fewer neutron hits ->
the ~10% pi+/pi- CEX deficit.  pi0 is selection-symmetric (both nucleons give CEX) -> immune (matches).

RULED OUT as the cause (all measured null): cross section (bit-exact, sec 10); Fermi-momentum sampling
(ACHILLES kf*cbrt(U) == ADoNIS, uniform Fermi sphere, no SRC); Pauli (deficit persists with
ACHILLES_NO_PAULI on both sides); fast_xsec top-K (~0.09 nucleons per 0.04fm slab, never active);
time_step (null for a lone pion); step granularity (step 0.04->0.16 left proton-frac 0.773->0.767);
first-vs-all-scatter binning (all-scatter gives the same 0.773).  The selection LAW is provably identical
to ACHILLES (closest-perp passer == perp-sorted first-passer).

So the residual is a SUBTLE discrete-Glauber transport realization difference: ADoNIS's per-step Gaussian
nucleon selection departs from sigma_total-weighting by ~5% at large/asymmetric sigma (the Delta), where
ACHILLES's exact impact-parameter walk stays sigma-weighted.  NOT pinned to a single fixable line; the
per-encounter ingredients are each faithful.  Impact: ~10% on the DILUTED pi+/pi- CEX channel at the
Delta -> expected sub-% on CC1pi.  A genuine fix needs per-encounter instrumentation on BOTH sides
(ACHILLES dump of the candidate set + struck nucleon + impact parameter, replayed through ADoNIS) to pin
the exact divergence -- an ACHILLES-rebuild-level effort.  Documented; NOT hacked with a fitted constant.
Tools: scripts/cascade_channel_perW.py (beam driver, records recoil charge + all scatters),
scripts/cascade_channel_plot.py, /tmp/pn_ratio.py.
