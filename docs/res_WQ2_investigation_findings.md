# RES dσ/dW, dσ/dQ² deficit — investigation findings

Goal: make ADoNIS RES single-pion `dσ/dW`, `dσ/dQ²` match ACHILLES, i.e. the ratio panels
in `res_WQ2_shapes.png` flat on 1.0 (like QE already is).

Symptom (robust, reproducible): for a **single free proton, mono 1 GeV** (ν p → μ⁻ p π⁺),
ADoNIS's generated `dσ/dQ²` shape vs the ACHILLES hepmc sags:
**ratio ≈ 0.71 at low Q² → ≈ 1.30 at high Q²** (area-normalised). χ²/ndf(Q²) ≈ 240.
This is NOT statistical (identical at 400k and 2M events) → a real bias.

---

## TL;DR (current best understanding)

- **The physics (matrix element + weights) is correct.** Every per-event quantity reconciles
  with ACHILLES to <1% (often bit-exact). The deficit is **NOT in the amplitude.**
- **The deficit is on the event-generation / sampling side**, i.e. *which events ADoNIS produces*.
- **The pion-mass lead is DEAD (tested 2026-06-08).** See below — regenerating ADoNIS free-proton
  events with the pion kinematic mass set to ACHILLES's value (134.977) instead of 139.57 left the
  Q² sag **completely unchanged** (low ratio 0.731 → 0.731). The 4.6 MeV threshold shift is
  irrelevant at the bin resolution used. **Do not pursue the pion mass further.**
- **MEASURED (2026-06-08): it is a clean Q² SHIFT, not a tilt or normalization.** Moments of
  flat-Dalitz×amps2 (1.5M, kept 416k) vs the hepmc (200k):

  | quantity | ADoNIS flat-Dalitz×amps2 | ACHILLES hepmc | Δ |
  |---|---|---|---|
  | ⟨Q²⟩ [GeV²] | 0.3927 | 0.3465 | **+0.0462** |
  | σ(Q²)       | 0.2205 | 0.2191 | +0.001 (≈equal) |
  | ⟨W⟩ [MeV]   | 1229.1 | 1226.2 | +2.9 (≈equal) |
  | σ(W)        | 58.6   | 57.6   | ≈equal |
  | Q² range    | [0.002,1.062] | [0.003,1.051] | same |
  | W range     | [1077.9,1554.7] | [1077.2,1553.7] | same |

  → **Same width, same W distribution, same kinematic ranges — ⟨Q²⟩ alone is pushed up 46 MeV².**
  ADoNIS systematically weights slightly higher Q². This is the entire effect; it fully explains the
  monotone 0.71→1.30 ratio.

- **What a pure-Q² shift implies.** W is fine and the support is identical, so it is NOT a
  threshold/phase-space-support problem (consistent with the dead pion-mass lead). A shift in ⟨Q²⟩
  with matched ⟨W⟩ means the **Q²-dependence of the per-event weight (amps2) differs** between ADoNIS
  and ACHILLES on the *uniform* flat-Dalitz sample — i.e. ADoNIS's amps2 falls off too slowly in Q²
  (or ACHILLES's falls faster). Candidate causes, in priority order:
    1. **amps2 evaluated at the wrong Q² argument** — ACHILLES feeds the **de-Forest-shifted**
       (off-shell) kinematics into the form factors / DCC table lookup; if ADoNIS looks the amplitude
       up at the *bare* leptonic Q² the form-factor falloff is misplaced → exactly a Q² reweighting.
       (NB the earlier "de-Forest no effect" test only checked the amps2 *value* on the proposal
       sample, ~0.65 MeV; it never checked the *Q²-binned shape* on a uniform sample. Re-examine.)
    2. **A Q²-dependent amps2 error invisible on ACHILLES's forward-pion proposal** but exposed by
       flat-Dalitz's uniform Q² coverage (the angular validation never binned in Q² on a uniform
       sample). → the bulletproof ACHILLES-driver test (step 3 below) settles this.
  **Next: bin my_amps2 vs Q² on a *uniform* sample and compare to ACHILLES_amps2 — needs the driver,
  OR check whether ACHILLES's RESDUMP points span the full Q² range and re-bin the ratio there.**

### Mass facts (measured, for reference — NOT the bug)
ACHILLES kinematic masses (measured from 20k hepmc events, std ~1e-13 → hardcoded constants):
- pion (pid **211**, π⁺): mass **134.977** (the π⁰ value — ACHILLES uses one common pion mass)
- final nucleon: **938.270**;  struck nucleon (at rest): **939.570**;  muon: **105.700**
ADoNIS kinematic masses: pion **139.57018**, nucleon **938.27**. The pion differs by 4.6 MeV but,
as established above, **this difference does not cause the sag.**

---

## DECISIVE (2026-06-08): the sampler is NOT the bug — the amplitude is

I ported ACHILLES's **own** event proposal verbatim — `ThreeBodyMapper::GeneratePoint`
(`TChannelMomenta` for the pion split + `Isotropic2Momenta` for the μ/N split, constants
m_alpha=0.9, ct∈[−1,1], amct=1) — as the actual free-proton **generator**, and weighted each event
by 1/gw with gw = `ThreeBodyMapper::GenerateWeight`. (`scripts/achilles_mirror_gen.py`.)
Validated bit-exact: conservation 2e-13, masses exact, and my forward gw vs the independent
inverse-map `GenerateWeight` agree to **2e-15 median**.

Everything else is byte-identical to `free_proton_gen.py` (masses, flux, spinavg, amps2, binning).
**The only thing changed was the proposal+weight: isotropic → ACHILLES's exact t-channel.**

Result — it **still sags identically**:

| generator | Q² low(<0.2) | Q² high(>0.6) |
|---|---|---|
| isotropic + J_3body | 0.731 | 1.303 |
| flat-Dalitz (trivial jacobian) | 0.685 | 1.288 |
| **ACHILLES-mirror (t-channel proposal + bit-exact gw)** | **0.725** | **1.265** |

**Three independent samplers — including a verbatim port of ACHILLES's own — agree with each other
and disagree with ACHILLES.** This rules out the sampler/jacobian/proposal as the cause and
**falsifies the "match ACHILLES's events+weights → recover agreement" plan.** The reason that plan
seemed guaranteed: the earlier "ACHILLES events + amps2 + weights → hepmc flat" checks were *not*
apples-to-apples — they used **ACHILLES's own dumped amps2** (test 8, near-tautological) or only the
**pointwise ratio** my_amps2/ACHILLES_amps2 on the dumped proposal subsample (test 5). The mirror
run is the first true test (my_amps2 on a fresh ACHILLES-identical sample, weighted ACHILLES's way)
— and it sags.

**Therefore ∫ my_amps2 dΦ ≠ ∫ ACHILLES_amps2 dΦ: the amplitude genuinely differs over phase space,
with a Q²-dependent tilt (the +46 MeV² ⟨Q²⟩ shift).** The pointwise "0.99 flat" of test 5 must have
been on a subsample that under-populates the region where they differ (the high-Q² tail). The bug is
the amplitude's Q²-dependence (or the Q² argument fed to it), NOT event generation.

**Next:** bin my_amps2 / ACHILLES_amps2 vs Q² on the **full** RESDUMP (not a subsample) and look
specifically for a tilt at high Q²; if the dump under-covers high Q², use the ACHILLES driver on
flat-Dalitz events. The leading mechanism remains the **de-Forest off-shell Q² argument** to the form
factors / DCC lookup (ACHILLES shifts it; ADoNIS may use the bare leptonic Q²).

## DRIVER (2026-06-09): the amplitude is exonerated by ACHILLES itself — inference-free

I patched ACHILLES (`XSecBackend::CrossSection`) to override the event momenta from a file **before
`TransformFrame`** (layout `momentum[]={nu,struck,mu,nucleon,pion}`), so ACHILLES runs its **own**
Fortran DCC amplitude on **my exact mirror events**; the existing RESDUMP then prints (my momenta,
ACHILLES amps2). Rebuilt incrementally and ran the mono 1H config with `ACHILLES_DRIVER` set.
(`scripts/achilles_mirror_gen.py` produced the events; `/tmp/drv_*` are the harness.)

- **Sanity:** where ACHILLES recomputes on its *own* re-injected momenta, `ach/original = 1.0000`
  → the injection is valid.
- **Decisive:** on my mirror events, wherever ACHILLES returns nonzero, **`ACHILLES_amps2 /
  my_amps2 = 0.999–1.000, FLAT across every Q² bin** (0.05→1.0 GeV²).

→ ACHILLES's own code, on my own generated events, reproduces my amps2 to 0.1% everywhere. **The
matrix element is definitively NOT the source of the sag** — this removes the last inference (no
longer "validated on ACHILLES's proposal sample"; validated on *my* events by *ACHILLES's* code).

**New clue the driver exposed:** only **~1/3 of `CrossSection` calls return nonzero** — including for
ACHILLES's *own* re-injected momenta (66/200 sanity) — and the zeros are **uncorrelated with W or
Q²** (zero/nonzero have identical median W≈1297, Q²≈0.24). So ACHILLES's amps2 for a *fixed* final
state is **stochastic per call** → the `RES_Spectral_Func` model is sampling something (removal
energy / off-shell struck nucleon) even for this "1H, Binding 0.0" run, which uses the **C12**
`info_C12_pke.data` spectral file. My mirror/free_proton_gen use a **fixed on-shell struck nucleon at
rest (939.57)** and omit that sampling entirely. **Prime remaining suspect for the dσ/dQ² shape:**
ACHILLES smears the struck nucleon via the (C12) spectral function → a distribution of off-shell
`s` → reshaped dσ/dW,dσ/dQ², which the fixed-`s` ADoNIS RES generator does not replicate. (Caveat:
the rejection *rate* is Q²-flat, so any shape effect acts through the kinematics shifted on the
*accepted* third, not through the rejection fraction.)

## CORRECTION (2026-06-09, later): the "weight is the bug" below was a VEGAS CONFOUND — retracted

The section immediately below concluded the bug is `gw`. **That is wrong** — resolved by reading:
- RES on **hydrogen uses `CoherentMapper`** (`FortranModel::PhaseSpace`, `FNuclearModel.cc:163`
  returns `"Coherent"` for hydrogen/free-neutron; only heavier nuclei get `"OneBodySpectral"`
  =`QESpectralMapper`). Struck at rest, `Jh=1` — matches the data.
- The dumped `psw` = `event.Weight()` = the **Vegas** integrator weight (`Process.cc:436`,
  `ps_wgt = m_integrator.GenerateWeight`) × `m_max_weight` (`EventGen.cc:250`) =
  `(1/gw) × Vegas_grid_weight × const`. So generated events carry the **converged Vegas adaptive
  weight** (the cv 7.3, U-shape); the `validate_res_psw` committed dump is **pre-Vegas** (uniform
  grid → Vegas weight = 1), so there `psw = Jb·Jh·(1/gw)` exactly → validate matched 1e-14.
- **Therefore `three_body_genweight` (gw) is CORRECT** — it is the mapper jacobian, confirmed by both
  validate and reading. The driver-own `amps2×(1/gw)` sag is just **dropping the Vegas weight on a
  Vegas-sampled set** (biased); it does NOT indict the mirror (which samples uniformly → 1/gw is right).

**So the `validate_res_psw` tension is RESOLVED (no gw bug), but the sag is REOPENED.** The decisive
remaining fact: `flat-Dalitz × amps2` (provably-uniform PS, no mapper/Vegas/weight) STILL sags, and
amps2 is driver-proven correct → **the hepmc dσ/dQ² ≠ textbook ∫amps2 dΦ.** Everything in ADoNIS is
confirmed correct; the difference is in ACHILLES's generation. Prime suspect: **de Forest off-shell**
(amplitude evaluated at generation-time-shifted kinematics; the driver captured the amps2 *value* on
final momenta but not the generation shift). Secondary: Vegas convergence / unweighting (~1%, unlikely).

---
## (RETRACTED — Vegas confound, see correction above) the bug is the final-state phase-space weight

High-statistics test on the **clean free-proton run** (64,567 ACHILLES-own events generated as a
by-product of the driver run — struck nucleon at rest, √s fixed, all validated):

| Q² | `amps2 × psw` (ACHILLES weight) / hepmc | `amps2 × (1/gw)` (my weight) / hepmc |
|---|---|---|
| 0.05 | 0.997 | 0.944 |
| 0.25 | 1.002 | 1.127 |
| 0.55 | 0.998 | 0.905 |
| 0.75 | 0.989 | 0.780 |
| 0.95 | 1.028 | 0.686 |

- **ACHILLES's dumped event weight `psw` reproduces the hepmc flat.** ✓
- **My `1/gw` produces the sag exactly.** ✗  → the weight is the bug, high-stat and unambiguous.
- `psw × gw` is **not** constant (median 0.239, cv 7.3, smooth Q² drift, U-shaped in W) — yet for a
  free proton `event.Weight() = lwgt(beam) × hwgt(Coherent=1) × mwgt(ThreeBody=1/gw)` with `lwgt`
  constant (mono beam, fixed-mass `Smin` parameter). So `psw×gw` *should* be constant.

**Therefore `my three_body_genweight (gw) ≠ ACHILLES's ThreeBodyMapper::GenerateWeight` on the full
kinematic range**, even though it matched on the small `validate_res_psw` sample (300 evts) and the
early mono dump (2656 evts, `psw×gw=1`). The deviation has clear structure — `psw×gw` vs W is
U-shaped (1.42 at low W, 0.82 mid, 1.61 high) — pointing at the t-channel split / s23=(μN)² grouping
or the `TChannelWeight` Jacobian in the port. **This `gw` is used by BOTH `free_proton_gen` and the
mirror, so fixing it fixes `res_WQ2_shapes.png`.**

**Driver also settled amps2 independently:** ACHILLES's own amplitude on my exact events = my amps2,
flat in Q² (above). So: amplitude correct, **phase-space weight `gw` wrong** — that is the whole bug.

**Simple inversion is RULED OUT:** re-weighting mirror by `amps2×gw` (instead of `1/gw`) does NOT
fix it — it gives a *different* wrong shape (1.72→0.80→1.28 vs the 0.71→1.28 sag). The correct
weight (ACHILLES `psw`) is **neither `gw` nor `1/gw`**. The missing factor `psw×gw` is median ≈0.24,
cv 7.3, U-shaped in W (1.42/0.82/1.61), ~0.83→1.23 in Q². Applied on top of `1/gw` it is the whole fix.

**Next:** instrument ACHILLES `PSMapper::GenerateWeight` to dump `lwgt`,`hwgt`,`mwgt` separately
(one incremental build — driver binary/mounts ready), to pin the missing factor to a specific mapper;
then bin `psw×gw` vs every mapper intermediate (s23, ct, tcw, i2w). The narrow committed/early-mono
validation samples (`psw×gw=1`) were too narrow to catch it; the high-stat driver-own sample reveals it.

## What is CONFIRMED CORRECT (with the decisive evidence)

All of these were checked against ACHILLES on **self-consistent** kinematics (amps2 and momenta
dumped from the *same* `XSecBackend::CrossSection` call — see Instrumentation below).

1. **DCC amplitude `zmtx`** — dumped ACHILLES's `zmtx` (`interpolate_amp`) and compared to my
   `build_zmtx` at the exact (W,Q²): **match to 4 decimals on all 8 components, all 14 partial
   waves**, incl. the longitudinal idx 3,4,7,8. (`scripts/zmtx_compare.py`)
2. **Hadron tensor W^{μν}** — dumped ACHILLES's hadron current `zj_mu`, built the spin-summed
   tensor, compared to mine on the same events: **all 10 independent components match to ~1–15%**
   incl. the transverse-longitudinal interference **W01/W02/W13/W23**. (`scripts/zj_compare.py`)
3. **Lepton current** — `leptonic.lepton_current` reproduces ACHILLES's dumped lepton current
   `L` exactly (ratio 1.0000). Spinors are bit-exact unit-tested. (`scripts/lh_reconcile.py`)
4. **Contraction** — `amps2 = Σ|L·H|²` with the (+,−,−,−) bilinear dot reproduces ACHILLES's
   `amps2` exactly (`sum|L_ach·H_ach|²/amps2 = 1.0000`).
5. **Full per-event amps2** — my `exclusive_amps2_batch` on ACHILLES's self-consistent events
   = ACHILLES amps2 **flat across all Q² (0.99–1.00)** and **flat across all pion angles
   cosθ_π ∈ [−0.9,+0.9] (1.000)**. → The amplitude is correct **everywhere in kinematics**.
   (`scripts/lh_reconcile.py`, `/tmp/ang.py`)
6. **flux, initwgt, spinavg** — match ACHILLES exactly (flux 1.0375e5, initwgt 1.0, spinavg 0.5;
   all constant for free proton). (`scripts/weight_reconcile.py`)
7. **psw (phase-space weight / `event.Weight()`)** — `validate_res_psw.py` reconstructs ACHILLES's
   `psw` from the ported `ThreeBodyMapper` + `QESpectralMapper` to **rel ~1e-14 (bit-exact)**.
8. **Unweighting is clean** — ACHILLES's **weighted** dσ/dQ² (from 370k RESDUMP integrand samples
   `amps2·flux·initwgt·spinavg·psw`) matches the **unweighted** hepmc to **<2% across all Q²**.
   So the hepmc faithfully represents the true dσ/dQ²; the `PercentileUnweighter` capping is not
   the issue. (`/tmp/wt_vs_unwt.py`)
9. **My sampler is unbiased for pure phase space** — my isotropic 3-body sampler weighted by
   `J_3body` reproduces the flat-Dalitz ground truth to **<1% in Q²** (amps2≡1).
   (`scripts/test_3body_q2_measure.py`)
10. **Constructed events are clean** — conservation to 1e-12, all final-state masses exact,
    Δ region populated.

## What I WRONGLY blamed (dead ends — and the lesson)

The investigation flip-flopped many times. Each of these was asserted as "the bug" and later
overturned. The recurring failure mode: **validating on ACHILLES's *proposal* sample (which is
t-channel-forward-pion + isotropic-muon weighted) and/or binning only in Q², which averages over
the kinematic structure where any discrepancy would live.**

- **Pion-pole sign flip** (`ADONIS_DBG_PION_POLE=-1`): flattened the mono bench but was a *fitted
  compensation* — `zmtx` (incl. pion pole) was later shown bit-exact to ACHILLES. Overturned.
- **dfun / irot_q=1 gather / longitudinal / W01-W13 interference**: `fold_tensor_compare` showed a
  2× difference between `current_and_tensor` (irot0) and the exclusive gather (irot1), but the
  direct angular amps2 test (5) showed my amps2 = ACHILLES at **all pion angles** → the gather is
  fine; that 2× was an artifact of *my own* irot1 fold construction, not vs ACHILLES.
- **de-Forest q-shift / struck mass**: ~0.65 MeV for free proton; tested both ways, no effect.
- **Lepton mass (m_μ longitudinal coupling)**: massless made it *worse*; lepton is correct.
- **Boost / frame**: amps2 is frame-invariant (lab vs 2CM contraction bit-identical).
- **The "resgen bench" 0.5× deficit was an ARTIFACT.** `scripts/resgen_ratio.py` read the
  `RESGEN` dump (`Process.cc`) whose `g_last_amps2` is the *last* CrossSection call's value, NOT
  the accepted event's — stale/mismatched vs the dumped momenta. Self-consistent dumps give 0.99.
  **Do not trust the resgen bench.**
- **The "strip test" amplitude deficit was an artifact** of the flat-Dalitz reference s-mismatch /
  my own event construction, contradicted by the direct angular test (5).

## The remaining contradiction (honest open issue)

Even with (1)–(10) all confirming amps2 + phase space are correct, the **gold-standard test still
sags**: flat-Dalitz (provably-unbiased, uniform-Dalitz = uniform phase space) × my amps2, area-
normalised, vs hepmc → **0.68 (low Q²) → 1.30 (high Q²)**, identical for the isotropic sampler.

Logically: if `my_amps2 = ACHILLES_amps2` everywhere AND `my_PS ≈ ACHILLES_PS`, then
`∫ my_amps2·dΦ` must equal the hepmc — but it doesn't. The marginals match:
- pure-PS **Q² marginal**: my flat-Dalitz vs ACHILLES `psw` → <6% (mostly <2%).
- pure-PS **W marginal**: <3% **except the lowest-W (threshold) bin: ACHILLES/mine = 1.16 (16%)**.

The most reliable single test (`scripts/lh_reconcile.py`, "my amps2 on ACHILLES's own events +
ACHILLES psw"): reproduces the hepmc **flat to <2%**. So ADoNIS matches ACHILLES **when it uses
ACHILLES's events**; the gap is purely in **which events ADoNIS itself generates** — the sampling.

**Caveat on that "reliable" test (important):** it reweights ACHILLES's *own proposal points* by
ACHILLES's psw and compares to the hepmc *produced from those same points*. That is partly
**circular** — it cannot independently catch an amps2 error in a region the ACHILLES proposal does
not populate. The genuinely independent estimator is **flat-Dalitz × my_amps2**, and that one sags.
So the gap is in one of:
  (A) **my_amps2 is wrong on kinematics ACHILLES's proposal never visits** (the angular validation
      was done only on ACHILLES's forward-pion proposal sample), or
  (B) a **kinematic-mapping / variable definition difference** (e.g. how Q²/W are formed, an
      off-shell q-shift applied to the binning, etc.) that shifts ADoNIS's spectrum vs ACHILLES's.
The smooth monotone-crossing ratio shape favors **(B) a shift/tilt** over a localized amps2 error.

## Next step (concrete) — pion mass is RULED OUT

1. ~~Pion-mass/threshold fix~~ — **DONE, NEGATIVE.** Matching m_π to 134.977 did nothing.
2. **Decide shift vs tilt (cheap, do first):** compute ⟨Q²⟩, ⟨W⟩, σ(Q²), σ(W) for
   flat-Dalitz×amps2 vs the hepmc. If the means differ → kinematic-mapping bug (B); hunt the Q²/W
   construction and any de-Forest q-shift in the *binning*. If means match but shape tilts → weight
   bug. Also plot the full **2D (W,Q²)** density of both.
3. **Bulletproof, removes all inference (do if 2 is ambiguous):** build a small ACHILLES driver that
   evaluates ACHILLES's *own* amps2 on **my flat-Dalitz events** (not ACHILLES's proposal points).
   If ACHILLES_amps2 == my_amps2 there too → (A) is false, the bug is purely the mapping/binning (B);
   if they differ at low Q² → I've localized the amplitude region to fix.

## Key methodology notes (for whoever continues)

- **Always validate against ACHILLES on a sample that spans the relevant kinematics**, not just
  ACHILLES's proposal sample. Per-event ratios binned in Q² can read 1.0 while the angular/W
  structure (and thus the integrated spectrum) differs. Use the **flat-Dalitz** sampler as the
  unbiased ground truth.
- **The `RESGEN` dump (`Process.cc`, `g_last_amps2`) is stale** — use the `XSecBackend` RESDUMP /
  LHDUMP which dump amps2 + momenta from the *same* CrossSection call.
- Distinguish **kinematic/phase-space masses** (event generation, thresholds) from
  **amplitude/form-factor masses** (M_PI=138.04 etc.) — they need not be equal and ACHILLES's
  kinematic pion mass appears to be the π⁰ mass 134.977.

## Instrumentation added to ACHILLES (revert as housekeeping)

`src/Achilles/fortran/amp_dcc_sl.f`: `ZMTXDUMP`/`ZMTXVAL` (zmtx), `ZJDUMP`/`ZJVAL` (zj_mu) — debug
prints only, gated counters.
`src/Achilles/XSecBackend.cc`: `LHDUMP` (lepton L, hadron H, amps2 + momenta, to stderr) and the
existing `RESDUMP` (amps2, flux, initwgt, spinavg, psw + momenta). RESDUMP cap raised to 4e6.
Rebuild: incremental `cmake --build` in `achilles-deps` with `_dockerbuild` + source mounted.

## ADoNIS diagnostic knobs left in place (defaults = faithful)

`adonis/primary/dcc/assembly.py`: `DBG{}` scale knobs (pion_pole, axial_*, idxp_start, axial_sign),
overridable via `ADONIS_DBG_*` env. `adonis/xsec/spinor.py`: `FORCE_MASSLESS` (env
`ADONIS_MASSLESS_LEPTON`). `adonis/xsec/dcc_current.py`: `exclusive_amps2_batch(return_zj=,
q_direct=)`, `ADONIS_CONTRACT_FRAME` env. All default to the faithful behaviour.

## Key scripts

- `scripts/free_proton_gen.py` — free-proton mono generator (res_xsec 3-body sampler) + figure.
- `scripts/test_3body_q2_measure.py` — flat-Dalitz ground-truth vs isotropic sampler (phase space).
- `scripts/zmtx_compare.py`, `scripts/zj_compare.py` — amplitude/hadron-tensor vs ACHILLES dumps.
- `scripts/lh_reconcile.py`, `scripts/weight_reconcile.py` — full per-event L/H/amps2/weight reconcile.
- `scripts/validate_res_psw.py` — bit-exact psw (phase-space weight) validation.
- `/tmp/ang.py` — amps2 ratio vs pion angle (the test that exonerated the amplitude angular structure).
