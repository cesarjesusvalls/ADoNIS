# Neutral-current (NC) support in ADoNIS — implementation plan

**Status:** plan only, no NC code written yet. **Date:** 2026-07-30. **Branch:** `sec1`.
**Rev 2** — revised after adversarial source review (see §8).

**Deliverable:** the two NC figures of arXiv:2508.19213 that Section 1 is missing —
`fig:NCpi0Xpdoublediff` (MicroBooNE NC1π⁰Xp double-differential in cos θ_π⁰ and p_π⁰) and
`fig:NCpi0Xpcospi` (dσ/dcos θ_π⁰), on argon, BNB flux. **Figure slots 11 and 12** — the gap our
fig10 → fig13 numbering already reflects.

**These are ADoNIS-vs-ACHILLES validation figures. No experimental data overlay**, exactly like
every other Section 1 figure.

> **Doc trust.** The `.md` files in this repo are known to be out of sync (README.md, RUNNING.md,
> phases/, logbook/). Everything below was read from **source** this session. Statements about
> ACHILLES cite `file:line` under `/sdf/data/neutrino/cjesus/ADoNIS/software/Achilles-src/` —
> note the `DIFFGEN/Achilles/` directory is `data/` + a flux symlink only (88 K, no source) and
> misled an earlier survey.

> **Prior NC work is NOT a foundation.** Commit `5a0cdb8` (2026-06-05, "Phase A2") predates the
> entire refactor arc — the `scripts/` removal, `adonis/xsec`→`adonis/channels`, the unified
> `generate_bank(cfg)`, the `PhysicsParams` unification, the pool-cascade migration, and the
> oracle-test purge. Its tests are gone, `data/oracle/` is empty, and its gate was a
> **single-constant bridge** (structurally blind to absolute normalisation). The two surviving
> artifacts — `NC_CHANNELS` (`dcc/structure.py:65-70`, **zero importers**) and the `mode<=-1`
> branch (`dcc/differential.py:218-227`, **zero test coverage**) — are treated as **unvalidated
> legacy**. They are re-derived from the ACHILLES Fortran and re-validated. Any agreement with
> them is a cross-check to be *earned*, never an input.

---

## 0. The biggest risk

**A wrong absolute NC normalisation that every shape gate is blind to.**

This is the `e3319a5` failure mode: the EM leptonic current omitted the photon `1/q²` propagator, so
the absolute (e,e′) scale was ~1e15 too large, and the ratio-only gate could not see it. NC now has
**two** normalisations to get right (RES and QE), each with three multiplicative unknowns a ratio
cannot separate: the Z propagator, the leptonic coupling `ee/(2·sw·cw)`, and `_NORM_NC` / `_COUPL_NC`.

**Non-negotiable mitigation.** No current phase closes on a ratio. Each closes on:

1. an **analytic propagator gate** — at `Q² ≪ M_Z²`, `amps2_NC/amps2_CC` at fixed hadronic kinematics
   must reduce to `(M_W²/M_Z²)²·|g_NC/g_CC|²·(isospin factor)` to ≲1e-3. Propagator-*sensitive* by
   construction. ~20 lines; would have caught `e3319a5` in an afternoon; **write it before the
   current it tests**, so it cannot be weakened to fit the implementation; and
2. an **absolute free-nucleon oracle gate** — σ(E_ν) in **absolute nb** vs an ACHILLES NC run at ≥5
   energies over 0.7–2.2 GeV, reporting **mean(ACH/ADO)** and **spread(ACH/ADO)** as two separate
   numbers. A constant offset ⇒ mean≠1, small spread (coupling/`_NORM` error); an energy slope ⇒
   propagator error. **Bridge constants are forbidden as gate closers.**

**Second risk, created by this plan:** the two renames (P2, P3) touch ~127 call sites and rewrite
~15 GB of banks in place. A silent migration error is caught by no physics gate — hence the
byte-comparison gates in G2 and G3.

---

## 1. Decisions

### D1 — the ACHILLES `coupl1` quirk: **SETTLED — implement as a switch**

`LeptonicCurrent.cc:94-96`:

```cpp
const std::complex<double> coupl1 = (ee * i / (4 * sin2w * cw)) * (0.5 - 2 * sin2w);
const std::complex<double> coupl2 = (ee * i / (4 * sw   * cw));
const std::complex<double> coupl3 =  ee * i / (cw * sw * 2);
```

`coupl1` has **`sin2w`** (= sin²θ_W) where `coupl2`, one line below, has **`sw`** (= sinθ_W).
Definitions verified at `include/Achilles/Constants.hh:41,57,58`. `coupl2 = ee·i/(4·sw·cw)` pins the
overall prefactor: the SM proton NC vector current is `(g/c_W)[(¼−s²)F1p − ¼F1n]`, and `coupl2`'s
`−¼·(g/c_W)` isovector partner fixes the normalisation at `g/(2c_W) = ee/(2·sw·cw)`. Therefore

```
correct coupl1 = ee·i/(sw·cw) · (¼ − sin2w) = ee·i/(2·sw·cw) · (0.5 − 2·sin2w)
ACHILLES coupl1 = ee·i/(4·sin2w·cw) · (0.5 − 2·sin2w)
ratio = [1/(2·sin2w)] / [1/sw] = 1/(2·sw) ≈ 1.0396          →  ~4 % too large
```

**This affects BOTH nucleons, not the proton alone.** `LeptonicCurrent.cc:110-111` uses `coupl1` for
`F1n` and `F2n` inside the **neutron** dictionary, exactly as it uses it for `F1p`/`F2p` in the
proton one. `FA`/`FAP` (carried by `coupl2`) and `coupl3` (RES, coherent) are untouched, so this is
**NC QE only** — but on both the proton and the neutron F1/F2 terms.

**Decision (project owner): make it a switch, defaulting to correct physics.**

- A `GenConfig` boolean, `achilles_coupl1_quirk: bool = False`.
- **Default `False` = correct physics** (`sw`). This is what ADoNIS *is*.
- **`True` = reproduce ACHILLES verbatim** (`sin2w`). Set explicitly in the bank config used to
  generate the ACHILLES-comparison figures, so the comparison is like-for-like.
- **The flag is written into `manifest.json`**, so a bank can never be ambiguous about which
  convention produced it. A figure config consuming a bank must assert the flag it expects.

**Why this is better than either single choice.** It turns a footnote into a **two-sided gate**:
with the quirk ON, ADoNIS must match ACHILLES; with it OFF, ADoNIS must differ from ACHILLES by
**exactly the predicted 1.0396 on both nucleons' F1/F2 terms and by nothing else**. That pins the bug
reproduction *and* the correct physics simultaneously, and it makes the discrepancy a measured
number rather than an assertion. Neither "always verbatim" nor "always correct" can do this.

Implementation notes: the switch lives at the coupling definition in `channels/currents/dirac.py`,
not smeared through call sites; a test pins **both** branches; the ACHILLES line is cited at the
switch; report upstream.

### D2 — the `vec`/`isv` convention: **RESOLVED BY SOURCE READING — the legacy form is wrong, and the exposure is CC**

> **This section was rewritten after review.** The plan originally left D2 open with an escape hatch.
> That hatch is now closed by source reading, and the conclusion is worse than the plan assumed: the
> defect is not confined to NC.

The Fortran vector assembly (`amp_dcc_sl_module.f:867-1050`) is:

- proton (`itiz=+1`): `zzz = vfac·zampv` for **all** waves;
- neutron (`itiz=-1`), `mode<10` (CC **and** NC): `zzz = vfac·zampv` for **all** waves;
- then, `if(mode.le.-1)` **only** (NC), a separate **additive** block over **I=½ waves only**:
  `+= vvfac(itiz)·zampv_is`.

And `adonis/channels/dcc/loader.py:107` reads `vec` / `isv` as the **raw** `zampv` / `zampv_is`
blocks. A term-by-term transcription is therefore

```
src_block = VFAC*vec + (I==½)·VVFAC(itiz)·isv
```

whereas the legacy `differential.py:218-227` computes

```
src_block = i32*(VFAC*vec) + (1-i32)*(VFAC*0.5*(vec-isv) + VVFAC*0.5*(vec+isv))
```

On an I=½ proton wave the coefficient of `vec` is `0.5*(VFAC+VVFAC) = 0.0376` vs `VFAC = 0.5376`
— a factor **~14**.

**Option (a) is closed.** The plan originally allowed that an isospin-decomposition step might exist
between the table and `build_zmtx`, making both branches consistent. It does not:

- `dcc/loader.py:107-109` reads `vec`/`isv`/`axial` as the **raw** `namp1`/`namp2`/`namp3` blocks —
  no decomposition. (It correctly does *not* apply the Fortran `isign`; `differential.py` does.)
- `dcc/amplitudes.py:33-38,55-86` — `full()` sums the three `za` components and the interpolators do
  FMM-cubic interpolation × knob scaling. **No isospin rotation anywhere.**

`build_zmtx` is the only place the isospin combination happens, and it does not match the Fortran.

**The CC "validation" does not cover the disputed term.** The plan previously took comfort from CC
passing at χ²/ndf ≈ 1.5 with a flat 1.0011 amps2 audit. That audit is **structurally blind** here:
`dcc/current.py:38-41` states it is against the ACHILLES **free-proton** RESDUMP, where `tcrz=+1`,
`tiz=+½` ⇒ `tpinz = 1.5`, and the wave filter `tpin + eps > |tpinz|` admits **only I=3/2 waves**.
`isv` is identically zero on all seven I=3/2 waves (checked directly against `data/achilles/dcc_EW.dat`),
so the audit never evaluates the `0.5*(vec−isv)` form at all.

**Size of the error, measured from the table.** On I=½ waves `isv ≈ (−0.51 … −0.96)·vec`, so
`0.5*(vec−isv)` is **0.75–0.98 × vec**, varying with wave and with (W,Q²). Aggregated over the I=½
waves: `Σ|vec| = 515.2` vs `Σ|0.5(vec−isv)| = 446.0` — **13 % low, and shape-distorted**, so no
constant `_NORM` can absorb it. It sits in the non-Δ I=½ background, exactly where a χ²/ndf ≈ 1.5
nuclear comparison is least sensitive.

**The justifying comment is a false citation.** `dcc/assembly.py:122-125` justifies the form by
citing *"interpolate_amp, lines 585-604"*. `amp_dcc_sl_module.f:585-604` is the `nLsdt` L-S table
setup loop — no amplitude, no isospin rotation. The form was invented, not transcribed.

**Conclusion: (b) — the form was written by analogy, and it is wrong for I=½ waves in BOTH CC and
NC.** The NC exposure is the factor ~14 above. The **CC** exposure is the 13 % shape distortion, and
that is the serious one: under **D5 a CC physics change is a declared stop-the-line event**, and it
would invalidate figs 07/08/09/A1/A2 and require full CC bank regeneration.

**What remains open is not the reading — it is the size of the CC exposure on physical events.**
See "the D2 measurement" in P0 item 1 and G4(3).

### D3 — NC QE channels and Pauli
NC elastic is `ν p → ν p` and `ν n → ν n`. Mirror the CC QE structure (per-species
`SpectralFunction`, `sample_importance`), correct `Z`/`N` target counting, Pauli blocking on as for
CC. **NC elastic on a neutron has no CC analogue in the repo**, so the neutron spectral path gets
its first NC-specific exercise here — gate it explicitly.

### D4 — antineutrino NC
In scope (P9), **off the critical path**. The MicroBooNE flux file is `flux/microboone_numu.dat`
(νμ only), so ν̄ cannot affect Figs 11/12. Recommend landing after the figures.

### D5 — bank regeneration budget
P2/P3 rewrite bank *keys* by I/O only, no physics recompute. If any physics phase changes a **CC**
number (it must not; the gates enforce it), CC banks need full regeneration at 2–3 h/shard with
documented preemption. **Treat any CC physics change as a stop-the-line event** requiring an
explicit owner decision, not an automatic regeneration.

> **D2 has already triggered this clause.** The plan as originally written assumed no CC number
> moves and, on that premise, scheduled P2/P3 to migrate ~15 GB of banks by I/O only. If the D2
> measurement confirms the CC I=½ exposure, those banks are regenerated from scratch anyway and the
> migration is wasted work. **Hence the reordering below: P2 and P3 now depend on G0, not on P1.**

---

## 2. Two faithfulness constraints, verified from ACHILLES source

### No strange form factors in NC QE — ACHILLES structurally cannot use them

- `FormFactor::Values` *does* carry and compute a strange axial term:
  `include/Achilles/FormFactor.hh:122` (`double FA{}, FAs{}, FAP{};`),
  `src/Achilles/FormFactor.cc:92` (`result.FAs = -gans / pow(1.0 + Q2/MA/MA, 2)`) and `:123`
  (`ZExpand(strange_params, z)`), with `Strange Params` read from YAML at `:102`.
- **But `FormFactorInfo::Type` has no strange entry.** The complete enum
  (`include/Achilles/FormFactor.hh:22-48`) is `F1, F2, F1p, F1n, F2p, F2n, FA, FAP, FCoh, FResV,
  FResA, FPiEM, FMecV3, FMecV4, FMecV5, FMecA5, F1Lam, F2Lam, FALam, F1Sigm, F2Sigm, FASigm,
  F1Sig0, F2Sig0, FASig0`. No `FAs`, no `F1s`, no `F2s`.
- Couplings attach only to a `Type`, and `NuclearModel::CouplingsFF` (`src/Achilles/NuclearModel.cc:70-160`)
  dispatches on `Type`. **No case reads `formFactors.FAs`.** A repo-wide grep for `FAs` consumers
  returns only the two computation sites and one zeroing (`FormFactor.hh:280`).

**`FormFactors::FAs` is computed and never consumed.** ADoNIS must omit `F1^s`, `F2^s`, `G_A^s`/ΔS
from NC QE. Including them would make ADoNIS more physically complete and **fail every χ² gate by
construction**, because every gate here is an ACHILLES comparison. Recorded as an explicit, cited,
**test-pinned** omission — not silence.

### No coherent pion production — ACHILLES's `Coherent` is coherent *elastic*

- `Coherent::AllowedStates` (`src/Achilles/NuclearModel.cc:512-523`) sets
  `result.m_hadronic = {{nucleus_pid}, {nucleus_pid}}` — hadronic final state is **the same
  nucleus**. No pion, ever.
- `Coherent::CalcCurrents` (`:483-511`) builds `subcur[i] = (pIn[i]+pOut[i])·ffVal[Type::FCoh]`, a
  pure vector nucleus→nucleus current; `FCoherent::HadronicCurrents` (`src/Achilles/Coherent.cc:12-36`)
  is the same.
- It requires leptonic charge 0 (`NuclearModel.cc:516-519`), i.e. NC-only — which is why the `FCoh`
  couplings exist for `{carbon,23}` / `{argon,23}` at `LeptonicCurrent.cc:117-119`. Those are
  **CEvNS-like coherent elastic**, easily mis-read as evidence for coherent π⁰.

**ACHILLES has no coherent pion production.** Per the owner's conditional ("implement iff ACHILLES
has it"), coherent NC π⁰ is out — on verified faithfulness grounds, not deferred on cost. It also
cannot contaminate the signal: no pion in the final state means it cannot enter an NC1π⁰ topology.
**Consequence: the forward `cos θ_π⁰ → 1` bins of Fig 12 are a non-issue** — with no data overlay
and neither generator having coherent, there is nothing for it to be missing against.

---

## 3. Phases and gates

Serial spine: **P−1 → {P0,P1} → P2 → P3 → {P4,P5} → P6 → P7 → P10.** P9 is off the critical path.
The renames are deliberately placed **before all NC code**, so NC is written once against the final
schema — but **after G0**, because G0's outcome decides whether the banks they migrate survive.

```
P-1 D2 measurement ─► G0 ─┬─► P2 rename probe ─► P3 rename k_lep ─┬─► P4 amplitude ─┐
P0 spec ──────────────────┘                                       ├─► P5 NC lepton ─┼─► P6 NC QE ─► P7 bank ─┬─► P10 figures
P1 fail-loud ─────────────┘                                       │                 │                        │
                                                                  └─────────────────┴──► P9 anti-ν (off)     P8 selection ─┘
```

Parallel: P0∥P1; P4∥P5 (**writing** G5's analytic test is parallel; **running** it needs P4);
P7-generation∥P8; P9∥everything post-P6.

**Ordering changed after review.** Originally `P1 → P2 → P3` with P0 merely parallel, which let the
12-manifest rewrite and the 15 GB `.npz` migration start before D2 was resolved. Gating P2/P3 on G0
costs ~2 idle days; not gating them costs the migration **plus** a full CC regeneration cycle if D2
lands as (b). P1 stays parallel — it is cheap and correct either way — but it no longer licenses
starting P2.

### P−1 — the D2 measurement (1–2 days, before anything else)

The **reading** is settled (§D2). What is not settled is **how much physical CC output moves**. The
deliverable is one number, and it decides whether this is a ~30-day NC project or a ~50-day
CC-repair-plus-NC project.

**Test:** compare per-event `amps2` against an instrumented ACHILLES RESDUMP on an
**I=½-accessible** channel — `n → pπ⁻` or `n → nπ⁰` (`tpinz = ½`), **not** the free proton the
existing 1.0011 audit uses — over a W range spanning the s11/p11 region, for CC. Then repeat with
`src_block` switched to the literal Fortran transcription. One of the two is flat at 1.000; the plan
predicts it is the literal form.

**Why a test at all, when the source reading already answers it:** the reading tells us what ACHILLES
computes; it does not tell us how far ADoNIS's validated CC figures currently are from it. Changing
a validated CC current on a reading alone — against a repo whose doctrine is bit-exactness — is
exactly the move this plan exists to prevent. The measurement is the arbiter and the regression gate,
not the decision.

---

### P0 — Spec from ACHILLES source. No code.

*Precedent: the EM extension began with a logbook-only commit naming the gates up front.*

1. **Record the D2 measurement** from P−1 — the reading is done (§D2); this item is now the write-up
   of the numerical result and the decision it forces (CC repair in scope, or not).
2. NC RES coupling table, one ACHILLES citation per entry: `tcrz=0` (`amp_dcc_sl_module.f:285`);
   `vfac=1-2sw2`, `vvfac(±1)=∓2sw2`, **`sw2=0.2312` hardcoded in ACHILLES** (`:288-294`, `sw2` at
   `:291`) — do **not** replace with `C.sin2w=0.23129`; axial active for `mode<10` with no `vfac`
   (`:802`, `:837`);
   pion pole `mode.gt.0` ⇒ CC only (`:844`); `isign=-1` for `mode==10/11` only, so NC keeps `+1`
   (`:644`); **no `sqrt(2)` isospin factor for NC** (`:275`).
3. Derive `_NORM_NC` symbolically from `LeptonicCurrent.cc:93-96`
   (`FResV = FResA = coupl3 = ee·i/(2·cw·sw)`) plus `:275`, and **record the predicted ratio
   `_NORM_NC/_NORM_EM = (2·sw·cw)² ≈ 0.7113` in advance of any measurement.**
4. NC leptonic coupling in ACHILLES's exact floating-point form:
   `coupl_left = (cw*ee*i)/(2*sw) + (ee*i*sw)/(2*cw)`, `coupl_right = 0`, `M=MZ`, `Γ=GAMZ`
   (`LeptonicCurrent.cc:31-36`). Algebraically `ee·i/(2·sw·cw)`, but write ACHILLES's form.
5. NC QE coupling table from `LeptonicCurrent.cc:93-115`, including **D1 both branches** (quirk and
   correct) with the 1.0396 factor computed, and the strange-omission constraint with its enum citation.
6. Build and run **two** ACHILLES NC cards, free-nucleon, no cascade (`Leptons: [14,[14]]`,
   `1H`/`1N`): one RES, one QE. Record the NC `proc` IDs — **and where they are consumed**:
   `SignalDef.ref_proc` (`config.py:214`) documents the CC values `200=QE, 401/402=RES`, and the
   QE/RES breakdown in `plotting.py:207-210` keys on the `chan` column derived from them.
7. Record that ACHILLES source lives at `/sdf/data/neutrino/cjesus/ADoNIS/software/Achilles-src/`,
   and that **`data/achilles/dcc_EW.dat` is absent from this worktree** — `adonis/io.py:22-27`
   requires `ACHILLES_DATA`, so any P−1/P0/P4 numerical work must set it. Document the path.
8. **Close the G5(1) isospin factor.** The propagator gate's target is written below as
   `(M_W²/M_Z²)²·|g_NC/g_CC|²·(isospin factor)` with the last term unspecified — a gate with a free
   multiplicative constant is precisely the "bridge constant" this plan forbids. For RES the
   leptonic+hadronic product is `cw⁴ · 1/(2cw²) · 2/(cw²·Vud²) = 1/Vud²`. **Write the closed form
   here, before P5.**

**G0 — spec review passes.** (a) every NC coupling, RES and QE, has an ACHILLES `file:line`;
(b) **the D2 measurement has been run and its outcome recorded** — this is a hard resolve, not a
"named test"; the original wording allowed the single most load-bearing unknown to pass the gate
unresolved; (c) `_NORM_NC` derived with its predicted ratio stated **before** measurement; (d) D1's
two branches and the strange-omission constraint written down with citations; (e) two ACHILLES NC
hepmc on disk with `proc` IDs; (f) the G5(1) isospin factor written in closed form.
**Owner sign-off before any code.**
*Effort 2–3 d. Depends: P−1. Parallel: P1. **Blocks: P2, P3.***

### P1 — Fail-loud probe registry

`exclusive_amps2_batch` currently does `_mode = 10 if probe == "EM" else 1` (`dcc/current.py:208-210`
— `_mode`, `_lep_kind` and `_norm` all `else`-branch to CC), so `probe="NC"` would **silently produce
CC**. Replace with a single registry (**new `adonis/channels/probes.py`**) that **raises on any
unknown probe**; `"NC"` is deliberately absent at this phase.

Sites, corrected after review:

- `dcc/current.py:208-210` — silent CC fallback. ✔
- `currents/matrix_element.py:53` (`lep_kind = "EM" if probe == "EM" else "CC_nu"`) — silent, with
  `spin_avg` an **explicit** registry field, never a fall-through. ✔
- `dcc/channel.py:117` (`is_em = (current == "EM")`) — silent. ✔
- **`dcc/assembly.py:126-133` — a fourth silent site the plan originally missed, and the one that
  matters most.** `if mode < 10:` catches `mode = −1`, so the scalar `build_zmtx` applies the **CC**
  isospin treatment to NC with no `VFAC`, no `VVFAC`, no `sw2`. Its docstring at `:79` already
  advertises `-1 (NC)`. `HadronStructure._full_grid` (`dcc/structure.py:98-113`) calls exactly this
  path and `NC_CHANNELS` (`structure.py:65-70`) feeds it — so **`HadronStructure(channels=NC_CHANNELS)`
  runs today and returns CC numbers without error.** P4 framed this as a DRY problem; it is a live
  silent-fallback and belongs here.
- ~~`currents/dirac.py:107-115`~~ — **not a fallback site.** It already `raise`s at `:118-119`.
  Removed from this phase.

**G1.** (1) CC and EM banks **byte-identical** at fixed seed; (2) tests asserting `probe="NC"` and
`probe="garbage"` both raise **at all four sites**, including `assembly.build_zmtx(mode=-1)` —
**written now, updated (never deleted) when NC lands in P4**; (3) fast suite stays fast.
*Effort 0.5–1 d. Depends: nothing. Fully parallel with P−1 and P0. Does **not** license starting P2.*

### P2 — Rename probe `weak`→`CC`; unify the `EM`/`ee` manifest split

No alias, no back-compat mapping, no documented wart.

Code: `workflow/config.py` `PROBES` (:97), `_PROBE_BEAMS` (:105), default (:118), the
`if self.probe in ("weak","EM")` check (:155), `bank_prefix`; `generate_bank.py:129`
(`m_extra = dict(probe="weak", ...)`); every consumer keying on `"weak"`/`"ee"`. **Correction:** the
plan cited `generate_bank.py:79` as the probe binary; `:79` is `n_neutron/n_proton`. The binary is
`EM = (cfg.probe == "EM")` at **`:69`**, branching at `:86` / `:108`, with `_generate_hardvertex` at
`:60`. That is what folds into the P1 registry.

**Existing inconsistency fixed in the same pass:** `GenConfig.probe` is `"EM"` but the manifest
records `"ee"`. **Set the manifest field from `cfg.probe`** rather than a literal, so the two can
never diverge again — that is the structural fix, not a string edit.

Config: all **12** `configs/banks/*.yaml`. `load_gen_config` rejects unknown keys, so a missed
rename fails loudly at load.

Data: **12 merged manifests** (3 `weak`, 3 `ee`, 6 `hadron`) **plus every `part_*/manifest.json`**.
JSON only. Write it as a checked-in idempotent script, not an ad-hoc shell loop.

**G2.** (1) **grep gate** in CI — zero `"weak"`/`"ee"` as a probe value in `adonis/`, `configs/`,
`analysis/`, `tests/`, or any manifest; (2) **permanent manifest/config agreement test**:
`manifest["probe"] == cfg.probe` for every probe — this makes the `ee` bug class unrepeatable;
(3) every `.npz` array byte-identical (only `manifest.json` changes); (4) `fig07/08/09/A1/A2`
reproduce with **identical histograms and χ²/ndf**; (5) migration is idempotent.
*Effort 1.5–2 d. Depends: **G0** and P1 — G0 because a (b) outcome regenerates these banks anyway.
Blocks: P3.*

### P3 — `k_mu` / `k_e` / `k_le` → `k_lep`, with bank data migration

The schema currently lies about what the particle is, and that lie is exactly what would make an NC
bank unsafe: a field named `k_mu` holding a neutrino produces wrong numbers nobody questions.

Code: ~**127** call sites — `k_mu` 96, `k_e` 19, `k_le` 12 (`k_lep` already appears 5×) across
`adonis/`, `analysis/`, `tests/`. Principal: `generate_bank.py` (`_accept_lepton(kkey=...)`, the
`save.update(...)` writers), `channels/{qe,res,ee,res_ee,free_proton}.py`,
`reweight/{bank_plot,tune,reweight_model,amps2_records}.py`, `workflow/{selection,analyze}.py`,
`analysis/paper/{physical_fit,info_content,physfit/physical_fit_run}.py`.

Data: the array key lives inside every chunk `.npz` (~15 GB). **Read, rewrite with the renamed key,
re-save** — I/O only, no physics recompute, order ~1 hour. **Not a loader-side shim**, which would
leave the on-disk schema permanently lying.

**G3 — the gate standing between this plan and 15 GB of silently corrupted banks.**
(1) per-chunk **byte-comparison of every other array** before/after — all `fs_*`, `f_*`, `n_*`,
`ks_*`, `prim_fate`, `nsc_prim`, `channel`, `w0`, `hv_*`, `res_*` byte-identical, and the lepton
array's *contents* byte-identical too; (2) **key-set gate** — post-migration key set equals
pre-migration with exactly one substitution; (3) `fig07` **and** `fig08` reproduce identically
(`fig07` exercises the TKI/STV path through `_obs(mu,...)`, `fig08` the pion path — between them
every lepton consumer is live); (4) grep gate, CI-enforced; (5) restartable, atomic per chunk
(temp + rename); (6) `pool_fsi_reweight(record,1,1)==1` bit-exact on a migrated bank.
*Effort 2–3 d + ~1 h migration. Depends: P2 (and transitively **G0**). Blocks: all NC driver phases.*

### P4 — NC amplitude layer, and elimination of the duplicate `build_zmtx`

- `dcc/differential.py` — the `mode<=-1` branch (`:218-227`) is **rewritten from the P0 spec**, not
  adjusted. If the re-derivation reproduces the legacy lines exactly, say so in the commit message
  as an **earned** cross-check with Fortran line numbers; if not, delete them. Keep `sw2=0.2312`
  with a `:291` citation and a comment on why it is **not** `C.sin2w`.
- `dcc/assembly.py` — **unify, do not duplicate and do not raise.** The real defect is that two
  implementations of `build_zmtx` exist; this repo has an explicit doctrine against exactly that
  ("the physics written once … no second copy to drift"). Adding an NC branch creates a second NC
  implementation guaranteed to drift; raising leaves a permanently crippled path. **Make
  `assembly.build_zmtx` a thin wrapper over `differential.build_zmtx_batched`** (batch-of-one or
  vmapped), deleting the duplicated body; `tests/test_build_zmtx_equiv.py` becomes the migration
  proof. `HadronStructure._full_grid` and `exclusive_H` then get NC for free, correctly.
  *Verify first* that the scalar path has no hard requirement the batched one cannot meet (a
  `jax.grad`-through-scalar pattern or a tracing constraint inside `_full_grid`'s `vmap`); if it
  does, implement NC properly in `assembly.py` (**not** raise) plus a standing equivalence test over
  CC, EM **and** NC. Either way the docstring lie at `:79` dies.
- `dcc/structure.py` — `NC_CHANNELS` (`:65-70`) re-derived against `currents_pi_dcc.f90`.
  **Also fix `mult`:** all four NC channels carry `mult=1.0`, copied from `EM_CHANNELS`. For argon
  these must be **Z=18 / N=22**. The plan mandates Z/N counting in P6/P7 for the drivers but omitted
  it here, where the channel table actually lives.
  *Verified favourable:* `currents_pi_dcc.f90:4,71-101` maps all four NC channels to `DCC_mode = −1`
  via `id_p=1, id_n=−1, id_pip=1, id_pi0=0, id_pim=−1` — `p→pπ⁰` (sumIDS 2), `n→nπ⁰` (−2), `p→nπ⁺`
  (1), `n→pπ⁻` (−1). **ν and ν̄ NC both give `mode=−1`**, so P9's hadronic side is identical and only
  `leptonic.py`'s `anti` differs. That de-risks P9 materially.

**G4.** (1) CC and EM `build_zmtx` **bit-identical** for a fixed `(vec,isv,axial,W,Q²,itiz)` batch;
extend `test_build_zmtx_equiv.py` (today `mode=1` only) to `mode ∈ {1,10,-1}`; (2) **differential
gates**, each separate: `axial_scale` **does** change NC (NC has an axial, EM does not); the
pion-pole knob has **exactly zero** effect on NC; `sw2→0` reduces the NC vector to pure isovector
and `sw2→0.5` makes `VFAC` vanish (two analytic limits, no oracle); proton vs neutron differ **only**
via `VVFAC` sign and the I=½ isoscalar block. **None of these discriminates D2** — all five hold
under both the legacy and the literal form. That is why (3) is separate and explicit:
(3) **the P−1 D2 measurement, re-run against the final `build_zmtx` and recorded** — the
I=½-accessible RESDUMP comparison defined in P−1, flat at 1.000 or the phase does not close.
The original wording ("the D2 discriminating test, run and recorded") named a test that was never
defined anywhere in the plan; (4) `HadronStructure`/`exclusive_H` unchanged for CC and EM after
unification.

> **Deliberate blind spot:** G4 is shape-and-structure only and is **blind to overall scale by
> construction**. P5 owns scale. A green G4 must never be reported as "NC works".

*Effort 2–3 d. Depends: P0, P3. Parallel: P5's leptonic half.*

### P5 — NC leptonic current + `_NORM_NC` (RES absolute normalisation)

- `currents/leptonic.py:33-39` — add `kind == "NC_nu"` returning ACHILLES's exact form
  `((cw*ee*i)/(2*sw) + (ee*i*sw)/(2*cw), 0, C.MZ, C.GAMZ, True)`. The existing
  `raise ValueError(kind)` already fails loudly — preserve it.
- `dcc/current.py:47-55` — `_FRESV_NC = C.ee/(2*C.sw*C.cw)`,
  `_NORM_NC = 2*(2π)/(_FRESV_NC² · (2·norm_m_N())²)`, derivation in a comment citing
  `LeptonicCurrent.cc:93-96` and `amp_dcc_sl_module.f:275`. Register
  `"NC" → (mode=-1, "NC_nu", _NORM_NC, spin_avg=0.5)` in the P1 registry.
- `currents/matrix_element.py` — NC branch with **explicit** `spin_avg = 0.5`.

- **NC free-nucleon generator.** G5(2) and G6(2) are the declared mitigation for the biggest risk,
  and **no phase produced the ADoNIS side of them.** P0 item 6 checks in the two *ACHILLES* cards;
  the ADoNIS free-nucleon machinery is CC-only — `channels/free_proton.py:14,24,30` imports `M_MU`,
  builds `Smin = (M_MU + m_Nf + m_pi)**2` and calls `_cc0pi_obs`; the σ(E_ν) scans at
  `channels/res.py:509,547` sit on the `M_MU`-hardcoded `_sample_3body`. **An NC free-nucleon
  RES + QE generator is an explicit deliverable of P5/P6**, or G5 and G6 cannot close. +2–3 d.
- **Flux threshold.** `flux/spectrum.py:88-92` hardcodes `Smin = (M_P + M_MU)**2`, a 0.112 GeV floor
  on E_ν, consumed by `channels/qe.py:37,108` and `channels/res.py:509,547`. **NC QE (`ν p → ν p`)
  has no threshold**, so the floor silently truncates the flux — small for Figs 11/12, but it
  directly biases the low-energy end of the G5/G6 absolute σ(E_ν) gates. Parametrise it.

**G5 — absolute normalisation, RES. All four required.**
(1) **analytic propagator gate** (no oracle), ≲1e-3 — the `e3319a5` insurance — with the isospin
factor in the **closed form fixed at G0(f)**, `1/Vud²` for RES, not left open;
(2) **absolute free-nucleon oracle gate**, ≥5 energies 0.7–2.2 GeV, **mean and spread reported
separately**, band set in advance (suggest mean ≤2 %, spread ≤1 %), **bridge constants forbidden**;
(3) **pre-registration check** — measured `_NORM_NC/_NORM_EM` vs the 0.7113 predicted in G0;
disagreement means the derivation is wrong *even if* the ratio passes, catching a derivation error
masked by a compensating implementation error; (4) per-channel ratios across the 4 NC RES channels
as a **secondary** check only, never the closer.
*Effort 1.5 d + 0.5 d oracle + 2–3 d free-nucleon generator. Depends: P0, P1, P3. Parallel: P4 —
but note G5(1) **compares `amps2_NC/amps2_CC` and therefore cannot run until P4 lands**. Write the
test early (that is the point); do not schedule it to pass early.*

### P6 — NC QE: `dirac.py` branch, form-factor recombination, `qe_nc.py`, and the D1 switch

- `currents/dirac.py:81-149` — add the `probe=="NC"` branch. The vertex assembly
  (`F1·γ + iF2/(2m)·σ·q + FA·γγ₅ + FAP·(q/m)γ₅`) is already probe-agnostic and reusable; only the
  coupling and FF recombination change. Add `_COUPL_NC` from `LeptonicCurrent.cc:94-96`, **with the
  D1 switch at the coupling definition** (not smeared through call sites):

  ```
  coupl1 = ee·i/(4·sin2w·cw) · (0.5 − 2·sin2w)     if quirk    # ACHILLES verbatim
  coupl1 = ee·i/(2·sw   ·cw) · (0.5 − 2·sin2w)     otherwise   # correct physics
  coupl2 = ee·i/(4·sw   ·cw)                                   # both branches
  ```

  > **Corrected after review.** The plan first wrote this as a single expression
  > `ee·i/(4·X·cw)·(0.5−2sin2w)` with `X = sin2w if quirk else sw`. **That is wrong by a factor 2**:
  > `X = sw` gives `ee·i/(4·sw·cw)·(0.5−2s²)`, half the correct coupling, making the ratio to
  > ACHILLES `1/sw = 2.0793` instead of 1.0396 — so the "correct physics" default would have been
  > half the right NC QE vector coupling and **G6(3) could never have closed**. The single-expression
  > form needs `X = sw/2`; write the two branches longhand instead, as above, so the substitution
  > cannot be misread again.

  Recombination, transcribed from `LeptonicCurrent.cc:97-112` — ACHILLES expresses the
  isoscalar/isovector recombination as a **per-nucleon coupling dictionary**, so ADoNIS should too
  rather than inventing an isospin decomposition (`dirac.py` already takes `is_proton`):
  - proton: `F1 → coupl1·F1p − coupl2·F1n`, `F2 → coupl1·F2p − coupl2·F2n`, `FA → +coupl2·FA`
  - neutron: `F1 → coupl1·F1n − coupl2·F1p`, `F2 → coupl1·F2n − coupl2·F2p`, `FA → −coupl2·FA`
- `currents/form_factors.py` — **no new form factors.** `nucleon_ff` already returns exactly what
  ACHILLES's NC QE consumes. Add a comment **and a test** recording the strange-omission constraint
  with the enum citation, so a future contributor cannot "complete the physics" and silently break
  every gate.
- **New `adonis/channels/qe_nc.py`** — `ν N → ν N`. Do **not** reuse `qe.py::sample_importance`: it
  hardcodes `M_MU` in the two-body split (`s2 = M_MU**2`) and the `n→μ⁻p` channel. **Watch the
  `m_lep=0` degeneracies** — this is the exact path that produced the `~1e17 MeV` blow-up fixed in
  the recent QE sanitisation arc. Sanitise explicitly and gate on it.
- `workflow/config.py` — `achilles_coupl1_quirk: bool = False`, threaded to the current and
  **written into the manifest**.

**G6 — absolute normalisation, QE. NC QE gets its own; it does not inherit RES's.**
(1) analytic gate at `Q² ≪ M_Z²`, ≲1e-3; (2) absolute free-nucleon oracle gate, ≥5 energies, mean
and spread separate; (3) **the D1 two-sided gate** — with `achilles_coupl1_quirk=True` ADoNIS
matches ACHILLES within the G6.2 band; with `False` it differs by **exactly 1.0396 on the F1/F2
terms of BOTH nucleons and by nothing else** (`FA`/`FAP` unchanged, `coupl3`/RES unchanged). Both
branches pinned by test, with the ACHILLES line cited, so neither can be "fixed" silently.
**Corrected after review:** this gate previously read "on the proton F1/F2 term ... neutron path
unchanged", which asserts something false — `LeptonicCurrent.cc:110-111` applies `coupl1` to `F1n`
and `F2n` in the neutron dictionary. As originally written the gate could not close;
(4) **strange-omission test**, citing the ACHILLES enum; (5) **proton/neutron differential** — the
neutron path has no CC analogue, exercise it explicitly; (6) **non-degeneracy** — no event with
`|k_lep| > E_ν` or non-finite weight, degenerate-draw fraction below a stated threshold; (7) **CC QE
bit-identical** after the `dirac.py` edit.
*Effort 3–4 d. Depends: P0, P1, P3, P5. The largest new-physics phase.*

### P7 — NC RES driver, workflow integration, banks

- **New `adonis/channels/res_nc.py`**, modelled on `res_ee.py` — a near-perfect template: it already
  parametrises the outgoing lepton mass in `_sample_3body_ee(..., m_lep, u)`, already carries a
  4-channel list and its own `SPIN_AVG`, and already passes `tcrz=0.0`. Differences: `SpectrumFlux`
  beam; `m_lep=0`; `NC_CHANNELS`; `SPIN_AVG=0.5`; `probe="NC"`; per-species spectral functions with
  explicit `Z`/`N`. **Do not reuse `res.py::_sample_3body`** — it hardcodes `M_MU` in ~15 places
  (`:87,98,104,108,164,167,224,237,241,243,254,334,362,394,403,409,414`) — **17** sites, not the
  ~15 originally stated; `:167` was missed. Same `m_lep=0` caution as P6.
- `workflow/config.py` — `_PROBE_BEAMS["NC"]=("spectrum",)`; **raise if `probe=="NC"` and
  `theta_acc != (0,180)`** (a lepton-angle cut on an invisible neutrino is meaningless and would
  silently bias the sample). `channels` may be `[qe,res]` — NC QE exists.
- `workflow/generate_bank.py` — generalise `_generate_hardvertex` (`:60`) from its hard binary
  (`EM = (cfg.probe == "EM")` at `:69`, branching `:86`/`:108`) into the registry lookup
  `{"EM":(ee,res_ee), "CC":(qe,res), "NC":(qe_nc,res_nc)}`. Write the outgoing neutrino into **`k_lep`**.
- **`reweight/amps2_records.py` — a gap the plan missed entirely, and one no gate would catch.**
  Every hard-vertex reweight record is built against the **CC** matrix element with no probe
  plumbing: `:33-34,47-49,61` call `me_cross_section(...)` with no `probe=` (defaults `"CC"`,
  `matrix_element.py:43`), and `:81-83,108-110,131-133` call `dcc.exclusive_amps2_batch(...)` with
  no `probe=` and no `tcrz=` (defaults `probe="CC"`, `tcrz=1.0`, `current.py:201`). `build_hv_sf`
  (`reweight/reweight_model.py:31-48`) is the sole caller and `generate_bank.py:181` invokes it for
  **every non-EM bank**. On an NC bank the `hv_*` records — `qe_ma`, `qe_vec`, `qe_gmp/gmn/gep/gen`,
  `res_ma`, `res_pp`, `res_delta`, `res_pw` — would be CC at `tcrz=1`, which for channels like
  `p→pπ⁰` also changes the isospin CG. **Nominal weights survive** (records are normalised to 1 at
  nominal), so the bank looks fine and **every knob response, every gradient and every tune on an NC
  bank would silently be CC**. Thread `probe`/`tcrz` through, and gate it — see G7(9).
- **Fix the `hv_*` latent bug properly.** `generate_bank.py` writes `hv_*` records **only inside
  `if do_qe and do_res:`**, so **any single-channel bank of any probe** silently loses its
  hard-vertex reweight records — a live latent bug for EM too, not just an NC inconvenience.
  Restructure to emit `hv_*` for whichever channels are present, with `_idma` identity padding
  applied **per channel**. Keep a loud guard in `bank_plot.load_bank` for genuinely absent `hv_*`
  (clear message, not a bare `KeyError`).
- **New `configs/banks/nc_uBooNE_Ar.yaml`** — `probe: NC, beam: spectrum, flux: microboone,
  material: Ar, channels: [qe, res], fsi: true, theta_acc: [0,180]`, **`achilles_coupl1_quirk: true`**
  (this bank exists to be compared against ACHILLES).
- **New ACHILLES card** — copy the CC MicroBooNE card, set `Processes: - Leptons: [14, [14]]`, and
  repoint the `HardCuts` block (`:122-125`), which reads `Type: AngleTheta, PIDs: 13, range: [0,180]`
  — a cut on a particle that does not exist in an NC event. **Downgraded after review:** the card
  already has `Main: HardCuts: false` at `:3` and the range is the full `[0,180]`, so this is
  hygiene, not a correctness risk. Do it; stop listing it as a landmine.
- **Oracle prefix-routing landmine.** `run_achilles.py::_RULES` routes by card-name **prefix**.
  `run_MicroBooNE_Ar_nc_fsi` does **not** match the existing `run_MicroBooNE_Ar_fsi` prefix, so it
  would fall through to the amd64 `:oracle` default and **SIGSEGV under emulation with the cascade
  on**. Name it **`run_MicroBooNE_Ar_fsi_nc.yml`** — zero code change. This is the `c9ceb2a` bug
  class. Add a test asserting every card resolves to its intended image.
- `oracle/extract.py` — extend `fs_rich` to capture a status-1 neutrino into `lep` when no μ/e is
  present (`:303-304` captures only `pid==MU or pid==ELEC`, so NC `lep` is always the zero vector),
  and **add a new `n_nonpion_meson` field** (see P8).
- **NC `proc` IDs** from P0 item 6 wired into `SignalDef.ref_proc` (`config.py:214`, documented for
  the CC values `200`/`401`/`402`) and the `chan` column that `plotting.py:207-210` splits QE/RES on.
  Recorded in P0, consumed nowhere — closed here.

**G7 — bank + oracle agreement, before any figure.**
(1) **CC/EM banks byte-identical** after the dispatcher generalisation and the `hv_*` restructure —
non-negotiable, this touches the shared generator; (2) **`hv_*` restructure proof** — a res-only and
a qe-only bank of an **existing** probe now carry correct `hv_*`, and the both-channels case is
byte-identical to before; (3) **the NC path demonstrably fires** (the `aebe605` pattern — prove no
change where none intended **and** prove the new path is live): `channel` contains both QE and RES,
`prim_pi_pid` shows the expected π⁰/π⁺/π⁻ mix, a deliberately narrow acceptance drops the expected
count; (4) **nominal identity exact** — `pool_fsi_reweight(record,1,1)==1` bit-exact on the NC
kind-1 FSI record (π⁰ is already transported by the cascade, so no new FSI capability is needed, but
it must still be proven on this sample); (5) **absolute σ_NC on Ar** — ADoNIS bank total vs ACHILLES
NC `fs_rich` total in nb, the nuclear-level repeat of G5/G6, catching target counting, spectral
function and Pauli errors invisible on a free nucleon; (6) **manifest asserts the D1 flag**;
(7) oracle routing test; (8) GPU/CPU parity smoke before mass submission (**procedural, not
pass/fail — state the tolerance or drop it**); (9) **`hv_*` knob-response gate, new** — perturb
`res_ma` on the NC bank and assert the response differs from the CC bank's by the predicted NC/CC
amplitude ratio. Nominal-weight identity **cannot** see the `amps2_records.py` defect above; only a
knob response can.

*Effort 3–4 d code. **Generation:** measured CC QE+RES rate is ~400–700 s/chunk, ~2–3 h per 20-chunk
shard, with documented repeated preemption on `turing` (hence `nu_T2K_C`'s
`part_*`/`part_top_*`/`part_top2_*` re-merges). Budget **4–6 calendar days** for a 10–20 M-event NC
Ar bank including preemption and re-sharding. Depends: P4, P5, P6. Parallel: P8.*

### P8 — π⁰ signal selection and observables

A **parallel** selection path, on correctness grounds: `_obs` computes `dpt, dalphat, dphit, pn,
dptt`, all of which take `mu` as a required argument and are **physically undefined** for NC.
Threading a "lepton optional" flag through them would create observables that are silently
meaningless rather than absent.

- `workflow/config.py::SignalDef` — add `pion_id: "pi0"`. **Fix the latent lie:** `"anypi"` passes
  validation but `selection.py` never branches on it, so `"anypi"` and `"pip"` are byte-identical
  today — **implement it properly** (~3 lines, `pion_counts` already returns all three). Likewise
  the dead fields (`require_proton`, `count_recoil_neutron`, `target`, `W_conv`, `n_ejected`,
  `eject_thresh`, `ref_proc`): implement, or **raise if set**. Config keys that silently do nothing
  are the same class of defect as the `k_mu` name lie.
- `workflow/selection.py` — `bank_signal_nc` / `oracle_signal_nc` that **bypass `_mu_pass`
  entirely**. NC observables: `p_π⁰`, `cos θ_π⁰`, proton multiplicity.
- `reweight/bank_plot.py` — `single_pi0(B)` mirroring `single_pip(B)` (`:137`); `pion_counts(B)`
  already returns `(n_π⁺, n_π⁰, n_π⁻)`.
- `kinematics.py` — `OBSERVABLES` (`:248-253`) **already carries `ppi_mag` and `ppi_costheta_lab`**.
  This is a naming decision, not new formulas: either reuse them or state why NC needs its own keys.
- **The `n_other_meson` landmine.** `extract.py:306` does `n_other += (pid != PIP)`, so
  `n_other_meson` **counts π⁰ and π⁻ as "other mesons"**; and `selection.py:95` computes
  `n_meson = (pipid != 0).sum(1) + n_other`, **double-counting** every non-π⁺ pion. Harmless for
  CC0π (both terms zero on signal). **Scope corrected after review:** the existing `pip` branch
  (`selection.py:105-113`) does not use `n_other_meson` at all, so the "vetoes 100 % of a π⁰ signal"
  claim holds **only if the new NC branch reuses it as a veto** — which is exactly the mistake to
  avoid. Use the **new** `n_nonpion_meson` field from P7; do not reinterpret the existing one, so no
  CC number moves.

**G8.** (1) **CC selection byte-identical** — `test_workflow_selection.py` plus full `fig07`/`fig08`
regeneration, unchanged histograms; (2) **signal-definition cross-check** — ADoNIS `bank_signal_nc`
and ACHILLES `oracle_signal_nc` agree on **selection efficiency** over a common truth-level NC1π⁰
sample; this catches "the two sides implement subtly different signal definitions", which no χ² can
distinguish from a physics disagreement; (3) **explicit veto test** — a synthetic 1π⁰ + 1η event is
vetoed correctly, proving `n_other_meson` was not naively reused; (4) **no-lepton assertion** —
identical output on a bank with `k_lep` deliberately zeroed, proving the NC path never reads the
lepton; (5) **`anypi` differential** — `"anypi"` and `"pip"` now give **different** results on a bank
containing π⁰/π⁻, proving the fix fires.
*Effort 2–3 d. Depends: P7's schema (not its big bank — 50 k events suffice). Parallel: P7 generation.*

### P9 — Antineutrino NC (off the critical path)

An implemented-but-ungated path is worse than a raise — but that justifies *gating*, not *omitting*.
So: implement and gate. `leptonic.py` `anti` handling for `"NC_nu"` (the flag exists at `:42` but no
production caller passes `anti=True`, so it has never been exercised for **any** probe);
`_PROBE_BEAMS`/flux plumbing for a ν̄ spectrum; a ν̄ free-nucleon ACHILLES card.

**G9.** The same **absolute** free-nucleon oracle gate as G5/G6, for ν̄ NC RES and ν̄ NC QE, mean and
spread reported. Plus a CC ν̄ cross-check if a ν̄ CC oracle is cheap — that would exercise `anti` for
CC too, currently untested for every probe.
*Effort 1.5–2 d + oracle. Depends: P5, P6. Parallel: everything post-P6.*

### P10 — Figures 11 and 12

**No data overlay.** ADoNIS-vs-ACHILLES, like the rest of Section 1.

- **New `configs/analysis/fig11_uboone_nc1pi0_doublediff.yaml`** and
  **`fig12_uboone_nc1pi0_cospi.yaml`** — "add a figure = drop a YAML, no code".
- `workflow/plotting.py` — the double-differential is the real work. `make_figure` is **1×N only**,
  and `figA1` already renders 14.3 in wide for want of row wrapping (a known open issue). A
  (cos θ, p) double-differential is naturally a **grid of p-slices**, so this is the first figure
  that genuinely requires it. **Doing it here also fixes `figA1`** — good leverage, not free.
- **No `data_overlay.py` work.**

**G10.** (1) per-observable **χ²/ndf** and integral ratio ACH/ADO; band 0.3–2.1 = validated, anything
above **explained or explicitly flagged**, never quietly cropped; (2) **honest-χ² discipline** — any
`shadow_frac`/`ratio_xmax` justified by bin populations, per the `load_scan()` doctrine that it
"refuses to hand back a figure that would quietly lie"; (3) **statistical errors** `sqrt(Σw²)`
(binomial is only for the constant-weight geometric beam σ); (4) **layout** — no figure wider than
the page, row wrapping verified against `figA1` too; (5) **faithfulness footnotes** in the
caption/methods: no strange FFs (matching ACHILLES), no coherent π⁰ (neither generator has it), and
the D1 switch state of the bank used.
*Effort 3–4 d (~1.5 of it plotting). Depends: P7, P8.*

---

## 4. Effort

**The estimate branches on D2, and the plan should state the branch rather than average over it.**

The original ~26–34 person-days / 6–8 calendar weeks was defensible **for the work the plan
enumerated**. It did not cover what review found:

| Item | Added |
|---|---|
| D2 fallout — CC I=½ re-derivation, re-validation of figs 07/08/09/A1/A2, CC bank regeneration | 8–15 pd + weeks of `turing` time |
| `amps2_records.py` probe plumbing + the G7(9) knob-response gate | 2–3 pd |
| NC free-nucleon generator (RES + QE) for G5/G6 | 2–3 pd |
| Flux threshold, `proc`-ID mapping, `NC_CHANNELS` `mult` | 1 pd |

- **If D2 confirms the CC exposure (expected): 40–55 person-days, 10–14 calendar weeks.**
- **If it somehow does not: 30–36 person-days, 7–9 calendar weeks.**

Dominated by P7 generation + preemption, P6 (largest new physics), the two rename/migration phases,
and — in the expected branch — the CC repair and regeneration.

---

## 5. First week

1. **Days 1–2 — P−1, the D2 measurement.** Now the first task, not the third. Instrument an ACHILLES
   RESDUMP on an **I=½-accessible** CC channel (`n→pπ⁻` / `n→nπ⁰`, `tpinz=½`) across the s11/p11
   region and compare per-event `amps2` against both the legacy `0.5*(vec−isv)` form and the literal
   Fortran transcription. **Its outcome decides whether this is a ~30-day or a ~50-day project**, so
   nothing else starts until it lands.
2. **Day 1, parallel — build and run the two ACHILLES free-nucleon NC cards** (RES and QE). The
   long-pole **external** dependency (container, possibly a fork build); produces the artifacts G5
   and G6 cannot close without. Check both into `configs/achilles/`.
3. **Day 1, parallel — P1, complete.** Kill the silent CC fallback **at all four sites**, including
   `assembly.py:126-133`. ~0.5 d, bit-exactness-gated, independent of every physics decision. Until
   it lands, every experiment is one typo from silently measuring CC — and
   `HadronStructure(channels=NC_CHANNELS)` runs today returning CC numbers.
4. **Day 3 — write up D1 (both branches, the switch) and the strange-omission constraint**, cited.
5. **Day 3 — write the G5 analytic propagator test** *before* the current it tests, so it cannot be
   weakened to fit the implementation. It cannot **run** until P4; that is fine and intended.
6. **Day 4 — owner review of G0**, now including the D2 result and the closed-form isospin factor.
7. **Day 5 — P2 only if G0 clears D2 as "no CC change".** Otherwise Day 5 opens the CC repair, and
   P2/P3 wait — migrating 15 GB of banks that are about to be regenerated is pure waste.

**Do NOT this week:** touch `selection.py`, or submit any large generation job. A bank generated
against an unvalidated current is hours of `turing` time plus a merge tree to clean up.

---

## 6. Out of scope, on correctness grounds

| Item | Why |
|---|---|
| **Coherent NC π⁰** | **ACHILLES does not have it** — verified (§2). Its `Coherent` model is coherent *elastic*. Implementing it would make ADoNIS **disagree** with the reference these figures compare against, and it cannot contaminate an NC1π⁰ topology anyway. Owner's conditional resolved, not deferred. |
| **Strange FFs in NC QE** | **ACHILLES structurally cannot use them** — verified (§2). Including them would fail every χ² gate by construction. Recorded as a cited, test-pinned faithfulness constraint. |
| **Reviving the Phase-A2 tests / `data/oracle/`** | They predate the entire refactor arc, `data/oracle/` is empty, and their gate was a single-constant bridge — the exact blindness this plan exists to prevent. Cross-check to be earned in P4, never an input. |
| **Pre-FSI (`--no-fsi`) banks; the high-\|p\| cascade π-absorption residual; `fig02` χ²/ndf 45.70; the Oset NaN below `m_π`** | Orthogonal pre-existing items affecting CC equally. NC inherits whatever CC gets; they will appear in NC π⁰ figures too (π⁰ shares the meson cascade) and must be **cited as known**, not re-litigated inside an NC deliverable. |

Moved **into** scope by the no-corner-cutting instruction: NC QE (P6), the `k_lep` rename (P3), the
`assembly.py` duplicate (P4), the `hv_*` single-channel bug (P7), antineutrino NC (P9), and the
`anypi` / dead-`SignalDef`-field lies (P8).

---

## 7. Provenance and residual uncertainty

Everything about ADoNIS was read from source. Everything about ACHILLES is cited `file:line` under
`Achilles-src`. `README.md`, `docs/RUNNING.md`, `docs/phases/*`, `docs/logbook/*` were treated as
untrusted throughout.

**Not independently verified:** the runtime/preemption figures and shard-merge workflow (from job-log
reading). **Unverifiable as stated:** D3's "Pauli blocking on as for CC" — no Pauli-blocking code was
located under `adonis/channels` or `adonis/nuclear` on review, so the statement is either vacuous or
refers to machinery not yet found. Resolve it in P0 or delete it from D3.

**D2 is resolved by reading and open by measurement.** The `0.5*(vec−isv)` form does not appear in
the Fortran, no decomposition step exists to justify it, its own code comment cites the wrong lines,
and the CC audit that appeared to validate it is structurally blind to it. What P−1 measures is not
*whether* it is wrong but *how much CC output moves* — because changing a validated CC current on a
reading alone is the failure mode this plan exists to prevent.

**A judgement, not a fact:** that `coupl1`'s `sin2w` is an ACHILLES typo. What the code says and the
1.0396 factor are facts; "it is a bug" is inference from the adjacent `coupl2` using `sw`. The D1
switch means we no longer have to bet on that inference — we implement both and measure the difference.

---

## 8. Review record

This plan was reviewed adversarially against ACHILLES and ADoNIS source after its first commit
(`65e344f`). The review fact-checked every load-bearing citation. **Confirmed and not to be
relitigated:**

- The strange-form-factor omission — every citation exact, `FAs` has zero consumers repo-wide.
- `Coherent` is coherent *elastic* — and **stronger than claimed**: the argon coupling
  `coupl3·(18·(0.5−2s²) − 11)` is literally the Ar-40 CEvNS weak charge `Z(½−2s²) + N(−½)` for
  18 p / 22 n.
- The `_NORM_NC/_NORM_EM = 0.7113` derivation, end to end — `res_spectral_model.f90:127,145-149`
  confirms `FResV` multiplies the **entire** DCC current while `FResA` is only a boolean.
- The oracle prefix-routing landmine and its zero-code fix; the `hv_*` single-channel bug; the
  `n_other_meson` double-count; the `anypi` dead branch; the five dead `SignalDef` fields;
  `res_ee.py` as the right template; the 127 rename sites; the 12 bank configs (3/3/6).

**Corrected in this revision:** the P6 coupling snippet (factor 2); "proton F1/F2 only" → both
nucleons, and G6(3) with it; D2 promoted from open question to resolved-with-CC-exposure;
`assembly.py:126-133` added to P1 and `dirac.py:107-115` removed from it; P2/P3 regated on G0;
`amps2_records.py`, the NC free-nucleon generator, the flux threshold, `NC_CHANNELS` `mult` and the
`proc`-ID consumers added; G0(b), G4(3) and G5(1) made falsifiable; the effort estimate branched;
and the Fortran/Python line numbers corrected (`:284`→`:285`, `:838`→`:837`, `:870`→`:867`,
`extract.py:307`→`:303-304`/`:306`, `selection.py:96`→`:95`, `generate_bank.py:79`→`:69`, `res.py`
17 sites).
