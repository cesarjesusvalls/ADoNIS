# Alternative-generator fake data — fit ADoNIS against unmodelled physics

Session **Callum_C** (worktree `ADoNIS-Callum_C`, branch `callum_c_altgen`). Task: build a
DUNE-ND-flavoured artificial dataset with a *different* generator (GENIE/NEUT) and fit it with
ADoNIS as if it were real data, then read the parameter pulls against the *known* GENIE-vs-ACHILLES
physics differences (SF-vs-LFG initial state, RES model, FSI model).

Evidence log — what was tested, how, and the outcome. No conclusions ahead of the evidence.

---

## 1. Feasibility scan (what is actually available)

Scanned the workspace for foreign generators (2026-07-02):

| Probe | Result |
|---|---|
| `genie/gevgen/neut/nuwro` on PATH, env vars, conda, `/usr/local`,`/opt` | none |
| Docker images | ACHILLES variants + **`lucid:latest` (14 GB)** |
| `nuisance/` build | source only, **not built**; NUISANCE is a comparison framework, does not itself generate events |
| `nuisance/data/{neut,nuwro}` | config **cards only** (`neut_default.card`, `default_params.txt`) — no event samples |
| `nuisance/data/DUNE`, `data/flux` | **flux ROOT files only** (`flux_dune_neutrino_ND.root`, CDR fluxes) — no MC |
| `*.ghep.root`, `*.gst.root`, `gxspl*` | none pre-existing |

**Decisive find (pointed to by the user):** the **LUCiD** container `lucid:latest`
(`org.opencontainers.image.title=LUCiD`, base `nuisancemc/tutorial:nuint2024 + GEANT4/PhotonSim`)
ships fully-built generators:
- **GENIE 3.04.00**, tune **AR23_20i_00_000**, splines present
  (`GENIE_XSEC_FILE=/opt/genie_xsec/3_04_00/AR23_20i_00_000/gxspl-min.xml.gz`), GENIE-Reweight 1.02.02
- **NEUT 5.x** (`/opt/neut`, cards incl. 12C GFG/SF CCQE), **NuWro 21.09.2**
- **NUISANCE built** (`PrepareGENIE`, `nuisflat`, `nuiscomp`), ROOT 6.30, LHAPDF 6.5.1, PYTHIA6 6.4.28

Verified all binaries resolve and splines load (`docker run --platform linux/amd64 lucid:latest ...`).
Image is **amd64**; host is **arm64** → runs under Docker **Rosetta** emulation.

**Verdict:** a GENUINE foreign-generator sample is feasible via LUCiD — **no proxy needed.**

## 2. Design constraint that fixes the scope

The frozen ADoNIS event bank is **T2K νμ flux (`flux/T2K_nu.dat`) on ¹²C** (confirmed:
`adonis/xsec/flux.py`, `bank_plot` CONV `1e-33/12`). Regenerating the bank on a DUNE/Ar config is
forbidden (~21 h, competes with the running 10M-bank + info-content jobs). A literally-DUNE-ND
sample would be **Ar40 on the DUNE flux**, which the ¹²C/T2K bank cannot predict.

→ Generate the foreign sample on the **SAME T2K-νμ / ¹²C config as the bank** so the frozen bank
machinery fits it directly. The probe ("fit unmodelled GENIE physics with ADoNIS knobs") is fully
realized on T2K flux; the flux label does not change it. DUNE/Ar deferred (needs a DUNE-flux bank).

Agreed with user (Remote Control): **GENIE AR23_20i on T2K-νμ/¹²C**.

## 3. Generation (GENIE AR23_20i, provenance)

Driver: `scripts/altgen/run_genie.sh` + `make_flux_root.C` (build a ROOT TH1D flux from
`T2K_nu.dat` — variable-width bins, gevgen samples the shape). Command essentials:
```
gevgen -n <N> -p 14 -t 1000060120 -e 0,30 -f t2k_flux.root,t2kflux \
       --cross-sections $GENIE_XSEC_FILE --tune AR23_20i_00_000 \
       --event-generator-list <LIST> ; gntpc -f gst
```

**Rosetta/PYTHIA6 blockers hit and resolved (documented, physics-preserving):**
- PyROOT broken in image (`libcppyy3_11` mismatch) → build flux via ROOT **C++ macro**, not PyROOT.
- **DIS** hadronization crashes: `PYTHIA6 illegal instruction` under Rosetta
  (`PythiaBaseHadro2019::ProcessEventRecord`). → exclude DIS (negligible at T2K peak ~0.6 GeV).
- **RES** crashed too: `PythiaDecayer` (η/ρ/ω decays) hits the same illegal instruction. → config
  override `scripts/altgen/genie_cfg/UnstableParticleDecayer.xml` drops `PythiaDecayer`, keeps
  `BaryonResonanceDecayer` (handles N*→Nπ, all that CC0π/CC1π STV needs). RES then generates on
  fast Rosetta. Cost: rare η/ρ/ω left undecayed — negligible for these selections.
- `CCinclMECnoDIS` preset fails (`NNBarOscDummyPXSec` — it pulls in `QEL-CC-CHARM`, whose splines
  are absent from the min file). → custom lists `scripts/altgen/genie_cfg/EventGeneratorListAssembler.xml`:
  - **`CCQERES`** = QEL-CC + RES-CC → exactly the channels ADoNIS models (PRIMARY probe).
  - `CCNODIS` = +MEC+COH (COH spline absent from min file → parked; MEC works standalone).

Throughput ≈ 82 events/s (single Rosetta thread). **Primary sample: 300k CCQERES**, seed 20240702
(`output/altgen/genie_t2k_12C_ar23_CCQERES`, log `gen_ccqeres_300k.log`).

gst branches used: channel flags `qel/res/mec/coh`, muon `pxl,pyl,pzl,cthl`, hadrons
`pdgf,pxf,pyf,pzf,pf`, topological counts `nfp,nfpip,nfpim,nfpi0`, per-event `XSec` [1e-38 cm²].

**Genuine vs proxy:** the sample is a GENUINE GENIE 3.04 AR23_20i MC (real event-by-event
generation). The only physics omissions vs a full GENIE CC sample are **DIS and MEC/COH**, dropped
for the reasons above — this makes the sample a QE+RES GENIE prediction, which is the correct
matched-channel comparison against ADoNIS (QE+RES only). Not a hand-authored proxy.

## 4. Fake dataset (`scripts/altgen/build_fakedata.py`)

Apply the SAME CC0π-Np topological selection ADoNIS applies to its bank
(`bank_plot.signal_cc0pi(topological=True)` + `tune._sel` acceptance): CC, 0 mesons, ≥1 proton;
μ p>250 MeV & cos>−0.6; leading proton 450<p<1000 MeV & cos>0.4. Observables δpT, δαT via the exact
ADoNIS formulas. Central values = GENIE dσ/dx; **error model = the real T2K CC0π-STV covariance**
(same binning) → a "T2K-like measurement realized by GENIE."

Absolute normalisation: each event ≡ ⟨σ_tot⟩_Φ / N_gen. ⟨σ_tot⟩_Φ estimated from per-event `XSec`
two ways — harmonic-mean identity (exact in expectation but high-variance: dominated by tiny
near-threshold 1/XSec outliers) and a robust flux-reweighted σ(E) integral (default). Absolute is
secondary; the **primary fit is SHAPE (profiled norm)**, normalisation-independent.

## 5. Fit (`scripts/altgen/fit_fakedata.py`)

Mirrors the T2K CC0π blueprint (`tune.py`): differentiable bank reweight `w(θ)` binned over the
CC0π-Np signal → dσ/dx (autodiff in θ), covariance χ², Adam→BFP, Hessian param covariance.
Fitted QE+RES knobs mapped to the known differences:
`M_A` (QE axial Q² shape), `qe_norm` (QE norm), `kF_sf` (initial-state Fermi scale = SF-vs-LFG),
`sscat` (nucleon FSI = hN-vs-INC → δpT tail), `res_norm`, `res_axial_strength`.

Worktree needs `ACHILLES_DATA=<main>/data/achilles` (DCC/SF tables, like `ADONIS_EVENT_BANK`).

**Dead-knob finding:** `nominal_knobs()` carries a legacy `sscat` key that `bank_reweight` IGNORES
(`bank_reweight.py:35` hardcodes the second `pool_fsi_reweight` arg to `1.0`; the live FSI knobs
are the split `sabs, s_piN_*, s_NN_elastic/inelastic, f_NN_cex`, exactly the `_specs` list).
Evidence: `jax.grad` of Σw wrt `sscat` = 0.0 exactly; in the first closure fit `sscat` never moved
and the zero Hessian row made `np.linalg.inv` raise `Singular matrix`. Fit knob set corrected to
`sabs` + a common `s_NN_elastic` scale (`s_NN_el`, applied to pp/pn/nn together); Hessian via
`pinv` with a near-singularity warning.

**Closure gate (v2, corrected knobs, abs metric, δpT):** inject qe_norm=1.3, kF_sf=1.1,
res_norm=0.8 (others nominal) into an ADoNIS-made fake dataset; fit the 6 knobs. By it 120:
χ²/ndf 39.5M → 604 (5 orders), **kF_sf=1.103** (inj 1.1) and **res_norm=0.808** (inj 0.8)
recovered; the injected qe_norm=1.3 split along the (M_A=1.110, qe_norm=1.185) QE degeneracy and
sabs=0.787 partially trading against res_norm — the two flat directions expected from ONE
8-bin observable. Machinery (exact reweight, autodiff, Adam, binning) validated; run stopped at
it~120 rather than crawling the degenerate valley (~13 s/it under 4-session CPU contention).
Log `/tmp/altgen_closure_fit2.log`. Joint δpT+δαT fitting is the known cure for these
degeneracies (not needed to gate the machinery).

## 6. GENIE fake dataset (built)

300k CCQERES events generated in 53 min (log `gen_ccqeres_300k.log`); gst 118 MB.
Channel fractions **QE=0.639 / RES=0.361** (MEC=COH=0 by construction). Selection cascade:
300000 CC → 210749 topological CC0π-Np → **86304 in T2K acceptance**. Per-bin counts 3.7k–20.5k
→ MC stat 0.7–1.6%, negligible vs the T2K covariance (~10%).

**Normalisation finding:** gst `XSec` is per **nucleus** (¹²C); T2K STV + bank are per **nucleon**
→ /12 (without it GENIE landed 10–15× above data; with it 0.9–1.3×). Robust flux-reweighted
⟨σ_tot⟩_Φ = 4.56e-38 cm²/¹²C = 0.380e-38/nucleon (harmonic-mean estimator confirmed unusable:
0.246e-38/¹²C, collapsed by near-threshold 1/XSec outliers).

GENIE/T2K-data ratios (context, not the metric): δpT peak 0.87–1.26, tail 0.45–0.51 (consistent
with the sample having **no MEC/2p2h**, which populate the tail); δαT 0.62–0.97 with the largest
deficits mid-range and at π. `output/altgen/fakedata_ccqeres.npz`.

## 7. GENIE AR23_20i actual constituents (read from the tune config in the container)

`$GENIE/config/AR23_20i/{README.md,ModelConfiguration.xml,CommonParam.xml}`, active (uncommented):

| sector | GENIE AR23_20i | ADoNIS (=ACHILLES) |
|---|---|---|
| QE 1p1h | `NievesQELCCPXSec/ZExp` (Valencia, RPA, z-exp axial) | Benhar 2-D SF(p,E), z-exp axial (MA=1.0) |
| initial state | `LocalFGM/SpectralFunctionLikeWithCorrelation`; **¹²C override `SpectralFunc1d`** (1-D SF) | full 2-D spectral function |
| RES | `BergerSehgalRESPXSec2014/NoPauliBlock` | DCC (ANL-Osaka) amplitudes |
| FSI | `HAIntranuke2018` (effective single-interaction hA) | full INC cascade (GiBUU NN, Oset π abs) |
| in sample | MEC/COH/DIS excluded (CCQERES list) | QE+RES only (all ADoNIS has) |

A-priori pull expectations from these differences (stated BEFORE reading the fit):
- δpT **peak width** ← 1-D SF vs 2-D SF momentum distribution → `kF_sf`
- δpT **tail** + δαT shape ← hA2018 vs INC cascade rescattering → `s_NN_el`
- QE low-Q² (RPA screening in Nieves, absent in ADoNIS) → `M_A`/`qe_norm`
- RES rate + π-absorption feed (BS vs DCC; hA abs vs Oset) → `res_norm`/`sabs`

## 8. Units audit (both found while validating, fixed at source)

- gst `XSec` is per-**nucleus** → /12 for per-nucleon (build_fakedata).
- `fit_fakedata` initially divided by GeV bin widths while CONV already carries MeV→GeV → model
  ×1000; **shape metric unaffected** (profiled A absorbs any global factor exactly), abs metric
  fixed before use. Evidence: first GENIE shape fit reported A_nom=0.001.

## 9. First fit: ADoNIS QE+RES → GENIE AR23 δpT (SHAPE, profiled norm)

`fit_fakedata.py dpt`, 6 knobs, 300 Adam iters, log `/tmp/altgen_genie_fit_dpt_shape.log`.
**Nominal ADoNIS-vs-GENIE shape χ²/ndf(7) = 3.55 → BFP χ² = 3.48 (ndf=1)**, converged by it≈200:

| knob | BFP | reading vs the §7 expectations |
|---|---|---|
| `sabs` | **0.300 = LOWER RAIL** | fit slashes ADoNIS's π-absorption → kills the RES→CC0π feed |
| `res_norm` | **0.300 = LOWER RAIL** | same direction — but see the §9b cross-check: the RES *rates* agree (7.5% vs 8.4%); the rails are a SHAPE compensation, not a rate mismatch |
| `s_NN_el` | 2.52 | ADoNIS needs ~2.5× MORE nucleon rescattering to reproduce GENIE's δpT tail-vs-peak — consistent with hA2018 (single effective interaction, more smearing per event) vs INC stepwise transport |
| `qe_norm` | 1.70 (drifting) | exactly degenerate with profiled A once RES is railed — flat direction, value not meaningful in the shape fit |
| `M_A` | 0.863 | softer axial Q² — the Nieves-RPA low-Q² screening in GENIE mimics a lower effective M_A, as anticipated |
| `kF_sf` | 0.932 | slightly narrower initial-state momentum: GENIE's 1-D SF for ¹²C vs ADoNIS's 2-D SF(p,E) peak-width difference, small as expected |

Final table (finalize run: warm start at BFP, FD-of-grad Hessian — `jax.hessian` on the full-bank
graph is OOM-killed on this 16 GB machine; also A_nom=0.875 after the bin-width fix, i.e. ADoNIS
nominal is ~14% ABOVE GENIE in absolute norm):

```
knob      nominal   BFP     +/-     pull    flag
M_A        1.000   0.862   0.794   -0.17
qe_norm    1.000   2.207  22.906    0.05    (flat: degenerate with profiled A once RES railed)
kF_sf      1.000   0.939   0.030   -2.02
s_NN_el    1.000   2.523   0.745   +2.04
sabs       1.000   0.300   3.859   -0.18    RAIL (subspace-projected sigma; pair-degenerate)
res_norm   1.000   0.300   4.421   -0.16    RAIL (with sabs — the product is the constrained combo)
Hessian near-singular (eig ratio -1.8e-4) — errors pinv-projected; rail errors invalid (blueprint)
```

χ²(shape) 24.8 (ndf 7) → **3.5 (ndf 1)**: the 6 knobs DO absorb most of the foreign-generator δpT
shape — but only by driving the RES sector to an unphysical corner (both RES knobs railed at 0.3)
and pulling nucleon FSI +2σ. Figure `output/figures/altgen_fit_dpt_shape.png`; history
`output/altgen/fit_dpt_shape.npz`.

### 9b. Rail cross-check — the RES rates actually AGREE (rails are a shape compensation)

Direct check of the rail reading: RES-origin fraction of the selected CC0π-Np sample —
- GENIE (gst `res` flag): **8.4%** of 86304 selected events
- ADoNIS bank (`channel==1`, w0-weighted): **7.5%** of the signal weight

The RES→CC0π *rates* are nearly identical, so the railed RES knobs are NOT correcting a rate
mismatch. RES-origin events are δpT-tail-heavy; GENIE's tail/peak ratio is lower than ADoNIS's, and
the least-bad shape move available to the knob space is: kill the tail-heavy RES component (rails)
while re-smearing QE with more NN rescattering (s_NN_el +2σ) and narrowing the initial state
(kF_sf −2σ). **Demonstration outcome: a foreign-generator fit can produce large, confident-looking
pulls on knobs whose underlying physics agrees between the generators — the pulls are shape
proxies, not physics measurements.** This is the central caution for fitting real data with any
single-generator model.

### 9c. δαT (shape): nominal ADoNIS-vs-GENIE χ²/ndf(7) = **0.37** (A_nom=0.818) — no power

The FSI-direction observable δαT already agrees in shape at nominal; the δpT tension has no δαT
counterpart. The fit (χ² 2.6→0.4) produced NO significant pulls — every σ is enormous
(qe_norm ±585, res_norm ±2750, sabs ±83; all |pull| < 0.4σ): at T2K-covariance precision δαT does
not discriminate the AR23-vs-ADoNIS differences, and the BFP values are flat-direction drift.
Notably `sabs` drifted UP to 2.2 here while the δpT fit railed it DOWN to 0.3 — mutually opposite
motion of the same knob across observables, in flat directions: a second, independent caution
against reading single-observable foreign-generator pulls as physics.
Figure `output/figures/altgen_fit_dat_shape.png`; log 9 h under 10M-bank contention.

### 9d. δpT (ABSOLUTE): χ² 107.8 (ndf 8) → 3.5 (ndf 2)

Same solution as the shape fit: **kF_sf 0.933 (−1.9σ)**, **s_NN_el 2.52 (+1.8σ)**, RES pair railed
at 0.3, M_A 0.98 (−0.02σ), qe_norm 1.72±1.67 (+0.4σ, absorbing the norm shuffle once RES is
railed + rescattering moves QE protons out of acceptance). The absolute and shape metrics agree on
the physics directions; nominal-ADoNIS is ~14% above GENIE absolutely (A_nom=0.875 from the shape
fit). Figure `output/figures/altgen_fit_dpt_abs.png`.

## 10. Summary of the demonstration (evidence recap)

1. Genuine GENIE 3.04 AR23_20i sample (300k QE+RES, T2K-νμ/¹²C) built and fit with the frozen
   ADoNIS differentiable bank — full pipeline works end-to-end.
2. δpT: nominal shape χ²/ndf 3.55; 6 QE+RES knobs absorb most of it (χ² 24.8→3.5) but only via an
   unphysical corner: RES sector railed + s_NN_el +2σ + kF_sf −2σ.
3. The rail is NOT a rate difference — RES→CC0π fractions agree (ADoNIS 7.5% vs GENIE 8.4%); it is
   a shape compensation for the δpT tail (hA2018 vs INC transport shows up exactly there).
4. δαT carries no discriminating power at this precision, and its (insignificant) drift moves the
   same knobs the opposite way.
5. Central lesson for real-data fits: confident-looking pulls from a single observable can be
   pure shape proxies — cross-observable consistency (δpT vs δαT) and rate-vs-shape decomposition
   (as in 9b) are the diagnostics that expose them.

---

### Status
- [x] Feasibility scan + LUCiD verified
- [x] GENIE generation pipeline (flux, decayer/list overrides, gst)
- [x] 300k CCQERES generated (53 min)
- [x] fit machinery closure-gated (kF_sf/res_norm recovered; degeneracies characterized)
- [x] GENIE fake-data build (§6), fits (§9: δpT shape+abs, δαT shape), interpretation (§9b, §10)

## 11. Integration of sessions A/B (merged origin/main @ 07a8cb4) + joint fit

Merged main cleanly. Relevant to this task:
- `full_knobs`: **M_A split → M_A_qe / M_A_res** (independent per-channel); `knob_specs` is the
  knob-metadata single source of truth; **sscat officially documented dead** (matches §5 finding).
- Session A (`info_content.py`): 5-dataset assembly `build_datasets` (CC0π dpt/dat per-nucleon +
  CC1π pN/dpTT/daT nb/CH with frozen free-H offsets) = single source of truth; JB-as-argument +
  per-knob `jvp` Jacobian pattern (avoids both my jit-constant OOM and the jacfwd 16 GB blowup);
  pion-FSI flat direction shown to be a per-event structural identity.
- Session B (`llh_surface.py`): on REAL T2K data, adding CC1π+Np tightens res_norm 3.2×
  (σ 0.43→0.13) — CC0π alone leaves the RES direction loose. Exactly the degeneracy behind my
  railed RES sector (§9/§9b).

**Joint-fit extension built on their machinery:**
- `build_fakedata.py` extended: CC1π+Np STV fake datasets from the SAME GENIE sample
  (selection mirrors `signal_cc1pi_stv` incl. cos70° forward cuts + kaon veto; observables via the
  imported validated `bank_plot` formulas; nb/CH; GENIE-C only — the frozen ADoNIS free-H offset is
  added identically to model and data in the fit, cancelling in the residual, so ONLY the C part is
  foreign). Selected: 54429 1π⁺ events → 7472 in acceptance.
- `fit_fakedata_joint.py`: 5-dataset joint LM fit (Gauss-Newton with per-iteration jvp Jacobian),
  same 6 knobs (M_A_qe, qe_norm, kF_sf, s_NN_el, sabs, res_norm), T2K covariances, GENIE centrals.
- Key question on the table: does CC1π **un-rail** the RES sector on foreign fake data, and do the
  δpT-driven pulls survive the RES-informed constraint?

## 12. JOINT CC0π+CC1π fit result (the completion of the demonstration)

Nominal joint (absolute, 28 bins): **χ²/ndf = 1.36** — remarkably close to the 1.44 the same model
gives on REAL T2K data (llh_surface). Per-dataset nominal: CC0π dpt **3.72**/bin; everything else
already agrees (dat 0.77, pN 0.43, dpTT 0.09, daT 0.04) — the foreign-generator tension is
localized in CC0π δpT.

LM converged (39 its, χ² 38.2 → 10.6, ndf 22 → **χ²/ndf 0.48**):

```
knob      BFP     +/-    pull                    single-obs dpt fit was:
M_A_qe   1.738   0.894   +0.83                    0.862 (other end of the qe banana)
qe_norm  0.906   0.279   -0.34                    1.7-2.2 (flat)
kF_sf    0.925   0.035   -2.13   <- ROBUST        0.933 (-1.9σ)  [same value in EVERY fit]
s_NN_el  1.655   0.388   +1.69                    2.52 (+1.8σ)  [moderated by CC1π]
sabs     0.300   1.211   RAIL    <- persists      0.300 RAIL
res_norm 1.229   0.288   +0.80   <- UN-RAILED     0.300 RAIL
per-dataset chi2/nbin at BFP: dpt 1.00, dat 0.16, pN 0.22, dpTT 0.02, daT 0.09
```

Readings (vs §7 a-priori expectations and §9/9b):
1. **CC1π un-rails res_norm** (0.300-RAIL → 1.229±0.288), exactly reproducing on foreign fake data
   the degeneracy-breaking session B demonstrated on real data. The RES production strength is now
   measured, physical, and mildly ABOVE nominal.
2. **The sabs rail persists** — and is now interpretable: with RES production anchored by the
   surviving-pion (CC1π) rate, the fit still wants minimal pion absorption, i.e. ADoNIS/Oset
   converts more of its RES production into absorbed-pion CC0π events than GENIE/hA2018 does. A
   genuine absorption-model difference (branching, not rate), consistent with §9b.
3. **kF_sf = 0.925±0.035 (−2.1σ)** — the most stable pull of the whole study (0.93 in every fit,
   every metric, every dataset combination): the 1-D-SF(GENIE-¹²C) vs 2-D-SF(ADoNIS) initial-state
   width difference. **s_NN_el = 1.66±0.39 (+1.7σ)** — persistent nucleon-transport difference
   (hA2018 vs INC), moderated from 2.5 once CC1π constrains the rest.
4. M_A_qe/qe_norm land on the *other* end of their −0.97-correlation banana than the CC0π-only fit
   (1.74/0.91 vs 0.86/1.7+) with individually insignificant pulls — single-observable "axial"
   pulls against a foreign generator are not stable statements.
5. End state: ADoNIS describes the 5-dataset GENIE QE+RES fake data at **χ²/ndf 0.48** with
   physical knob values everywhere except the single interpretable sabs rail.

Figure `output/figures/altgen_fit_joint.png`; history `output/altgen/fit_joint.npz`;
log `/tmp/altgen_joint_fit.log`.

## 13. +MEC variant — the tail attribution CONFIRMED, and a deeper result

300k GENIE `CCQERESMEC` (custom list QEL+RES+MEC; smoke-tested; seed 20240704; 53 min). Sample:
**QE 54.7% / RES 31.1% / MEC 14.2%**. Norm-builder bug found+fixed at source: σ_tot(E)
reconstruction summed only QE+RES channel XSecs — MEC now included (σ_tot 4.55→5.53e-38/¹²C,
+21%); fix is a no-op for channel-absent samples. Fake data rebuilt
(`fakedata_qeresmec.npz`: CC0π 89 895 sel, CC1π 6 519 sel).

**Joint fit vs the +MEC fake data** (`altgen_fit_joint_mec.png`, `fit_joint_mec.npz`):

| | QE+RES-only fake | **+MEC fake** |
|---|---|---|
| nominal joint χ²/ndf | 1.36 | **0.61** |
| CC0π dpt nominal /bin | 3.72 | **1.10** |
| BFP joint χ²/ndf | 0.48 | 0.14 |
| s_NN_el BFP | 1.66±0.39 (+1.7σ) | **1.06±0.70 (0.1σ) — pull GONE** |
| kF_sf BFP | 0.925±0.035 | **0.935±0.042 — unchanged** |
| rails | sabs LOW (0.3) | M_A_qe HIGH (2.0), sabs HIGH (3.0) — noise-chasing at χ²/ndf 0.14, huge σ |

Readings:
1. **§9b tail attribution confirmed experimentally**: adding GENIE's MEC (a channel ADoNIS does
   not model) REMOVES the CC0π δpT tension (3.72→1.10/bin) and the s_NN_el pull (1.66→1.06).
   The "FSI-strength difference" of the matched-channel fit was the missing-MEC tail.
2. **Deeper result**: nominal ADoNIS (2-D SF with SRC tail + INC cascade, NO explicit MEC channel)
   agrees BETTER with GENIE's full physics (0.61) than with GENIE's matched QE+RES subset (1.36).
   The matched-channel comparison is the harsher test, not the fairer one — both generators
   approximate the same reality, and ADoNIS's SF/cascade covers part of the strength GENIE books
   as 2p2h. Direct implication for real-data fits: no MEC knob does not mean MEC-blind.
3. **kF_sf = 0.935 ± 0.042** — identical in EVERY fit of this study (0.925–0.939 across
   single-obs/joint, shape/abs, ±MEC): the one robust, sample-composition-independent physics
   pull (GENIE-¹²C 1-D SF vs ADoNIS 2-D SF initial state).
4. The upper-rail excursions (M_A_qe→2.0, sabs→3.0) occur at χ²/ndf≈0.14 with σ≥1.3 — flat-
   direction noise-chasing after the fit is already statistically perfect; pulls at such χ²
   carry no information (the LM run should arguably early-stop on χ²/ndf<~0.3).

## 14. NEUT variant — second foreign generator, distinct fingerprint

300k NEUT (stock 5.4.0 pure-¹²C card + T2K flux histogram; `run_neut.sh`: neutroot2 → PrepareNEUT
→ nuisflat GenericVectors; crsdat path fixed to the container install; ~2 h under parallel load).
CC mode composition: CCQE 100 863, 2p2h 18 217, CC1π(11–13) 63 644, coh 1 072, multi-π 18 654,
DIS 16 865. Normalisation via NUISANCE per-event `fScaleFactor`. Two slices from the one run
(`build_fakedata_neut.py` → `fakedata_neut.npz`): **matched** (modes 1,11–13 ≈ QE+RES) and
**full** (all CC — 2p2h/multi-π/DIS are channels ADoNIS lacks).

**Matched slice, joint fit** (`altgen_fit_joint_neut.png`, `fit_joint_neut.npz`):
nominal joint χ²/ndf **1.59** (CC0π dpt 2.35, **CC1π pN 2.87** — unlike GENIE, NEUT's CC1π
disagrees at nominal too) → BFP χ² 16.2/22 = 0.74:

```
knob      BFP     +/-    pull
M_A_qe   2.000   0.823   RAIL(hi)   (banana with qe_norm=1.03±0.21: NEUT Q2 shape)
kF_sf    0.944   0.027   -2.0σ      <- SAME ~0.94 as vs GENIE: two independent foreign
                                       generators prefer a slightly narrower initial state
s_NN_el  1.781   0.164   +4.8σ      <- STRONG, well-constrained: NEUT cascade vs INC transport
sabs     0.582   1.034   -0.4σ      (not railed, weakly constrained)
res_norm 1.860   0.310   +2.8σ      <- NEUT's matched-slice RES rate genuinely above ADoNIS
per-dataset at BFP: dpt 0.59, dat 0.38, pN 1.43 (worst, not absorbable), dpTT 0.16, daT 0.64
```

**Full-CC slice** (`altgen_fit_joint_neutfull.png`, `fit_joint_neutfull.npz`): nominal joint
χ²/ndf **2.45** (dpt 2.42, dat 3.61, pN 2.65, dpTT 1.18, daT 1.25) — the OPPOSITE nominal response
to GENIE's +MEC (1.36→0.61): NEUT's 2p2h+multi-π+DIS overshoot ADoNIS everywhere. Fit → 11.2/22 =
0.51:

```
knob      BFP     +/-    pull
M_A_qe   2.000   1.099   RAIL(hi, flat)    qe_norm 0.795±0.169 (banana)
kF_sf    0.936   0.035   -1.8σ             <- sixth fit, same 0.93-0.94
s_NN_el  1.005   0.452   +0.0σ             <- the +4.8σ matched-slice pull VANISHES
sabs     3.000   3.312   RAIL(hi, flat)
res_norm 1.965   0.459   +2.1σ             <- survives both slices: genuine
per-dataset at BFP: dpt 0.40, dat 0.33, pN 1.10 (still worst), dpTT 0.15, daT 0.07
```

Cross-generator fingerprints (matched vs full, both generators — the study's synthesis):
- **kF_sf ≈ 0.93–0.94 in ALL SIX fits** (GENIE single-obs ×3 / joint / +MEC; NEUT matched / full):
  the one robust, generator- and channel-composition-independent pull — ADoNIS's initial-state
  momentum width is slightly wide relative to both foreign implementations.
- **s_NN_el is NEVER a transport measurement**: +1.7σ (GENIE matched) → 0.1σ (+MEC);
  +4.8σ (NEUT matched) → 0.0σ (full). In BOTH generators the "FSI-strength pull" evaporates when
  the sample's channel content is completed — it is a channel-completeness proxy. (This corrects
  the §14-matched reading above that called it "real"; the full-CC evidence supersedes it.)
- **res_norm**: GENIE ~1.2–1.3; NEUT ~1.9–2.0 in BOTH slices (+2.1…2.8σ) — a genuine NEUT-vs-ADoNIS
  RES-sector normalisation difference. CC1π pN keeps a ~1.1–1.4/bin residual in every NEUT fit —
  a real, not-absorbable RES/initial-state shape difference.
- Fitted end states: GENIE-matched 0.48, GENIE+MEC 0.14, NEUT-matched 0.74, NEUT-full 0.51 —
  the 6-knob ADoNIS reweight describes every variant after fitting; what distinguishes the
  generators is WHERE the knobs land and WHICH residuals persist, not the final χ².

### Open follow-ups
- DUNE-ND/Ar variant — requires a DUNE-flux/Ar event bank (~21 h regen, deferred)
- NuWro (third generator, same pipeline) — available in LUCiD if wanted
