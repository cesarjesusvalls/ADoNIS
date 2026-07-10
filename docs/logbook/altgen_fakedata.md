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

## 15. HIGH-STATISTICS regime — turning un-hideable mismatch into a physics signal

Mandate refinement (Remote Control, 2026-07-07). §9–14 all used the **real T2K CC0π-STV
covariance** (~10% errors) as the error model; under it the 6 knobs absorb the foreign-generator
mismatch (χ²/ndf → 0.48, 0.14). That absorption is a *large-error artifact*. The real question:

> When a future experiment drives statistical errors to negligible, the best possible agreement is
> never good if the forward model doesn't contain the fake-data physics. Can we fit so that such
> "broken" physics becomes **informative** rather than just a failed χ²?

### The mechanism (why this is tractable)
The 6 knobs span a 6-D manifold in observable space; the foreign generator is a point that generally
lies OFF it. The best fit projects onto the manifold, leaving an irreducible residual `r_⊥` in the
knob Jacobian's NULL SPACE (the directions the knobs cannot reach). With `C → α·C` (α→0 = future
high-stat/high-precision):
- χ²_min = `r_⊥ᵀ C⁻¹ r_⊥` **grows like 1/α** — does NOT converge to 1 for any structural mismatch.
- So high stats does not "hide better then plateau"; it PINS the knobs (large-error flat directions
  freeze) and EXPOSES `r_⊥`, whose direction in observable space is the missing-physics fingerprint.
- Low-stat regime: knobs move a lot, residual buried. High-stat regime: knobs pinned, residual
  exposed. The diagnostic turns ON precisely where the worry lives.

This is the high-stat generalization of the region-split rule (`region-split-not-summary-chi2`):
don't read the summary χ², read WHERE it concentrates — here in eigen-directions, not just bins.

### Why this session is the ideal testbed
The physics differences are KNOWN and documented (§7/§13/§14): GENIE-matched is missing MEC (δpT
tail), GENIE+MEC removes it, the initial-state SF difference is *absorbable* (lives in kF_sf). So
this is a CONTROLLED experiment: verify a high-stat residual diagnostic lights up exactly where the
known missing physics lives. If it does, the protocol is validated and can be trusted on real data
where the missing physics is unknown. Deliverable = the protocol, not "ADoNIS fits GENIE".

### Plan (reuses fit_fakedata_joint.py + the frozen bank)
1. **Statistics ladder**: refit at `C(α)=α·C_T2K`, α down to the MC-stat floor. Plot χ²_min/ndf vs
   1/α. Adequate model → flat ~1; structural mismatch → linear rise. Crossover α* = the precision at
   which the missing physics becomes >Nσ. Contrast GENIE-matched (rises) vs +MEC (stays flat).
2. **Irreducible-residual fingerprint**: at the high-stat BFP whiten `J̃=C^{-1/2}J`, project residual
   onto the knob null space, un-whiten `r_⊥` per bin/observable. Validate it concentrates in the δpT
   tail (matched) and nearly vanishes (+MEC).
3. **Characterize, not just detect**: add a free-form template (MEC-shaped / δpT spline, cf.
   `cc0pi_q2spline_fit.py`) as extra nuisance directions; show χ²_⊥ collapses and the recovered
   template matches the known missing channel. Closes detect → measure.

### Scope decision (Remote Control, 2026-07-07)
Generate ~3M each for **GENIE matched (CCQERES) vs +MEC (CCQERESMEC)** so fake-data central values
reach ~0.3% (vs the 300k 0.7–1.6% MC floor) and the ladder extends to ~30×-better-than-T2K.
NEUT/NuWro/DUNE not in this phase.

### Generation (in progress)
- Wrapper reconstructed (docker-run was never persisted) and smoke-gated at NEV=1000 (gst produced).
- `run_genie.sh` flux filename parametrized by NAME (`flux_${NAME}.root`) so the two concurrent 3M
  jobs cannot race on a shared histogram — deterministic content, name only isolates the file.
- 300k CCQERES gevgen took 52 min (~95 evt/s) → 3M ≈ 8.8 h/job; run in parallel (8 cores, both
  single-thread under Rosetta).
- CCQERES 3M: seed 20250707, `NAME=genie_t2k_12C_ar23_CCQERES_3M`, log `gen_ccqeres_3M.log`.
- CCQERESMEC 3M: seed 20250708, `NAME=genie_t2k_12C_ar23_CCQERESMEC_3M`, log `gen_qeresmec_3M.log`.
- Measured rate (+1h): ~12–14% each → ~7 h/job wall-clock in parallel (8 cores absorb both single-thread
  Rosetta jobs; contention penalty negligible vs the 8.8 h single-thread estimate).

### Diagnostic machinery — `scripts/altgen/highstat_diagnostic.py` (built + code-validated on 300k)
Three deliverables (see the module docstring for the mechanism):
1. Statistics ladder (analytic: a global covariance scale does not move the argmin, so
   χ²_min(α)/ndf = χ²_min(1)/ndf · 1/α; validated by ONE warm-started refit from the converged base).
2. Irreducible-residual fingerprint (block-diagonal whitening + null-space projection at the BFP).
3. MEC-channel decomposition (below).
- **Code validation (300k, NIT=3, under-converged — CODE not physics):** nominal matched χ²/ndf =
  **1.365, bit-matches logbook §12** → dataset assembly + central swap correct. Full pipeline
  (fit → fingerprint eigh/pinv → null-space decomposition → figure) runs end-to-end, fig produced.
  Ladder-gate ratio 0.96 & |Δθ|=3.7e-2 are under-convergence artifacts (will tighten at NIT~40).
- **Methodology fix found via the smoke:** the naive "irreducible residual ∝ raw MEC template" is
  WRONG by construction — the 6 knobs can partly MIMIC the MEC shape, so the un-absorbable object is
  the MEC template projected onto the knob NULL SPACE, not the raw template (raw dpt corr was only
  0.23). Corrected: decompose the KNOWN MEC channel into knob-absorbable vs irreducible in the T2K
  metric; report the irreducible fraction ("X% of GENIE's MEC channel is un-fakeable by the 6 knobs")
  and compare *that* to the matched-fit residual. Null-space block unit-tested (H·Hⁱⁿᵛ=I, projector
  idempotent, irr⊥knob-space, frac∈[0,1]). Real frac_irr_mec + correlations await the converged 3M run.

## 16. GENIE as data at GENIE's OWN precision (mandate refinement, Remote Control 2026-07-08)

Redirection: NOT the T2K covariance — treat GENIE-3M as a measurement with its OWN precision.
Final config (after iteration with the user): **20 bins/observable** (dpt, dat), error model =
GENIE stat ⊕ **5% uncorrelated bin-by-bin syst** ⊕ ADoNIS-bank-MC (quadrature, diagonal).
`scripts/altgen/fit_genie_precision.py` (live per-iteration progress; early nominal-overlay figure;
plateau early-stop). 3M generation: both samples completed in ~7 h wall (parallel, ~2× the
single-thread estimate of 8.8 h avoided); fake datasets rebuilt at 3M
(`fakedata_{ccqeres,qeresmec}_3M.npz`; CC0π 862k sel, CC1π 74k sel; QE/RES = 0.638/0.362 ✓ vs 300k).

Bugs found en route (all flagged by the user's normalization instinct):
- **1000× units bug**: GENIE dpt dsig was per-MeV while the model conv is per-GeV (dat unaffected —
  no unit factor). Symptom: dpt χ²/bin 272 vs dat 8.4 + qe_norm railed at 0.3 fighting a factor it
  can't reach. Fixed (bw/1000 for dpt, as build_fakedata always did).
- **bin-(−1) clip bug**: overflow-fold clipped values exactly onto edges[0] → searchsorted−1 = −1 →
  bincount throws. Fixed: clip strictly inside the range (eps), identical for GENIE and ADoNIS.
- **Real ~18–20% norm offset** (not a bug): ADoNIS QE+RES sits above GENIE-matched — the §13
  "ADoNIS SF+cascade carries strength GENIE books as 2p2h" difference; absorbed by qe_norm≈0.78.

Binning studies (equal-count vs uniform): equal-count (20 quantile bins) buries the dpt tail in one
[0.5,1.5] GeV bin → χ²/ndf 40.3→6.56 (qe_norm 0.82, sabs+res_norm railed low). **Uniform** 20 bins
over [0,p99] + overflow fold resolves the tail and is the fairer test:

**RESULT (uniform, stat+5%+MC): χ²/ndf 65.9 → 7.41** (`altgen_genieprec_20uni.png/.npz`):
```
M_A_qe   2.000±0.249  RAIL(hi)     qe_norm 0.776±0.071 (-3.1σ)   kF_sf 0.939±0.013 (-4.7σ)
s_NN_el  1.865±0.224  (+3.9σ)      sabs    0.300 RAIL(lo)        res_norm 0.300 RAIL(lo)
per-dataset χ²/nbin:  dpt 90.4 → 10.99      dat 21.7 → 1.61
```
- **dat is fittable** (21.7→1.6): its ~20% nominal gap was normalization → absorbed.
- **dpt is NOT fittable** (90→11): coherent S-shaped post-fit residual (peak −6σ…, mid +3…+6σ,
  tail −2σ), 3 knobs railed — the ADoNIS(2D-SF+SRC+INC) vs GENIE(1D-SF+hA2018) initial-state/FSI
  difference is NOT representable in the 6-knob space. At T2K's correlated ~10% this was χ²/ndf 1.4.
- ADoNIS-bank floor at 20 bins: 1.0–2.6% MC/bin (effN 1.9k–9.4k) — subdominant to the 5% syst
  ("more ADoNIS" not needed at this binning; it WAS the floor at the 775-bin/3% design: 5.9%/bin).

**Honest mandate assessment (2026-07-08):** the pulls deliver understandable physics ONLY for
differences the knob space can represent (kF_sf≈0.93–0.94 in every fit = initial-state width;
qe_norm = the 2p2h-bookkeeping offset). The LARGEST difference (dpt shape) breaks the pulls
(3 rails, thrashing) and migrates into the irreducible residual. A clean demonstration still
requires the controlled closure (matched vs +MEC = known-injected difference) — machinery built
(§15) but not yet run at GENIE precision. → superseded by the PHYSICAL-FIT program (§17).

## 17. PHYSICAL-FIT program (mandate restated, Remote Control 2026-07-10) — PLAN

Problem statement (user): any forward model (ACHILLES/ADoNIS) is approximate — (a) tunable-parameter
uncertainty, (b) computational simplifications (impulse approx., nuclear-model gaps), (c) unknown
unknowns (e.g. 2p2h historically). How do we learn from data WITHOUT being biased by (b)/(c)?
Core requirement: a fit whose parameter motion is only accepted when it is PHYSICALLY INFORMED —
i.e. the bins a knob touches must pull it COHERENTLY. A knob pulled up by half its bins and down by
the other half (net ≠ 0 because a tail dominates) is NOT a measurement; it is mismodeling leaking
into a dial. Regions that no coherent variation can fix get FLAGGED as unknown-unknown candidates.

### Methodology design (v1)
Two gates on every parameter, then a flagging pass:
- **Gate I — informativeness (Asimov Fisher).** All knobs get uncorrelated priors (20% multiplicative;
  additive/fractional knobs in natural units: Eb_shift ±4 MeV, f_NN_cex ±0.1). Asimov at nominal:
  posterior = (JᵀC⁻¹J + Π⁻¹)⁻¹. Fit only knobs with σ_post < 0.5·σ_prior (sample actually informs
  them); freeze the rest at prior. Uses session A's per-knob jvp Jacobian machinery (info_content).
- **Gate II — coherence (v2, refined with the user 2026-07-10).** Per-bin demand estimate
  δθ_kb = r_b/J_kb, Fisher weight w_kb=(J_kb/σ_b)², responsive bins |J_kb|·σ_prior > f·σ_b (f~0.3).
  (1) **Per-knob heterogeneity = Cochran's Q**: Q_k = Σ_b w_kb (δθ_kb − δθ̂_k)² ~ χ²(n_resp−1)
  (+ I² = incoherent variance fraction). This IS the user's requested "how likely is this dial
  variation given the observed per-bin pulls" — a calibrated LR test of one-shared-shift vs
  heterogeneous demands (meta-analysis heterogeneity). Supersedes sign-agreement (calibrated,
  catches magnitude incoherence).
  (2) **Vector split-fit**: full Gate-I vector fit on disjoint regions (bulk/tail per observable) →
  Q_split = Δθᵀ(V₁+V₂)⁻¹Δθ ~ χ²(p); catches CROSS-KNOB COMPENSATION that per-knob Q misses (dial A
  fixes bulk / dial B breaks tail lands the two regions at different points in parameter space);
  worst eigen-direction names the inconsistent combination. Diagonal errors → disjoint regions
  independent → exact calibration. NB: at any BFP, Σ_b J_kb r_b/σ_b² = 0 BY CONSTRUCTION (the fit
  forces half-up/half-down balance) — aggregate demand is useless post-fit; only the demand-field
  STRUCTURE (Q_k, Q_split) carries the incoherence signal.
  Knobs/combinations failing coherence are FROZEN (their motion is mismodeling leakage, not
  measurement); refit; iterate to a fixed point.
- **Loss-embedding evaluated (user proposal):** literal penalty L = χ² + λ·Incoherence REJECTED for
  v1 (non-likelihood → invalid Hessian errors; arbitrary λ; penalty active on noise → bias under a
  CORRECT model; opaque compromises). The principled in-loss variant IS included as a competitor:
  **M2 = Huber/Tukey robust loss** on r_b/σ_b (bounded influence of any mismodeled region,
  differentiable, standard constant) — downweights by residual size not demand-coherence, so it is
  a baseline, not the full answer.
- **Test matrix:** M0 traditional global χ² (control) / M1 gated fit (Gate I + Q_k + Q_split,
  freeze+flag) / M2 Huber — identical data and binning, raced through ladder steps 2–4; step 3
  (known true knobs + known injected artifact) is decisive.
- **Flagging pass.** At the physical-fit BFP: per-bin standardized residual map; contiguous regions
  >2σ that NO frozen-knob release can coherently fix = unknown-unknown candidates (report location,
  amplitude, and which knobs tried to absorb it).
Diagonal error model required for per-bin coherence to be rigorous → primary config: 20 uniform
bins/obs, stat ⊕ 5% uncorrelated syst (§16 style), 5 observables (CC0π dpt/dat ⊕ CC1π pN/dpTT/daT).

### Validation ladder (blueprint order)
1. **Build** `physical_fit.py` (Gates I+II + flagging on the differentiable bank; live progress).
2. **Closure**: ADoNIS Asimov + injected shifts in Gate-I-passing knobs (e.g. M_A_qe·1.2,
   res_norm·0.8) → physical fit must recover them (blind), no false freezes, no flags.
3. **Unphysical injection**: ADoNIS Asimov with bins ×2 above a kinematic threshold (e.g.
   δp_T > 0.3 GeV) → traditional global-χ² fit biases the knobs; physical fit must (a) keep knob
   estimates unbiased, (b) freeze/reject the incoherent pulls, (c) flag exactly the injected region.
4. **GENIE fake data** (the §16 samples, matched + MEC): does the physical fit isolate the
   representable differences (kF_sf, norm) while flagging the dpt-shape region instead of railing
   sabs/res_norm/M_A_qe?

Success criterion (user): identify understandable physics from parameter pulls WITHOUT bias from
the non-representable component. Step 3 is the decisive test (known ground truth + known injected
"unknown unknown").

### Step 1 RESULT — Gate I table (2026-07-10, `physfit_gate1.npz`, `scripts/altgen/physical_fit.py`)
27 knobs × 100 bins (5 obs × 20 uniform-p99 bins, overflow folded), σ² = (5%·d)² + ADoNIS-MC²,
Asimov at nominal, 20% priors (Eb_shift ±4 MeV, f_NN_cex ±0.1). **5/27 pass (shrinkage<0.5):**
Eb_shift 0.06, kF_sf 0.07, s_NN_el[pn] 0.34, f_NN_cex 0.44, M_A_res 0.50(borderline).
**Key observation:** M_A_qe(0.65)/qe_norm(0.88)/res_norm(0.82)/axial_strength(0.75)/sf_norm(0.76)
freeze NOT from insensitivity but from mutual degeneracy — the marginalized posterior punishes each
member of a collinear group individually. These are exactly the knobs the unphysical §16 fit railed:
Gate I removes that freedom a priori. (Future v2: fit Fisher eigen-combinations instead of freezing
group members; not v1.) Consequence: closure injections must live in the passing set →
step 2 injects kF_sf=1.10, M_A_res=0.85 (blind to the fit).

### Step 2 RESULT — closure PASSED, all methods (2026-07-10, `physfit_closure.npz`)
Inject kF_sf=1.10, M_A_res=0.85 (Gate-I passers; fit blind). M0/M1/M2 all: exact recovery
(kF_sf 1.100±0.008, M_A_res 0.851±0.013; non-injected knobs at truth ≤0.05σ), χ²_data=0.00
(noiseless Asimov + exact reweight; residual χ²=0.81 = exactly the prior penalty of the injected
truth, (0.5σ)²+(0.75σ)²). M1: ZERO false freezes (all Q_k p=1, Q_split p=1), zero flags. Caveat:
noiseless data makes the Q-tests pass trivially — fluctuated closure (data jittered by σ) queued
as the real null calibration of the p<0.01 thresholds.

### Step 3 RESULT — the decisive test: M1 unbiased, M0 biased up to 27σ (2026-07-10)
`physfit_inject2x.npz`. Data = nominal Asimov with the 12 dpt bins above 300 MeV ×2 (σ recomputed).
- **M0 (traditional): every knob corrupted** — kF_sf −11.2σ, s_NN_el[pn] −26.9σ, Eb_shift +8.0σ,
  M_A_res −2.8σ, f_NN_cex −2.7σ; χ²_data 608 (excess only partially absorbable). Its own Qk p-values
  (~1e-75) scream incoherence — a traditional fit computes nothing that acts on them.
- **M1 (physical): REFUSED every pull** — all 5 knobs fail Cochran's Q; Q_split=62.9/5 (p=3e-12) →
  all frozen → θ=nominal → unbiased by construction; flags = the injected dpt region.
- **M2 (Huber): bias reduced ~10× but not eliminated** (kF_sf −2.4σ, s_NN_el[pn] −1.8σ,
  f_NN_cex −1.75σ); still flags dpt [330,823] at +8.3σ. Bounded per-bin influence, but 12 bad bins
  still drag collectively.
Two v1 defects found: (a) all-frozen path reported θ/flags at the DISCARDED biased point instead of
nominal (spurious dat flag); (b) "freeze everything" is correct but blunter than optimal — the clean
bulk still measures the knobs. → **v2 EXCISE-REFIT**: flag → excise flagged bins → rerun the gated
fit on the clean region with the full Gate-I subset, iterate to no-new-flags.

**Step 3 v2 RESULT — COMPLETE PASS** (`physfit_inject2x_v2.npz`): x0 all-5 frozen (Q_split p=3e-12)
→ evaluated at nominal → excised EXACTLY the 12 injected bins (dpt [330,823], mean pull +9.0; no
spurious flags after fix (a)) → x1 on the clean region: zero Q failures, Q_split p=1 →
**every knob at truth to 0.00σ with real clean-region errors** (kF_sf 1.000±0.008, M_A_res
1.000±0.016, Eb 0.010±0.362, s_NN_el[pn] 1.000±0.060, f_NN_cex 0.500±0.044), χ²_data(clean)=0.
Same data: M0 −27σ biases vs M1 exact truth + artifact localized/labeled. The mandate's central
demonstration on controlled ground truth.

### Step 4 RESULT — GENIE-3M as data: the ladder completes (2026-07-10, `physfit_genie.npz`)
- **M0 (traditional)**: M_A_res +9.3σ, kF_sf −5.0σ (0.964), s_NN_el[pn] +4.4σ, f_NN_cex +6.3σ —
  four confident pulls, EVERY one Cochran-Q incoherent (p 1e-16..1e-36). What a standard tune would
  publish; provably compromises, not measurements. χ²_data 398.
- **M1 (physical)**: x0 all-frozen → excised 51/100 bins at nominal: dpt [83,782] (−8.7σ, the shape
  mismatch), ALL 20 dat bins (−4.5σ, the ~20% norm offset), 4 CC1π structures (pN [155,505] −3.6 /
  [996,1136] +3.6, dpTT [−435,−69] −3.3 / [113,387] −3.1). x1 on the 49 coherent bins (mostly CC1π):
  ALL knobs coherent (Q_split p=0.166, Qk p 0.02–1) → **M_A_res 0.868±0.039 (−3.4σ, genuine coherent
  RES-shape difference: GENIE BS softer than ADoNIS DCC), kF_sf 1.031±0.026, s_NN_el[pn] 0.939±0.100,
  f_NN_cex 0.514±0.057, Eb 1.5±0.8 MeV**. χ²_data(clean) 81.7/49 bins.
- **M2 (Huber)**: between (M_A_res +6.3σ, kF_sf −5.3σ, all Q-incoherent).
- **Reinterpretation 1**: the §12–14 "robust kF_sf≈0.93" was NOT a clean measurement — coherent-region
  value is 1.031±0.026; the 0.93 was largely absorption of the (excised) shape/norm mismatch. The
  physical fit catches on real foreign data exactly the failure mode it was built for.
- **Reinterpretation 2 (methodology)**: the whole-dat excision is REPRESENTABLE physics (a coherent
  norm offset qe_norm could absorb) misclassified as unknown-unknown because Gate I froze the entire
  degenerate norm group. → v3: fit one representative / the Fisher eigen-combination of each
  degenerate group before excising. (Not yet implemented.)

### Step 2/3 machinery — `scripts/altgen/physical_fit_run.py`
Modes: closure (exact-reweight data at injected θ) / inject2x (Asimov ×2 above threshold, e.g.
dpt>300 MeV; σ recomputed on injected data). Methods: M0 traditional (LM, χ²_data+prior),
M1 physical (M0 → Gate II: per-knob Cochran Q [resp. bins |J|·prior>0.3σ, p<0.01] + vector
split-fit Q_split [one-step GN, lo/hi half-bins per obs, prior-anchored → conservative; worst-|z|
knob frozen on failure] → freeze → refit ≤4 rounds → FLAG contiguous ≥2-bin |pull|>2 runs),
M2 Huber IRLS (c=1.345). All fits on the Gate-I subset only, live per-iteration logging.

### Fluctuated closure (null calibration, seed 20260710) — PASS
Truth (kF_sf=1.10, M_A_res=0.85) + 1σ jitter: M1 zero false freezes (Qk p 0.92–0.96), zero flags,
Q_split p=0.23, recovery ≤2.3σ, χ²_data 73.7/95. Gates quiet on healthy noisy data (one seed;
multi-seed calibration = future work). `physfit_closure_fluct.npz`.

### 5-parameter closure (2026-07-10) — PASS, M0 and M2 (`physfit_closure5.npz`, fig7)
GENIE fitting parked (user). Inject ALL Gate-I knobs (M_A_res 0.85, kF_sf 1.10, Eb 3 MeV,
s_NN_el[pn] 1.25, f_NN_cex 0.40); fit blind, current binning (dat full [0,pi]). M0 and M2 converge
to the IDENTICAL point: every knob ≤0.5σ from truth (largest s_NN_el −0.5σ), χ²_data 0.31.
Eb_shift recovered 2.77±0.57 vs 3.0 — the weakest direction, consistent. Run npz now persists
binned data/σ/model curves → figures render instantly without bank recompute.

## 18. Q^2-dependent unknown-unknown + the g(Q^2)-nuisance closure (2026-07-10)
Injected mod: w(Q^2)=1-0.2*exp(-Q^2/0.3 GeV^2) per event (RPA-like; mean suppression 9.4%,
Q^2 median 0.18). Suite EXPANDED to 9 observables (+CC0pi pmu/cosmu, +CC1pi ppi/cospi = the
Q^2-carrying kinematics; cosmu forward "spike" verified REAL physics: dsig/dcos rises smoothly
3.4x over [0.9,1.0], zero pileup at cos=1). Gate I: 7/27 now pass (M_A_qe 0.48 and
res_axial_strength 0.36 unlocked by the new variables). 7-param closure (no mod): PASS <=0.4 sig.
- **Naive fits vs the mod (q2mod / closure_q2mod, M0=M2)**: the mod is absorbed INSIDE the manifold
  — chi2 8.9-10.7, ALL gates silent, zero flags; bias conspiracy across the axial sector
  (closure_q2mod: M_A_res +1.8σ, res_axial −2.2σ, kF_sf −1.7σ, M_A_qe −1.0σ; FSI knobs unharmed).
  Unlike the x2 tail (loud/incoherent), a smooth physics-like unknown-unknown defeats residual- and
  coherence-based defenses: nothing is left in the residuals to test.
- **THE FIX — flexible Q^2-shape nuisance**: per-event multiplicative g(Q^2;c), 5 knots at
  Q^2={0.02,0.08,0.18,0.45,1.2}, linear in log Q^2, weak ±0.5 priors, fit jointly (12 params).
  `p9_closq2_nuis`: **closure RESTORED** — all 7 knobs <=0.4σ of truth (M_A_qe +0.08, M_A_res +0.33,
  res_axial −0.31, kF_sf −0.02, Eb −0.28, s_NN_el −0.40, f_NN_cex +0.32); errors honestly inflated
  where g degenerates with axial shape (M_A_qe ±0.022→0.058, res_axial ±0.067→0.110) and barely for
  kF_sf (±0.009→0.011). **g(Q^2) RECONSTRUCTS the injected w(Q^2)** within errors at every knot
  (mid-knot 0.878±0.068 vs true 0.890 = 1.8σ DETECTION of the suppression). M0 and M2 identical.
  chi2_data 0.3. Fig `physfit_fig9_q2nuis.png`; runs p9_* npz (curves persisted).
- Consistency run (`p9_q2mod_nuis`, no knob injection): PERFECT separation — all 7 knobs at nominal
  to <=0.05σ, g(Q^2) alone carries the full modification (0.843/0.862/0.887/0.965/1.004 vs true
  0.813/0.847/0.890/0.955/0.996, all within errors), chi2_data 0.0.
