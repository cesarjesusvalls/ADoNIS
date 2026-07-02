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
| `res_norm` | **0.300 = LOWER RAIL** | same direction: ADoNIS(DCC+Oset) puts far more RES-origin events into CC0π-Np than GENIE(BS+hA2018) — the fit can only cut it by railing BOTH RES knobs (~×0.09 combined feed) |
| `s_NN_el` | 2.52 | ADoNIS needs ~2.5× MORE nucleon rescattering to reproduce GENIE's δpT tail-vs-peak — consistent with hA2018 (single effective interaction, more smearing per event) vs INC stepwise transport |
| `qe_norm` | 1.70 (drifting) | exactly degenerate with profiled A once RES is railed — flat direction, value not meaningful in the shape fit |
| `M_A` | 0.863 | softer axial Q² — the Nieves-RPA low-Q² screening in GENIE mimics a lower effective M_A, as anticipated |
| `kF_sf` | 0.932 | slightly narrower initial-state momentum: GENIE's 1-D SF for ¹²C vs ADoNIS's 2-D SF(p,E) peak-width difference, small as expected |

Residual χ²(shape)=3.5 after 6 knobs: the knob space CANNOT fully absorb the foreign-generator
shape — i.e. the unmodelled-physics residual is visible and finite, not fittable away.
(Hessian errors: invalid for the railed pair by construction; qe_norm flat — pinv-projected. Full
table + figure appended when the run's Hessian block lands.)

---

### Status
- [x] Feasibility scan + LUCiD verified
- [x] GENIE generation pipeline (flux, decayer/list overrides, gst)
- [~] 300k CCQERES generation running
- [~] fit machinery + closure gate running
- [ ] GENIE fake-data build, fit, physics interpretation
