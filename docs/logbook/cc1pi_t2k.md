# Logbook: T2K CC1π⁺Np STV (paper figure pair pN / δp_TT)

Blueprint port (CLAUDE.md): same flux/generators/cascades as CC0π; signal = pion SURVIVES
as π⁺ + leading proton; RES-H nonzero (free-proton π⁺ always survives). ACHILLES side
pre-existing: t2k_cc1pi_tki_achilles.npz (54790 events from T2K_CH_virt.hepmc, tight cuts).

## #1 — first blueprint iteration (2026-06-12, commit 521a94e + this entry)

- scripts/cc1pi_fig_tki.py: RES-C (200k x 4 seeds, Cylinder cascade, early_exit) + RES-H
  (50k x 4); selection/observables mirror extract_t2k_cc1pi_tki.py exactly (tight windows,
  theta<70 deg; NUISANCE hydrogen flat-daT throw; carbon-mass pN formula; daT release bins
  in DEGREES). Output npz: t2k_cc1pi_tki_adonis_blueprint.npz.
- sigma_CC1pi(tight): RES-C 1.2591e-6 + RES-H 5.515e-7 nb. Integrals [nb per CH]:
  ACH 1.747e-6 / ADO 1.810e-6 (+3.6%) / T2K 1.596-1.82e-6 -- normalization bookkeeping
  confirmed (per-nucleon cm2 x 1e33 x 13).
- ACH/ADO ratio chi2/ndf: pN 2.83, dpTT 1.63, daT 1.98 (MC errors, absolute).
  Localized SIGNIFICANT residuals (>2sigma): pN bin 240-600 MeV ratio 0.90 (ADO high ~10%);
  dpTT last bin (+500) ratio 0.81 (ADO high ~20%); daT mid/high bins ~0.93-0.96.
  ADO sits ABOVE ACHILLES in the FSI-sensitive tails.
- Known modeled difference (declared in the script docstring): the ADoNIS pion cascade does
  not emit pi-scatter recoil nucleons into the final state; ACHILLES does. Plausible driver
  of the tail residuals (knockouts repopulate the leading-proton window and smear pN/dpTT) --
  NOT yet proven; needs a dedicated check (e.g. ACHILLES-side extraction excluding knockout
  protons, or adding knockout emission to DiscreteCascadeFSI).
- vs data: both generators describe the data shape; data errors are large compared to the
  ACH/ADO differences except in the tails.

Open next steps:
1. Diagnose the tail residuals: instrument the ACHILLES extraction to tag whether the
   leading proton is the RES nucleon or a cascade knockout; if knockouts dominate the
   discrepant bins, add pi-scatter knockout emission to DiscreteCascadeFSI (the struck
   index/recoil are available at the scatter step).
2. Walk/M_A records for the CC1π chain (same pattern) once the forward agreement is settled.

## #2 — knockout emission + ROOT CAUSE of the carbon excess (2026-06-12, commit 58d49c1)

- Knockout emission implemented (DiscreteCascadeFSI.last_scat_ko, ACHILLES FinalizeMomentum /
  UpdateKicked mirrored; bit-exact gates pass). Verdict: fidelity up, residual NOT fixed --
  integral gap +3.6% -> +4.5% (the sign analysis predicted this: knockouts only ADD ADO signal).
- C/H decomposition (exact hydrogen tag = |dptt| < 0.5: free proton has dptt == 0; only 0.24%
  of carbon events fall there; the status-code is_h tag was BROKEN, 0/18012 tagged):
  ADO/ACH = 0.971 on H (good), 1.079 on CARBON -> the offender is an ~8% carbon CC1pi excess.
- Pion-kinematics split of the carbon cell (100k x 4 ADO vs re-extracted ACH npz with
  pi_p/pi_cth/lp_p): ACH/ADO vs pi momentum [150..1200, 7 bins] =
  0.98 0.97 0.91 0.78 0.62 0.57 0.55 -- the excess is ENTIRELY high-momentum surviving pi+;
  angle and leading-proton spectra comparatively flat. nsc split: excess not explained by
  scatter counts (78% of ADO-C signal has nsc=0).
- ROOT CAUSE (code-level): ACHILLES MesonBaryonAmplitudes includes the INELASTIC 2-body
  channels piN -> etaN (W>~1486), K Lambda (>~1609), K Sigma (>~1689)
  (src/Achilles/MesonBaryonAmplitudes.cc channel table; CrossSectionsW[i_i][i_f];
  GetAllCSW/SelectChannel). A high-W pion in ACHILLES can CONVERT and leave the pi+ signal.
  ADoNIS's cascade competition has ONLY piN->piN (cascade_mb sig_io) -- PHASE_E recorded
  "etaN/KLambda/KSigma soft-blocked", but the ANL tables ARE local (ANL_0-{1,2,3}.dat) and
  anl_xsec.py already parses eta_n/K0Lambda sigma. The earlier "sigma matches <0.5%/W"
  validation was at Delta-region W where the inelastic channels are closed -- consistent.
- FIX PLAN (next session): port CrossSectionsW for i_f in {1,2,3} (charge-channel CG
  projections; etaN/KLambda pure I=1/2, KSigma I=1/2+3/2); add the conversion branch to the
  cascade competition (converted pion -> not absorbed, not pi+; correct for BOTH CC0pi and
  CC1pi signal definitions; eta/K products' secondary protons neglected, declared); extend
  brec records + pion_branch_reweight with the inelastic sigma (unscaled by sscat initially);
  gates: sigma vs the published eta/KLambda shapes + bit-exact records + CC1pi rerun
  (expect the pi_p ratio slope 0.55->1 and the carbon integral 1.08->~1).

## #3 — conversion channels implemented; REFRAME: the offender is the PRIMARY spectrum

- Conversion channels PORTED (anl_xsec.conversion_sigma_grid: ACHILLES initIso CGcof +
  CalcCrossSectionW_grid verbatim incl. the ANL-code masses 138.5/938.5 and the KSigma
  I=1/2 sign quirk AS CODED; gate: identical to the Phase-E eta/KLambda functions when the
  mass convention is aligned -- the 4.3% gap was purely the PF masses, ACHILLES uses 138.5).
  Wired as a 3rd cascade branch (same uniform roll -> bit-exact when si=0; brec now
  (branch, sa, ss, si, nh); pion_branch_reweight = per-branch likelihood ratio; converted
  pion -> pid_pi = -1, not absorbed; 10 gates pass). MEASURED effect: ~0.3% removal --
  correct fidelity, NOT the offender (sigma_conv 0.5-3 mb vs ~20-40 mb elastic).
- Run-card check: T2K cascade = PionInteraction {MesonBaryonInteraction + 
  PionAbsorptionOneStep} + NucleonNucleon GiBUU -- structurally what ADoNIS models.
- DECISIVE: PRIMARY-level comparison (T2K_CH_virt_nofsi.hepmc extraction vs ADoNIS res
  events with NO cascade, same tight selection): the ACH/ADO pi-momentum SHAPE ratio falls
  ~1.0 -> ~0.4 across 150..1200 MeV ALREADY AT THE PRIMARY LEVEL (same slope as post-FSI;
  absolute scale of the nofsi extraction not comparable -- run-specific GenXS constant).
  => the cascade is NOT the offender. ADoNIS's T2K-flux-folded RES primary produces ~2x
  more high-momentum in-acceptance pi+ than ACHILLES at p_pi ~ 1 GeV. The fixed-energy
  RES validations (sigma, dsigma/dW, dsigma/dQ2 at E_nu = const; channel fractions) never
  probed the flux-folded high-p_pi acceptance corner.
- NEXT (fresh session): compare ADO-vs-ACH(nofsi) primary W and E_nu distributions of the
  signal events (extend the extractor to store W/E_nu; ADO from res events directly).
  Candidate causes to discriminate: high-W amplitude treatment (spline coverage, W>1600
  tail noted in res_amps2_frame_fix.md), the flux-fold high-E_nu tail, 3-body phase-space
  vs ACHILLES at high W, or the W<2000 hard-cut handling.

## #4 — disaggregated CC1pi (cc0pi-style) localizes the offender to the W TAIL

- scripts/cc1pi_disaggregated.py (cells {RES-C, RES-H} x {no-FSI, FSI} with vertex W/Q2/Enu)
  + extractor extended (status-2 struck nucleon -> vertex W; beam pid 14 -> Enu); both
  hepmcs re-extracted (FSI 54790 ev; nofsi 9136 ev, smaller-stats run -> SHAPE comparisons).
- No-FSI carbon SHAPE ratios ACH/ADO (unit-normalized):
    Enu  [200..3000]:  0.94 1.01 1.04 1.06 0.95 1.11   -> FLUX FOLD EXONERATED
    Q2   [0..2 GeV^2]: 0.98 1.02 1.01 1.02 0.97 0.90 1.03 (0.46 last bin)
    W    [1100..2000]: 0.97 1.10 0.99 0.47 0.28 0.41 0.51 0.89 0.58
    pi_p [150..1200]:  1.02 1.05 1.04 0.76 0.58 0.41 0.36  (the kinematic image of W)
  => ADoNIS RES primary over-populates W ~ 1400-1800 (second-resonance region) by 2-3x
  in the flux-folded in-acceptance sample. mean W: ACH 1234.0 vs ADO 1245.8 (the bulk
  agrees; it is the TAIL). Cascade fully exonerated (the FSI cells inherit the primary).
- Consistency: ADO RES-C no-FSI sigma 2.8244e-6 nb reproduces the ad-hoc primary run.
- NEXT: controlled fixed-energy gate at E_nu = 2 GeV (mono, free p + 12C): ACHILLES
  dsigma/dW oracle vs ADO res_xsec with ratio+chi2 -- isolate whether the 2-3x W-tail
  excess reproduces at fixed energy (then bisect: amplitudes/spline vs phase space vs
  W-dependent factors in currents_pi_dcc at high W). The earlier fixed-E validations were
  at lower E_nu where W>1400 is barely populated.

## #5 — CORRECTION: the "2-3x primary W-tail excess" (#3/#4) was a DIAGNOSTIC ARTIFACT

- My no-FSI diagnostic selections (ad-hoc primary script AND cc1pi_disaggregated res_C
  fsi=False) were MISSING the proton-PID requirement: for n -> n pi+ events (large at high W)
  ADoNIS counted NEUTRONS in the proton window, inflating its selected W tail 2-3x. The
  figure-level chains (cc1pi_fig_tki) always had the pid check -- their numbers stand.
- With the proton PID required on both sides (ACH: any in-window PROTON via the new prot_ok
  extraction; ADO: Npid==2212), the FULL-signal acceptance vs W agrees to <=13%:
  ACH/ADO = .146/.147 | .078/.084 | .054/.061 | .025/.017  (W 1100-1400|1400-1600|
  1600-1800|1800-2000).  n_pi == 1 for ALL ACH no-FSI RES events (multi-meson veto nil).
- Corrected conclusions: vertex distributions agree unselected (W shape 0.92-1.01);
  the GENUINE residual is an 8-15% pi+-acceptance excess at W 1400-1800 driven by
  10-20% shape differences in the piN final-state kinematics (pi cos-theta wiggles
  0.73-1.22 within W slices; pi |p| slice edges) -- consistent with PHASE_I's known
  cos-theta* chi2/ndf 4.5-7.3 and integrating to the observed ~8% carbon CC1pi excess.
- cc1pi_disaggregated.py res_C(fsi=False) pid bug FIXED (Npid==2212).
- NEXT: the angular distribution at high W -- fixed-(W,Q2) angular comparison of the
  exclusive amps2 against an ACHILLES RESDUMP-style oracle in the second-resonance region
  (the Delta-region angular validation passed; W>1400 was never gated).

## #6 — controlled mono-2GeV carbon gates: primary EXONERATED at <=1%; offender = NN INELASTIC in the nucleon cascade

- New oracles (achilles:resgen, mono 2 GeV, carbon, 12C-only run card): no-FSI 400k events
  (1m49s free-p; carbon similar) + WITH-cascade run (159,974 events written -- NEvents
  semantics; ALWAYS count E-lines for denominators).
- Free proton 2 GeV: W shape +-3%, cos-theta* flat in all W slices -> 3-body measure,
  amplitudes, angular distributions all validated at high W (amps2 RESDUMP audit: ratio
  1.000, spread 1e-4, ALL W and cos-theta* bins to 1950 MeV).
- Carbon 2 GeV no-FSI (400k vs ADO mono port): unselected W shape 0.991-1.007; TOTAL tight
  acceptance 0.2641 vs 0.2641; per-W-bin acceptance ratios 1.001/0.999/0.967/0.960 ->
  the ENTIRE PRIMARY CHAIN agrees at <=1% (the earlier "no-FSI carbon excess" was the
  small-stats T2K nofsi sample: ~330 events in the discrepant bins, 1.5-2 sigma noise).
- Carbon 2 GeV WITH cascade: signal fraction ACH 0.1478 (23640/159974; my first 2.9x claim
  was a 400k-denominator error, caught by the one-path leg recount) vs ADO 0.1697 ->
  ADO +15%. Pion fates MATCH (pi+ survival 57.6/57.1%, pi0 20.0/20.4, abs ~20/21,
  surviving pi+ spectra mean 342/350, frac>700 MeV 8.1/8.8%) -> pion cascade fine at 2 GeV.
- OFFENDER: the proton leg + meson veto. ACHILLES NucleonNucleon = GiBUU with
  ResonanceMode: Decay -> NN -> N Delta -> N N pi INELASTIC channels: fast protons are
  degraded harder (drop out of the 450-1200 window) and extra pions are created (measured:
  1.6% multi-pion events in the ACH cascade sample; impossible in ADoNIS, whose nucleon
  cascade is ELASTIC-only). At T2K flux most protons sit below the NN-inelastic threshold
  -> dilutes to the observed +8% carbon excess. CONSISTENT with all observations.
- NEXT IMPLEMENTATION: port the GiBUU NN inelastic channel (NN -> N Delta, Delta -> N pi)
  into DiscreteNucleonFSI (sigma_inel from ACHILLES NucleonNucleon.cc GiBUU mode; Delta
  production degrades the leading proton + emits a pion that enters the meson veto);
  extend the kind-1 records accordingly; gates: sigma_inel vs ACHILLES tables, fates +
  signal fraction at mono-2GeV carbon (expect 0.1697 -> ~0.148), then the T2K CC1pi figure.

## #7 — NN -> N Delta inelastic PORTED; explains ~2% of the 15%, remainder OPEN

- adonis/fsi/nn_inelastic.py: faithful Dmitriev-Sushkov port (MatNN2NDelta constants verbatim;
  L=1 Blatt-Weisskopf effective width; Particles.yml masses; table = delta++ @ pcm=1GeV scaled
  by isofactor/pcm exactly as SigmaNN2NDeltaInterp; charge combinatorics summed 4/3 pp|nn,
  2/3 pn). Gates: sigma(pp->NDelta) peak ~24 mb at sqrts~2.4 (literature-consistent); mass
  sampling BW-shaped; share at p_lab 1.2/1.5 GeV = 19%/46%; 1.2 GeV proton beam through 12C:
  made-pion fraction 9%.
- Wired into _propagate_nucleon_discrete (flagged nn_inelastic=True; fold_in keys -> elastic
  stream untouched; Pauli on both product nucleons; walk continues with the leading product;
  made_pi flag -> DiscreteNucleonFSI.last_made_pion; CC1pi chain vetoes made-pion events).
  7 cascade gates pass.
- MEASURED at mono-2GeV carbon: signal fraction 0.1697 -> 0.1670 (-1.6% rel). The in-window
  protons (450-1200 MeV) are mostly BELOW the NN->NDelta threshold; the strongly-inelastic
  >1.2 GeV protons rarely land in the window. The ACH-ADO gap remains ~ -12% at 2 GeV
  (ACH 0.1478), i.e. NN inelastic was correct fidelity but NOT the main offender.
- Declared approximations: created pion treated as final (no re-absorption, ~25% too strong a
  veto -- conservative); leading-product isospin bookkeeping unchanged; mpi=138.04 in the
  Delta decay split.
- REMAINING SUSPECTS for the -12% @2GeV (diluted ~+5-7% at T2K flux): (a) ACHILLES's FULL
  recursive multi-nucleon cascade (every kicked nucleon re-cascades; ADO tracks leading + 2
  knockout generations) -- more in-window protons LOST and more soft protons made; (b) NN
  elastic sigma at p>1.2 GeV; (c) Delta re-interactions. NEXT MEASUREMENT: with-cascade
  P(in-window proton) and post-cascade leading-proton spectrum, ACH (0.6008 measured) vs ADO
  at mono-2GeV; then a proton-beam-through-carbon oracle (resgen image, proton beam config)
  to gate the nucleon transport in isolation at 1-2 GeV.

## #8 — standing after all fixes (final rerun, 100k x 4)

- T2K CC1pi integrals [nb/CH]: ACH 1.747e-6 / ADO 1.818e-6 = +4.1% (was +4.7% pre-NN-inel,
  +3.6% at iteration 1 with fewer fidelity mechanisms). Carbon cell: 1.2666e-6 vs ACH-C
  1.1808e-6 = +7.3%. H cell unchanged (-2.9%).
- All three fidelity mechanisms now in (knockouts +1%, conversion -0.3%, NN-inel veto -0.6%);
  net ~cancel. The remaining +7% carbon = the open nucleon-transport proton-leg gap at
  p >~ 1 GeV (logbook #7: -12% at mono-2GeV, diluted by the T2K flux). Next: proton-beam
  12C transport oracle.

## #9 — proton-beam transport oracle: components VALIDATED; remainder = chain COMPOSITION

- New oracles: achilles-cascade CrossSection mode, PROTON beam (PID 2212), Cylinder, T2K NN
  interactions, 600/900/1200/1500 MeV (60k/38k/1.5k/5.2k events; frequent SIGSEGV at high p,
  partial output valid). KEY: cascade-mode hepmc contains INTERACTED events only -- condition
  the ADoNIS side on >=1 interaction (my first unconditioned comparison was misleading;
  caught via P(nsc=0)=0.80 vs the MFP expectation).
- CONDITIONED comparison (ACH vs ADO): <p_lead> 434/432 (600), 638/634 (900), 792/809 (1200),
  913/985 (1500); P(450-1200): 0.528/0.521, 0.811/0.801, 0.851/0.904, 0.652/0.702;
  P(made pi): 0.000/0.000, 0.012/0.016, 0.149/0.193, 0.326/0.393.
  => ELASTIC transport agrees <=1.5% at 600-900; ~5-8% residuals at 1200-1500 (ACH stats
  +-3-5%); NN-inelastic pion rates same ballpark (ADO slightly higher -- no re-absorption of
  created pions, declared).
- CONCLUSION: every individual component is now validated at the few-% level (primary chain
  <=1%, pion cascade fates ~1%, nucleon transport <=1.5-8%). The remaining in-event gap
  (ADO/ACH +12% signal fraction at mono-2GeV carbon; +7% T2K CC1pi carbon) is the chain
  COMPOSITION: ACHILLES evolves ONE shared nuclear state (pion + all nucleons co-cascade,
  joint consumed set, true vertex positions, every knockout a proton candidate, full
  recursion), while ADoNIS factorizes into independent cascades with resampled backgrounds
  and a 3-candidate leading-proton set.
- NEXT PHASE (architecture): a shared-state event cascade for the discrete chain -- one
  nucleus per event, pion and nucleons co-evolving with a common consumed/positions state and
  full kicked-particle recursion (still kind-1: walks at nominal, records per branch). This
  is the remaining item between the current +7% and the <=1% goal for CC1pi; CC0pi is
  already at ~1% because absorption events have little nucleon-side activity in-window.

## #10 — shared-state event cascade: DESIGN (implementation in progress)

adonis/fsi/cascade_event.py: ONE nucleus per event, all hadrons co-evolve.
- Fixed particle slots (n, K_PART=8): slot0 = pion (RES), slot1 = primary nucleon; free slots
  take products (pi-scatter recoils, NN recoils, NN->NDelta decay nucleons + created pions).
- Per step (0.04 fm): per-slot slab search against the SHARED background (n, A=12) with the
  SHARED consumed mask; per-slot sigma by species (pion: oset abs + MB elastic + conversion;
  nucleon: NN elastic + NN->NDelta); branch picks as in the existing kernels; products spawn
  into free slots (incl. created pions, which can later be ABSORBED -> fixes the
  too-strong made-pion veto); Pauli per product; formation zones per ACHILLES.
- Escape: in-event particles are INTERNAL (ACHILLES status) -> sphere escape |pos|>radius &
  outward for ALL particles (the external z-plane rule is cascade-mode-only).
- Both particles start at the TRUE struck-nucleon vertex (consumed), not resampled positions.
- early_exit while_loop (no active particle can interact); max_steps a safety bound.
- v1 nominal-only (no kind-1 records); records per-branch can be added once forward physics
  gates (same compressed-slot pattern, now per particle slot).
- Gates: (1) pion-only and nucleon-only limits reproduce the existing single-particle
  cascades statistically; (2) mono-2GeV carbon signal fraction 0.167 -> ~0.148 (ACH);
  (3) T2K CC1pi carbon +7% -> ~1%.

## #11 — event cascade CONVERGENCE state (fz fix; anchor set established)

- fz-ordering bug fixed (4c940bb..): formation zones now from PRE-update momenta (the
  degenerate fz(p_new,p_new) froze nucleons after first scatter).
- v2 absorption (Pauli rejection + products + partner consumption): correct physics,
  ~zero net at T2K (abs products >> kF at the Delta).
- ANCHOR SET at T2K carbon (cascade on), ACH unselected per-leg (425k RES events with
  >=1 pion; denominators conditioned on a SURVIVING pion) vs ADO event kernel (fz-fixed):
    P(pi+ exactly-one & window | surv):  ACH 0.3267   ADO 0.3359  (+2.8%)
    P(in-window proton | surv):          ACH 0.5286   ADO 0.5315  (+0.5%)
    P(prot | pi-window):                 ACH 0.4507   ADO 0.4516  (+0.2%)
    JOINT (with mu) | surv:              ACH 0.0876   ADO 0.0863  (-1.5%)
    P(abs | RES-C in-event):             ACH 0.2168   ADO 0.2239  (+3.3% rel)
  Pion fates vs factorized kernel: identical (survival .5645/.5689).
- mono-2GeV gates: v1 0.1474, v2(escape) 0.1472 vs ACH 0.1478 (fz fix re-gate pending).
- FULL-STATS T2K figure (60k x 4, fz-fixed): RES-C 1.0992e-6 vs ACH-C 1.1808e-6 = -6.9%;
  total ADO/ACH = 1.650/1.747 = -5.6%. Trajectory: factorized +7.9% -> event v1 -6.3% ->
  fz-fixed -6.9%. Decomposition of the -6.9%: ~1% absorption rate, ~1.5% joint
  conditional, ~1-2% sigma_tot bookkeeping, remainder within the 1-2% per-measurement
  stats of these run sizes.
- NEXT: converge below 2% with HIGHER-stats anchors (300k+ per measurement), starting
  from P(abs) (+3.3% rel: the oset/partition path in the event kernel vs the factorized --
  note the factorized had the SAME 0.223, both ~3% above ACH in-event 0.2168, despite the
  beam-oracle match: revisit the in-event absorption environment: struck-vertex density /
  consumed-vertex handling), then the +2.8% pi-window leg (DCC scatter-angle frame check
  vs _two_body_cm_scatter).
