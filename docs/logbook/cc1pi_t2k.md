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

## #12 — high-stats anchors: absorption NOT the cause; decomposition does not yet close

- DCC scatter frame: event-kernel _dcc_scatter == validated _two_body_cm_scatter exactly
  (cos_cm reproduced to print precision on identical inputs) -- frame exonerated.
- Absorption, apples-to-apples "NO PION IN FINAL STATE" (counts NN-inel created pions
  correctly; the earlier 0.2239 was the slot flag): ACH 0.21640 (N=530794, +-0.06%) vs
  ADO event-kernel 0.20990 (400k weighted) -- ADO UNDER-absorbs by 3% relative, i.e.
  MORE pion events -> would push the integral UP. The -6.9% carbon integral therefore
  cannot be explained by the absorption rate; combined with the matched conditionals
  (#11), the decomposition does NOT close -- one of the leg comparisons still hides a
  selection/denominator subtlety, or the per-leg stats are undersold.
- NEXT HARNESS (definitive, ~2h compute): ONE paired 300k x 4 measurement of the FULL
  signal fraction P(sel | RES-C): ADO = event-kernel chain exactly as the figure;
  ACH = signal-count / RES-C-count from T2K_CH_virt.hepmc using proc ids (both
  unambiguous event counts, no SCALE, no per-leg denominators); then bisect WITHIN that
  single harness by relaxing one cut at a time (mu -> pi-window -> exactly-one ->
  proton) so every step shares identical denominators. This removes the cross-script
  denominator hazards that have produced three artifacts already.

## #13 — DECISIVE: cut-by-cut bisect closes it; per-event physics agrees +3.5%

Single-harness raw-count bisect (identical cuts, NO SCALE, carbon RES-C both sides):
- ACH cumulative: all 1.0 | +mu 0.6261 | +1pi+ 0.3568 | +piwin 0.1604 | +prot 0.0678
- ADO cumulative: all 1.0 | +mu 0.6292 | +1pi+ 0.3637 | +piwin 0.1688 | +prot 0.0702
- per-STEP survival ADO/ACH: mu 1.0048 | 1pi+ 1.0143 | piwin 1.0321 | prot 0.9839
=> FINAL signal fraction ADO/ACH = 0.0702/0.0678 = +3.5%, EVERY step within 1.6-3.2% of 1,
   the proton step even slightly BELOW 1.  The shared-state event cascade brings the
   per-event CC1pi physics to ~3.5% (from the factorized chain's +7.9%).
- CONTRADICTS the figure's -6.9%: with the per-event counts agreeing at +3.5%, the figure's
  ABSOLUTE integral gap is a NORMALIZATION/SCALE artifact in cc1pi_fig_tki.py (the ACH hepmc
  weight SCALE = 6.266697e-02*1e-3/8.905269e+05, and/or the RES-C/RES-H sigma normalization),
  NOT a cascade physics gap.  The cut-bisect is the clean apples-to-apples comparison; the
  figure SCALE must be re-derived.
- NET RESULT of the autonomous push: factorized +7.9% carbon -> shared-state event cascade
  +3.5% per-event (cut-bisect), all per-step legs <=3%.  Remaining to <=1%: the ~1.4% pi+
  survival (piwin step 1.032) -- consistent with the known PHASE_I cos-theta* softness in
  the DCC angular distribution feeding the pion acceptance -- plus run-size stats.
- NEXT: (1) re-derive the figure SCALE so the absolute integral matches the +3.5% count
  (decouple normalization from cascade physics in cc1pi_fig_tki.py); (2) the residual ~1.4%
  piwin step -> DCC angular distribution at the relevant W (PHASE_I follow-up).

## #14 — SCALE_ACH removed; the -6.9% is NOT a normalization artifact (hypothesis falsified)

- All hardcoded `SCALE_ACH = 6.266697e-02*1e-3/8.905269e+05` magic numbers removed from the
  codebase (7 figure scripts + the two extractors). New adonis/data/oracle/normalization.py:
  hepmc_norm() reads GenCrossSection from the header + sums event weights in one lightweight
  pass; weight_to_nb = GenXS[nb]/sum_w_all. Extractors store it in the npz; figures read via
  weight_to_nb_of(). Existing npz migrated in place. grep clean: zero hardcoded constants.
- DECISIVE NUMBER: file-derived FSI factor 7.037066e-11 == the old hardcoded value EXACTLY
  (GenXS 6.266697e-2 pb, sum_w 8.905269e5, N=2e6). The no-FSI run is 7.285582e-10 (its own
  GenXS 6.2578e-2 / sum_w 8.589e4). cc0pi figure ACH integral unchanged (1.588e-5).
- THEREFORE the #13 reasoning that "the figure -6.9% is a SCALE/normalization artifact" is
  FALSIFIED. The ACHILLES normalization was correct all along. The CC1pi absolute -6.9%
  carbon (event cascade) is a REAL absolute-cross-section difference, consistent with the
  cut-bisect interpretation only if ADoNIS's absolute RES-C->CC1pi sigma is ~7% below
  ACHILLES's selected sigma -- i.e. a genuine physics item, not plotting.
- RECONCILING #13 (+3.5% fraction) WITH -6.9% (absolute): sigma_abs_ADO/ACH =
  (frac_ADO/frac_ACH) x (sigtot_ADO/sigtot_ACH); +3.5% fraction x -6.9% absolute =>
  sigtot_RES-C_ADO/ACH ~ 0.90.  The OPEN question is now purely: does ADoNIS's absolute
  RES-C CC1pi cross section sit ~7-10% below ACHILLES, and if so where (RES total sigma at
  the T2K flux, vs the +-1% fixed-energy validations)?  This is the next physics target;
  the normalization plumbing is now clean and first-principles on both sides.

## #15 — pN-tail discrepancy is CASCADE COMPOSITION (primary ruled out by the sign flip)

- The pN high-momentum tail FLIPPED sign across cascade engines, on the IDENTICAL primary
  (same res_xsec + spectral function):
    pN 240-600 ACH/ADO:  factorized #1 = 0.90 (ADO 11% HIGH)  ->  event cascade = 1.30 (ADO 23% LOW)
    dpTT wide bin:        factorized #1 = 0.81 (ADO HIGH)      ->  event cascade = 1.54 (ADO LOW)
    carbon integral:      factorized +3.6%/+7.3% (HIGH)        ->  event cascade -5.6% (LOW)
- A tail that flips sign when ONLY the cascade changes CANNOT be primary (Fermi/SRC/spectral)
  -> the no-FSI primary pN test is unnecessary (would be misdirected). RULED OUT.
- Diagnosis: the two compositions BRACKET ACHILLES on the high-pN leading proton:
    * factorized (leading N + 2 knockout generations) OVER-populates the high-pN tail;
    * shared-state event cascade (all nucleons co-evolve, shared consumed set, full re-cascade)
      OVER-depletes it.
  So the headline +3.5% cut-bisect fraction hides a real SHAPE problem: the event cascade
  removes too much high-momentum leading-proton strength; pN (and dpTT wings) are where it shows.
- NEXT (running): factorized vs event leading-proton |p| + pN spectra on IDENTICAL primary
  events, to localize which event-cascade mechanism over-depletes high-pN (leading-proton
  consumption / "highest-momentum surviving proton" competition / re-cascade depth).

## #16 — REVERT: event cascade removed; differentiable factorized chain restored

User decision: the shared-state event cascade is NOT faithful to ACHILLES (cross-section
ports are faithful, but the TRANSPORT was a from-scratch discrete-Glauber reimplementation,
not a port of Cascade.cc; multiple declared approximations + slot-cap truncation). Validating
a second, non-differentiable cascade is pointless when differentiability is required anyway.
- DELETED adonis/fsi/cascade_event.py and scripts/cc1pi_cut_bisect.py (the latter was also
  FLAWED: it counted ACHILLES events UNWEIGHTED while summing ADoNIS cross-section weights;
  ACHILLES is 99-percentile-unweighted with a ~1% high-weight tail sitting preferentially at
  high pN, so the "+3.5% per-event agreement" was apples-to-oranges and is RETRACTED).
- cc1pi_fig_tki.py res_C restored to the differentiable factorized chain (DiscreteCascadeFSI
  pion + DiscreteNucleonFSI nucleon + knockout re-cascade), verbatim from ae4ee6d^, keeping
  the weight_to_nb_of (no hardcoded SCALE) cleanup. cc1pi_disaggregated.py inherits it via F.res_C.
- The cascade_discrete.py fidelity additions (knockout emission, piN conversion, NN->NDelta
  inelastic) are KEPT -- they are part of the factorized chain and were independently gated.
- STANDING (correctly-weighted, factorized): CC1pi carbon ~+7% (ADO high), per logbook #8.
- NEXT: disaggregated ACH/ADO ratio checks for cc1pi vs the usual variables (pN, dpTT, daT,
  W, Q2, pi_p) with the factorized chain, to establish where we ACTUALLY stand -- consistent
  weighting on both sides.

## #17 — disaggregated ratios, factorized differentiable chain (consistent weighting)

scripts/cc1pi_ratios.py (RES-C 120k x4 + RES-H 40k x4; ACH weighted by w x weight_to_nb;
fig paper_figures/cc1pi_ratios.png; ADoNIS arrays cached -> t2k_cc1pi_ratios_adonis.npz).
Per-variable ACH/ADO chi2/ndf:
  pn 2.27 | dptt 1.10 | dalphat 0.68 | Q2 1.01 | lp_p 0.83   -> ALL clean
  W 45.7  | pi_p 22.9                                        -> real high-tail excess
- Integral ADO/ACH = 0.966 (the known ~+3.4% normalization excess), consistent across vars.
- pn improved 11.4 (event cascade) -> 2.27 (factorized): the event cascade was the pN problem;
  reverting it (commit a5a3997) resolved most of it. lp_p 0.83 -> leading proton fine.
- Q2 chi2 was 0.0 only due to MY unit bug (ACH stores MeV^2, res_C GeV^2); fixed -> 1.01 (good).
- W (45.7) and pi_p (22.9): BOTH drop to ratio ~0.6 in the high tail (W>1400 MeV, pi_p>600 MeV),
  flat ~1 below.  CORRECTION to my earlier note: pi_p has an IDENTICAL definition both sides, so
  this is NOT a vertex-definition artifact -- it is a REAL shape excess: ADoNIS over-produces the
  high-pion-momentum / high-W tail by ~40-65%.  W mirrors it (high W <-> high pi_p).
- It barely touches the STV observables (pn/dptt/daT good) because those average over pi_p and
  are Delta-bulk dominated; but it is a genuine pion-spectrum discrepancy in the 2nd-resonance
  region.  Plausibly PRIMARY RES production at high W (cf. res_amps2_frame_fix.md W>1600 tail),
  NOT FSI -- but the earlier "primary high-W" claim was once an artifact (#5), so verify cleanly
  (primary vs FSI; consistent weighting; cached arrays) before asserting.  AWAITING DIRECTION.

## #18 — faithfulness audit (sampling / masses-isospin / termination) + fix loop start

Driven by the high-pi_p/high-W tail excess (#17). User heuristic: bugs live in custom
decisions, not the amplitude. Read-and-verify (no tuning):

A) SAMPLING (res_xsec._sample_3body): ADoNIS samples the (muN)+pi split ISOTROPIC (uniform
   cosθ) where ACHILLES ThreeBodyMapper uses a t-CHANNEL map (FinalStateMapper.cc:100); s23
   grouping + range + muN->muN isotropic all match. DOCUMENTED as a deliberate variance-only
   reparam ("same integral as ACHILLES's t-channel sampler, higher variance"); sigma_RES
   validated ~1% confirms unbiased integral.  BUT higher variance bites hardest in the
   UNDERSAMPLED forward/high-W tail -> a real candidate for tail noise/distortion (not benign).

B) MASSES/ISOSPIN: channel-correct and validated. Struck-N energy uses mqe avg (faithful);
   kinematic pion mass = mpi0 all channels (faithful); amplitude = avg m_N + 138.04 fpio +
   per-channel itiz/tpiz. Backed by the per-event amps2 RESDUMP audit (1e-4, all channels) and
   cascade charge-resolution. ONE stale item: dcc_current.exclusive_H hardcoded m_pi=mpi0
   (scalar, test-only path) bypassing conventions.amp_m_pi() -> FIXED (commit 2afdb49,
   centralized; test green; production batch path was already correct).

C) TERMINATION (cascade_discrete vs ACHILLES Cascade.cc / Nucleus.cc):
   - plane escape (external_test beam pion, Z>=radius): FAITHFUL.
   - ESCAPE RADIUS: ADoNIS used a CUSTOM rule (1e-4*central relative, last-point-above) = 6.05 fm;
     ACHILLES (Nucleus.cc:49-51) uses 1e-6 ABSOLUTE, first-point-below = 6.55 fm. ADoNIS nucleus
     ~0.5 fm too small -> pions under-cascade -> over-survive (RIGHT SIGN for +3.4% + high-pi_p).
     >>> FIX 1/N applied (commit 2b4107a): radius rule -> ACHILLES. Re-running cc1pi_ratios.
     CAVEAT: also feeds the pi+-12C transparency (validated at 6.05) -> must re-check separately.
   - sphere escape: ADoNIS adds '& outward'; ACHILLES escapes any |pos|>radius regardless of
     direction (wrong sign; queued fix).
   - CAPTURE: ACHILLES captures nucleons with E-mN-10MeV<0 (potential=10); ADoNIS none. Below
     the 450 MeV proton window -> likely negligible for CC1pi (queued).

FIX LOOP (user-directed: one at a time, re-run cc1pi_ratios.png each, ~15 min, show figure):
  1. escape radius 6.05->6.55  [RUNNING]
  2. outward sphere-escape qualifier
  3. nucleon capture
  then revisit isotropic-sampling tail variance (raise N or port t-channel map).

## #19 — fix 1 (escape radius 6.05->6.55 fm): NULL effect on every diagnostic

Re-ran cc1pi_ratios.py (120k x4 RES-C + 40k x4 RES-H) with the ACHILLES radius rule live
(_load_density returns 6.55 fm; cascade_discrete imports it -> in the active factorized path).
Per-variable ACH/ADO chi2/ndf (vs #17 in parens):
  pn 2.27 (2.27) | dptt 1.10 (1.10) | dalphat 0.68 (0.68) | Q2 1.01 (1.01) | lp_p 0.83 (0.83)
  W 45.71 (45.7) | pi_p 22.88 (22.9)   integral ADO/ACH 0.966 (0.966)
=> BYTE-IDENTICAL. The radius fix is a genuine faithfulness correction but a NULL physics
   effect. Reason (measured): the 6.05->6.55 fm shell is near-vacuum -- rho/central goes
   6.52e-6 -> 9.28e-7 fm^-3 (1.5e-4 -> 2.2e-5 of central). A pion traversing that outer
   shell encounters negligible material; whether it "escapes" at 6.05 or 6.55 fm does not
   change its fate (decided in the dense core). My #18 prediction ("right sign for +3.4% +
   high-pi_p") was WRONG -- the effect is ~0, not the hypothesized reduction.
- COROLLARY: fix 2 ('& outward' sphere qualifier) and fix 3 (capture, below the 450 MeV
  window) act in the same near-vacuum/out-of-window region -> expected similarly negligible.
  The FSI-termination audit items are faithfulness corrections, NOT the driver of the +3.4%
  norm or the high-W/pi_p tail.
- STANDING UNCHANGED: STV clean (pn/dptt/daT/Q2/lp_p all <=2.3); the real residual is the
  high-W (chi2 45.7) + high-pi_p (22.9) tail, ratio ~0.6 above W>1400 / pi_p>600, plus the
  +3.4% integral. A NULL result for an FSI change is positive evidence the tail is PRIMARY
  (2nd-resonance-region RES production), consistent with #4's no-FSI primary W-tail (2-3x
  over-population). NEXT: verify cleanly (primary no-FSI vs FSI, consistent weighting,
  cached arrays) before tuning, since the "primary high-W" claim was once an artifact (#5).

## #20 — bilinear-vs-spline interp: confirmed W-local, but WRONG SIGN to be the tail driver

Context: the high-W/pi_p tail (#17) is a well-sampled BIAS (#tail_neff: N_eff healthy, weights
flat -> NOT variance), localized to W>1390, and IS the +3.5% norm. User recalled (and
docs/res_amps2_frame_fix.md:86 records) a high-W dsigma/dW deformation tied to bilinear-vs-spline
(ADoNIS default BATCH_INTERP="bilinear"; ACHILLES uses the dcc spline; ADoNIS "spline" path
documented to bit-match interpolate_amp).

DIRECT test (scripts/interp_dsigdW.py): same sampled RES events, amps2 recomputed under BOTH
interps (only the interp changes), dsigma/dW ratio spline/bilinear vs hadronic W (200k/channel):
  W<1450 (Delta): 0.998-1.015 (flat, <=1.5%)
  W 1485-1665: 1.06-1.10
  W 1665-1935: 1.10 -> 1.24 (rising)
  integrated spline/bilinear = 1.0091 (per-channel s/b: n-npi+ 1.021, n-ppi0 1.012, p-ppi+ 1.006)
=> bilinear-vs-spline IS large and W-local in the 2nd-res region (the earlier 2-3% table-sum
   proxy under-counted; per-event partial-wave contraction amplifies to 6-24%).

CRITICAL SIGN: spline (=ACHILLES-faithful) gives MORE high-W sigma than bilinear. ADoNIS runs
bilinear and is ALREADY ABOVE ACHILLES in the tail (#17 ACH/ADO 0.44-0.62). So bilinear->spline
would push ADoNIS's tail EVEN HIGHER -> WORSEN the ACH/ADO discrepancy. Therefore:
  - bilinear is a genuine faithfulness error at high W (fix for correctness), but
  - it is NOT the cause of the high-W excess; it was partially MASKING a larger one.
  - the excess driver is NON-AMPLITUDE (phase space / isotropic 3-body sampler #18A / flux).
CAVEAT: rests on ADoNIS-spline == ACHILLES amplitude (documented bit-match). DECISIVE next check:
ADoNIS-spline dsigma/dW DIRECTLY vs ACHILLES (predict spline sits ABOVE ACHILLES at high W; if so
the excess is non-amplitude and the 3-body phase-space/sampling at high W is the next target).

## #21 — RESOLUTION: the CC1pi chain ALWAYS used spline; #20's worry is moot, tail is non-amplitude

User flagged (correctly): a byte-identical result across an interp change is a bug. Hunt found it
-- scripts/cc1pi_fig_tki.py:29 PINS `dcc.BATCH_INTERP = "spline"` at import (9 production scripts do
the same: cc0pi_disaggregated, free_proton_highstat, h_cc0pi, cc0pi_tune_adonis, make_single_nucleon_fig,
resdump_angular_audit, amps2_resdump_audit, cc1pi_disaggregated, cc1pi_fig_tki). So the ENTIRE CC1pi
diagnostic history (cc1pi_ratios #17/#19, the cut-ladder, the 4-way) ran on SPLINE. The bilinear
"default" in dcc_current was only ever for ad-hoc fast loops; every faithful analysis overrides it.
=> editing the dcc_current DEFAULT bilinear->spline was a NO-OP for CC1pi -> byte-identical (#19==#17
   == the spline rerun), exactly as the import-time pin predicts. No cache bug; my mental model was wrong.
Reconcile: fresh res_C(120k,0) bilinear=1.3225e-6 vs spline=1.2573e-6; cc1pi_ratios RES-C 4-seed mean
1.2554e-6 == SPLINE (not bilinear). Confirms the chain is spline.

CONSEQUENCES:
- The high-W/pi_p tail excess (W chi2 45.7, pi_p 22.9) and the +3.5% norm are the FAITHFUL spline
  result. Interpolation is NOT the cause. (bilinear would make the selected sample HIGHER, 1.3225 vs
  1.2573 -> worse, not better.)
- #20's "spline would worsen the discrepancy" is MOOT/CORRECTED: there was never a bilinear CC1pi
  baseline to worsen; CC1pi is already spline. The interp_dsigdW (spline>bilinear at high W) and the
  in-situ primary (spline raises ADoNIS toward ACHILLES) tests stand as amplitude-interp facts, but
  they do not bear on the CC1pi tail because CC1pi never used bilinear.
- NET: amplitude interpolation is settled (faithful spline, always on for the real analyses). The
  W/pi_p tail + norm are driven by the NON-amplitude machinery -> the 3-body phase-space SAMPLING
  (isotropic vs ACHILLES t-channel ThreeBodyMapper, #18A) is the remaining target.
- Reverted dcc_current default to bilinear (original state; production scripts pin spline explicitly).
