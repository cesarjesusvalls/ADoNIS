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
