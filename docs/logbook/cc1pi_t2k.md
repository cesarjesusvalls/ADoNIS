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
