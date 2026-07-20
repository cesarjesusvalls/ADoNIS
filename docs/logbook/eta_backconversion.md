# η propagation & back-conversion (π⁺ high-|p| absorption residual)

## Symptom
Tagged-beam validation (`analysis/beams`): π⁺ on ¹²C **absorption** cross section ran **+10–16 %**
high (3–5 σ) for beam |p| **above 715 MeV/c**, while the total **reaction** cross section matched
ACHILLES to ≈1 % in every bin. χ²/ndf ≈ 6.7 on the absorption panel.

## Diagnosis (bisection)
- The excess turns on at **|p| = 715 MeV/c → W ≈ 1506 MeV**, exactly the **ηN production threshold**
  (m_η + m_N = 548 + 939 = 1487 MeV).
- Split of the "absorbed" tag by fate: below 500 MeV/c it is 100 % Oset (πNN→NN); above 715 MeV/c
  **~30–40 % is the η/K conversion channel** (`FATE_CONVERT`), which removes the pion.
- Elementary σ all faithful: MBSCAT total (ADoNIS ss+si vs ACHILLES `MesonBaryonInteraction` stderr)
  matches to 1–3 %; the conversion σ formula matches (ACHILLES `CalcCrossSectionW_grid` uses the
  **initial-channel** PF masses `Mass_m[iMB_i]`, same as ADoNIS `_sigma_cf`).
- **Root cause**: ACHILLES `GenerateMomentum` emits the η **and** the recoil baryon as cascade
  particles (`MesonBaryonInteractions.cc:224`), and `InitialStates()` registers ηN as an initial
  channel, so the η **propagates and back-converts ηN→πN**, regenerating a pion. ADoNIS's
  `_pion_step` did `alive &= ~is_conv` — the pion was killed and the η never created. Both codes
  define "absorbed" identically (reacted, no final-state pion), so every conversion was *always*
  absorbed in ADoNIS, whereas in ACHILLES ~35 % of the ηs back-convert. Solving the ratio bin-by-bin
  gives a back-conversion fraction f ≈ 0.3–0.4, fully accounting for the +10–16 %.

The N(1535) sits right at the ηN threshold and couples strongly to both ηN and πN, so the
back-conversion σ(ηN→πN) is huge near threshold (~107 mb at W=1490) — a slow near-threshold η has a
large probability to regenerate a pion.

## Fix (faithful η propagation)
- `adonis/fsi/mb/anl_xsec.py`: `eta_production_sigma_grid` (πN→ηN, the morphable piece of the
  conversion), `eta_elastic_sigma_grid` (ηN→ηN), `eta_backconv_sigma_grid` (ηN→πN per out-pion
  charge). PF uses the ηN ANL-code masses `Mass_m[1]=548, Mass_b[1]=938.5`. `_sigma_cf` refactored to
  take explicit PF masses.
- `adonis/fsi/mb/cascade_mb.py`: JAX interfaces `jax_pi_to_eta_sigma`, `jax_eta_elastic_sigma`,
  `jax_eta_backconv_sigma`.
- `adonis/fsi/cascade_real.py`: `_CH_MASS`/`_CH_PID` index **3 = η** (221, 547.862 MeV).
- `adonis/fsi/cascade_discrete.py::_pion_step`: a meson track with charge idx 3 is an η — sa=0,
  ss=ηN elastic (isotropic, no charge exchange, stays η), si=ηN→πN. On a **pion** conversion, with
  prob σ_η/σ_conv the pion→η morph emits a propagating η (the rest, KΛ/KΣ, stays terminal); on an
  **η** conversion it emits a regenerated pion (charge sampled from the per-charge back-conversion σ).
  The product meson is emitted through the (previously unused for pion slots) meson spawn slot; the
  recoil N′ baryon is emitted as before. The primary pion still latches `FATE_CONVERT` (reacted),
  and the η/regenerated-pion is a *new* pool particle — so nothing in the primary-fate logic changed.
- `adonis/fsi/cascade_full.py`: routes the pion-step meson spawn into the `pio` spawn slot.
- `analysis/beams/beam_bank.py`: `pi_alive` excludes charge 3 so a surviving η counts as "no pion".

**Determinism**: the new morph draws use independent RNG folds (331/332), so pion trajectories below
the ηN threshold are bit-identical to before. Only conversions (W>1487) change.

**Differentiability**: η-track steps are excluded from the pion kind-1 FSI reweight record
(`perp2_c → 1e6`) — their σ carry ηN values, which a pion FSI knob must not scale. The η path is
forward-faithful and gradient-neutral for the pion knobs (a follow-up could add an η knob).

## Verification
- ηN cross sections: exact 2:1 isospin charge split; back-conversion peaks at the N(1535)/threshold.
- π⁺ absorption over |p| ∈ [710,1000] MeV/c (new η-fix bank vs old pre-fix bank × known old/ACHILLES
  baseline): new/ACHILLES = 0.99, 1.01, 1.00, 1.01, 0.96, 1.15 at 734…976 MeV/c — the +10–16 %
  excess closes to ~0–1 % in 5 of 6 bins. The residual at the top bin (976 MeV/c, +15 %) is where
  the KΛ/KΣ channels open (treated as terminal here, whereas ACHILLES also propagates the kaon) plus
  ~5 % stats.
- T2K carbon RES: only 0.07 % of events get a surviving η → §1 validation unchanged (ACHILLES has
  the same ηs). MicroBooNE/MINERvA slightly more faithful.

## Open (second-order)
- KΛ/KΣ production is terminal; ACHILLES propagates the kaon (small back-conversion). Accounts for the
  residual +15 % in the top |p| bin only.
- η path not yet in the differentiable FSI-knob gradient.
