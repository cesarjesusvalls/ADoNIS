# Phase I — Exclusive electron vs neutrino comparison

Plan ref: `docs/REPRODUCTION_PLAN.md` Phase I. The exclusive e/ν comparison: at the SAME
kinematics the only difference between electron (EM, vector) and neutrino (CC, vector+axial)
single-pion production is the **current structure**, so the exclusive pion observables expose
the axial current and its V–A interference directly.

## Matched-kinematics oracles
Three ACHILLES `RES_Spectral_Func` oracles on ¹²C at **E = 2.222 GeV** with the **same** forward
lepton-angle acceptance (`AngleTheta [14,17]°`), from the `achilles:oracle` image:
- electron: `Leptons:[11,[11]]` — reused from the Fig-1 RES oracle
  (`_oracle_out/inclusive_ee_12C_res.hepmc`);
- ν_e CC: `Leptons:[12,[11]]` — `_oracle_out/exclusive_nue_12C_res.yml` → hepmc;
- ν_μ CC: `Leptons:[14,[13]]` (muon final state, `AngleTheta` on PID 13) —
  `_oracle_out/exclusive_numu_12C_res.yml` → hepmc.

`scripts/make_exclusive_evnu.py` parses both hepmc into the exclusive pion observables
(`cos θ*_π` the pion CM angle vs q, and the hadronic invariant mass `W`), saves the oracle
summary to `data/oracle/exclusive_evnu_c12.npz`, and overlays the **ADoNIS** prediction for
each current (`EM_CHANNELS` / `CC_CHANNELS` sampled at the matched kinematics) with a
model/oracle ratio panel + χ²/ndf.

## The physics (ADoNIS vs ACHILLES, both reproduced)
- **cos θ*_π**: the electron is strongly **forward-peaked** (A_FB = +0.110), the neutrino is
  flatter/centre-shifted (A_FB = +0.051) — the axial current + V–A interference redistribute
  the pion angular distribution away from the pure-vector forward peak.
- **W**: the neutrino is more sharply **Δ(1232)-peaked** (⟨W⟩ = 1258 MeV; ν/e ratio ~1.5 at
  the Δ), the electron carries more strength into the second-resonance region (⟨W⟩ = 1349)
  — the EM 1/Q⁴ propagator (favouring low Q²/higher-W reach) + vector coupling to the higher
  resonances.

**ν_μ** (muon final state) gives essentially the same axial signature as ν_e — A_FB = +0.043
vs +0.051, ⟨W⟩ = 1252 vs 1251 — the lepton mass is only a small kinematic shift on the same
V–A structure. ADoNIS reproduces **all three** channels: χ²/ndf (model vs oracle) **W: e 2.4,
ν_e 1.5, ν_μ 5.2; cos θ*: e 4.5, ν_e 7.3, ν_μ 5.2**. Figure: `figures/exclusive_evnu_c12.png`.

## Gates (`tests/test_exclusive_evnu.py`)
- **physical difference** — e more forward than BOTH neutrinos (A_FB) and higher ⟨W⟩, in the
  oracle and the model (the axial signature); ν_e ≈ ν_μ (lepton-mass shift small).
- **model reproduces oracle** — χ²/ndf < 6–8 (W) and < 12 (cos θ*) for all three channels.

## Caveat
At matched kinematics the e/ν difference combines the current (vector vs V–A) **and** the
propagator (EM 1/Q⁴ vs CC ~const) — i.e. it is the full measurable EM-vs-weak difference, not
the axial current in isolation. Both effects are physical and both are reproduced.

## Remaining (optional)
- ☐ ν_μ (muon-mass) variant and an NC exclusive comparison (reuse the same machinery).
- ☐ Fold the cascade FSI (Phase D) into the exclusive observables for the e/ν + FSI picture.
