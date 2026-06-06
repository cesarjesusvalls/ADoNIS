# Paper figures — arXiv:2508.19213v2 reproduction

Dedicated reproductions of the paper's figures: **experiment data + ACHILLES + ADoNIS**, with
a ratio panel + χ²/ndf. Filenames match the paper numerology (`figN_*.png`). Each `make_figN.py`
regenerates its figure.

Experimental data is pulled from the local **NUISANCE** clone (`../nuisance/data/`) and vendored
under `../data/experiment/`; the paper's run-cards/flux come from the Zenodo release
(10.5281/zenodo.18149407). The JLab/CLAS/DUET hadron data the paper also uses are not in
NUISANCE (the paper digitises them separately).

| Fig | Observable / target | Data | Status |
|---|---|---|---|
| **1** | inclusive (e,e′) dσ/dω, ¹²C | JLab (NUISANCE 12C.dat, 2.020 GeV/15.02°) | ✅ **done** — experiment+ACHILLES+ADoNIS; ADoNIS χ²/ndf 142 ≈ ACHILLES 131 (QE peak; cascade-free) |
| **2** | free-nucleon ν_μ CC1π σ(E_ν), 3 channels | ANL+BNL (NUISANCE) | ✅ **done** — experiment+ACHILLES+ADoNIS+χ²/ndf+ratio |
| **3** | π⁺-C/Ar absorption & reaction σ(p_π) | DUET/Ashery/LADS/Dytman (not in NUISANCE) | ⏳ needs the real cascade (parked) + hadron data |
| 4–6 | e4ν exclusive (E_QE, E_cal, P_T), ¹²C | CLAS Nature 2021 (not in NUISANCE) | ⏳ needs cascade (proton FSI) + CLAS data |
| 7 | T2K CC0π δp_T, δα_T (CH) | T2K (NUISANCE + Zenodo flux) | ⏳ needs flux-fold + nuclear-TKI pipeline (+cascade) |
| 8 | T2K CC1π⁺ p_N, δp_TT (CH) | T2K | ⏳ same |
| 9,16 | MINERvA CC0π δα_T / p_n / 6-panel (CH) | MINERvA (NUISANCE + flux) | ⏳ same |
| 10 | MicroBooNE CC1p0π δp_T in δα_T bins (Ar) | MicroBooNE | ⏳ same |
| 11,12 | MicroBooNE NC1π⁰ cosθ, p (Ar) | MicroBooNE (NUISANCE + flux) | ⏳ NC1π⁰ + flux-fold |
| 13 | πN/η-p σ(W) + dσ/dΩ (model only) | — (ACHILLES INC vs ANL-Osaka DCC) | ◐ have the DCC side (Phase E); ACHILLES-INC side = cascade |
| 14 | NN→NNπ σ (model only, GiBUU) | — | ☐ GiBUU NN parameterisation |
| 15 | Δ-production diagrams | — | n/a |
| 17 | T2K appendix δα_T/δϕ_T (CH) | T2K | ⏳ flux-fold pipeline |

**Reachable cascade-free (done): 1, 2.** Most remaining figures need the real cascade (parked)
and/or the flux-averaged nuclear-TKI pipeline; Figs 4–6 / 3 also need the JLab/CLAS/DUET hadron
datasets that the paper digitised separately (not in NUISANCE).
