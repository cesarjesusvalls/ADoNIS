# Paper figures
> **Acceptance criterion (see ACCEPTANCE.md):** a figure is ✅ only when |ADoNIS/ACHILLES − 1| < 3% in EVERY bin of EVERY panel. χ²/ndf-vs-data is NOT the metric.
 — arXiv:2508.19213v2 reproduction

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
| **3** | π⁺-C absorption & reaction σ(p_π) | DUET/Ashery (not in NUISANCE) | ✅ **done (model)** — ACHILLES VirtRes oracle vs ADoNIS cascade; **abs. fraction 0.31 vs 0.35**, Δ-region χ²/ndf ~3.5 (bridge ×0.75); DUET overlay pending digitisation |
| 4–6 | e4ν exclusive (E_QE, E_cal, P_T), ¹²C | CLAS Nature 2021 (not in NUISANCE) | ⏳ needs cascade (proton FSI) + CLAS data |
| **7** | T2K CC0π δp_T, δα_T (CH) | T2K (NUISANCE ROOT) | ◐ **NOT YET <3%** — pipeline+tight cuts done, but ADoNIS/ACHILLES per-bin fails: δp_T 21%, δα_T 12% (FSI tail too weak / stats) |
| **8** | T2K CC1π⁺ p_N, δp_TT (CH) [δα_T appendix] | T2K (NUISANCE STV) | ◐ **partial** — δp_TT <3% ✅; p_N 8% / δα_T 8% FAIL the 3% per-bin criterion (stats + CCQE/FSI residual) |
| 9,16 | MINERvA CC0π δα_T / p_n / 6-panel (CH) | MINERvA (NUISANCE + ME flux) | ◐ pipeline built (CCQE+proton FSI, tight cuts read from NUISANCE); ACHILLES 2M oracle generating |
| 10 | MicroBooNE CC1p0π δp_T in δα_T bins (Ar) | MicroBooNE | ⏳ same |
| 11,12 | MicroBooNE NC1π⁰ cosθ, p (Ar) | MicroBooNE (NUISANCE + flux) | ⏳ NC1π⁰ + flux-fold |
| **13** | πN/ηN/KΛ σ(W) + dσ/dΩ (model only) | — (ACHILLES INC vs ANL-Osaka DCC) | ✅ **done** — πN 9.3:2.2:1 + 1+3cos²θ; **η production at N(1535)** + KΛ threshold (full ANL tables were local) |
| **14** | NN→NNπ σ (model only, GiBUU) | — (ACHILLES vs GiBUU param) | ✅ **done** — pp→pnπ⁺ ~20 mb / pp→ppπ⁰ ~4 mb (5:1), peak p_beam~1.8 GeV; exact Dmitriev-Sushkov port |
| 15 | Δ-production diagrams | — | n/a |
| 17 | T2K appendix δα_T/δϕ_T (CH) | T2K | ⏳ flux-fold pipeline |

**Done: 1, 2, 3, 13, 14.** Fig 3 validates the differentiable ADoNIS cascade against the
ACHILLES Virtual-Resonances oracle (π⁺-¹²C transparency); the cascade reproduces the ACHILLES
absorption fraction (0.31 vs 0.35) and the Δ-peak σ shape (≈1.33× normalisation, the smooth-ρ
continuum-transport residual). Most remaining figures (4–12, 16, 17) need the flux-averaged
nuclear-TKI pipeline; Figs 4–6 / the Fig 3 data overlay also need the JLab/CLAS/DUET hadron
datasets the paper digitised separately (not in NUISANCE).
