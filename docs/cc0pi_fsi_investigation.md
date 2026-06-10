# CC0π-Np FSI investigation: ADoNIS vs ACHILLES

**Status:** in progress. **Goal:** remove every implementation difference between the ADoNIS and
ACHILLES intranuclear cascades so the CC0π-Np predictions agree to within MC statistics (no fitted
constants). This doc is the running summary of what has been measured, read, fixed, and what
remains open.

---

## 1. Setup

- Target: **CH** (¹²C + free H), T2K νμ flux. Oracle = ACHILLES `T2K_CH_virt.hepmc` (2M, FSI) and
  `T2K_CH_virt_nofsi.hepmc` (200k, no-FSI).
- Absolute σ (first-principles, no fit): ACHILLES `w·GenXS[pb→nb]/Σw_all`; ADoNIS per-event nb.
- CC0π-Np selection: pμ>250, cosθμ>−0.6; leading proton 450–1000 MeV/c, cosθp>0.4; **no surviving
  meson**.
- Disaggregated by truth: ADoNIS via the generator cells (`cc0pi_disaggregated.py`); ACHILLES via
  `signal_process_id` (200=QE, 401/402=RES) added to `parse_hepmc.py` + `extract_t2k_cc0pi_tki.py`,
  and the struck nucleon (status 2) for the vertex W = √((q+p_struck)²).

---

## 2. Top-level observation (by true process)

Absolute ACHILLES/ADoNIS ratios (`cc0pi_ratios_byproc.png`), pre-fix:

| process | σ_ACH | σ_ADO | ratio ACH/ADO | note |
|---|---|---|---|---|
| QE  no-FSI | 1.781e-5 | 1.775e-5 | **1.003** | bare QE vertex — exact |
| QE  with-FSI | 1.458e-5 | 1.356e-5 | **1.076** | nucleon FSI removes ~8% too much |
| RES with-FSI | 1.295e-6 | 1.819e-6 | **0.712** | pion-absorption → CC0π 40% too much |

The total FSI σ agreed to 3% only by **cancellation** (QE deficit −0.10e-5 vs RES excess +0.05e-5).
Disaggregation exposed two independent FSI problems. **No-FSI QE is exact (1.003), so the primary
vertex is correct; all disagreement is in the cascade.**

---

## 3. RES excess — full decomposition (the main thread so far)

Carbon-only RES bookkeeping (`_ach_res_eff.py`, `_ado_res_eff.py`):

| quantity | ACHILLES (C) | ADoNIS (C) | ratio |
|---|---|---|---|
| σ_RES_total (production) | 1.689e-5 | 1.668e-5 | **0.99 ✓** |
| absorbed fraction | 0.2176 | 0.270 | 1.24 |
| acceptance \| absorbed | 0.352 | 0.337 | **0.96 ✓** |
| σ(RES→CC0π) | 1.295e-6 | 1.517e-6 | 1.17 |

Free-H: **ACHILLES abs_frac(H) = 0.0000 exactly** — a free proton cannot absorb a pion (πNN→NN
needs two nucleons).

So the 40% CH RES excess = two equal halves:
1. **H absorption artifact** (~0.27e-6): ADoNIS's `h_cc0pi` ran the free-H pion through a *12C-config*
   cascade → faked CC0π that ACHILLES gives as zero.
2. **Carbon over-absorption** (~0.25e-6): ADoNIS absorbs 24% more carbon pions (abs_frac 0.270 vs
   0.218). Production matches to 1%; post-absorption proton acceptance matches to 4% — the excess is
   *purely* the absorbed fraction.

---

## 4. Fixes implemented

- **① Pauli gate on pion absorption** (`cascade_discrete.py`): ACHILLES `FinalizeMomentum` Pauli-blocks
  both outgoing nucleons of πNN→NN and rejects the absorption if either is below the local Fermi
  momentum (the pion then continues). ADoNIS had no such gate. Implemented faithfully (channel choice
  separated from the Pauli-gated outcome; product A at the pion position, B at the struck-nucleon
  position). **Correct ACHILLES physics, but tested negligible here** (+1.2% RES-C, +0.3% RES-H): the
  absorption products come out ~480 MeV, well above kf~250, so the gate rarely fires. Kept anyway.
- **A. Free-H cannot absorb** (`cc0pi_disaggregated.py::res_H`): RES-H → CC0π is now identically 0
  (both FSI and no-FSI), matching ACHILLES. Removes the ~0.27e-6 artifact for the right reason.

---

## 5. Reading audit — what is provably identical

Read both cascades end-to-end. Every per-encounter ingredient matches ACHILLES:

| ingredient | ADoNIS | ACHILLES | verdict |
|---|---|---|---|
| run-card knobs | step 0.04, Cylinder, InMedium None, PotentialProp False | `run_T2K_virt.yml` | **match** |
| absorption σ (Oset, 2N+3N p-wave + s-wave, vrel, 4/9, ImB0) | `oset_xsec.py:abs_cross_section` | `OsetCrossSections.cc:12-77` | **identical term-for-term** |
| density ρ(r) feeding σ | `c12_density.txt` ×2 | `c12.prova.txt` ×(p+n) | **byte-identical data** |
| pion birth position | at production nucleon (`cascade_discrete.py:227`) | at vertex (`Process.cc:106`, "TODO: propagating deltas") | **match — no Δ displacement** |
| scatter σ (ANL-Osaka MB) | `cascade_mb.jax_channel_sigmas` | `MesonBaryonAmplitudes.GetAllCSW` | **same table** |
| scatter σ below threshold | `left=0.0` | `GetCSW:481` returns 0 below threshold | **match** |
| NN elastic σ + isotropic CM | `nucleon_cascade.nn_elastic_sigma` | `NucleonNucleon.cc` (GiBUU, isotropic) | **match** |
| post-absorption proton acceptance | 0.337 | 0.352 | **match (4%)** |

### Differences found by reading
- **Δ-decay displacement of pion birth** — *refuted* (ACHILLES emits the pion at the vertex too).
- **isospin-averaged σ_scatter** (`cascade_mb` uses ½p+½n for every nucleon; ACHILLES uses the actual
  nucleon's σ) — real, but **wrong direction**: `p_abs=σ_abs/(σ_abs+σ_scat)` is convex, so by Jensen
  the averaged σ̄_scat makes ADoNIS absorb *less*, not more.
- **No formation zone** in the nucleon cascade; **no NN→NΔ→NNπ** channel — both relevant to the QE
  side (Section 7), not the RES absorbed fraction.

---

## 6. Produced-pion T_π spectrum — measured, matches

Direct comparison of the emitted spectra (`_tpi_compare.py`, `_tpi_ach.py`), RES carbon, weighted:

| | ⟨T_π⟩ | median | p10 | p90 | frac(T<60) | frac(T<100) |
|---|---|---|---|---|---|---|
| ACHILLES (status-29 production pion) | 204.2 | 154.2 | 59.1 | 360.6 | 0.103 | 0.266 |
| ADoNIS (`res_xsec` p_pi) | 199.2 | 154.6 | 60.2 | 351.3 | 0.099 | 0.255 |

Agree to ~2–3%; ADoNIS is if anything **slightly harder** (fewer soft pions → would absorb *less*).
**So the production spectrum is not the cause.**

---

## 7. Current hypothesis — and why it is NOT yet established

Reframed by the natural metric, the carbon "absorbed +24%" is **survival 0.730 (ADoNIS) vs 0.782
(ACHILLES) = 6.6% relative**. The discrete-Glauber cascade docstring claims it reproduces the
ACHILLES π⁺-¹²C transparency oracle "to ~5%". So the residual *sits at the documented
discrete-Glauber approximation level*.

**BUT this is a hypothesis, not a conclusion.** "Tested to ~5%" does not prove the residual is the
differentiable/vectorized **algorithm class** (discrete-Glauber walk vs ACHILLES's closest-approach
time-ordered cascade). It could equally be an **undiscovered bug or subtle discrepancy** we have not
isolated. We will not let this shade of suspicion loom unproven.

### Decisive diagnostic (next action)
Implement a **parallel pion cascade that is bit-exact to ACHILLES's `Cascade.cc` BaseAlgorithm**
(closest-approach, time-ordered, over the *real* event config, with the secondary-particle list),
**non-differentiable, for diagnosis only**. Run the same RES-carbon sample through it and measure the
absorbed fraction.
- If the bit-exact cascade gives abs_frac ≈ 0.218 (matching ACHILLES) → the discrete-Glauber
  approximation *is* the cause; the differentiable path then needs the exact algorithm as a detached
  kind-1 proposal (compatible with differentiability — only the weight must be differentiable).
- If the bit-exact cascade *also* gives ~0.270 → there is a real bug/discrepancy still hidden in a
  shared ingredient, and the discrete-Glauber story is exonerated.

This isolates the effect cleanly instead of assuming it.

---

## 8. QE side (8% deficit) — pending, same method

QE with-FSI removes ~5.5 pts too many protons from the signal window (survival 76.4% vs ACHILLES
81.9%) and over-hardens δpT. Per-encounter NN σ matches (bit-exact GiBUU). Candidate differences from
reading: **no formation zone** on the leading nucleon (ACHILLES suppresses early rescatter), and
**missing NN→NΔ→NNπ** (real omission, but pushes ADoNIS *high*, opposite sign). To be decomposed with
the same read-then-measure approach, and the same bit-exact diagnostic via the nucleon cascade.

---

## 9. Artifacts

- Figures: `paper_figures/cc0pi_ratios.png` (no-FSI + FSI, 4 obs), `cc0pi_ratios_byproc.png` (QE/RES split).
- Scripts: `cc0pi_disaggregated.py`, `cc0pi_ratios.py`, `cc0pi_ratios_byproc.py`,
  `extract_t2k_cc0pi_tki.py` (proc tag + vertex W), diagnostics `_ach_res_eff.py`, `_ado_res_eff.py`,
  `_tpi_compare.py`, `_tpi_ach.py`, `_val_pauli_abs.py`.
- Oracle npz: `data/oracle/t2k_cc0pi_tki_achilles{,_nofsi}.npz` (now carry W + proc),
  `cc0pi_disaggregated.npz` (regenerating with fix ①+A at time of writing).

## 10. Open questions
- Does the bit-exact cascade reproduce ACHILLES abs_frac (0.218)? (Section 7 diagnostic.)
- QE: formation zone + NN→NΔ contributions, quantified.
- Whether any residual after the bit-exact cascade is genuinely the differentiable proposal or a bug.
