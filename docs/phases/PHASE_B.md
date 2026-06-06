# Phase B — QE channel + inclusive (e,e′) → Fig 1

Plan ref: `docs/REPRODUCTION_PLAN.md` Phase B. B1 QE 1-nucleon vertex + SF fold;
B2 EM inclusive dσ/dω = QE + 1π (Fig 1, ¹²C); B3 ⁴⁰Ar SF swap (Fig 1, ⁴⁰Ar).

## B1 — free-nucleon CCQE σ(E_ν) — DONE
`adonis/primary/qe/llewellyn_smith.py`: the analytic **Llewellyn-Smith** CCQE
dσ/dQ² for ν_μ n → μ⁻ p, with the **Kelly vector form factors** (F1, F2; new in
`form_factors.py`) and a dipole axial FF (F_A; F_P from PCAC). Integrated over the
kinematic Q² range → σ(E_ν).

- **Physical absolute units from first principles** (prefactor G_F²cos²θ_c — *no*
  calibration): σ rises from threshold and plateaus at **~1.05 ×10⁻³⁸ cm²** above 1 GeV,
  the known free-nucleon CCQE value.
- **Oracle (absolute)**: vs ACHILLES `QE_Spectral_Func` on a stationary neutron (ν_μ on
  `1N`), `data/oracle/freenucleon_ccqe_sigma.csv`. Since both sides are absolute nb, the
  gate checks **model/ACHILLES ≈ 1 directly** (no bridging constant). With ACHILLES's
  default **z-expansion** axial FF (`ff="zexp"`): **0.997–1.002 across 0.5–3 GeV, max 0.3%**.
  (With the **dipole** FF the model is 1.01–1.03, max 3.2% — the dipole is kept as the
  M_A-tunable closure handle; the z-exp matches ACHILLES's FF choice.)
- **Closure**: dσ/dM_A autodiff==FD (analytic, exact; dipole FF). Gates: `tests/test_ccqe.py`.

## Remaining
- ☐ **B1b** Optional: z-expansion axial FF in the LS σ (close the ~3%); EM/NC elastic
  variants of the 1-nucleon current (reuse the Kelly FFs) for the inclusive fold.
- ☑ **B2 — EM inclusive dσ/dω = QE + 1π** — DONE (Fig-1 structure) on ¹²C, vs the JLab-config ACHILLES (E=2.222 GeV,
  θ=15.541°). Build the QE 1-nucleon hadron tensor W^{μν}_1N folded over the ¹²C spectral
  function (reuse `SpectralFunction`) + the EM 1π piece (A1), summed → dσ/dω. **→ Fig 1 (¹²C).**
- ☐ **B3 — ⁴⁰Ar spectral function** swap: point the fold at the `pke40{p,n}` tables (present
  in the image) → Ar inclusive. **→ Fig 1 (⁴⁰Ar).**

## Status
B1 (the QE vertex, free-nucleon CCQE) is done and validated absolutely vs ACHILLES. The
inclusive (e,e′) fold (B2/B3, Fig 1) is the remaining build — it composes the QE 1-nucleon
tensor (here) with the spectral fold and the A1 EM-1π piece.

## B2 update — QE inclusive response (PWIA) DONE
`adonis/nuclear/qe_inclusive.py`: the plane-wave impulse-approximation QE (e,e') response, folding the single-nucleon elastic response (Kelly FFs) over the 12C spectral function S(p,E). At the JLab kinematics (E=2222 MeV, theta=15.541 deg) it gives a QE peak at omega~208 MeV (the relativistic quasi-free peak sqrt(q^2+M^2)-M+E_b, with the proper v_L/v_T Rosenbluth response -- see the oracle section below) with a **Fermi-motion width FWHM~153 MeV** (expected k_F q/M ~147). Gate: tests/test_qe_inclusive.py. Remaining: add the 1pi inclusive piece (from A1, integrated over the pion) for the full QE+1pi dsigma/domega, and overlay the JLab-config ACHILLES run -> Fig 1.

## B2 complete — QE + 1pi inclusive (Fig 1 structure)
`adonis/nuclear/inclusive_1pi.py`: the 1pi/Delta contribution to dsigma/domega, folding the EM transverse/longitudinal structure functions W_T,W_L (EM_CHANNELS hadron tensor) over S(p,E) at the lepton (omega,q). The Delta bump sits at omega~459 MeV (expected (m_D^2-M^2+Q^2)/2M ~491), well above the QE peak (~223). The total QE+1pi (figures/inclusive_ee_c12.png) reproduces the two-peak structure of the paper's Fig 1. test_inclusive_two_peak_structure.

## B2 oracle — inclusive (e,e') validated vs ACHILLES (ratio + chi2)
Generated **ACHILLES (e,e') oracles** for both components: `QE_Spectral_Func` and
`RES_Spectral_Func` electron scattering at the model kinematics (E=2.222 GeV, theta in [14,17]
deg via the ACHILLES `AngleTheta` HardCut, which makes a fixed-angle inclusive oracle
efficient), 40k events each from the `achilles:oracle` image
(`_oracle_out/inclusive_ee_12C_{qe,res}.yml`; `scripts/gen_inclusive_ee_oracle.py` parses the
hepmc into omega = E_beam - E_e' **with per-bin MC errors** ->
`data/oracle/inclusive_ee_12C_{qe,res}.csv`). The comparison uses a **ratio panel + chi2/ndf**
(`scripts/make_inclusive_ee_validation.py` -> `figures/inclusive_ee_validation_c12.png`), the
standard ADoNIS-vs-ACHILLES diagnostic — not just an eyeballed overlay.

**1pi/Delta (RES): chi2/ndf ~15, centroid agrees to ~1 MeV.** Found + fixed a real bug:
`onepi_dsigma_domega` used `tot0 = w + M_N` (struck nucleon at the FULL rest energy), ignoring
that the nucleon is **bound** — so the Delta bump sat ~60 MeV too low in omega. Corrected to
`tot0 = w + M_N - E_rm` with `E_rm` = the S(p,E)-weighted mean removal energy (~41 MeV, read
from the spectral function). Model peak 459 -> 517 (oracle 531); centroid 508 -> 531 (oracle
532). The ratio is flat ~1.0 across the peak. **High-omega tail:** unit-area-normalised, the
model tracks ACHILLES well through omega~700 (ratios 0.91-1.07); integrated strength above
600 MeV is **36% (model) vs 39% (oracle)** -- only a ~3% gap. The genuine residual is the
*far* tail (omega > 800, ratio -> 0.73): the third-resonance / DIS-onset region a
resonance-region (DCC EM) fold under-carries (the DCC W_T does have the N(1440)/N(1520)
second-resonance bump at W~1470-1510, but weak ~0.2 of the Delta, and no DIS). The model
1pi curve is now plotted to omega=900 (`make_qe_figure.py`) so this tail is visible rather
than truncated at 520.

**QE: chi2/ndf ~9 after the v_L/v_T fix (was ~28), peak 208 vs 192.** Diagnosed via the ratio
panel (originally <1 below the peak, >1 above — a shifted peak). ACHILLES uses the **same**
spectral function (`info_C12_pke.data` -> `pke12p_tot.data`), so it is purely a prescription
difference. Two factorial tests isolated it:
- **Energy balance** — keep the Benhar form `E_f = omega + M - E` (E = removal energy from
  S(p,E)); the binding is already carried by E, so the bare mass M is correct. de Forest's
  on-shell-initial variant `sqrt(M^2+p^2)` *double-counts* the Fermi energy and **worsens**
  chi2 (8.6 -> 96) — rejected.
- **Response** — replaced the isotropic `(G_E^2 + tau G_M^2)/(1+tau)` with the proper
  **longitudinal/transverse Rosenbluth separation** `v_L R_L + v_T R_T`
  (`v_L=(Q^2/q^2)^2`, `v_T=Q^2/2q^2 + tan^2(theta/2)`, `R_L~G_E^2`, `R_T~tau G_M^2`). This is
  the real fix: **chi2/ndf 27.6 -> 8.6**, peak 222 -> 208, ratio flat ~1.0 across the peak.

The residual ~16 MeV / chi2~9 is the deeper off-shell single-nucleon cross section (de Forest
cc1/cc2) + the high-omega SRC tail the model under-carries.

Gates: `test_onepi_bump_matches_achilles_res` (chi2/ndf < 20, centroid < 20 MeV),
`test_qe_peak_vs_achilles` (chi2/ndf < 15, peak offset < 30 MeV) — both skipped if the CSVs
are absent.
