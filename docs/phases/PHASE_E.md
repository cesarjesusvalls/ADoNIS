# Phase E — Real meson-baryon scattering (DCC PWA)

Plan ref: `docs/REPRODUCTION_PLAN.md` Phase E. The πN→{πN,ηN,KΛ,KΣ} cross sections from
the ANL-Osaka partial-wave amplitudes — the keystone input for the real cascade (G).

## Input format (scoped)
ANL tables in the image at `data/MesonBaryonAmplitudes/ANL/ANL_{i}-{f}.dat` (4×4
initial→final meson-baryon channels, i,f ∈ {0:πN, 1:ηN, 2:KΛ, 3:KΣ}). Extracted format
(`ANL_0-0.dat`, πN→πN):
- header comment lines, then rows: `W[MeV]` + **40 columns = 20 partial waves × (Re, Im)**.
- wave order (`src/Achilles/MesonBaryonAmplitudes.cc`): `S11 S31 P11 P13 P31 P33 D13 D15
  D33 D35 F15 F17 F35 F37 G17 G19 G37 G39 H19 H39`. Label `L_{2I,2J}`: e.g. **P33** =
  L=1, I=3/2, J=3/2 — the **Δ(1232)** (peaks near W≈1232).
- W grid 1080–2200 MeV. Column layout: `W, Re(S11), Im(S11), Re(S31), Im(S31), …`.

## Build plan
- ☑ **E0 parser** (`adonis/fsi/mb/anl_xsec.py::load_anl`): reads `ANL_i-f.dat` → `(W,
  amps[nW,20])` complex in WAVES order. `wave_qn('P33')` → (L,2I,2J).
- ☑ **E1 total σ(W)** — DONE for π⁺p (pure I=3/2). Partial-wave sum with the exact ACHILLES
  normalisation (`CalcCrossSectionW_grid`: σ = (ħc)²·10·2π·4W²/PF·Σ(2J+1)|Σ_I CG_I A_LJI|²
  [mb]). **Validated: the π⁺p Δ(1232) peak — W=1220 MeV, σ=208 mb** (PDG ~200 mb), falling
  to ~10 mb above. Gate `tests/test_mb_xsec.py`; σ-norm knob is exactly linear (the
  differentiable handle). CI extracts `MesonBaryonAmplitudes/ANL` (cache key bumped).
  - ☑ **π⁻p channels** — `pim_p_elastic` (c_{3/2}=1/3, c_{1/2}=2/3), `pim_p_cex`
    (π⁻p→π⁰n, c_{3/2}=+√2/3, c_{1/2}=−√2/3), `pim_p_total` (elastic+cex). **The textbook
    9:2:1 Δ isospin ratio is reproduced: π⁺p:cex:elastic = 9.3:2.2:1** (the small >2 cex
    excess is the physical opposite-sign I=1/2 fill-in). Gate
    `test_cex_isospin_921_ratio`. Figure `figures/mb_pin_scattering.png` (`make_mb_figure.py`).
  - ◐ **soft-blocked**: ηN/KΛ/KΣ final states (`ANL_0-{1,2,3}`) are **not in the extracted
    data dump** (only `ANL_0-0.dat` πN→πN is present) — like the A3 ANL/BNL overlay, this
    needs external/image data, not model work. The paper's Fig DCC_total η/K overlay waits
    on those tables.
- ☑ **E2 angular dσ/dΩ** — DONE (`dsigma_dOmega`): spin-non-flip f + spin-flip g from the
  multipoles (`f=Σ[(L+1)a_{L+}+L a_{L-}]P_L`, `g=Σ[a_{L+}−a_{L-}]P_L^1`, dσ/dΩ=|f|²+|g|²).
  At the Δ it gives the **P33 1+3cos²θ** shape (fwd/90°≈4); off-resonance the S–P
  interference tilts it forward (W>1232) or backward (W<1232). Gate
  `test_delta_angular_distribution`; figure `figures/mb_pin_scattering.png` (right panel).

## Oracle note (important)
The paper's Fig DCC_total/DCC_angular compares the ACHILLES **INC** (cascade) to the
ANL-Osaka model. Our forward σ(W) **is** the ANL-Osaka model (the tables), so it self-
validates against the known πN cross section (Δ peak, PDG values) without the cascade
binary. The full INC oracle would need `achilles-cascade` — a **confirmed showstopper** in
this environment (no `cmake`/`gfortran`; image lacks the binary — see `README.md`). So E is
gated on the **physical πN cross section** (Δ peak + PDG), not the standalone INC run.

## Status
E0/E1/E2 **DONE** for the πN sector with the available data: parser, total σ(W) for all
three πN charge channels (9:2:1 Δ ratio), and the angular dσ/dΩ (P33 1+3cos²θ). The only
remaining E item is the ηN/KΛ/KΣ final states, which are **soft-blocked on missing data
tables** (`ANL_0-{1,2,3}` absent from the dump) — not a model gap. Self-validated against
the physical πN cross section (Δ peak, PDG, isospin ratios) without the cascade binary.
