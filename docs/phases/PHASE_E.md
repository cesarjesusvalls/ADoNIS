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
  - ◐ remaining E1: the π⁻p / charge-exchange channels (I=1/2+3/2 with the GetCG isospin
    Clebsches, `CalcCrossSectionW_grid` lines 409-421), and ηN/KΛ/KΣ final states
    (`ANL_0-{1,2,3}`); the paper's Fig DCC_total overlay.
- ☐ **E2 angular dσ/dΩ**: Legendre `P_L`, `P_L'`, full multipole interference + inverse-CDF
  angle sampling. **Fig DCC_angular.**

## Oracle note (important)
The paper's Fig DCC_total/DCC_angular compares the ACHILLES **INC** (cascade) to the
ANL-Osaka model. Our forward σ(W) **is** the ANL-Osaka model (the tables), so it self-
validates against the known πN cross section (Δ peak, PDG values) without the cascade
binary. The full INC oracle would need `achilles-cascade` — a **confirmed showstopper** in
this environment (no `cmake`/`gfortran`; image lacks the binary — see `README.md`). So E is
gated on the **physical πN cross section** (Δ peak + PDG), not the standalone INC run.

## Status
Format scoped + extracted. E0/E1/E2 are the remaining build (a new parser + the partial-wave
→ cross-section physics with isospin). Self-validatable against the πN Δ peak without the
cascade binary.
</content>
