# Inclusive (e,e') on ¹²C — the vector + spectral-function probe

## Why this sample

Gate I on T2K (`docs/paper_plan.md` §3) leaves a block of knobs **DEGENERATE**: the data sees them
clearly, but another knob spends the sensitivity. The worst case is a **rate-only flat direction**:

| knob | raw (data sees it) | marginalized (fit delivers it) | penalty |
|---|---|---|---|
| `qe_norm` | **0.033** | 0.86 | **26×** |
| `axial_strength` | 0.034 | 0.69 | 20× |
| `vector_strength` | 0.032 | 0.80 | 25× |
| `sf_norm` | 0.025 | 0.69 | 28× |
| `src_tail` | 0.124 | 0.76 | 6× |

Four different ways to spell "scale the QE rate" (posterior corr. `qe_norm`–`axial_strength` = −0.50,
`res_norm`–`sf_norm` = −0.78). **No amount of T2K data breaks this** — it needs a sample where the
vector current and the spectral function appear *without* the axial current.

Inclusive (e,e') is exactly that: a **photon** probe, so the current is **purely vector**, and there is
no neutrino flux uncertainty. Pin vector + SF externally and the ν data only has to determine the axial
part. The x > 1 / high-ω tail is also the classic short-range-correlation probe → `src_tail`.

**Honest limit:** `gen` (G_E^n) likely stays INVISIBLE even here — G_E^n is small in electron scattering
too (measured |∂ln amps²/∂θ| = 0.012 on the CC side).

## ACHILLES oracle — DONE (2026-07-15)

`configs/achilles/run_inclusive_ee_C_{qe,res}.yml` (recovered in `43d68d4`; they were already in the repo):
e⁻ beam **E = 2.222 GeV**, θ_e ∈ [14°, 17°] via `Main/HardCuts` (narrow acceptance ≈ 15.541°), QE spectral
function + RES on ¹²C, **Cascade: Run: False** (inclusive). This is the JLab Murphy:2019wed kinematic point.
Run in the amd64 `:oracle` image (no cascade → Rosetta is fine). 5 seed batches each, on disk:
`output/achilles/inclusive_ee_C_{qe,res}*.hepmc`.

## ADoNIS EM current — DONE (`adonis/xsec/dirac.py`, commit db34e61)

The faithful spec is `Achilles/src/Achilles/LeptonicCurrent.cc:120` (`pid == 22`):

```cpp
const std::complex<double> coupl = i * ee;
proton : {F1p, coupl}, {F2p, coupl}, ...      // NO FA entry
neutron: {F1n, coupl}, {F2n, coupl}, ...      // NO FA entry
```

So versus CC, three changes and nothing else:
1. the photon couples to the struck nucleon's **own** form factors, **not** the isovector difference
   `F1p − F1n`;
2. coupling `i·e` instead of `Vud·e·i/(s_w√2·2)`;
3. `CouplingsFF` has **no FA entry** ⇒ `FA = FAP = 0` — the current is purely vector.
4. protons **and** neutrons are both struck, incoherently (ν-CC hits neutrons only).

`hadron_current_qe_dirac(..., probe="CC"|"EM", is_proton=...)`. The CC path is bit-for-bit unchanged; the
EM branch reuses the same vertex, spinors, de Forest shift and Kelly form factors. The leptonic side
already had `kind="EM"` (`adonis/xsec/leptonic.py`: photon, no propagator denominator).

Gates passed: `axial_scale` has **no** effect on the EM current (purely vector) and **does** on CC;
`vector_scale=2` exactly doubles the EM current; the CC current is unchanged.

Crucially, `vector_scale` still multiplies F1/F2, so `vector_strength` / `mu_p` / `mu_n` / `gep` / `gen`
reweight on an (e,e') sample with **no new knob machinery**.

## What is NOT done — the (e,e') driver

1. **Sampling**: draw the struck nucleon from the spectral function for **both** species (p and n) — the
   CC generator samples neutrons only. Weight each by Z / N.
2. **Normalization**: the EM flux factor / phase space (the CC driver carries `flux_factor` and `SPIN_AVG`
   for a neutrino beam; a monochromatic e⁻ beam at fixed angle is a different normalization).
3. **Observable**: inclusive dσ/dω (or dσ/dE') at the fixed beam energy and angle acceptance, so it can be
   compared bin-by-bin with the oracle.
4. **Validation**: ADoNIS vs ACHILLES dσ/dω with ratio + χ²/ndf, exactly as the T2K and beam samples do.
   **This is the gate — do not put an (e,e') row into the knob × sample Fisher until it passes.**
5. Then: add it as a sample in `analysis/beams/beam_fisher.py` (Fisher is additive, so the column drops
   straight in) and check whether `qe_norm` / `axial_strength` / `vector_strength` / `sf_norm` / `src_tail`
   cross the shrinkage < 0.5 line — the whole point of the exercise.

## Caveat to carry into the paper

An (e,e') sample constrains the vector current and the spectral function **as implemented in our model**.
It cannot repair a wrong model, only its dials. And a real joint (e,e') + ν fit needs each sample's own
systematics; for a Fisher **informativeness** study ("what would this sample buy") that is fine, but it
must be stated as sensitivity, not as a combined measurement.
