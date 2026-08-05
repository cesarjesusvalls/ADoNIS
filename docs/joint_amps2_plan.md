# Joint amps2 hard-vertex reweight — flexible reduced-quadratic design (QE + RES)

**Status:** planned. Implement QE first (gates → refresh QE records → commit/push), then RES.
**Scope:** the *hard-vertex* reweight `r_vertex` only. `w = w0·r_vertex·r_FSI·r_SF`; FSI (`cascade.pool_fsi_reweight`)
and SF (`sf_reweight`) already take all their knobs jointly-exact in closed form, and the blocks share no knobs, so
once `r_vertex` is exact the full weight is exact and any-order differentiable in all 28 dials.

## 1. The defect (confirmed against the code)
`reweight_model._hv_qe`/`_hv_res` build `r_vertex` as a **product of independently-built single-parameter records**
(`reweight_model.py:77-93`); each record is a per-event quadratic `(a,b,c)` from 3 evals of *one* scale with all others
at nominal (`amps2_records.py`). But `amps2 = Σ_ab|L_a·H_b|²` (`matrix_element.py::_contract`) and the current is
**linear in the form factors** — `vtx = F1·γ^μ + (iF2/2m_N)·σ^{μν}q_ν + FA·γ^μγ5 + FAP·(q^μ/m_N)·γ5`
(`dirac.py:203-206`) — so `amps2` is an **exact quadratic form** in the structure amplitudes. A product of two
single-axis records drops every inter-structure cross term (e.g. `amps2(v,a)=v²V+a²A+v·a·I`: the product carries `I`
twice, never the `v·a` term; `V=A=I=1,s=2` → truth 4, product 49/9). Consequences: **first derivatives at nominal are
exact** (Jacobian/Fisher unaffected); values off-nominal wrong at 2nd order; **every mixed 2nd derivative is
identically 0** where the truth has a cross term → the D2/D3 non-Gaussian tensors are wrong. Invisible to the suite
(`test_{strength,res_strength}_reweight.py` vary one knob at a time).

## 2. The mechanism: flexible reduced quadratic
**Atoms.** The current is linear in a fixed per-event set of *unit currents* `H_i` (kinematics only): `H = Σ_i F_i H_i`,
`F_i` the form-factor amplitudes the dials control. Then
```
amps2(F) = Σ_ij F_i F_j M_ij,   M_ij = Σ_ab Re[(L_a·H_i)*(L_a·H_j)]   (symmetric, KNOB-INDEPENDENT, per event)
```
Assemble `M` once from kinematics; `amps2` at any knobs is `FᵀMF` — exact (float precision; gate: `==` direct amps2).

**Dials→atoms** is a smooth closed form: `w(dials)=F(dials)ᵀMF(dials)/F₀ᵀMF₀`. Closed-form ⇒ **exact derivatives at any
order for any simultaneous variation** (D2/D3 exact). Today's product = the diagonal-only approximation (keeps `M_ii`,
drops off-diagonals).

**Reduced (only what's needed).** For a declared dial set, partition atoms into **active** `A` (some dial scales them)
and **frozen** `Z` (held at nominal). The quadratic splits *exactly*:
```
amps2 = F_Aᵀ M_AA F_A + 2 F_Aᵀ v + c,     v = M_AZ F_Z⁰  (|A|),   c = F_Z⁰ᵀ M_ZZ F_Z⁰  (scalar)
```
Store/compute only `(M_AA, v, c)` + the nominal FFs and `Q²` to rebuild `F(dials)`. Exact (frozen atoms compressed into
`v,c`, not dropped), minimal, and it auto-grows/shrinks with the dial list. One channel-agnostic pair:
`build_reduced_amps2(dials, kin) -> record`, `reweight(record, dials) = FᵀM_AAF + 2Fᵀv + c` over `F₀ᵀMF₀`.

## 3. QE design (implement first)
- **Atoms** `{F1,F2,FA,FAP}` (4). All 7 QE dials map onto these ⇒ no frozen QE atoms ⇒ store the full `M` (10), the
  nominal Sachs FFs `G_E^p,G_M^p,G_E^n,G_M^n` + `FA,FAP`, and `Q²`. (`v,c` unused for QE.)
- **Unit currents**: add `return_structures=True` to `hadron_current_qe_dirac` returning `H_i` (the 4 basis vertices at
  unit coeff), shape `(...,4struct,4combo,4mu)`. New `build_qe_mij_records` contracts with `L` → `M_ij`.
- **F(dials)**: `F1=vector_strength·[F1p(gep,gmp)−F1n(gen,gmn)]`, `F2=vector_strength·[F2p−F2n]` with
  `F1x=(G_E^x+τG_M^x)/(1+τ)`, `F2x=(G_M^x−G_E^x)/(1+τ)`, `τ=Q²/4m_N²`; `FA=axial_strength·dipole(Q²;M_A_qe)·FA_nom`,
  `FAP=axial_strength·dipole(Q²;M_A_qe)·FAP_nom`. `mij_reweight(rec, knobs)=FᵀMF/F₀ᵀMF₀`.
- **Probe**: EM zeroes `FA,FAP` and uses the struck nucleon's own FFs (per `dirac.py`); NC per its coupling dict.
- Rewire `_hv_qe` to the one joint reweight.

## 4. RES design (after QE)
- **Atoms** `{wave w × block∈{V,A,P}}` (≤42). Knob coefficients compose **nested** (differential.py:207-223):
  `g_{w,V}=(1+pw_norm[w])·[δ if w=5]`, `g_{w,A}=g_{w,V}·r_axial`, `g_{w,P}=g_{w,A}·pion_pole`,
  `r_axial=dipole(Q²;M_A_res)·res_axial_strength`.
- **Paper active set** (`pw_norm` dormant): the axial+pole blocks are scaled; the bare vector block is frozen; wave 5 is
  split out for `delta_strength`. So `A≈{rest_A, rest_P, w5_V, w5_A, w5_P}`, `Z={rest_V}` → small reduced form. The
  builder derives `A/Z` from the dial list, so turning `pw_norm` on just enlarges `M_AA` (same ~3 current passes).
- **Unit currents**: `return_structures` on `build_zmtx_batched`/`exclusive_amps2_batch` emitting per-(wave,block)
  currents in ~3 shared passes (V/A/P); `build_res_mij_records` forms `(M_AA,v,c)`. Rewire `_hv_res`.

## 5. Banks — refresh the records (NOT regenerate events)  ← decision
The hard-vertex records are **persisted** in the banks (`hv_qe_ma_a/b/c/Q2`, `hv_qe_vec_*`, `hv_qe_g{ep,en,mp,mn}_*`,
`hv_res_{ma,pp,delta}_*`). The physics events (kinematics, final states, FSI `f_*`) are complete and **not regenerated**.
- **Inputs for the rebuild are already in the banks**: `k_nu`, `k_mu(=k_lep)`, `p_struck`, and RES `res_p_N/res_p_pi/`
  `res_ipid/res_ppid`. QE `p_out` is the deterministic 2-body vertex solution of `(k_nu,k_lep,p_struck)` → reconstructed.
- **Refresh pass** (`jobs/`-style, per-chunk, in place): read kinematics → build the reduced-quadratic records →
  replace the `hv_*` fields with the new layout (`hv_qe_mij_*`, `hv_qe_ff_*`, `hv_qe_Q2`; RES `hv_res_M/v/c/*`). Cheap
  (one vertex-record build), parallel over chunks, no re-sampling/cascade. Works on the swapped high-stat banks.
- **Compat**: `bank_reweight`/`reweight_model` read the new fields; keep a reader that falls back to the old per-knob
  fields (exact single-axis) until a bank is refreshed, so nothing breaks mid-migration.

## 6. Validation gates (each channel)
1. **Exactness**: `FᵀMF == ` direct `amps2` for random `F` (~1e-12).
2. **Nominal identity**: `reweight == 1` at nominal (exact).
3. **Single-knob byte-match**: for each dial alone, joint `==` the old per-knob reweight (no regression on the suite).
4. **NEW joint gate**: vary ≥2 dials (vector+axial; Sachs pair; RES `M_A_res`+`res_axial_strength`+`pion_pole`)
   *together*; `==` direct `amps2` ratio (today's product fails this).
5. **Mixed 2nd derivative**: autodiff `∂²log w/∂θ_i∂θ_j` `==` direct-amps2 finite difference (nonzero).
6. **First-order invariance**: the Jacobian/Fisher (sec2/sec3) numerically unchanged.
7. **FSI/SF joint-exactness**: pin that `r_FSI`,`r_SF` are exact under simultaneous FSI/SF variation.
8. **Figure invariance**: nominal sec1 histograms unchanged.

## 7. Execution order
1. QE: implement §3 → gates §6 (1-6,8) → refresh QE records on the paper banks → commit/push.
2. RES: implement §4 → gates §6 → refresh RES records → commit/push.
Non-goals: no event regeneration; no change to `w0`, FSI, SF; first-order outputs must be numerically unchanged (gate 6).
