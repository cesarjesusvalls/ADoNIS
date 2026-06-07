# ACHILLES single-pion (RES / DCC "virtual resonances") cross-section chain

Target process: `ν_μ + ¹²C → μ⁻ + N + π` (CC single-pion), nuclear model = spectral function,
hadronic current = ANL–Osaka DCC ("virtual resonances"). This is the `run_T2K_virtresonances`
configuration of the paper (arXiv:2508.19213v2).

Every file/line below is in `Achilles/src/Achilles/` unless noted.

---

## 0. The master integrand

`DefaultBackend::CrossSection` — `XSecBackend.cc:57-174`. For one phase-space point:

```
xsec = amps2 · flux · initial_wgt · SpinAvg · event.Weight()        (XSecBackend.cc:168)
```

The total cross section is the Monte-Carlo average of `xsec` over points drawn by the
MultiChannel/Vegas integrator. A cross section is an integral, so any unbiased sampler that
covers the same domain with the matching Jacobian yields the same σ — Vegas only reduces variance.

The five factors:

| factor | source | meaning |
|---|---|---|
| `amps2` | `XSecBackend.cc:88-122` | spin-summed \|M\|² (leptonic ⊗ hadronic current) |
| `flux` | `FluxFactor`, `XSecBackend.cc:35-46` | incident flux factor + nb conversion |
| `initial_wgt` | `InitialStateFactor`→`QESpectral::InitialStateWeight`, `NuclearModel.cc:603-614` | nucleon count × spectral function S(p,E) |
| `SpinAvg` | `XSecBackend.cc:26-33` | initial-state spin average (½) |
| `event.Weight()` | product of mapper Jacobians (Beam × Hadron × FinalState) | phase-space volume / dⁿ(random) |

---

## 1. `amps2` — the matrix element (spin SUM)

`XSecBackend.cc:91-101` (the `spect.size()==0` branch — no spectator, used for QE and RES):

```cpp
amps2 += std::norm(lcurrent_spin * hcurrent_spin);   // summed over lepton & hadron spins, bosons
```

- **Leptonic current** `LeptonicCurrent::CalcCurrents(k_in, k_out)` (`XSecBackend.cc:77`): CC V−A,
  W-propagator folded in.
- **Hadronic current** `m_model->CalcCurrents(...)` (`XSecBackend.cc:86`): for RES this routes to the
  DCC current (`currents_pi_dcc.f90` → `amp_dcc_sl.f::amplitude`). It is **table-driven**.
- `amps2` is a **spin SUM** over initial+final hadron spins and lepton spins. (For the DCC current
  the per-event \|M\|² is a Frobenius norm over hadron spin indices; the unitary spin-rotation blocks
  cancel in that norm.)

### DCC matrix element internals
- `currents_pi_dcc.f90:104-120` computes `W = √((k_π + p_N)²)` and `Q² = -(q)²` and applies the
  **hard validity gates**:
  - `Q² < 0  or  Q² > 5·10⁶ MeV²  → J_mu = 0`  (`:111`)
  - `W < 1076.957  or  W > 2000.0 MeV → J_mu = 0`  (`:117`)
- `amp_dcc_sl.f::interpolate_amp` (`:634`) splines the `dcc_EW.dat` table. The table grid is
  **W ∈ [1076.957, 5000] MeV (67 pts)**, **Q² ∈ [0, 5·10⁶] MeV² (28 pts)**, 14 partial waves
  (S11,S31,P11,P13,P31,P33,D13,D15,D33,D35,...). NOTE the table extends to W=5 GeV, but the current
  wrapper **deliberately restricts physics to W ≤ 2000 MeV** (the DCC validity region). Outside the
  grid the spline would `stop` (`amp_dcc_sl.f:660-667`), confirming the upstream gate is mandatory.
- Final scaling `J_mu = J_mu · 2·xmn / ħc` (`currents_pi_dcc.f90:130`).

---

## 2. `flux` — `FluxFactor` (`XSecBackend.cc:35-46`)

```cpp
double mass = ParticleInfo(hadron).Mass();             // PDG mass from Particles.yml
double flux = 2*E_lep_in  ·  2*sqrt(p_had² + mass²);   // on-shell hadron energy
return Constant::HBARC2 / flux * 1e6;                  // 1e6 = to nb
```
- Uses the **PDG `ParticleInfo` masses** (neutron 939.565…, proton 938.272…), not `Constant::mn/mp`.
- Uses the **on-shell** struck-nucleon energy `√(p²+m²)` regardless of binding.

---

## 3. `initial_wgt` — `QESpectral::InitialStateWeight` (`NuclearModel.cc:603-614`)

```cpp
removal = Constant::mN - p_struck.E();                 // E_struck is OFF-shell (= mN − E_removal)
return (struck==proton) ? nprotons · spectral_proton(|p|, removal)
                        : nneutrons · spectral_neutron(|p|, removal);
```
- **N_nucleon × S(|p|, E_removal)**. For ¹²C: `nprotons = nneutrons = 6`.
- **proton channels use `spectral_proton` (pke12p), neutron channels use `spectral_neutron`
  (pke12n).** Both spectral tables are normalised to N (∫4π p² S dp dE · norm convention).
- The spectral function S(p,E) is a `SpectralFunction.cc` 2-D table, evaluated by a Numerical-Recipes
  `Polint` 2-D interpolation (cubic in p, linear in E). Energy grid [2.50, 397.50] MeV, momentum grid
  [10, 790] MeV.

---

## 4. `SpinAvg` (`XSecBackend.cc:26-33`)

```cpp
spin = 1;  if(lepton not neutrino) spin *= 2;  if(NSpins>1) spin *= 2;  return 1/spin;
```
For an incident **neutrino** with the spectral model (`NSpins = 2`): `SpinAvg = 1/2`.

---

## 5. `event.Weight()` — phase-space Jacobian (the mappers)

`event.Weight()` is the product of the per-mapper phase-space weights. Each mapper's
`GenerateWeight` is its local density; the event weight is the **forward Jacobian**
(phase-space volume per unit random number), i.e. `dΦ / dⁿu`, so that σ = ⟨integrand · Weight⟩.

### 5a. Beam — `BeamMapper.cc`
Samples `E_ν` from the T2K flux. Weight = `dE · flux(E) / flux_integral`.

### 5b. Struck nucleon — `HadronicMapper.cc`
Samples `(|p|, cosθ, φ, E_removal)` of the bound nucleon. Weight = the `d³p dE` measure factor.
(Combined with `initial_wgt`'s `N·S` this gives the full nuclear phase-space integrand.)

### 5c. Final state μNπ — `ThreeBodyMapper` (`FinalStateMapper.cc:79-147`)

This is the structurally important piece. `s = (p_struck + k_ν)²`, masses `s2=m_N², s3=m_π²,
s4=m_μ²`.

**Grouping: (Nπ) together as the hadronic mass, μ split off.**
```
s23 = M(Nπ)² = W²                                                    (FinalStateMapper.cc:96)
s23 ∈ [ (m_N+m_π)² , (√s − m_μ)² ]   sampled UNIFORM in s23           (:89-90, :96)
TChannelMomenta( total → (Nπ-system) + μ )      t-channel sampling    (:100)
Isotropic2Momenta( (Nπ-system) → N + π )        isotropic in Nπ frame (:102)
```

**Weight** (`GenerateWeight`, `:113-147`):
```
GenerateWeight = (2π)^5 · TChannelWeight · Isotropic2Weight / (s23_max − s23_min)
```
with (full angular range ctmin=−1, ctmax=+1):
- `Isotropic2Weight = (2/π)·(s/√λ)·2/(ctmax−ctmin)`  (`:337-350`, `SqLam = √λ_Källén / s`)
- `TChannelWeight` = t-channel propagator density (`:281-335`).

The event weight contributed by this mapper is the **forward Jacobian**
`(s23_max−s23_min) / [(2π)^5 · TChannelWeight · Isotropic2Weight]`.

### 2-body anchor (QE, validated)
For comparison, the QE `TwoBodyMapper::GenerateWeight` (`:53-77`) gives forward Jacobian
`J_2body = pcm / (4π·ecm)`. ADoNIS reproduces the **full** `event.Weight()` (`psw`) bit-exactly for
QE via the instrumented `QEDUMP` — this anchors the Jacobian convention for the whole port.

---

## 6. Channel content
The DCC CC single-pion final states off ¹²C (ΔS=0, ΔQ=+1 hadronic):
- `n → n π⁺`   (struck neutron, spectral_neutron, ×6)
- `n → p π⁰`   (struck neutron, spectral_neutron, ×6)
- `p → p π⁺`   (struck proton,  **spectral_proton**, ×6)

σ_RES = Σ over channels of ⟨amps2·flux·initial_wgt·SpinAvg·Weight⟩.
