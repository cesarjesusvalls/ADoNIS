# ADoNIS single-pion (RES / DCC) cross-section chain

Mirror of `docs/achilles_res_chain.md`, for the ADoNIS port. Files under `adonis/xsec/`.
Entry point: `res_xsec.generate(n, seed)` (`res_xsec.py:93`).

---

## 0. The master integrand (per channel, vectorised over n points)

`res_xsec.py:99-112`:
```python
w = amps2 · flux_factor · iw · SPIN_AVG · J          # J = J_beam · J_had · J_3body
sigma_channel = w.mean()
sigma_RES     = Σ_channels sigma_channel
```
This is the term-by-term analogue of ACHILLES
`xsec = amps2 · flux · initial_wgt · SpinAvg · event.Weight()`.

| ADoNIS factor | code | ACHILLES counterpart | status |
|---|---|---|---|
| `amps2` | `dcc_current.exclusive_amps2_batch` | DCC `amplitude()` | **bit-exact** vs RESDUMP |
| `flux_factor` | `backend.flux_factor` | `FluxFactor` | **bit-exact** (PDG masses) |
| `iw = N_NUC = 6` | `res_xsec.py:103` | `initial_wgt = N·S` | absorbed via importance (see §3) |
| `SPIN_AVG = 0.5` | `res_xsec.py:31` | `SpinAvg` | equal |
| `J = J_beam·J_had·J_3body` | `res_xsec.py:_sample_channel` | `event.Weight()` | convention equal; **3-body grouping differs** |

---

## 1. `amps2` — DCC matrix element (`dcc_current.py`)

- `exclusive_amps2_batch` (`dcc_current.py:160-210`): builds the DCC current from the spline of the
  same `dcc_EW.dat` table, contracts with the leptonic current, returns the spin-summed \|M\|².
- **Validated CHANNEL-INDEPENDENT and bit-exact** against the instrumented-ACHILLES `RESDUMP`
  (`tests/test_xsec_res_current.py`). Normalisation constant `_NORM = 3.77040e-05`.
- **Validity gate** (`dcc_current.py:209`): `W ∈ [1076.957, 2000] MeV  &  Q² ∈ [0, 5·10⁶] MeV²`.
  → **bit-exact match** to `currents_pi_dcc.f90:111,117`. (The table itself runs to W=5 GeV, but
  ACHILLES hard-cuts at 2000; ADoNIS does the same.)

---

## 2. `flux_factor` (`backend.py`)

`flux_factor(k_nu, p_struck, had_mass)` = `HBARC2 / (2·E_ν · 2·√(p_had²+m²)) · 1e6`, with
`had_mass = MASS_PDG_NEUTRON (939.57)` or `MASS_PDG_PROTON (938.27)` per channel.
→ **bit-exact** with ACHILLES `FluxFactor` (same PDG masses, same on-shell hadron energy).

---

## 3. Struck nucleon + `iw` — importance sampling (`spectral.py`, `res_xsec.py:62-64,103`)

ACHILLES samples the nucleon flat and puts `N·S(p,E)` into `initial_wgt`. ADoNIS instead **importance
-samples** the nucleon from the physical distribution (its Vegas analogue):

```python
pvec, E_rm = SpectralImportanceSampler.sample(n, rng)   # draws (|p|,E) ∝ |p|² S(p,E), Ω isotropic
p_struck   = [mN − E_rm, pvec]
J_had = 1 ;  iw = N_NUC = 6                              # the |p|²S and N·S cancel
```

Why `iw = N` is exact (`spectral.py:146-178`): sampling density `g(p,E,Ω) = p²S/(4π·Z_p)`,
normalised over `dp dE dΩ`, with `Z_p = ∫p² n_p dp = 1/(4π)` because the table's own `norm`
satisfies `∫4π p² (∫S dE) dp = norm` and `S` is divided by `norm`. The struck-nucleon contribution
then collapses to `N · 4π·Z_p = N`. The **same machinery validated σ_QE to 1.4 %.**

- E-removal window `2.5 < E_rm < 397.5` matches the pke12 grid exactly (`res_xsec.py:105`).
- ⚠ **All three channels sample from `pke12n` (neutron).** ACHILLES uses `pke12p` for the
  `p→pπ⁺` channel. Both tables have identical norm (6.000), so this is a **shape-only** effect on one
  of three channels (small), not a normalisation bias.

---

## 4. `SPIN_AVG = 0.5` (`res_xsec.py:31`)
Equal to ACHILLES `SpinAvg` for ν + spectral.

---

## 5. Phase-space Jacobian `J = J_beam · J_had · J_3body`

### 5a. Beam (`res_xsec.py:57-59`)
`E_ν ∈ [seed_min, max]` uniform; `J_beam = dE·flux(E)/flux_integral`. → equal to `BeamMapper`.

### 5b. Struck nucleon
`J_had = 1` (folded into the importance sampler, §3).

### 5c. Final state μNπ — **DIFFERENT GROUPING from ACHILLES** (`res_xsec.py:68-87`)

ADoNIS groups **(μN) together** and splits the **pion** off first — the *opposite* of ACHILLES,
which groups (Nπ) and splits μ off:

```python
s23 = M(μN)²  ∈ [ (m_μ+m_N)² , (√s − m_π)² ]   sampled UNIFORM       # NOT M(Nπ)²
split A:  total → (μN) + π     ISOTROPIC   (ACHILLES: → (Nπ)+μ, TChannel)
split B:  (μN)  → μ + N        ISOTROPIC   (ACHILLES: (Nπ)→N+π, isotropic)
density  = (2π)^5 · I2W_A · I2W_B / (s23_max − s23_min)
J_3body  = 1 / density
```
where `I2W = (2/π)·(s/√λ)` matches ACHILLES `Isotropic2Weight` exactly for the full angular range.

**Convention check (no inversion bug):** `J_3body = 1/density` is the forward Jacobian, consistent
with the QE anchor `J_2body = pcm/(4π·ecm) = 1/(ACHILLES TwoBody GenerateWeight)`. If ACHILLES used
isotropic for *both* splits, its `event.Weight()` 3-body part would equal ADoNIS `J_3body`. ACHILLES
uses TChannel for split A → the **per-event weights differ, but the integral is identical** (both are
valid, unbiased samplers of the same dΦ₃). The only consequence is sampling efficiency/variance —
and, importantly, ADoNIS cannot currently validate the RES `psw` bit-exactly against the RESDUMP the
way QE does, because the sampling variables differ.

---

## 6. Channel content (`res_xsec.py:33-37`)
- `n → n π⁺`   (struck n, pke12n, iw=6)
- `n → p π⁰`   (struck n, pke12n, iw=6)
- `p → p π⁺`   (struck p, **pke12n** ← should be pke12p, iw=6)

`generate()` sums the three channel means. Measured: σ_RES(importance) = 1.27·10⁻⁵ nb
(σ_RES(flat) = 1.46·10⁻⁵ nb). The code comment "ACHILLES ~1.698·10⁻⁵" is **unverified provenance**.
