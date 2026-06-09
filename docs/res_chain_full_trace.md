# RES single-pion σ: full operation-by-operation trace, ACHILLES vs ADoNIS

Goal: write down **every** operation ACHILLES performs to compute the CC single-pion
cross section σ_RES (ν_μ + ¹²C, T2K flux, `Cascade:Run:False`), with the exact file/line
each fact comes from. Then, section by section, state **exactly** what ADoNIS
(`adonis/xsec/res_xsec.py`) does for the same step. No inference where a line can be cited.

All ACHILLES paths are under `Achilles/`. All ADoNIS paths under `ADoNIS/`.

Config under trace: `Achilles/run_nofsi_res.yml`
- `Processes: - Leptons: [14, [13]]` (ν_μ → μ⁻)
- Beam: `Spectrum`, `flux/T2K_nu.dat`
- NuclearModel: `FortranModel` / `RES_Spectral_Func`, `ConfigFile: data/info_C12_pke.data`
- Nucleus: `data/default/12C.yml`

---

# PART 1 — THE ACHILLES CHAIN

## A0. Process enumeration and grouping (run once, at setup)

`ProcessGroup::ConstructGroups` — `src/Achilles/Process.cc:308-338`
1. For the one YAML process `[14,[13]]`, call `model->AllowedStates(info)` (line 315).
2. `NuclearModel::AllowedStates`, Resonance mode, leptonic charge = −1
   (`src/Achilles/NuclearModel.cc:269-283`). It returns, **in this order**:
   - `[0]`  n → {p, π⁰}      (`m_hadronic = {{neutron},{proton, pion0}}`, line 277-278)
   - `[1]`  n → {n, π⁺}      (line 279-280)
   - `[2]`  p → {p, π⁺}      (line 281-282)
3. Each allowed state becomes a `Process`. Processes are bucketed into groups
   **keyed by `info.Multiplicity()`** (`Process.cc:323-327`).
   `Multiplicity = 1 + n_had_in + n_final` (`ProcessInfo.cc:7-9`) = 1+1+3 = **5** for all
   three. ⇒ **all three channels live in ONE ProcessGroup.**

`ProcessGroup::SetupIntegration` — `src/Achilles/Process.cc:352-374`
- The phase-space channels (mappers) are built from **`m_processes[0].Info()` only**
  (`Process.cc:357`: `m_backend->SetupChannels(m_processes[0].Info(), …)`).
- ⇒ **The shared mapper uses process[0] = (n→pπ⁰) masses and threshold for ALL three
  channels.** This is verified empirically: in the 300-line RESDUMP, all three channels
  carry ⟨m_N⟩ = 938.27 MeV (proton) and ⟨m_π⟩ = 134.98 MeV (π⁰), regardless of their PID
  label (`tests/data/res_dump_achilles.txt`).

`ProcessInfo::Masses()` — `src/Achilles/ProcessInfo.cc:11-25`
- Returns squared masses, **outgoing leptons first, then outgoing hadrons** in
  `m_hadronic.second` order. For process[0]: `Masses() = [m_μ², m_p², m_π0²]`.

## A1. The integration driver (per call)

`MultiChannel::operator()` — `include/Achilles/MultiChannel.hh:149-188`
For each of `params.ncalls` points (default 10000, `MultiChannel.hh:26`):
1. Draw `ndims` uniform randoms `rans` (line 161).
2. Select a channel index `ichannel ~ channel_weights` (line 164).
3. `func.GeneratePoint(ichannel, rans, point)` (line 167) → maps rans to momenta.
4. `wgt = func.GenerateWeight(channel_weights, point, densities)` (line 170) — the
   multichannel weight (see A2).
5. `val = func(point, wgt)` (line 171) = the cross-section function value (see A6).
6. `results += val` (line 174). After the loop, `summary.sum_results += results`.

The reported number `Total xsec = m_xsec.Mean()` (`Process.cc:425`) is the running mean of
`val` over all integration points across iterations. Optimization runs ≥7 iterations
(`MultiChannel.hh:27 nint_default=7`) plus Vegas refines.

`m_units = nb`, `UnitScale(nb)=1` (`MultiChannel.hh:59-72`): no extra unit scaling on the
reported nb number.

## A2. The per-point weight (multichannel + Vegas)

`Integrand::GenerateWeight` — `include/Achilles/Integrand.hh:85-96`
```
weight = Σ_i  wgts[i] * densities[i] / vw_i           // i over integration channels
return 1.0 / weight
```
- `densities[i] = channels[i].mapping->GenerateWeight(point, rans)` (line 90) = the
  phase-space density of channel i's mapper at this point (= PSMapper weight, A3).
- `vw_i = channels[i].integrator.GenerateWeight(rans)` (line 92) = the Vegas-grid Jacobian
  (A2a).
- `wgts[i] = channel_weights[i]`, the multichannel α-weights.

So `event.Weight()` (the dumped `psw`) is this combined `1/Σ(α·dens/vw)`. With a single
channel and a uniform (un-adapted) Vegas grid, it reduces to `1/density`.

`m_xsec += weight` per point where `weight = Σ_processes CrossSection` (A5).

### A2a. Vegas grid Jacobian
`AdaptiveMap::operator()` (`src/Achilles/AdaptiveMap.cc:38-54`) remaps each uniform ran
through the per-dim inverse-CDF; Jacobian = Π width(i,bin)·m_bins. Initially uniform
(width=1/bins ⇒ Jacobian=1). `GenerateWeight` (line 56-63) returns the same Jacobian.
Vegas is unbiased: adapting the grid changes per-point weight but not the mean.

## A3. Phase-space point: PSMapper = Beam ⊗ Hadron ⊗ FinalState

`PSMapper::GeneratePoint` — `src/Achilles/PhaseSpaceMapper.cc:5-32`
Momentum vector order: `[0]` initial lepton, `[1]` initial hadron, `[2]` outgoing lepton,
`[3]` outgoing hadron-0 (nucleon), `[4]` outgoing hadron-1 (pion).
`GenerateWeight` (line 34-66): `wgt = lwgt * hwgt * mwgt` (product of the three sub-mappers).

### A3.1 Beam (incident neutrino) — `BeamMapper` + `Spectrum`
`BeamMapper::GeneratePoint` — `src/Achilles/BeamMapper.cc:7-14`
- `point[0] = m_beam->Flux(beam_id, rans, smin_seed)` with
  `smin_seed = (Smin() - Masses()[1]) / (2*sqrt(Masses()[1]))` (line 10).
  `Masses()[1] = m_p²` (process[0] outgoing nucleon). `Smin()` = process[0] threshold
  `(m_μ+m_p+m_π0)²` (set on the mapper; see A3.2 note on Smin).
`Spectrum::Flux` — `src/Achilles/Beams.cc:283-288`
- `min_energy = max(smin_seed * units, m_min_energy)`; `energy = ran*delta + min_energy`;
  returns `{E,0,0,E}` (massless ν along z).
`Spectrum::GenerateWeight` — `Beams.cc:290-296`
- returns `(delta_energy * m_flux(E)) / m_flux_integral`. `BeamMapper::GenerateWeight`
  returns `1/wgt` (`BeamMapper.cc:25`).
- `m_flux_integral` = Σ width·height over the histogram (`Beams.cc:101,147`).

### A3.2 Struck nucleon — `QESpectralMapper`
`QESpectralMapper::GeneratePoint` — `src/Achilles/HadronicMapper.cc:30-78` (E_ν ≡ point[0].E())
- `radical = E_ν² + 2 E_ν m_N + m_N² − Smin` (line 33-34), clamped ≥0.
- `pmin = E_ν − √radical`, `pmax = E_ν + √radical`, clamp `pmin≥0`, `pmax≤800` (35-39).
- `mom = (pmax−pmin)·ran0 + pmin` (41).
- `cosT_max = (2 E_ν m_N + m_N² − mom² − Smin)/(2 E_ν mom)`, clamp to [−1,1] (42-44).
- `cosT = (cosT_max+1)·ran1 − 1` (45) ⇒ struck nucleon polar angle ∈ [−1, cosT_max].
- `phi = 2π·ran2` (47).
- `det = E_ν² + mom² + 2·(p⃗·E_ν ẑ) + Smin` (50); `emax = m_N + E_ν − √det` (51),
  `emax = min(emax, m_N − mom)` (52), `emax = min(emax, 400)` (53).
- `energy = emax·ran3 − 1e-8` (54) ⇒ removal energy ∈ [−1e-8, emax].
- `point[HadronIdx()] = {m_N − energy, mom·(sinT cosφ, sinT sinφ, cosT)}` (64).
  ⇒ struck nucleon 4-momentum has time component `m_N − removal`.
`QESpectralMapper::GenerateWeight` — `HadronicMapper.cc:80-126`
- `wgt = 1 / (p² · dp · dCos · dPhi · dE)` (line 115), with `dp = pmax−pmin`,
  `dCos = cosT_max+1`, `dPhi = 2π`, `dE = emax`. The `p²` is the d³p Jacobian.
- **Smin for this mapper = process[0] threshold** (the mapper's `Smin()`), i.e.
  `(m_μ+m_p+m_π0)²`. Same Smin used for beam seed and all three channels.

### A3.3 μ+N+π final state — `ThreeBodyMapper`
Masses set in constructor (`FinalStateMapper.hh:58`): `s2 = Masses()[0] = m_μ²`,
`s3 = Masses()[1] = m_p²`, `s4 = Masses()[2] = m_π0²`. Constants (`FinalStateMapper.hh:97`):
`m_amct=1, m_alpha=0.9, m_ctmax=1, m_ctmin=−1`.
`ThreeBodyMapper::GeneratePoint` — `src/Achilles/FinalStateMapper.cc:79-111`
- `s = (p_lep_in + p_had_in).M2()`, `√s` (84-86).
- `s23_max = (√s − √s4)²` (89), `s23_min = max((√s2+√s3)², 1e-8)` (90)
  ⇒ s23 = invariant mass² of the (μ,N) pair; `s4 = m_π0²` is split off.
- `s23 = s23_min + (s23_max − s23_min)·ran0` (uniform) (96).
- `TChannelMomenta(p_lep_in, p_had_in, p23, p_pi, s23, s4, t_mass=0, …, ran1, ran2)` (100):
  splits total → (p23 of mass² s23) + (pion of mass² s4) by a **t-channel** angular law.
- `Isotropic2Momenta(p23, s2, s3, p_mu, p_N, ran3, ran4, ctmin, ctmax)` (102): splits the
  (μN) system → μ (s2) + N (s3) **isotropically** in [ctmin,ctmax]=[−1,1].
`ThreeBodyMapper::GenerateWeight` — `FinalStateMapper.cc:113-147`
```
wt  = 1/(s23_max − s23_min)                                  (135)
wt *= TChannelWeight(p_in1,p_in2, p23, p_pi, …)              (140)
wt *= Isotropic2Weight(p_mu, p_N, …)                         (142)
if(wt!=0) wt = 1.0/wt / (2π)^(3·3−4=5)                       (144)
return 1/wt                                                  (146)
```
⇒ the returned density = `(2π)^5 · TChannelWeight · Isotropic2Weight / (s23_max−s23_min)`.
- `Isotropic2Weight` (`FinalStateMapper.cc:337-350`):
  `= 2/π / SqLam(s,s1,s2) · 2/(ctmax−ctmin)` = `2/π / SqLam` for ct∈[−1,1].
  `SqLam(s,s1,s2)=√((s−s1−s2)²−4s1s2)/s` (229-235).
- `TChannelWeight` (`FinalStateMapper.cc:281-335`): the t-channel angular Jacobian (peaked
  sampling toward forward angles, full [−1,1] coverage), with `ctexp=m_alpha=0.9`.

## A4. The summed cross section per point

`ProcessGroup::SingleEvent` — `src/Achilles/Process.cc:440-473`
- Build `Event` with the shared momenta and `event.Weight()=ps_wgt` (multichannel, A2).
- During optimization (`b_optimize`), `SelectProcess` is not called; `process_opt=nullopt`.
- `CrossSection(event, nullopt)`.

`ProcessGroup::CrossSection` — `src/Achilles/Process.cc:287-302`
```
weight = 0
for i in processes:                                  // all 3 channels
    weight += m_backend->CrossSection(event, processes[i])
event.Weight() = weight
m_xsec += weight                                      // accumulates the integral
```
⇒ **Per point, all three channels' cross sections are summed.** Each uses the SAME shared
momenta and the SAME `event.Weight()` (psw) computed from process[0]'s mapper.

## A5. The backend cross-section per channel

`DefaultBackend::CrossSection` — `src/Achilles/XSecBackend.cc:57-174`
- `event = event_in` (copy); `TransformFrame` (63); `ExtractParticles` (72) assigns **this
  channel's PIDs** to the **shared momenta** (so π⁺ channels carry m_π0 momenta — verified
  from the dump, A0).
- Lepton current from bare `lepton_in, lepton_out` (77-78); `q = lep_in − lep_out` (85).
- `hadron_current = m_model->CalcCurrents(hadron_in, hadron_out, spect, q, ff_info)` (86).
- `amps2 = Σ_lepton_spin Σ_hadron_spin |L·H|²` (spin SUM, 92-99).
- `flux = FluxFactor(...)` (126); `initial_wgt = InitialStateFactor(...)` (129).
- `xsec = amps2 * flux * initial_wgt * SpinAvg * event.Weight()` (168).

`FluxFactor` — `XSecBackend.cc:35-46`
- `flux = 2 E_lep · 2 √(p_had² + m²)`, `m = m_hadronic.first[0]` mass; `to_nb=1e6`;
  return `HBARC2 / flux * 1e6`. `HBARC2 = (ħc)²·10` mb·MeV² (`Constants.hh:17`).

`InitialStateFactor` → `FortranModel::InitialStateWeight` → `res_spec_init_wgt`
- `FNuclearModel.cc:131-154` → `res_spectral_model.f90:165-197`:
  `E = mqe − p4(1)` (removal energy, mqe=`Constant::mN`), `pmom = |p⃗|`;
  `wgt = nproton·spectral_p(pmom,E)` if struck=proton else `nneutron·spectral_n(pmom,E)`
  (lines 181-188). nproton=nneutron=6 for ¹²C.
- Neutron channels use `spectral_n` (pke12n); proton channel uses `spectral_p` (pke12p).

`SpinAvg` — `XSecBackend.cc:26-33`
- `spin_avg=1`; ν ⇒ ×1 (line 28 only for non-ν); `NSpins()>1` ⇒ ×2 (line 30);
  return `1/spin_avg`. For this model = **0.5** (validated bit-exact in RESDUMP).

## A6. The matrix element amps2 (DCC)

`res_spec_currents` — `src/Achilles/fortran/res_spectral_model.f90:101-149`
- `p4 = struck nucleon`, `pp4 = outgoing nucleon`, `kpi4 = pion`, `q4 = q` (129-133).
- `coupling = ff%lookup("FResV")` (127).
- `current_init(p4, pp4, q4, kpi4)` (141); `hadr_curr_matrix_el(pid_in, pid_N, pid_pi, …)`
  (142); `cur = coupling · J_mu` (145-149).

`current_init` (de Forest off-shell shift) — `src/Achilles/fortran/currents_pi_dcc.f90:25-39`
```
w = q(1)
q(1) = w + p1(1)                                   // p1(1)=struck E as given (m_N−removal)
p1(1) = sqrt(|p⃗_struck|² + xmnuc²)                 // put struck nucleon on-shell, xmnuc=(mn+mp)/2
q(1) = q(1) − p1(1)                                // de Forest shifted q energy
```
`xmnuc=(xmn+xmp)/2` (`currents_pi_dcc.f90:20`).
`hadr_curr_matrix_el` — `currents_pi_dcc.f90:41+`: builds W=(pp1+kpi).M, Q²=−q_shift²,
angles in the (Nπ) CM, partial-wave (ANL-Osaka DCC) sum via `amp_dcc_sl.f`.

Validity gate (table support): outside `W∈[1076.957,2000]`, `Q²∈[0,5e6]` ⇒ J_mu=0.

---

# PART 2 — THE ADoNIS CHAIN

All ADoNIS refs are `adonis/xsec/res_xsec.py` unless noted. Each subsection mirrors the
ACHILLES step above and ends with **MATCH** or **DIVERGENCE**.

## B0. Channels and masses (mirror of A0)

`CHANNELS` — `res_xsec.py:33-37`:
```
(2112, -1, M_N, 211, M_PIP, MASS_PDG_NEUTRON)   # n -> n pi+   m_Nf=939.57  m_pi=139.57
(2112, -1, M_P, 111, M_PI0, MASS_PDG_NEUTRON)   # n -> p pi0   m_Nf=938.27  m_pi=134.98
(2212, +1, M_P, 211, M_PIP, MASS_PDG_PROTON)    # p -> p pi+   m_Nf=938.27  m_pi=139.57
```
`generate()` (line 103-134) loops the 3 channels, samples each **independently** with its
**own** `(m_Nf, m_pi)`, computes `sc = w.mean()` per channel (122), and sums `sig += sc`.

> **DIVERGENCE 1 (primary).** ACHILLES builds ONE shared phase space from process[0] =
> (n→pπ⁰) and uses **m_p (938.27) and m_π0 (134.98) for ALL three channels** — both for the
> phase-space measure and for the momenta fed to amps2 (A0, verified from the dump). ADoNIS
> uses **each channel's own** masses, i.e. the heavier m_π+ (139.57) for the two π⁺ channels
> (`n→nπ⁺`, `p→pπ⁺`) and m_n for `n→nπ⁺`. A heavier pion shrinks `s23_max=(√s−m_π)²` and the
> SqLam factors ⇒ smaller 3-body phase space ⇒ **lower σ for 2 of 3 channels**. Direction
> matches the observed 0.74× deficit. The π⁺/π⁰ mass split is 4.6 MeV (3.3%), against a
> near-threshold phase space, so the effect is amplified well beyond 3.3%.

## B1. Integration driver (mirror of A1)

`generate()` draws `u = rng.random((n,10))` per channel (`_sample_channel`, line 54), builds
per-event weight `w` (120), takes `w.mean()` (122). The estimator is
`σ = Σ_channels E[a2·flux·iw·spinavg·J]`.

> **MATCH (in expectation).** ACHILLES sums the 3 channels per shared point and means over
> points; ADoNIS means each channel separately and sums. `E[Σ_c w_c] = Σ_c E[w_c]`, so the two
> are identical in expectation **provided the per-channel integrand and measure agree** — which
> is exactly what DIVERGENCE 1 breaks.

## B2. Per-point weight (mirror of A2)

ADoNIS has **no Vegas and no multichannel**. The weight is `J = J_beam · J_had · J_3body`
(`res_xsec.py:100`, assembled in `_sample_channel`), each factor = `1/density` of its own
mapper. `w = a2 * fl * iw * SPIN_AVG * J` (120).

> **MATCH.** With a single channel and a uniform Vegas grid, ACHILLES's `event.Weight()`
> reduces to `1/density` (A2/A2a). ADoNIS's `J` is exactly that inverse density. Vegas only
> changes variance, not the mean. (Validated: RES `psw` reproduced to 1.4e-14,
> `scripts/validate_res_psw.py`.)

## B3.1 Beam (mirror of A3.1)

`_sample_channel` — `res_xsec.py:53-64`:
- `Smin = (M_MU + m_Nf + m_pi)²` (line 55) — **per-channel** threshold.
- `minE = max((Smin − m_Nf²)/(2 m_Nf)/1000, flux.min_energy)` (60).
- `E = u4·(maxE−minE)+minE`, `k_nu = {E,0,0,E}` (62-63).
- `J_beam = (dE_beam · flux.f(E)) / flux.flux_integral` (64) — matches `Spectrum::GenerateWeight`.

> **DIVERGENCE 1b.** Beam seed uses **per-channel** `(Smin, m_Nf)`. ACHILLES uses
> `Masses()[1] = m_p²` and process[0]'s `Smin` for ALL channels (A3.1). Same root cause as
> DIVERGENCE 1. Effect on σ is small (seed only sets the low-E flux cutoff) but nonzero.

## B3.2 Struck nucleon (mirror of A3.2)

`res_xsec.py:65-67, 92-98`:
- `pvec, energy = _IMP.sample(n, rng)` — **importance** sampler, draws `(|p|,E,Ω)~p²·S_norm`
  (`adonis/xsec/spectral.py:146-178`); direction isotropic over full 4π.
- `iw = N_NUC = 6` (line 113), constant — the `p²·S_norm` cancels exactly:
  `∫d³p dE N·S·[rest] = E_q[N·[rest]]` with `q=p²·S_norm` (derived; S in integrand = S in q).
- emax gate (94-96): `det_e = E² + mom² + 2 p_z E + Smin`; `emax = m_N+E−√det_e`;
  `emax = min(emax, m_N−mom, 400)`; `valid &= energy < emax` (98). Mirrors `HadronicMapper.cc:50-53`.

> **MATCH (struck-nucleon measure).** Importance sampling with `iw=N` is the exact analytic
> equivalent of ACHILLES's flat (p,cosθ,φ,E) sampling weighted by `N·S` (QE closes at 0.996×
> through this same machinery). The cosT_max restriction and the [0,emax] box are reproduced by
> the `s>Smin` and `energy<emax` gates (unbiased: gating a 4π-normalised importance draw equals
> restricting the sampling domain).
>
> **DIVERGENCE 1c.** The `Smin` inside `det_e`/`emax` and `s>Smin` is **per-channel**
> `(M_MU+m_Nf+m_pi)²`; ACHILLES uses process[0]'s `Smin` for all three. Same root cause.
>
> **DIVERGENCE 3 (minor).** `_IMP` is built once from **pke12n** (`res_xsec.py:25`) and used
> for ALL channels — including `p→pπ⁺`, where ACHILLES's integrand uses **pke12p**
> (`res_spec_init_wgt`, A5). Because `iw=N` folds in the sampler's S, the proton channel is
> effectively integrated with the neutron S. For ¹²C (near-isoscalar) the p/n SF differ little,
> so this is a few-% channel-3 effect, not the bulk deficit.

## B3.3 μ+N+π final state (mirror of A3.3)

`_sample_channel` — `res_xsec.py:70-91`:
- `s, √s` from `P=k_nu+p_struck` (70-71).
- `s23max=(√s−m_pi)²`, `s23min=max((M_MU+m_Nf)², 1e-8)`, `s23 = s23min+(s23max−s23min)·u5` (72-73).
- **split A (total → μN + π), ISOTROPIC**: `ctA=2u6−1`, `phA=2π u7` (76); boost to lab (80);
  `I2W_A = 2/π / SqLam(s, s23, m_pi²)` (81).
- **split B (μN → μ + N), ISOTROPIC**: `ctB=2u8−1`, `phB=2π u9` (84); `I2W_B = 2/π / SqLam(s23, M_MU², m_Nf²)` (89).
- `density = (2π)^5 · I2W_A · I2W_B / (s23max−s23min)` (90); `J_3body = 1/density` (91).

> **MATCH (measure normalisation).** `(2π)^5`, `1/(s23max−s23min)`, and the `2/π/SqLam`
> isotropic weights are exactly ACHILLES's `ThreeBodyMapper::GenerateWeight` (A3.3) with
> ct∈[−1,1]. The only sampler difference: ADoNIS splits the **pion** isotropically (split A)
> whereas ACHILLES splits it via **t-channel** (`TChannelMomenta`). Different importance
> density, identical integral in expectation (both cover the full [−1,1]; verified: an
> independent **t-channel** replication `scripts/res_exact_sigma.py` gives the SAME 0.74×).
>
> **DIVERGENCE 1d.** `m_pi` and `m_Nf` here are **per-channel** (heavier m_π+ for the π⁺
> channels) ⇒ smaller `s23max` and SqLam ⇒ smaller `J`-integrated phase space. ACHILLES uses
> m_π0, m_p for all. Same root cause as DIVERGENCE 1; this is where the volume actually shrinks.

## B5/B6. Backend pieces + amps2 (mirror of A5/A6)

- `flux = flux_factor(k_nu, p_struck, had_mass=mstr)` (`res_xsec.py:119`) — bit-exact vs A5.
- `iw = N_NUC = 6` carries `nproton/nneutron·S` (A5) via the importance cancellation.
- `SPIN_AVG = 0.5` (line 31) — matches A5.
- `a2 = exclusive_amps2_batch(k_nu, k_mu, p_struck, p_N, p_pi, itiz, ppid)` (117) —
  `adonis/xsec/dcc_current.py`. de Forest shift `qsh[:,0]=q[:,0]+p_struck[:,0]−E_on` with
  `E_on=√(p⃗²+m_N²)`, `m_N=(mp+mn)/2` (`dcc_current.py:157-158`) — matches `current_init` (A6).
  W, Q², angles, partial-wave sum, table gate `W∈[1076.957,2000]`, `Q²∈[0,5e6]` — validated
  bit-exact vs RESDUMP (mean ratio 1.004, max 1.3%).

> **MATCH (function), DIVERGENCE 1e (evaluation point).** `exclusive_amps2_batch` reproduces
> ACHILLES's amps2 **at ACHILLES's kinematics**. But ADoNIS feeds it **per-channel-mass
> momenta** (heavier π⁺), while ACHILLES evaluates every channel's amps2 at the **shared
> (m_p, m_π0) momenta** (A5/A0). So even the matrix element is sampled at slightly different
> (W, Q²) for the π⁺ channels. The amps2 **function** is correct; the **points** differ — same
> root cause as DIVERGENCE 1.

---

# SUMMARY OF DIVERGENCES

| # | Step | ACHILLES | ADoNIS | Bias direction |
|---|------|----------|--------|----------------|
| **1** | masses for ALL channels | process[0]=(m_p, m_π0) shared | each channel's own (m_π+ for 2/3) | ADoNIS phase space smaller ⇒ **σ lower** ✓ |
| 1b | beam seed | Masses()[1]=m_p², proc[0] Smin | per-channel m_Nf², Smin | small |
| 1c | emax / s-gate Smin | proc[0] Smin (all) | per-channel Smin | small |
| 1d | s23max, SqLam | m_π0, m_p | per-channel m_π, m_Nf | **main volume loss** ✓ |
| 1e | amps2 eval point | shared (m_p,m_π0) momenta | per-channel-mass momenta | small/medium |
| 3 | proton-channel SF | pke12p | pke12n (importance) | few-% on channel 3 |
| — | pion split sampler | t-channel | isotropic | none (unbiased) |
| — | Vegas/multichannel | yes | no | none (unbiased) |

**Root cause (single):** ACHILLES groups all three CC-1π channels by multiplicity into ONE
ProcessGroup, builds the phase space from **process[0] = n→pπ⁰ only**, and evaluates all three
channels' matrix elements at those **shared (m_p, m_π0) momenta** with the shared psw
(`Process.cc:323-327, 357`; `Process.cc:287-302`; `XSecBackend.cc:72`; dump-verified). ADoNIS
treats each channel with its own, physically-correct masses. The two π⁺ channels therefore get
a lighter pion (m_π0) and a larger phase space in ACHILLES than in ADoNIS.

**The fix:** to reproduce ACHILLES's NUMBER, sample all three channels from the **shared
process[0]=(m_p, m_π0)** phase space (same Smin, same s2/s3/s4, same momenta) and evaluate each
channel's amps2 on those shared momenta — i.e. mirror the ProcessGroup. (Whether ACHILLES's
shared-mass treatment is *physically* preferable to ADoNIS's per-channel-exact one is a separate
question; for closure to ACHILLES we must match its convention.)

**Next:** quantify by re-running res_xsec with all channels forced to (m_p, m_π0) and Smin_0,
and check σ → 1.6947e-5.

---

## UPDATE 1 — DIVERGENCE 1 TESTED AND **REJECTED**

`scripts/res_shared_mass_sigma.py` reproduces the ProcessGroup convention exactly: ONE shared
phase-space sample from process[0]=(m_p, m_π0), all three channels' amps2 evaluated on those
shared momenta and summed. Result (N=40000×4 seeds):

```
ACHILLES target sigma_RES = 1.6947e-05 nb
per-channel-mass (res_xsec)  = 1.2465e-05  ratio 0.736
SHARED (m_p,m_pi0) all chans = 1.2433e-05  ratio 0.734
```

The shared-mass convention moves σ by **<1%** — it is **not** the deficit. The π⁺/π⁰ mass
difference is real but the near-threshold amplification I expected does not materialise (the T2K
flux puts most strength well above the π⁺/π⁰ threshold split). DIVERGENCE 1/1b/1c/1d/1e are all
real bookkeeping differences but together negligible for σ.

**Status:** the 0.736× deficit is still unexplained by anything *read so far*. Every per-event
factor (amps2 1.004 weighted, psw 1.4e-14, flux, initwgt, spinavg) and both independent 3-body
samplers (isotropic `res_xsec` and t-channel `res_exact_sigma`) agree at ~0.74×. The remaining
*untested-by-read* quantity is **the target itself**: `1.6947e-5` and the per-channel `2.178e-6`
were **measured from ACHILLES output, never read from a converged integration log**. Re-running
`run_nofsi_res.yml` to read the actual `Total xsec` + per-process lines is the decisive check
that distinguishes "ADoNIS is low" from "the target was wrong."
