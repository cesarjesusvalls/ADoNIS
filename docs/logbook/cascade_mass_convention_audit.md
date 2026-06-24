# Cascade mass-convention audit (ADoNIS vs ACHILLES)

Triggered by the QE low-p neutron deficit, root-caused to a nucleon-mass convention at the capture
boundary (ADoNIS used the average nucleon mass for the neutron energy; ACHILLES uses the physical
neutron mass). This doc maps EVERY nucleon/pion mass usage on both sides and lists the fixes.
**Priority: match ACHILLES's choices exactly (not first-principles physics).**

## ACHILLES mass constants (include/Achilles/Constants.hh)
mp=938.27208816, mn=939.56542054, **mN=(mp+mn)/2=938.91875** (average), mN2=mN²,
mpip=139.57018, mpi0=134.9764, mdelta=1232.25, meta=548.

## The ACHILLES rule
- **Particle 4-vectors + 2→2 final-state kinematics → PHYSICAL per-species** (mp/mn, mpip/mpi0, via
  `ParticleInfo::Mass()`).
- **Cross-section parametrizations, form factors, formation zone, capture-threshold subtraction →
  AVERAGE mN** (or the parametrization's own constants).

### ACHILLES site map (file:line)
| Context | ACHILLES mass | cite |
|---|---|---|
| Background nucleon 4-vec E (GenerateConfig) | physical mp/mn | Nucleus.cc:145 |
| NN elastic/inelastic GenerateMomentum ma/mb, Eacms/Ebcms, pfCMS | physical | NucleonNucleon.cc:111,120 |
| NN elastic cross-section plab/sqrts | per-pair AVERAGE (m1+m2)/2 | NucleonNucleon.cc:184 |
| Formation zone (E/|mN²−p1·p2|·ℏc) | AVERAGE mN² | Particle.cc:10 |
| Escape/capture: energy = E − mN − 10 (E is the particle's physical 4-vec E) | E physical, subtract AVERAGE mN | Cascade.cc:639 (Escaped, ungated) |
| Hamiltonian capture (PotentialProp:True only, OFF here) | AVERAGE mN | Cascade.cc:183,199 |
| NN→NΔ DSigmaDM threshold/xsec | AVERAGE (mN avg, mpi weighted-avg) | ResonanceHelper.cc:186 |
| Pion 4-vec + piN (MesonBaryon) scatter ma/mb | physical mpip/mpi0 + physical nucleon | OsetMesonBaryon.cc:87 |
| Δ→Nπ decay products | physical (mn, mpip from ParticleInfo) | ResonanceHelper.cc:25 |
| Oset absorption kinematics | AVERAGE nucleon + physical pion | OsetCrossSections.cc:33,43 |

## ADoNIS current state + divergences
Constants: adonis/constants.py — mp/mn/mN/mpip/mpi0 all correct. `M_N = ox.M_N = mN` (avg) is used as the
single nucleon mass throughout the cascade.

| # | ADoNIS site | current | ACHILLES | verdict |
|---|---|---|---|---|
| 1 | capture recap (cascade_discrete:213) | was `p4[0]−M_N` (E uses avg) | E physical − avg mN | **FIXED** → physical species E |
| 2 | background nucleon E (cascade_discrete sample_nucleons; cascade_real:155) | avg M_N | physical mp/mn | **FIX** |
| 3 | NN scatter ma/mb (cascade_discrete:262, nucleon_cascade:108 via _two_body_cm_scatter m2=M_N) | avg M_N | physical | **FIX** |
| 4 | piN scatter recoil nucleon (cascade_real:233 via _two_body_cm_scatter m2=M_N) | avg M_N | physical | **FIX** |
| 5 | Δ→Nπ pion mass (cascade_discrete:299) | hardcoded 138.04 | per-charge mpip/mpi0 | **FIX** |
| 6 | NN→NΔ mass-sampling threshold (nn_inelastic:131 MN_HEAVY=neutron) | neutron for both | AVERAGE (DSigmaDM) | **FIX** → avg |
| ✓ | formation zone mN² | avg | avg | match |
| ✓ | NN elastic plab/sqrts (nucleon_cascade) | avg | per-pair avg | match |
| ✓ | NN→NΔ DSigmaDM thresholds/xsec (nn_inelastic MN_AVG/MPI_AVG) | avg | avg | match |
| ✓ | pion 4-vec (cascade_real _CH_MASS) | physical per-charge | physical | match |
| ✓ | RES amp pion 138.039 / kin pion mpi0 (conventions) | as ACHILLES | as ACHILLES | match |
| ✓ | QE/DCC primary amplitude nucleon = avg (mqe) | avg | avg (mqe) | match (validated bit-exact — leave) |

Note: the QE/DCC **primary** vertex uses avg `mN` for the amplitude — this is ACHILLES-faithful (mqe=mN)
and validated bit-exact, so it is NOT changed. Only the **cascade** 4-vectors/kinematics are corrected.

## Fixes implemented (2026-06-24) — all match ACHILLES's choices (priority over first-principles)

All in `adonis/fsi/cascade_discrete.py` unless noted. Physical masses imported: `_MP_PHYS=mp=938.272`,
`_MN_PHYS=mn=939.565`; `M_N` stays the average (938.919) for the contexts ACHILLES averages.

1. **Capture E → physical per-species** (recap, ~L213). ACHILLES `Escaped` (Cascade.cc, every step,
   ungated): captured if `E − mN_avg − 10 < 0`, with **E the particle's PHYSICAL 4-vec energy**. ADoNIS
   now recomputes `E_phys = sqrt(m_species² + |p|²)` (mp/mn by `isp`) for the threshold, subtracting the
   AVERAGE `M_N`. Was: `p4[0] − M_N` with p4[0] carrying avg mass → over-captured neutrons in |p|∈
   (132.9, 137.4) MeV. **Validated** (identical-input ablation): ejected-n |p|<137 0→**25-30 vs ACH 32**.

2. **Background nucleon 4-vec E → physical per-species** (`sample_nucleons`, ~L175). `E = sqrt(m_species²
   + p²)`, m_species = mp/mn by `nisp`. ACHILLES Nucleus.cc:145 uses `Info().Mass()` (physical).

3. **NN scatter outgoing masses → physical per-species** (`_nucleon_step` scat_one, ~L269). Elastic:
   leading keeps `isp` mass, recoil keeps struck `nisp[j]` mass; passed via the new
   `_two_body_cm_scatter(..., m_recoil=)` (cascade_real.py). ACHILLES GenerateMomentum ma/mb physical.

4. **piN scatter recoil nucleon mass → physical per-species** (`_pion_step` scat_one, ~L517). Recoil
   species from charge-exchange (`struck_p + out_ch − ch`, same expr as its Pauli k_F); `m_recoil` =
   mp/mn. Outgoing pion already physical per-charge (`_CH_MASS`). ACHILLES OsetMesonBaryon physical.

5. **Δ→Nπ products → physical per-species/charge** (NN-inelastic, ~L288). The two nucleons get mp/mn by
   their channel charge (`q_pair−dch`, `dch−pi_q`); the pion gets `_CH_MASS[1−pi_q]` (mpip/mpi0) instead
   of the hardcoded isospin-avg 138.04. Charge computation moved before the splits — RNG fold keys
   (107/108) are order-independent → **charges bit-identical, only masses shift** (verified). ACHILLES
   decays the Δ via DecayHandler → ParticleInfo (physical) masses.

### Checked and CORRECT (left unchanged — already match ACHILLES)
- Formation zone `mN²` (avg); NN-elastic plab/sqrts (per-pair avg); NN→NΔ DSigmaDM thresholds/xsec
  (`MN_AVG`/`MPI_AVG`); pion 4-vectors (`_CH_MASS` physical per-charge); RES amp pion 138.039 / kin pion
  mpi0; QE/DCC **primary** amplitude nucleon = avg `mqe` (ACHILLES-faithful, validated bit-exact).
- `nn_inelastic.py:131` Δ-mass-range bound uses physical neutron — ACHILLES `ResonanceHelper.cc:25` also
  uses the physical neutron for the Δ threshold, so this MATCHES (not changed).

### Not fixed here (logged in docs/TODO.md)
- Legacy real cascade `_sample_fermi_nucleon` (cascade_real.py:155) uses avg M_N — legacy path
  (transparency script only); production pool uses the fixed `sample_nucleons`.
- `nucleon_cascade.py:108` `_two_body_cm_scatter(..., M_N)` — dead code (no callers).

### Validation
On-shell masses after fix (identical-input ablation): ejected neutrons mean 939.564 (=mn), protons
938.276 (=mp). Golden re-baselined (intentional physics change). QE comparison re-run after bank regen.
Residual neutron deficits at 137–200 MeV and >700 MeV SURVIVE the mass fix → separate transport residual
(not mass).
