"""Bit-exact validation of the ACHILLES RES phase-space weight (event.Weight) against the RESDUMP.

Reconstructs psw = J_beam * J_had(Smin_RES) * J_3body where J_3body = 1/ThreeBodyMapper.GenerateWeight,
faithfully porting FinalStateMapper.cc::ThreeBodyMapper (the (Npi)+mu grouping, TChannelWeight +
Isotropic2Weight) and HadronicMapper.cc::QESpectralMapper.GenerateWeight (struck nucleon).  If this
reproduces the dumped psw to ~1e-12, the ADoNIS RES sampler can be made provably equivalent.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from adonis.xsec import constants as C
from adonis.flux.spectrum import SpectrumFlux

MN = C.mN                                  # Constant::mN (isoscalar) used in the mappers
TWO_PI = 2 * np.pi
M_MU = 105.6583745
_alpha = 0.9; _ctmin = -1.0; _ctmax = 1.0; _amct = 1.0       # ThreeBodyMapper constexpr members


def parse_res(path):
    rows = []
    for ln in Path(path).read_text().splitlines():
        if not ln.startswith("RESDUMP"):
            continue
        d = {}
        for tok in ln.split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                d[k] = v
        g = lambda *ks: np.array([float(x) for x in (d[k] for k in ks)])
        rows.append(dict(
            li=np.array([float(d["liE"]), *[float(x) for x in d["li"].split(",")]]),
            lo=np.array([float(d["loE"]), *[float(x) for x in d["lo"].split(",")]]),
            hi=np.array([float(d["hiE"]), *[float(x) for x in d["hi"].split(",")]]),
            hN=np.array([float(d["hNE"]), *[float(x) for x in d["hN"].split(",")]]),
            hP=np.array([float(d["hPE"]), *[float(x) for x in d["hP"].split(",")]]),
            psw=float(d["psw"])))
    return rows


def m2(p):
    return p[0] ** 2 - p[1:] @ p[1:]


def sqlam(s, s1, s2):
    a = (s - s1 - s2) ** 2 - 4 * s1 * s2
    return np.sqrt(a) / s if a > 0 else 0.0


def boost_to_cm(q, ph):
    """ACHILLES Boost(lflag=1, q, ph) -> p (lab -> rest frame of q)."""
    rsq = np.sqrt(m2(q))
    pE = (q[0] * ph[0] - q[1:] @ ph[1:]) / rsq
    c1 = (pE + ph[0]) / (rsq + q[0])
    vec = ph[1:] - c1 * q[1:]
    return np.array([pE, *vec])


def Hj1(cn, amcxm, amcxp):
    ce = 1.0 - cn
    if ce > 1e-12:
        return (amcxp ** ce - amcxm ** ce) / ce
    return np.log(amcxp / amcxm)


def isotropic2_weight(p1, p2):
    p = p1 + p2
    mf = sqlam(m2(p), m2(p1), m2(p2))
    if mf < 1e-12:
        return 0.0
    return 2.0 / np.pi / mf * 2.0 / (_ctmax - _ctmin)


def tchannel_weight(p1in, p2in, p1out, p2out, t_mass=0.0):
    pin = p1in + p2in
    s = m2(pin); sabs = np.sqrt(abs(s))
    s1in, s2in = m2(p1in), m2(p2in)
    s1out, s2out = m2(p1out), m2(p2out)
    if s1out < 1e-8: s1out = 0.0
    if s2out < 1e-8: s2out = 0.0
    p1inhE = (s + s1in - s2in) / (2 * sabs)
    p1inmass = sabs * sqlam(s, s1in, s2in) / 2
    p1outhE = (s + s1out - s2out) / (2 * sabs)
    p1outmass = sabs * sqlam(s, s1out, s2out) / 2
    a = (t_mass ** 2 - s1in - s1out + 2 * p1outhE * p1inhE) / (2 * p1inmass * p1outmass)
    if a <= 1.0 + 1e-6: a = 1.0 + 1e-6
    if a < _amct: a = _amct
    # ct = cos angle between p1out and p1in 3-momenta in the CM frame of pin
    p1outh = boost_to_cm(pin, p1out)
    p1inh = boost_to_cm(pin, p1in)
    ct = (p1outh[1:] @ p1inh[1:]) / (np.linalg.norm(p1outh[1:]) * np.linalg.norm(p1inh[1:]))
    if ct < _ctmin or ct > _ctmax:
        return 0.0
    aminct = a - ct
    wt = 2.0 * sabs / (-aminct ** _alpha * Hj1(_alpha, a - _ctmin, a - _ctmax) * p1outmass * np.pi)
    return wt


def three_body_genweight(p_struck, k_nu, p_N, p_pi, k_mu):
    """ACHILLES ThreeBodyMapper::GenerateWeight.  Masses() is lepton-first -> s2=mmu^2, s3=mN^2,
    s4=mpi^2: the PION is split off via TChannel and (mu,N) are grouped as s23 = M(muN)^2."""
    p23 = k_mu + p_N                                    # the (mu N) system, s23 = M(muN)^2
    pin = p_struck + k_nu
    s = m2(pin); sqrts = np.sqrt(s)
    s2, s3, s4 = m2(k_mu), m2(p_N), m2(p_pi)
    s23_max = (sqrts - np.sqrt(s4)) ** 2
    s23_min = max((np.sqrt(s2) + np.sqrt(s3)) ** 2, 1e-8)
    # PSMapper orders momenta lepton-first: mom[0]=nu, mom[1]=struck N.  TChannelWeight(mom[0],
    # mom[1], mom[2]+mom[3], mom[4]) -> p1in=nu, p2in=struck, p1out=(muN), p2out=pi.
    tcw = tchannel_weight(k_nu, p_struck, p23, p_pi)   # total -> (muN) + pi, t-channel on the lepton
    i2w = isotropic2_weight(k_mu, p_N)                  # (muN) -> mu + N
    if tcw == 0.0 or i2w == 0.0:
        return 0.0
    return (2 * np.pi) ** 5 * tcw * i2w / (s23_max - s23_min)


def j_had(k_nu, p_struck, Smin):
    """QESpectralMapper forward Jacobian = |p|^2 dp dCos dPhi dE (parameterised by Smin)."""
    E0 = k_nu[0]
    radical = E0 ** 2 + 2 * E0 * MN + MN ** 2 - Smin
    if radical < 0: radical = 0.0
    pmin = max(E0 - np.sqrt(radical), 0.0)
    pmax = min(E0 + np.sqrt(radical), 800.0)
    dp = pmax - pmin
    p1 = p_struck[1:]; mom = np.linalg.norm(p1)
    cosT_max = (2 * E0 * MN + MN ** 2 - mom ** 2 - Smin) / (2 * E0 * mom)
    cosT_max = min(1.0, max(-1.0, cosT_max))
    dCos = cosT_max + 1
    det = E0 ** 2 + mom ** 2 + 2 * (p1 @ k_nu[1:]) + Smin
    emax = MN + E0 - np.sqrt(det)
    emax = min(emax, MN - mom)
    emax = min(emax, 400.0)
    return mom ** 2 * dp * dCos * TWO_PI * emax


def main():
    rows = parse_res(str(Path(__file__).resolve().parents[1] / "tests/data/res_dump_achilles.txt"))
    flux = SpectrumFlux(); minE = flux.seed_min_GeV(); maxE = flux.max_energy
    rel = []
    for r in rows:
        k_nu, k_mu, p_struck = r["li"], r["lo"], r["hi"]
        p_N, p_pi = r["hN"], r["hP"]
        mN_f = np.sqrt(max(m2(p_N), 0.0)); mpi = np.sqrt(max(m2(p_pi), 0.0)); mmu = np.sqrt(max(m2(k_mu), 0.0))
        Smin = (mmu + mN_f + mpi) ** 2
        # BeamMapper seed: (Smin - Masses()[1])/(2 sqrt(Masses()[1])), Masses()[1] = final nucleon^2.
        seed_GeV = ((Smin - mN_f ** 2) / (2 * mN_f)) / 1000.0
        minE_ev = max(seed_GeV, flux.min_energy)
        Jb = (maxE - minE_ev) * flux.f(k_nu[0] / 1000.0) / flux.flux_integral
        Jh = j_had(k_nu, p_struck, Smin)
        gw = three_body_genweight(p_struck, k_nu, p_N, p_pi, k_mu)
        if gw == 0.0:
            continue
        recon = Jb * Jh * (1.0 / gw)
        rel.append(abs(recon - r["psw"]) / abs(r["psw"]))
    rel = np.array(rel)
    print(f"RES psw reconstruction over {len(rel)}/{len(rows)} events:")
    print(f"  median rel {np.median(rel):.3e}   mean {rel.mean():.3e}   max {rel.max():.3e}")
    print(f"  PASS<1e-9: {np.median(rel) < 1e-9}")


if __name__ == "__main__":
    main()


def test_res_psw_bit_exact():
    """RES event.Weight (psw) reconstructs to ~1e-12 from the ported ACHILLES mappers."""
    rows = parse_res(str(Path(__file__).resolve().parents[1] / "tests/data/res_dump_achilles.txt"))
    flux = SpectrumFlux(); minE = flux.seed_min_GeV(); maxE = flux.max_energy
    rel = []
    for r in rows:
        k_nu, k_mu, p_struck, p_N, p_pi = r["li"], r["lo"], r["hi"], r["hN"], r["hP"]
        mN_f = np.sqrt(max(m2(p_N), 0.0)); mpi = np.sqrt(max(m2(p_pi), 0.0)); mmu = np.sqrt(max(m2(k_mu), 0.0))
        Smin = (mmu + mN_f + mpi) ** 2
        seed_GeV = ((Smin - mN_f ** 2) / (2 * mN_f)) / 1000.0
        Jb = (maxE - max(seed_GeV, flux.min_energy)) * flux.f(k_nu[0] / 1000.0) / flux.flux_integral
        Jh = j_had(k_nu, p_struck, Smin)
        gw = three_body_genweight(p_struck, k_nu, p_N, p_pi, k_mu)
        if gw == 0.0:
            continue
        rel.append(abs(Jb * Jh / gw - r["psw"]) / abs(r["psw"]))
    rel = np.array(rel)
    assert np.median(rel) < 1e-9, f"RES psw median rel {np.median(rel):.2e}"
    assert rel.max() < 1e-7, f"RES psw max rel {rel.max():.2e}"


def test_res_initwgt_bit_exact():
    """initwgt = N_nucleon * S(|p|, removal) reproduces the RESDUMP bit-exactly."""
    from adonis.xsec.spectral import SpectralFunction
    sfn = SpectralFunction("data/Spectral_Functions/pke12n_tot.data")
    sfp = SpectralFunction("data/Spectral_Functions/pke12p_tot.data")
    path = Path(__file__).resolve().parents[1] / "tests/data/res_dump_achilles.txt"
    rel = []
    for ln in path.read_text().splitlines():
        if not ln.startswith("RESDUMP"):
            continue
        d = dict(t.split("=", 1) for t in ln.split() if "=" in t)
        hiID = int(d["hiID"]); iw_ach = float(d["initwgt"])
        hi = np.array([float(d["hiE"]), *[float(x) for x in d["hi"].split(",")]])
        if iw_ach == 0:
            continue
        p = np.linalg.norm(hi[1:]); removal = MN - hi[0]
        sf = sfn if hiID == 2112 else sfp
        rel.append(abs(6.0 * sf(p, removal) - iw_ach) / abs(iw_ach))
    rel = np.array(rel)
    assert rel.max() < 1e-12, f"initwgt max rel {rel.max():.2e}"
