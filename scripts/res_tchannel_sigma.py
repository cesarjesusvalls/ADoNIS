"""sigma_RES via ACHILLES's EXACT forward sampler (TChannelMomenta + Isotropic2Momenta) and the
bit-exact ThreeBodyMapper weight, vs res_xsec's isotropic sampler.  If the TChannel result matches
ACHILLES (channel p->p pi+ = 1.151e-5) while isotropic gives ~0.85e-5, the isotropic sampler
under-resolves the forward t-channel peak (the source of the uniform 0.73x deficit).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from adonis.xsec import constants as C
from adonis.xsec.flux import T2KFlux
from adonis.xsec.spectral import SpectralFunction, SpectralImportanceSampler
from adonis.xsec.backend import flux_factor, MASS_PDG_PROTON
from adonis.xsec.dcc_current import exclusive_amps2_batch
from scripts.validate_res_psw import three_body_genweight, m2 as _m2

MN = C.mN; M_MU = 105.7; M_P = 938.27; M_PIP = 139.57018
_alpha = 0.9; _ctmin = -1.0; _ctmax = 1.0; _amct = 1.0
N_NUC = 6; SPIN = 0.5
_SF = SpectralFunction("data/Spectral_Functions/pke12n_tot.data")
_IMP = SpectralImportanceSampler(_SF)


def sqlam(s, a, b):
    v = (s - a - b) ** 2 - 4 * a * b
    return np.where(v > 0, np.sqrt(np.clip(v, 0, None)) / s, 0.0)


def Tj1(cn, amcxm, amcxp, ran):
    ce = 1.0 - cn
    return (ran * amcxm ** ce + (1.0 - ran) * amcxp ** ce) ** (1.0 / ce)


def boost(lflag, q, ph):
    """ACHILLES Boost: lflag=1 lab->cm(q), lflag=0 cm(q)->lab.  q,ph: (N,4)."""
    rsq = np.sqrt(np.clip(q[:, 0] ** 2 - np.sum(q[:, 1:] ** 2, axis=1), 1e-9, None))
    qv = q[:, 1:]; pv = ph[:, 1:]
    if lflag == 1:
        E = (q[:, 0] * ph[:, 0] - np.sum(qv * pv, axis=1)) / rsq
        c1 = (E + ph[:, 0]) / (rsq + q[:, 0])
        vec = pv - c1[:, None] * qv
    else:
        E = (q[:, 0] * ph[:, 0] + np.sum(qv * pv, axis=1)) / rsq
        c1 = (ph[:, 0] + E) / (rsq + q[:, 0])
        vec = pv + c1[:, None] * qv
    return np.concatenate([E[:, None], vec], axis=1)


def tchannel_momenta(p1in, p2in, s1out, s2out, ran1, ran2):
    """Forward TChannelMomenta: total->p1out(mass^2 s1out)+p2out(mass^2 s2out), t_mass=0."""
    pin = p1in + p2in
    s = pin[:, 0] ** 2 - np.sum(pin[:, 1:] ** 2, axis=1); sabs = np.sqrt(np.clip(s, 1e-9, None))
    s1in = np.clip(p1in[:, 0] ** 2 - np.sum(p1in[:, 1:] ** 2, axis=1), 0, None)
    s2in = np.clip(p2in[:, 0] ** 2 - np.sum(p2in[:, 1:] ** 2, axis=1), 0, None)
    p1inhE = (s + s1in - s2in) / (2 * sabs); p1inmass = sabs * sqlam(s, s1in, s2in) / 2
    p1outhE = (s + s1out - s2out) / (2 * sabs); p1outmass = sabs * sqlam(s, s1out, s2out) / 2
    a = (-s1in - s1out + 2 * p1outhE * p1inhE) / (2 * p1inmass * p1outmass)
    a = np.maximum(a, 1.0 + 1e-6); a = np.maximum(a, _amct)
    aminct = Tj1(_alpha, a - _ctmin, a - _ctmax, ran1); ct = a - aminct
    st = np.sqrt(np.clip(1 - ct ** 2, 0, None)); phi = 2 * np.pi * ran2
    # p1out in CM, built at angle ct from the p1in CM direction
    p1in_cm = boost(1, pin, p1in)
    din = p1in_cm[:, 1:] / np.linalg.norm(p1in_cm[:, 1:], axis=1, keepdims=True)
    # orthonormal frame around din
    ref = np.tile(np.array([0.0, 0.0, 1.0]), (len(din), 1))
    alt = np.tile(np.array([1.0, 0.0, 0.0]), (len(din), 1))
    ref = np.where(np.abs(din[:, 2:3]) > 0.9, alt, ref)
    e1 = np.cross(ref, din); e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(din, e1)
    dirv = (ct[:, None] * din + (st * np.cos(phi))[:, None] * e1 + (st * np.sin(phi))[:, None] * e2)
    p1out_cm = np.concatenate([p1outhE[:, None], p1outmass[:, None] * dirv], axis=1)
    p1out = boost(0, pin, p1out_cm)
    p2out = pin - p1out
    return p1out, p2out


def iso2_momenta(p, s1, s2, ran1, ran2):
    """Isotropic 2-body: p (mass^2=p.M2) -> q1(s1)+q2(s2) isotropic in p rest frame."""
    sM = p[:, 0] ** 2 - np.sum(p[:, 1:] ** 2, axis=1); rs = np.sqrt(np.clip(sM, 1e-9, None))
    E1 = (sM + s1 - s2) / (2 * rs); pc = rs * sqlam(sM, s1, s2) / 2
    ct = 2 * ran1 - 1; st = np.sqrt(np.clip(1 - ct ** 2, 0, None)); ph = 2 * np.pi * ran2
    d = np.stack([st * np.cos(ph), st * np.sin(ph), ct], axis=1)
    q1c = np.concatenate([E1[:, None], pc[:, None] * d], axis=1)
    q2c = np.concatenate([np.sqrt(s2 + pc ** 2)[:, None], -pc[:, None] * d], axis=1)
    return boost(0, p, q1c), boost(0, p, q2c)


def sigma_tchannel(n, seed, itiz=+1, ppid=211, m_Nf=M_P, mpi=M_PIP, mstr=MASS_PDG_PROTON):
    rng = np.random.default_rng(seed); u = rng.random((n, 6))
    flux = T2KFlux(); maxE = flux.max_energy
    Smin = (M_MU + m_Nf + mpi) ** 2
    minE = max((Smin - m_Nf ** 2) / (2 * m_Nf) / 1000.0, flux.min_energy)
    E_GeV = u[:, 0] * (maxE - minE) + minE; Enu = E_GeV * 1000.0
    k_nu = np.stack([Enu, np.zeros(n), np.zeros(n), Enu], axis=1)
    J_beam = ((maxE - minE) * flux.f(E_GeV)) / flux.flux_integral
    pvec, E_rm = _IMP.sample(n, rng)
    mom = np.linalg.norm(pvec, axis=1)
    p_struck = np.concatenate([(MN - E_rm)[:, None], pvec], axis=1)
    P = k_nu + p_struck
    s = P[:, 0] ** 2 - np.sum(P[:, 1:] ** 2, axis=1); sqrts = np.sqrt(np.clip(s, 1e-9, None))
    s23min = (M_MU + m_Nf) ** 2; s23max = (sqrts - mpi) ** 2
    s23 = s23min + (s23max - s23min) * u[:, 1]
    # emax constraint (match ACHILLES)
    det_e = Enu ** 2 + mom ** 2 + 2 * pvec[:, 2] * Enu + Smin
    emax = MN + Enu - np.sqrt(np.clip(det_e, 0, None)); emax = np.minimum(np.minimum(emax, MN - mom), 400.0)
    valid = (s > Smin) & (s23max > s23min) & (E_rm < emax) & (E_rm > 2.5)
    # TChannel: total -> (muN, s23) + pi
    p23, p_pi = tchannel_momenta(k_nu, p_struck, s23, mpi ** 2, u[:, 2], u[:, 3])
    k_mu, p_N = iso2_momenta(p23, M_MU ** 2, m_Nf ** 2, u[:, 4], u[:, 5])
    # bit-exact weight = 1/ThreeBodyGenerateWeight (per event)
    gw = np.array([three_body_genweight(p_struck[i], k_nu[i], p_N[i], p_pi[i], k_mu[i]) for i in range(n)])
    a2 = np.zeros(n)
    idx = np.where(valid & (gw > 0))[0]
    if len(idx):
        a2[idx] = exclusive_amps2_batch(k_nu[idx], k_mu[idx], p_struck[idx], p_N[idx], p_pi[idx], itiz, ppid)
    fl = np.asarray(flux_factor(k_nu, p_struck, had_mass=mstr))
    w = np.where(valid & (gw > 0) & (a2 > 0), a2 * fl * N_NUC * SPIN * J_beam / np.clip(gw, 1e-300, None), 0.0)
    w = np.where(np.isfinite(w), w, 0.0)
    return w.mean()


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
    sigs = np.array([sigma_tchannel(n, seed=s) for s in range(4)])
    print(f"p->p pi+ TChannel sampler: sigma = {sigs.mean():.4e}  sem {sigs.std()/2:.2e}")
    print(f"  vs ACHILLES 1.151e-5 (ratio {sigs.mean()/1.151e-5:.3f}); res_xsec isotropic ~8.5e-6 (0.74)")
